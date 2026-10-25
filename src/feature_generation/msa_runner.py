"""Frozen MSA Transformer features and scores using validated generated pairs.

ProteinGym's weighted replacement sampler supplies one fixed context per seed.
Scores use its MSA-interval token windows and are averaged over sampling seeds.
MSA-region global means and full-query centered sites are an NKN extension.
"""

import csv
from datetime import datetime, timezone
import json
import random

import numpy as np
import torch

from data_loading.features import ALPHABET, parse_substitutions
from data_loading.sequence_windows import centered_window_bounds
from .inputs import prepare_assay, safe_component, sha256_file
from .msa_config import EMBEDDING_RECIPE, SAMPLING, SCORE_RECIPE
from . import score_storage, storage


class MSAContext:
    def __init__(self, rows, weights, columns, reference, config, seed=None):
        self.reference = reference
        self.seed = config.seeds[0] if seed is None else seed
        self.total_rows = len(rows)
        if len(columns) != len(rows[0][1]) or not columns or columns != sorted(set(columns)):
            raise ValueError("MSA target columns must be unique, increasing and match the alignment width")
        if columns[0] < 1 or columns[-1] > len(reference):
            raise ValueError("MSA column is outside the target sequence")
        if columns != list(range(columns[0], columns[-1] + 1)):
            raise ValueError("ProteinGym scoring requires a contiguous MSA interval")
        if rows[0][1] != "".join(reference[position - 1] for position in columns):
            raise ValueError("MSA query does not match mapped target residues")
        if any(len(sequence) != len(columns) or set(sequence) - set(ALPHABET + "-") for _, sequence in rows):
            raise ValueError("Processed MSA must contain aligned canonical residues or gaps")
        weights = np.asarray(weights, dtype=np.float64)
        if (weights.shape != (len(rows),) or not np.isfinite(weights).all()
                or not ((weights > 0) & (weights <= 1)).all()):
            raise ValueError("Invalid row weights")
        if len(rows) > 1:
            probabilities = weights[1:] / weights[1:].sum()
            # Match random.seed(seed); random.choices(...) in pinned ProteinGym,
            # without mutating process-global RNG state. Duplicates and draw
            # order are intentional, including when source depth is below 400.
            chosen = random.Random(self.seed).choices(
                range(1, len(rows)), weights=probabilities, k=config.max_depth - 1)
        else:
            chosen = []
        self.indices = [0, *map(int, chosen)]
        self.columns = columns
        self.start, self.end = columns[0] - 1, columns[-1]
        self.homologs = []
        for index in self.indices[1:]:
            padded = ["-"] * len(reference)
            for position, residue in zip(columns, rows[index][1]):
                padded[position - 1] = residue
            self.homologs.append((rows[index][0], "".join(padded)))

    def window(self, query, start, end, *, drop_empty_homologs=False):
        if len(query) != len(self.reference) or not 0 <= start < end <= len(query):
            raise ValueError("Invalid full-query MSA window")
        homologs = [(name, seq[start:end]) for name, seq in self.homologs]
        if drop_empty_homologs:
            # Preserve sampled order and duplicates among rows with evidence.
            # With no evidence in this window, the same model gets query only.
            homologs = [(name, seq) for name, seq in homologs if seq.strip("-")]
        return [("query", query[start:end]), *homologs]


def load_contexts(config, inputs, validate_pair):
    """Resolve exactly one run, verify its saved pair, and map its target columns."""
    safe_component(inputs.assay_id, "assay ID")
    runs = sorted((config.msa_root / "assays").glob(inputs.assay_id + "_*"))
    if len(runs) != 1:
        raise ValueError(f"Expected exactly one generated run for {inputs.assay_id}; found {len(runs)}")
    directory = runs[0]
    manifest_path = directory / "manifest.json"
    manifest_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_path = config.msa_root / "DMS_substitutions.csv"
    if not reference_path.is_file():
        reference_path = config.msa_reference_csv
    if (manifest.get("assay_id") != inputs.assay_id
            or manifest["inputs"]["reference"]["sha256"] != sha256_file(reference_path)):
        raise ValueError("Generated MSA assay/catalog identity mismatch")
    with reference_path.open(newline="", encoding="utf-8-sig") as handle:
        records = [row for row in csv.DictReader(handle) if row["DMS_id"] == inputs.assay_id]
    if len(records) != 1 or records[0]["target_seq"] != inputs.reference_sequence:
        raise ValueError("Generated MSA and extraction catalog target sequences differ")
    rows, weights = validate_pair(directory, manifest)
    with (directory / "columns.csv").open(newline="", encoding="utf-8") as handle:
        mapping = list(csv.DictReader(handle))
    if [int(row["processed_column_one_based"]) for row in mapping] != list(range(1, len(rows[0][1]) + 1)):
        raise ValueError("Processed MSA column indices are inconsistent")
    columns = [int(row["target_position_one_based"]) for row in mapping]
    expected = list(range(int(records[0]["MSA_start"]), int(records[0]["MSA_end"]) + 1))
    if columns != expected:
        raise ValueError("Generated MSA columns disagree with the catalog MSA interval")
    contexts = [MSAContext(rows, weights, columns, inputs.reference_sequence, config, seed)
                for seed in config.seeds]
    if sha256_file(manifest_path) != manifest_hash:
        raise ValueError("MSA manifest changed while being read")
    return contexts, manifest_hash, directory


def proteingym_token_window(token_position, msa_length, model_window=1024):
    """Match compute_fitness.py and get_optimal_window at commit 144fe22b07df.

    MSA-1b has BOS but no EOS. ProteinGym nevertheless passes msa_length + 2
    to its window helper, then slices the already-tokenized MSA. Preserve that
    behavior (including Python's clipped final slice) for scoring parity.
    """
    if not 1 <= token_position <= msa_length:
        raise ValueError("Score position must identify a residue in the MSA")
    if msa_length + 1 <= model_window:
        return 0, msa_length + 1
    sequence_length = msa_length + 2
    half = model_window // 2
    if token_position < half:
        return 0, model_window
    if token_position >= sequence_length - half:
        return sequence_length - model_window, sequence_length
    return max(0, token_position - half), min(sequence_length, token_position + half)


def validate_score_coverage(inputs, context):
    positions = sorted({position for row in inputs.rows for position in
                        parse_substitutions(row["mutant"], row["mutated_sequence"], inputs.reference_sequence)})
    missing = [position for position in positions if not context.start < position <= context.end]
    if missing:
        raise ValueError(f"ProteinGym scoring cannot cover mutation positions outside the MSA interval "
                         f"{context.start + 1}-{context.end}: {missing[:10]}; "
                         "no complete mean-score cache will be published")
    return positions


class MSAInference:
    def __init__(self, model, alphabet, config):
        self.model = model.requires_grad_(False).eval().to(config.device)
        self.alphabet, self.config = alphabet, config
        self.converter = alphabet.get_batch_converter()
        self.offset = int(alphabet.prepend_bos)
        if not alphabet.prepend_bos or alphabet.append_eos:
            raise ValueError("Expected the MSA-1b BOS-only token layout")
        model.max_tokens_per_msa_(config.max_tokens_per_msa)

    def forward(self, context, query, bounds, mask_position=None):
        start, end = bounds
        rows = context.window(query, start, end, drop_empty_homologs=mask_position is None)
        _, _, tokens = self.converter([rows])
        if tokens.shape != (1, len(rows), end - start + self.offset + int(self.alphabet.append_eos)):
            raise ValueError("MSA tokenizer changed alignment dimensions")
        column = None
        if mask_position is not None:
            if not start < mask_position <= end:
                raise ValueError("Masked position is outside its window")
            column = self.offset + mask_position - 1 - start
            tokens[0, 0, column] = self.alphabet.mask_idx
        with torch.inference_mode():
            result = self.model(tokens.to(self.config.device), repr_layers=[12] if column is None else [])
            values = (result["representations"][12][0, 0, self.offset:self.offset + end - start]
                      if column is None else result["logits"][0, 0, column].float().log_softmax(-1))
            values = values.detach().to(device="cpu", dtype=torch.float32).clone()
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite MSA Transformer output")
        if column is None and values.shape != (end - start, 768):
            raise ValueError("Unexpected MSA Transformer representation shape")
        return values

    def embeddings(self, inputs, contexts, progress):
        """Average NKN global/site features over the same contexts as scoring."""
        total = packed = None
        for context in contexts:
            progress(f"{inputs.assay_id}: embedding MSA sample seed {context.seed}")
            values, sites = self._embeddings_one(inputs, context, progress)
            if total is None:
                total, packed = values, sites
            else:
                total.add_(values)
                packed["site_embeddings"].add_(sites["site_embeddings"])
            del values, sites
        total.div_(len(contexts))
        packed["site_embeddings"].div_(len(contexts))
        return total, packed

    def _embeddings_one(self, inputs, context, progress):
        length, width = inputs.length, self.config.max_length
        mutations = [parse_substitutions(row["mutant"], row["mutated_sequence"], inputs.reference_sequence) for row in inputs.rows]
        # Every variant uses the same columns and boundaries inside MSA coverage.
        # Site windows below still use full-reference coordinates.
        global_bounds = [(start, min(start + width, context.end))
                         for start in range(context.start, context.end, width)]
        # Reuse WT sums only for windows whose query is unchanged. A mutation in
        # a different window cannot affect a separate model invocation.
        wt_sums = {}
        embeddings = torch.empty((len(inputs.rows), 768), dtype=torch.float32)
        positions = [position for sites in mutations for position in sites]
        bounds = [centered_window_bounds(position, length, width) for position in positions]
        site_data = {
            "site_embeddings": torch.empty((len(positions), 768), dtype=torch.float32),
            "site_variant_indices": torch.tensor([i for i, sites in enumerate(mutations) for _ in sites], dtype=torch.long),
            "site_positions": torch.tensor(positions, dtype=torch.long),
            "site_window_starts": torch.tensor([start for start, _ in bounds], dtype=torch.long),
            "site_window_ends": torch.tensor([end for _, end in bounds], dtype=torch.long),
        }
        cursor = 0
        for index, (sequence, sites) in enumerate(zip(inputs.sequences, mutations)):
            site_bounds = {centered_window_bounds(position, length, width) for position in sites}
            retained = {}
            total = torch.zeros(768, dtype=torch.float64)
            for window in global_bounds:
                start, end = window
                changed = any(start < position <= end for position in sites)
                if not changed and window in wt_sums and window not in site_bounds:
                    total += wt_sums[window]
                    continue
                values = self.forward(context, sequence, window)
                summed = values.sum(0, dtype=torch.float64)
                total += summed
                if not changed:
                    wt_sums[window] = summed
                if window in site_bounds:
                    retained[window] = values
            # Sum residue vectors, then divide by covered length: exactly the
            # length-weighted mean of window means, including a short last one.
            embeddings[index] = total / (context.end - context.start)
            for position in sites:
                window = centered_window_bounds(position, length, width)
                if window not in retained:
                    retained[window] = self.forward(context, sequence, window)
                site_data["site_embeddings"][cursor] = retained[window][position - 1 - window[0]]
                cursor += 1
            if (index + 1) % 25 == 0 or index + 1 == len(inputs.rows):
                progress(f"{inputs.assay_id}: embedded {index + 1}/{len(inputs.rows)} variants")
        return embeddings, site_data

    def scores(self, inputs, contexts, progress):
        mutations = [parse_substitutions(row["mutant"], row["mutated_sequence"], inputs.reference_sequence) for row in inputs.rows]
        total = torch.zeros(len(inputs.rows), dtype=torch.float64)
        for context in contexts:
            positions = validate_score_coverage(inputs, context)
            # Crop to MSA coverage BEFORE tokenizing, as in ProteinGym. Middle
            # and tail windows are slices of these tokens: do not add a new BOS.
            _, _, tokens = self.converter([context.window(inputs.reference_sequence, context.start, context.end)])
            msa_length = context.end - context.start
            if tuple(tokens.shape) != (1, len(context.indices), msa_length + 1):
                raise ValueError("Unexpected MSA scoring token layout")
            log_probs = {}
            for count, position in enumerate(positions, 1):
                column = position - context.start
                start, end = proteingym_token_window(column, msa_length, self.config.scoring_token_window)
                masked = tokens[:, :, start:end].clone()
                masked[0, 0, column - start] = self.alphabet.mask_idx
                with torch.inference_mode():
                    logits = self.model(masked.to(self.config.device), repr_layers=[])["logits"]
                    probs = logits[0, 0, column - start].float().log_softmax(-1)
                    log_probs[position] = probs.detach().cpu().clone()
                del logits, probs, masked
                if not torch.isfinite(log_probs[position]).all():
                    raise ValueError("Nonfinite MSA Transformer score")
                if count % 25 == 0 or count == len(positions):
                    progress(f"{inputs.assay_id}: seed {context.seed}: scored {count}/{len(positions)} WT positions")
            # ProteinGym label_row subtracts float32 logits, converts each term
            # to a Python float, and averages the per-seed scores in pandas.
            values = [sum(float(log_probs[position][self.alphabet.get_idx(sequence[position - 1])]
                                - log_probs[position][self.alphabet.get_idx(inputs.reference_sequence[position - 1])])
                          for position in sites)
                      for sequence, sites in zip(inputs.sequences, mutations)]
            total.add_(torch.tensor(values, dtype=torch.float64))
        return total / len(contexts)


def prepare(config, assay_ids, runtime_report, inference, validate_pair, *, resume=False, progress=print):
    """Prepare all selected single-mutant caches; retain failures in the result."""
    results = []
    for assay_id in assay_ids:
        embeddings = site_data = scores = None
        try:
            inputs = prepare_assay(config.assay_index, assay_id, singles_only=True)
            contexts, manifest_hash, directory = load_contexts(config, inputs, validate_pair)
            context = contexts[0]
            num_sites = sum(len(parse_substitutions(row["mutant"], row["mutated_sequence"], inputs.reference_sequence)) for row in inputs.rows)
            output_bytes = (len(inputs.rows) + num_sites) * 768 * 4 + num_sites * 4 * 8
            if output_bytes > config.max_output_gib * 1024**3:
                raise ValueError("MSA embeddings exceed max_output_gib")
            common = {
                "model_id": "esm_msa1b_t12_100M_UR50S", "layer": 12, "hidden_size": 768,
                "msa_recipe_version": 2, "inference_dtype": "float32",
                "checkpoint_sha256": runtime_report["checkpoint"]["sha256"],
                "proteingym_commit": runtime_report["proteingym_commit"],
                "packages": runtime_report["packages"],
                "msa_manifest_sha256": manifest_hash,
                "sampled_row_indices": [sample.indices for sample in contexts],
                "msa_total_rows": context.total_rows,
                "sampling": SAMPLING, "sampling_rng": "python_random_choices",
                "max_depth": config.max_depth, "seeds": list(config.seeds),
                "max_length": config.max_length, "max_tokens_per_msa": config.max_tokens_per_msa,
                "msa_covered_positions": context.columns,
            }
            feature_config = storage.make_leaf_config(config.cache_id, {
                **common, **EMBEDDING_RECIPE,
                "global_region_start": context.start, "global_region_end": context.end,
                "pooling": "mean", "truncation": "none", "includes_special_tokens": False,
                "uncovered_positions": config.uncovered_positions,
                "embedding_aggregation": "arithmetic_mean_over_sampling_seeds",
                "window_policy": "centered_clamped", "window_indexing": "zero_based_half_open",
                "global_window": "non_overlapping", "global_window_aggregation": "length_weighted_mean",
                "site_embeddings": "centered", "cache_layout": "mutation_sites_v1", "site_context": "mutated_sequence",
            }, model_family="msa_transformer")
            score_config = score_storage.make_score_config(config.score_cache_id, "msa_transformer", {**common, **SCORE_RECIPE})
            provenance = {
                "uses_dms_labels": False, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "device": config.device, "msa_directory": str(directory),
                "source_assay_sha256": inputs.source_assay_sha256,
                "source_index_sha256": inputs.source_index_sha256,
                "msa_manifest_sha256": manifest_hash,
            }
            root = config.output_root / assay_id
            progress(f"{assay_id}: {len(inputs.rows)} variants; MSA depth {len(context.indices)}; "
                     f"seeds {list(config.seeds)}; covered {len(context.columns)}/{inputs.length} target residues")
            stages = {}
            # Each leaf can resume independently after an interrupted job.
            for stage in ("sequence", "scores"):
                try:
                    if stage == "sequence":
                        exists = storage.check_existing(root, config.cache_id, inputs, feature_config, resume)
                        if not exists:
                            embeddings, site_data = inference.embeddings(inputs, contexts, progress)
                            storage.write_sequence_cache(root, config.cache_id, inputs, feature_config,
                                                         embeddings, provenance, site_data=site_data)
                        storage.validate_sequence_cache(root, config.cache_id, "msa_transformer")
                    else:
                        exists = score_storage.check_existing(root, "msa_transformer", config.score_cache_id, inputs, score_config, resume)
                        if not exists:
                            scores = inference.scores(inputs, contexts, progress)
                            score_storage.write_score_cache(root, "msa_transformer", config.score_cache_id,
                                                            inputs, score_config, scores, provenance)
                        score_storage.validate_score_cache(root, "msa_transformer", config.score_cache_id)
                    stages[stage] = "skipped" if exists else "completed"
                except (OSError, ValueError, RuntimeError, KeyError) as error:
                    stages[stage] = f"failed: {error}"
                    progress(f"{assay_id}: {stage} {stages[stage]}")
                finally:
                    embeddings = site_data = scores = None
            result = {"assay_id": assay_id, "status": "failed" if any(value.startswith("failed") for value in stages.values()) else "completed", **stages}
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError, EOFError, csv.Error) as error:
            result = {"assay_id": assay_id, "status": "failed", "error": str(error)}
        results.append(result)
        progress(json.dumps(result))
    return results
