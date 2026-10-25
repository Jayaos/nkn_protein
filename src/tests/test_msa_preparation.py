"""MSA-to-NKN integration with synthetic alignments and a deterministic model."""

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from config import ExperimentConfig
from data_loading import load_feature_bundle
from data_loading.sequence_windows import COMPACT_SITE_FIELDS, compact_sequence_recipe, validate_compact_sequence_payload
from feature_generation.inputs import prepare_assay, sha256_file
from feature_generation.msa_config import EMBEDDING_RECIPE, LEGACY_SCORE_RECIPE, MSAConfig, validate_msa_metadata
from feature_generation.msa_runner import MSAContext, MSAInference, load_contexts, prepare, proteingym_token_window
from feature_generation.storage import check_existing, make_leaf_config, validate_sequence_cache, write_sequence_cache
from feature_generation.score_storage import make_score_config, validate_score_cache
from gp.model import ExactProteinGP
from gp.preprocessing import FoldTransform

ROOT = Path(__file__).resolve().parents[2]
HELPERS = ROOT / "sbatch_run/run_prepare_rep"
sys.path.insert(0, str(HELPERS))
from generated_msa import validate_pair


def write_csv(path, names, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        writer.writerows(rows)


class Alphabet:
    prepend_bos, append_eos, mask_idx = True, False, 2
    tokens = "ACDEFGHIKLMNPQRSTVWY-"

    def get_idx(self, residue):
        return self.tokens.index(residue) + 3

    def get_batch_converter(self):
        def convert(batch):
            return None, None, torch.tensor([
                [[1, *[self.get_idx(residue) for residue in seq]] for _, seq in msa] for msa in batch
            ])
        return convert


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def max_tokens_per_msa_(self, value):
        self.chunk_size = value

    def forward(self, tokens, repr_layers):
        assert not self.training and not torch.is_grad_enabled()
        self.calls.append((tokens.clone(), bool(repr_layers)))
        values = torch.zeros((*tokens.shape, 768))
        values[..., 0] = tokens.float()
        values[..., 1] = torch.arange(tokens.shape[-1]).float()
        values[..., 2] = tokens[:, 1:].sum(1).unsqueeze(1)
        # Make predictions depend on the sampled homologs and masked context.
        scale = (tokens.sum(dim=(1, 2)) % 5 + 1).float()
        logits = (torch.arange(24).float() * scale[:, None, None, None]).expand(*tokens.shape, 24)
        return {"representations": {12: values}, "logits": logits}


class MSAPreparationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.index = self.root / "DMS_substitutions.csv"
        write_csv(self.index, ["DMS_id", "DMS_filename", "target_seq", "seq_len", "MSA_start", "MSA_end"],
                  [["tiny", "tiny.csv", "ACDEFG", 6, 2, 5]])
        write_csv(self.root / "tiny.csv", ["variant_id", "mutant", "mutated_sequence", "DMS_score"], [
            ["head", "C2V", "AVDEFG", 0.1], ["middle", "E4W", "ACDWFG", 0.2],
            ["tail", "F5A", "ACDEAG", 0.3], ["double", "C2V:F5A", "AVDEAG", 0.4],
        ])
        self.config = MSAConfig("features-test", "scores-test", assay_index=self.index,
                                msa_root=self.root / "pairs", msa_reference_csv=self.index,
                                output_root=self.root / "features", max_depth=3, max_length=4,
                                seeds=(1, 2, 3, 4, 5), device="cpu")
        self.directory = self.config.msa_root / "assays/tiny_run"
        self.directory.mkdir(parents=True)
        self.rows = [(">query", "CDEF"), (">homolog1", "CD-F"), (">homolog2", "-DEF")]
        self.weights = np.array([1.0, 0.5, 0.75])
        self.save_pair()
        self.inputs = prepare_assay(self.index, "tiny", singles_only=True)
        self.model = Model()
        self.inference = MSAInference(self.model, Alphabet(), self.config)
        self.report = {"checkpoint": {"sha256": "a" * 64}, "proteingym_commit": "b" * 40, "packages": {"torch": "fixture"}}

    def save_pair(self):
        (self.directory / "alignment.a2m").write_text("".join(f"{name}\n{seq}\n" for name, seq in self.rows))
        np.save(self.directory / "weights.npy", self.weights)
        write_csv(self.directory / "rows.csv", ["weight_index_zero_based", "processed_header", "processed_sequence_sha256", "recomputed_weight"], [
            [i, name, hashlib.sha256(seq.encode()).hexdigest(), self.weights[i]] for i, (name, seq) in enumerate(self.rows)
        ])
        write_csv(self.directory / "columns.csv", ["processed_column_one_based", "raw_column_one_based", "target_position_one_based"],
                  [[i, i, i + 1] for i in range(1, 5)])
        manifest = {"status": "generated", "assay_id": "tiny", "processed_rows": len(self.rows), "processed_columns": 4,
                    "inputs": {"reference": {"sha256": sha256_file(self.index)}},
                    "artifacts": {name: sha256_file(self.directory / name) for name in ("alignment.a2m", "weights.npy", "rows.csv", "columns.csv")}}
        (self.directory / "manifest.json").write_text(json.dumps(manifest))

    def run_preparation(self, **kwargs):
        return prepare(kwargs.pop("config", self.config), kwargs.pop("assays", ["tiny"]), self.report,
                       self.inference, validate_pair, progress=lambda _: None, **kwargs)

    def test_saved_pair_mapping_preserves_full_query_and_gap_pads_homologs(self):
        contexts, _, _ = load_contexts(self.config, self.inputs, validate_pair)
        context = contexts[0]
        self.assertEqual(context.window("VCDEFG", 0, 6), [
            ("query", "VCDEFG"), (">homolog1", "-CD-F-"), (">homolog2", "--DEF-"),
        ])
        self.assertEqual(context.window("ACDEFG", 2, 6)[1][1], "D-F-")

    def test_sampling_matches_proteingym_with_replacement_and_preserves_draw_order(self):
        # Golden first draws of ProteinGym's normalized non-query weighted sampler.
        expected = [[1, 2, 2, 1, 2, 2, 2, 2], [2, 2, 1, 1, 2, 2, 2, 1],
                    [1, 2, 1, 2, 2, 1, 1, 2], [1, 1, 1, 1, 1, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 1, 2]]
        cfg = replace(self.config, max_depth=400)
        rng_state = random.getstate()
        for seed, first_draws in zip(cfg.seeds, expected):
            first = MSAContext(self.rows, self.weights, [2, 3, 4, 5], "ACDEFG", cfg, seed)
            changed_weights = self.weights.copy()
            changed_weights[0] = 0.00001
            second = MSAContext(self.rows, changed_weights, [2, 3, 4, 5], "ACDEFG", cfg, seed)
            self.assertEqual(first.indices, second.indices)
            self.assertEqual(len(first.indices), 400)
            self.assertEqual(first.indices[:9], [0, *first_draws])
            self.assertEqual(first.indices.count(0), 1)
        self.assertEqual(random.getstate(), rng_state)

    def test_query_only_and_reject_wrong_mapping(self):
        context = MSAContext(self.rows[:1], self.weights[:1], [2, 3, 4, 5], "ACDEFG", self.config)
        self.assertEqual(context.window("ACDEFG", 0, 6), [("query", "ACDEFG")])
        for columns in ([1, 2, 3, 4], [2, 3, 3, 5], [2, 3, 4, 7]):
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                MSAContext(self.rows, self.weights, columns, "ACDEFG", self.config)

    def test_global_site_and_scores_round_trip_through_nkn(self):
        result = self.run_preparation()[0]
        self.assertEqual(result["status"], "completed", result)
        root = self.config.output_root / "tiny"
        self.assertEqual(validate_sequence_cache(root)["embedding_dim"], 768)
        self.assertEqual(validate_score_cache(root, "msa_transformer")["num_variants"], 3)
        bundle = load_feature_bundle(root, enabled_branches=["sequence_global", "sequence_site"],
                                     feature_selection={"sequence": "msa_transformer/features-test"},
                                     score_sources=["scores/msa_transformer/scores-test"])
        self.assertEqual(bundle.variant_ids, ("head", "middle", "tail"))
        expected = torch.tensor([np.mean([Alphabet().get_idx(aa) for aa in sequence[1:5]]) for sequence in self.inputs.sequences])
        torch.testing.assert_close(bundle.global_features[:, 0], expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(bundle.global_features[:, 1], torch.full((3,), 2.5, dtype=torch.float64))
        torch.testing.assert_close(bundle.local_features["sequence_site"][:, 1], torch.tensor([2., 3., 3.], dtype=torch.float64))
        with (root / "scores/msa_transformer/scores-test/mean_score.csv").open() as handle:
            scores = [float(row["score"]) for row in csv.DictReader(handle)]
        masked = [tokens for tokens, representations in self.model.calls if not representations]
        self.assertEqual(len(masked), 15)
        contexts, _, _ = load_contexts(self.config, self.inputs, validate_pair)
        seed_scores, global_contexts, site_contexts = [], [], []
        for seed_index, context in enumerate(contexts):
            per_seed = []
            for i, (column, old, new) in enumerate(((1, "C", "V"), (3, "E", "W"), (4, "F", "A"))):
                _, _, expected_tokens = Alphabet().get_batch_converter()([context.window("ACDEFG", 1, 5)])
                expected_tokens[0, 0, column] = Alphabet.mask_idx
                torch.testing.assert_close(masked[seed_index * 3 + i], expected_tokens)
                scale = float(expected_tokens.sum() % 5 + 1)
                per_seed.append((Alphabet().get_idx(new) - Alphabet().get_idx(old)) * scale)
            seed_scores.append(per_seed)
            homologs = context.window("ACDEFG", 0, 6)[1:]
            context_sum = np.array([[Alphabet().get_idx(aa) for aa in seq] for _, seq in homologs]).sum(0)
            global_contexts.append(context_sum[1:5].mean())
            site_contexts.append(context_sum[[1, 3, 4]])
        self.assertGreater(len({tuple(values) for values in seed_scores}), 1)
        np.testing.assert_allclose(scores, np.mean(seed_scores, axis=0), atol=1e-5)
        # The generated ProteinGym scores must reach the GP mean, preserving
        # variant order and using only training-fold scores for normalization.
        expected_scores = torch.tensor(np.mean(seed_scores, axis=0), dtype=torch.float64)
        self.assertEqual(bundle.score_names, ("scores/msa_transformer/scores-test",))
        torch.testing.assert_close(bundle.scores[:, 0], expected_scores, atol=1e-5, rtol=1e-6)
        train, heldout = bundle.select([0, 1]), bundle.select([2])
        transform = FoldTransform.fit(train)
        experiment = ExperimentConfig.load(ROOT / "configs/msa_transformer_nkn.json")
        gp = ExactProteinGP(transform.apply(train), experiment.model, experiment.training)
        with torch.no_grad():
            gp.intercept.fill_(0.4)
            gp.coefficients.fill_(1.7)
        expected_mean = 0.4 + 1.7 * (expected_scores[2:] - expected_scores[:2].mean()) / expected_scores[:2].std(unbiased=False)
        torch.testing.assert_close(gp.mean(transform.apply(heldout)), expected_mean, atol=1e-5, rtol=1e-6)
        np.testing.assert_allclose(bundle.global_features[:, 2], np.mean(global_contexts), atol=1e-5)
        np.testing.assert_allclose(bundle.local_features["sequence_site"][:, 2], np.mean(site_contexts, axis=0), atol=1e-5)

    def test_fixed_msa_global_windows_weight_short_tail_and_keep_outside_mutations_in_sites(self):
        reference = Alphabet.tokens[:-1]
        config = replace(self.config, seeds=(1,), max_depth=2)
        # Nonzero MSA start; shorter than, equal to, and longer than max_length.
        for msa_length in (3, 4, 7):
            with self.subTest(msa_length=msa_length):
                start, end = 6, 6 + msa_length
                positions = [1, 6, 7, end, end + 1, len(reference)]
                rows = []
                for position in positions:
                    old = reference[position - 1]
                    new = "V" if old != "V" else "A"
                    sequence = reference[:position - 1] + new + reference[position:]
                    rows.append({"mutant": f"{old}{position}{new}", "mutated_sequence": sequence})
                inputs = replace(self.inputs, reference_sequence=reference, rows=rows)
                msa = [("query", reference[start:end]), ("homolog", reference[start:end])]
                context = MSAContext(msa, [1., 1.], list(range(start + 1, end + 1)), reference, config)
                model = Model()
                inference = MSAInference(model, Alphabet(), config)
                globals_, sites = inference.embeddings(inputs, [context], lambda _: None)
                expected = torch.tensor([
                    [Alphabet().get_idx(aa) for aa in sequence[start:end]] for sequence in inputs.sequences
                ], dtype=torch.float32).mean(1)
                torch.testing.assert_close(globals_[:, 0], expected)
                # Position-dependent output detects both wrong boundaries and
                # an unweighted average of means for the final 3-residue chunk.
                window_lengths = [min(4, msa_length - offset) for offset in range(0, msa_length, 4)]
                position_mean = sum(width * (width + 1) / 2 for width in window_lengths) / msa_length
                torch.testing.assert_close(globals_[:, 1], torch.full((len(rows),), position_mean))
                torch.testing.assert_close(globals_[[0, 1, 4, 5]], globals_[0].expand(4, -1))
                expected_sites = torch.tensor([Alphabet().get_idx(sequence[position - 1])
                                               for sequence, position in zip(inputs.sequences, positions)], dtype=torch.float32)
                torch.testing.assert_close(sites["site_embeddings"][:, 0], expected_sites)
                self.assertEqual(sites["site_window_starts"].tolist(),
                                 [max(0, min(position - 3, len(reference) - 4)) for position in positions])
                self.assertTrue(all(tokens.shape[-1] <= 5 for tokens, _ in model.calls))
                self.assertTrue(all(representations for _, representations in model.calls))
                # Terminal site windows have no MSA overlap and use one row.
                query_only = [tokens for tokens, _ in model.calls if tokens.shape[1] == 1]
                self.assertEqual(len(query_only), 2)
                self.assertEqual(sites["site_embeddings"][[0, -1], 2].tolist(), [0., 0.])
                # Adjacent outside mutations retain partial homolog evidence.
                self.assertGreater(sites["site_embeddings"][1, 2].item(), 0.)
                self.assertGreater(sites["site_embeddings"][4, 2].item(), 0.)

    def test_embeddings_drop_all_gap_rows_per_window_without_changing_score_rows(self):
        config = replace(self.config, seeds=(1,), max_depth=5)
        context = MSAContext([("query", "CDEF"), ("homolog1", "-D--"), ("homolog2", "---F")],
                             self.weights, [2, 3, 4, 5], "ACDEFG", config)
        inference = MSAInference(self.model, Alphabet(), config)
        # Partial coverage keeps both draws of homolog1, with flank gaps.
        inference.forward(context, "VCDEFG", (0, 3))
        expected_rows = [("query", "VCD"), ("homolog1", "--D"), ("homolog1", "--D")]
        _, _, expected = Alphabet().get_batch_converter()([expected_rows])
        torch.testing.assert_close(self.model.calls[-1][0], expected)
        # Actual residue evidence matters, even for a column inside coverage.
        for bounds in ((3, 4), (5, 6)):
            values = inference.forward(context, "ACDWFG", bounds)
            self.assertEqual(self.model.calls[-1][0].shape[1], 1)
            self.assertEqual(values[0, 2].item(), 0.)
        self.assertEqual(len(context.window("ACDEFG", 0, 3)), config.max_depth)
        inference.scores(self.inputs, [context], lambda _: None)
        self.assertTrue(all(tokens.shape[1] == config.max_depth
                            for tokens, representations in self.model.calls if not representations))

    def test_msa_region_metadata_payload_validation_and_legacy_features(self):
        self.assertEqual(self.run_preparation()[0]["status"], "completed")
        root = self.config.output_root / "tiny"
        leaf = root / "sequence/msa_transformer/features-test"
        config = json.loads((leaf / "config.json").read_text())
        payload = torch.load(leaf / "features.pt", weights_only=True)
        self.assertEqual((config["msa_recipe_version"], payload["represented_length"]), (3, 4))
        self.assertEqual((payload["global_region_start"], payload["global_region_end"]), (1, 5))
        torch.testing.assert_close(payload["positions"], torch.arange(1, 7))
        for changes in ({"global_region": "full_query"}, {"global_region_start": None},
                        {"global_region_start": True}, {"global_region_end": 1},
                        {"global_region_end": 6}, {"global_window_aggregation": "mean_of_window_means"},
                        {"embedding_homolog_filter": "keep_all_rows"}, {"empty_embedding_window": "esm2"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                make_leaf_config("invalid", {**config, **changes}, model_family="msa_transformer")
        with self.assertRaises(ValueError):
            compact_sequence_recipe({**config, "global_region_end": 7}, reference_length=6)
        for changes in ({"represented_length": 6}, {"global_region_start": 0}, {"global_region_end": 6},
                        {"global_region": "full_query"}, {"site_window_starts": torch.zeros(3, dtype=torch.long)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_compact_sequence_payload({**payload, **changes}, config, 6, [[2], [4], [5]])
        missing_region = {key: value for key, value in config.items() if key != "global_region"}
        with self.assertRaises(ValueError):
            compact_sequence_recipe(missing_region)

        # v2 full-query features retain their original coverage and load beside v3.
        legacy = {key: value for key, value in config.items()
                  if key not in {*EMBEDDING_RECIPE, "global_region_start", "global_region_end"}}
        legacy["msa_recipe_version"] = 2
        legacy = make_leaf_config("legacy-v2", legacy, model_family="msa_transformer")
        legacy_leaf = write_sequence_cache(root, "legacy-v2", self.inputs, legacy, payload["embeddings"], {},
                                           site_data={key: payload[key] for key in COMPACT_SITE_FIELDS})
        self.assertEqual(torch.load(legacy_leaf / "features.pt", weights_only=True)["represented_length"], 6)
        validate_sequence_cache(root, "legacy-v2", "msa_transformer")
        bundle = load_feature_bundle(root, enabled_branches=["sequence_global", "sequence_site"],
                                     feature_selection={"sequence": "msa_transformer/legacy-v2"},
                                     score_sources=["scores/msa_transformer/scores-test"])
        self.assertEqual(bundle.global_features.shape, (3, 768))
        with self.assertRaisesRegex(ValueError, "new cache ID"):
            check_existing(root, "features-test", self.inputs, {**legacy, "checkpoint": "features-test"}, resume=True)

    def test_new_feature_cache_reuses_unchanged_v2_scores(self):
        self.assertEqual(self.run_preparation()[0]["status"], "completed")
        leaf = self.config.output_root / "tiny/scores/msa_transformer/scores-test"
        before = {path.name: path.read_bytes() for path in leaf.iterdir()}
        self.assertEqual(json.loads(before["config.json"])["msa_recipe_version"], 2)
        self.model.calls.clear()
        result = self.run_preparation(config=replace(self.config, cache_id="another-feature-cache"), resume=True)[0]
        self.assertEqual((result["sequence"], result["scores"]), ("completed", "skipped"))
        self.assertTrue(self.model.calls)
        self.assertTrue(all(representations for _, representations in self.model.calls))
        self.assertEqual({path.name: path.read_bytes() for path in leaf.iterdir()}, before)

    def test_resume_skips_both_stages_and_rejects_changed_sampling(self):
        self.assertEqual(self.run_preparation()[0]["status"], "completed")
        self.model.calls.clear()
        result = self.run_preparation(resume=True)[0]
        self.assertEqual((result["sequence"], result["scores"]), ("skipped", "skipped"))
        self.assertFalse(self.model.calls)
        result = self.run_preparation(config=replace(self.config, seeds=(6, 7, 8, 9, 10)), resume=True)[0]
        self.assertEqual(result["status"], "failed")
        self.assertFalse(self.model.calls)

    def test_cleanup_prunes_other_recipes_without_rewriting_resumed_caches_or_inputs(self):
        self.assertEqual(self.run_preparation()[0]["status"], "completed")
        other = replace(self.config, cache_id="other-features", score_cache_id="other-scores")
        self.assertEqual(self.run_preparation(config=other)[0]["status"], "completed")
        root = self.config.output_root / "tiny"
        current = [root / "sequence/msa_transformer/features-test", root / "scores/msa_transformer/scores-test"]
        protected = [self.directory, *current]
        marker = root / "sequence/esm2/keep/features.txt"
        marker.parent.mkdir(parents=True)
        marker.write_bytes(b"keep")
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                  for directory in protected for path in directory.iterdir() if path.is_file()}
        self.model.calls.clear()
        result = self.run_preparation(resume=True, cleanup_and_save=True)[0]
        self.assertEqual((result["sequence"], result["scores"]), ("skipped", "skipped"), result)
        self.assertFalse(self.model.calls)
        for directory in current:
            self.assertEqual(list(directory.parent.iterdir()), [directory])
        self.assertEqual({path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}, before)
        self.assertEqual(marker.read_bytes(), b"keep")

    def test_cleanup_regenerates_corrupted_stages_and_removes_other_recipes_after_saving(self):
        self.assertEqual(self.run_preparation()[0]["status"], "completed")
        other = replace(self.config, cache_id="other-features", score_cache_id="other-scores")
        self.assertEqual(self.run_preparation(config=other)[0]["status"], "completed")
        root = self.config.output_root / "tiny"
        feature = root / "sequence/msa_transformer/features-test"
        score = root / "scores/msa_transformer/scores-test"
        (feature / "completion.json").unlink()
        (score / "mean_score.csv").write_bytes(b"corrupted")
        self.model.calls.clear()
        result = self.run_preparation(resume=True, cleanup_and_save=True)[0]
        self.assertEqual((result["sequence"], result["scores"]), ("completed", "completed"), result)
        self.assertTrue(self.model.calls)
        self.assertEqual(list(feature.parent.iterdir()), [feature])
        self.assertEqual(list(score.parent.iterdir()), [score])
        validate_sequence_cache(root, "features-test", "msa_transformer")
        validate_score_cache(root, "msa_transformer", "scores-test")

    def test_cleanup_without_resume_preserves_existing_current_and_sibling_caches(self):
        self.assertEqual(self.run_preparation()[0]["status"], "completed")
        other = replace(self.config, cache_id="other-features", score_cache_id="other-scores")
        self.assertEqual(self.run_preparation(config=other)[0]["status"], "completed")
        root = self.config.output_root / "tiny"
        before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
        self.model.calls.clear()
        result = self.run_preparation(cleanup_and_save=True)[0]
        self.assertEqual(result["status"], "failed")
        self.assertFalse(self.model.calls)
        self.assertEqual({path: path.read_bytes() for path in root.rglob("*") if path.is_file()}, before)

    def test_long_scores_slice_existing_tokens_including_bos_boundary_and_clipped_tail(self):
        # Expected bounds follow pinned ProteinGym's +2 length convention even
        # though MSA-1b has BOS only. Interior windows must not add another BOS.
        cases = [(1023, 1023, (0, 1024)), (1024, 1, (0, 1024)),
                 (1024, 1024, (2, 1026)), (1600, 800, (288, 1312)),
                 (1600, 1600, (578, 1602))]
        for length, position, bounds in cases:
            with self.subTest(length=length, position=position):
                self.assertEqual(proteingym_token_window(position, length), bounds)
                reference = (Alphabet.tokens[:-1] * (length // 20 + 1))[:length]
                old, new = reference[position - 1], "V" if reference[position - 1] != "V" else "A"
                mutant = reference[:position - 1] + new + reference[position:]
                inputs = replace(self.inputs, reference_sequence=reference,
                                 rows=[{"mutant": f"{old}{position}{new}", "mutated_sequence": mutant}])
                context = MSAContext([("query", reference)], [1.], list(range(1, length + 1)), reference, self.config)
                actual = self.inference.scores(inputs, [context], lambda _: None)
                _, _, expected = Alphabet().get_batch_converter()([[("query", reference)]])
                expected[0, 0, position] = Alphabet.mask_idx
                expected = expected[:, :, bounds[0]:bounds[1]]
                torch.testing.assert_close(self.model.calls[-1][0], expected)
                self.assertEqual((expected == Alphabet.mask_idx).sum().item(), 1)
                scale = float(expected.sum() % 5 + 1)
                expected_score = (Alphabet().get_idx(new) - Alphabet().get_idx(old)) * scale
                self.assertEqual(actual.dtype, torch.float64)
                self.assertAlmostEqual(actual.item(), expected_score, places=5)

    def test_uncovered_mutation_keeps_embeddings_but_rejects_complete_scores(self):
        write_csv(self.root / "tiny.csv", ["variant_id", "mutant", "mutated_sequence", "DMS_score"],
                  [["uncovered", "A1V", "VCDEFG", 0.1]])
        result = self.run_preparation()[0]
        self.assertEqual(result["sequence"], "completed")
        self.assertIn("outside the MSA interval", result["scores"])
        self.assertFalse((self.config.output_root / "tiny/scores/msa_transformer/scores-test").exists())
        self.assertFalse(any(not representations for _, representations in self.model.calls))

    def test_preset_and_nkn_selectors_use_depth32_single_seed(self):
        cfg = MSAConfig.load(ROOT / "configs/extraction/msa_transformer.json")
        self.assertEqual((cfg.max_depth, cfg.seeds, cfg.scoring_token_window, cfg.max_tokens_per_msa),
                         (32, (1,), 1024, 16384))
        for name in ("msa_transformer_nkn.json", "msa_transformer_proteinmpnn_nkn.json"):
            experiment = json.loads((ROOT / "configs" / name).read_text())
            self.assertEqual(experiment["feature_selection"]["sequence"], f"msa_transformer/{cfg.cache_id}")
            self.assertEqual(experiment["score_sources"], [f"scores/msa_transformer/{cfg.score_cache_id}"])
        for change in ({"seeds": []}, {"seeds": [1, 1]}, {"seeds": [True]}, {"scoring_token_window": 1022},
                       {"sampling": "weighted_without_replacement_sorted_query_first"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(cfg, **change)

    def test_ensemble_metadata_and_legacy_cache_recipe_validation(self):
        self.assertEqual(self.run_preparation()[0]["status"], "completed")
        root = self.config.output_root / "tiny"
        config = json.loads((root / "scores/msa_transformer/scores-test/config.json").read_text())
        validate_msa_metadata(config)  # Repeated rows in seed 4/5 are valid.
        for change in ({"seeds": [1]}, {"sampled_row_indices": [[0, 1]] * 5}, {"seed": 0},
                       {"sampled_row_indices": [[0, 0, 1]] * 5}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_msa_metadata({**config, **change})
        legacy = {**config, **LEGACY_SCORE_RECIPE, "msa_recipe_version": 1, "seed": 0,
                  "sampled_row_indices": [0, 1, 2], "sampling": "weighted_without_replacement_sorted_query_first",
                  "uncovered_positions": "full_query_gap_padded_homologs", "window_policy": "centered_clamped",
                  "window_indexing": "zero_based_half_open"}
        for key in ("seeds", "sampling_rng", "msa_total_rows", "scoring_token_window", "scoring_coverage",
                    "score_aggregation", "window_sequence_length"):
            legacy.pop(key)
        self.assertEqual(make_score_config("legacy", "msa_transformer", legacy)["scoring_recipe_version"], 1)

    def test_scores_still_run_after_embedding_failure_and_resume_independently(self):
        with patch.object(self.inference, "embeddings", side_effect=RuntimeError("GPU failure")):
            result = self.run_preparation()[0]
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["scores"], "completed")
        result = self.run_preparation(resume=True)[0]
        self.assertEqual((result["sequence"], result["scores"]), ("completed", "skipped"))

    def test_legacy_torch_without_mmap_can_validate_compact_features(self):
        self.assertEqual(self.run_preparation()[0]["status"], "completed")
        original = torch.load
        def legacy_load(path, map_location=None, weights_only=False):
            return original(path, map_location=map_location, weights_only=weights_only)
        with patch("torch.load", legacy_load):
            validate_sequence_cache(self.config.output_root / "tiny")

    def test_corrupt_missing_ambiguous_inputs_and_memory_guard_fail_before_inference(self):
        with (self.directory / "alignment.a2m").open("a") as handle:
            handle.write("\n")
        result = self.run_preparation()[0]
        self.assertEqual(result["status"], "failed")
        self.assertIn("checksum", result["error"])
        self.save_pair()
        (self.directory.parent / "tiny_other").mkdir()
        self.assertIn("found 2", self.run_preparation()[0]["error"])
        (self.directory.parent / "tiny_other").rmdir()
        result = self.run_preparation(config=replace(self.config, max_output_gib=1e-9))[0]
        self.assertIn("max_output_gib", result["error"])
        results = self.run_preparation(assays=["missing", "tiny"])
        self.assertEqual([result["status"] for result in results], ["failed", "completed"])


if __name__ == "__main__":
    unittest.main()
