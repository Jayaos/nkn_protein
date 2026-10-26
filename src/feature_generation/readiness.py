"""Read-only audit of all five model preparation jobs' assets."""

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
from importlib import import_module
import json
import os
from pathlib import Path
import sys

import torch

from data_loading.catalog import AssayCatalog
from assay_policies import PROTEINMPNN_SCORE_SKIP_REASONS
from .checkpoints import local_checkpoint as esm2_checkpoint
from .config import ExtractionConfig, PROJECT_ROOT
from .inputs import prepare_assay
from .inverse_checkpoints import local_checkpoint as inverse_checkpoint
from .inverse_config import InverseFoldingConfig
from .inverse_storage import validate_inverse_cache
from .msa_config import EMBEDDING_RECIPE, SCORE_RECIPE, MSAConfig
from .msa_runner import load_contexts, validate_score_coverage
from .prosst_checkpoints import local_checkpoint as prosst_checkpoint
from .prosst_config import ProSSTConfig
from .prosst_structure import validate_structure_token_cache
from .score_runner import default_cache_id
from .score_storage import validate_score_cache
from .storage import validate_sequence_cache
from .structure_fragments import load_inverse_structure
from .structure_inputs import load_structure


BRCA2 = "BRCA2_HUMAN_Erwood_2022_HEK293T"
CONFIG_DIR = PROJECT_ROOT / "configs/extraction"
SUMMARY_COLUMNS = (
    "array_index", "assay_id", "inputs", "structure", "esm2_embeddings", "esm2_scores",
    "proteinmpnn_features", "proteinmpnn_scores", "esm2_model", "proteinmpnn_model",
    "proteinmpnn_scoring_model", "esm_if1_structure", "esm_if1_features", "esm_if1_scores",
    "esm_if1_model", "prosst_structure", "prosst_embeddings", "prosst_scores", "prosst_model",
    "msa_inputs", "msa_transformer_embeddings", "msa_transformer_scores", "msa_transformer_model",
    "outputs_status", "status", "errors", "notes",
)


def preparation_helper(name):
    """Reuse the standalone MSA validators without requiring their environment."""
    sys.path.insert(0, str(PROJECT_ROOT / "sbatch_run/run_prepare_rep"))
    try:
        return import_module(name)
    finally:
        sys.path.pop(0)


def msa_checkpoint(config):
    runtime = json.loads(config.runtime_config.read_text(encoding="utf-8"))
    # validate_runtime loads a model and runs inference; only verify its files here.
    return preparation_helper("setup_msa_transformer").verify_checkpoint(config.runtime_dir, runtime["checkpoint"])


def msa_inputs(config, inputs):
    if inputs is None:
        raise ValueError("Cannot verify MSA inputs: current assay input validation failed")
    return load_contexts(config, inputs, preparation_helper("generated_msa").validate_pair)


def msa_recipe(config, inputs, kind, context, checkpoint):
    if context is None:
        raise ValueError("Cannot verify cache MSA: generated alignment/weight validation failed")
    contexts, manifest_hash, _ = context
    first = contexts[0]
    runtime = json.loads(config.runtime_config.read_text(encoding="utf-8"))
    expected = {
        "model_id": "esm_msa1b_t12_100M_UR50S", "layer": 12, "hidden_size": 768,
        "inference_dtype": "float32", "proteingym_commit": runtime["proteingym_commit"],
        "packages": runtime["packages"], "msa_manifest_sha256": manifest_hash,
        "sampled_row_indices": [sample.indices for sample in contexts],
        "msa_total_rows": first.total_rows, "sampling": config.sampling,
        "sampling_rng": "python_random_choices", "max_depth": config.max_depth,
        "seeds": list(config.seeds), "max_length": config.max_length,
        "max_tokens_per_msa": config.max_tokens_per_msa, "msa_covered_positions": first.columns,
    }
    if checkpoint is not None:
        expected["checkpoint_sha256"] = checkpoint["sha256"]
    if kind == "scores":
        validate_score_coverage(inputs, first)
        expected.update(msa_recipe_version=2, **SCORE_RECIPE)
    else:
        expected.update(**EMBEDDING_RECIPE, global_region_start=first.start, global_region_end=first.end,
                        uncovered_positions=config.uncovered_positions, pooling="mean", truncation="none",
                        global_window="non_overlapping", site_embeddings="centered")
    return expected


def validate_prosst_tokens(config, saved, structure):
    """Check the exact token cache referenced by each output, without quantizing."""
    signature = saved["structure_token_input_signature"]
    if (not isinstance(signature, str) or len(signature) != 64
            or any(char not in "0123456789abcdef" for char in signature)):
        raise ValueError("Invalid ProSST structure-token signature")
    paths = list(config.structure_token_dir.glob(f"prosst2048-*/{signature}"))
    if len(paths) != 1:
        raise ValueError(f"Expected one ProSST structure-token cache for {signature}; found {len(paths)}")
    tokens = validate_structure_token_cache(paths[0], {
        "input_signature": signature, "structure_input_signature": structure.input_signature,
        "reference_sequence": structure.reference_sequence,
    })
    if tokens.metadata != saved["structure_tokens"]:
        raise ValueError("Cache uses different ProSST structure tokens")
    if tokens.metadata["recipe"]["batch_size"] != config.quantizer_batch_size:
        raise ValueError("Cache recipe differs from preparation config: quantizer_batch_size")


def validate_output(config, assay, inputs, kind, structure=None, *, msa_context=None, msa_model=None):
    """Validate contents and compare them with the current preparation inputs."""
    family = getattr(config, "model_family", "esm2")
    root = config.output_root / assay
    if kind == "scores":
        cache_id = config.score_cache_id if family == "msa_transformer" else default_cache_id(config, num_orders=1)
        report = validate_score_cache(root, family, cache_id)
    elif family in ("esm2", "prosst", "msa_transformer"):
        report = validate_sequence_cache(root, config.cache_id, model_family=family)
    else:
        report = validate_inverse_cache(root, family, config.cache_id)
    if inputs is None:
        raise ValueError("Cannot verify coverage: current assay input validation failed")
    if report["input_signature"] != inputs.input_signature:
        raise ValueError("Cache differs from current catalog/assay or complete single-mutant coverage")
    saved = json.loads((Path(report["path"]) / "config.json").read_text(encoding="utf-8"))
    expected = {} if family == "msa_transformer" else {
        key: getattr(config, key) for key in ("model_id", "revision", "inference_dtype")}
    if family == "msa_transformer":
        expected.update(msa_recipe(config, inputs, kind, msa_context, msa_model))
    elif kind == "scores":
        if family == "proteinmpnn":
            expected.update(seed=config.seed, num_orders=1)
        elif family in ("esm2", "prosst"):
            # The score cache ID also identifies full versus centered windows.
            expected["scoring_window"] = config.scoring_window
        else:
            expected["seed"] = config.seed
    elif family in ("esm2", "prosst"):
        expected.update({key: getattr(config, key) for key in
                         ("layer", "pooling", "truncation", "max_length", "site_embeddings", "global_window")})
        expected["dtype"] = config.output_dtype
    else:
        expected.update(conditioning_mode=config.conditioning_mode, seed=config.seed, dtype=config.output_dtype)
        if config.conditioning_mode == "sequence_conditional":
            expected["num_orders"] = config.num_orders
    legacy_defaults = {"site_embeddings": "none", "global_window": "prefix", "scoring_window": "full"}
    for key, value in expected.items():
        if saved.get(key, legacy_defaults.get(key)) != value:
            raise ValueError(f"Cache recipe differs from preparation config: {key}")
    if family in ("proteinmpnn", "esm_if1", "prosst"):
        if structure is None:
            raise ValueError("Cannot verify cache structure: registered structure validation failed")
        signature = saved["structure_input_signature"] if family == "prosst" else report["structure_input_signature"]
        if signature != structure.input_signature:
            raise ValueError("Cache uses a different registered structure")
        if family == "prosst":
            validate_prosst_tokens(config, saved, structure)
    return report


def audit(esm2, features, scores, esm_if1, prosst, msa, summary_stream=None):
    """Check every stage independently, continuing after missing/corrupt assets."""
    if any(config.assay_index != esm2.assay_index for config in (features, scores, esm_if1, prosst, msa)):
        raise ValueError("Preparation configs must select the same assay catalog")
    if features.structure_manifest_dir != scores.structure_manifest_dir:
        raise ValueError("ProteinMPNN configs must select the same structure manifests")
    catalog = AssayCatalog(esm2.assay_index)
    failures = 0
    statuses = {}
    errors = []
    writer = csv.DictWriter(summary_stream, fieldnames=SUMMARY_COLUMNS) if summary_stream is not None else None
    if writer is not None:
        writer.writeheader()
        summary_stream.flush()

    def check(key, label, action):
        nonlocal failures
        try:
            value = action()
            statuses[key] = "READY"
            print(f"  READY  {label}", flush=True)
            return True, value
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError, EOFError, AttributeError, csv.Error) as error:
            failures += 1
            statuses[key] = "FAILED"
            errors.append(f"{label}: {error}")
            print(f"  FAILED {label}: {error}", flush=True)
            return False, None

    print("Shared model assets (offline validation):", flush=True)
    check("esm2_model", "ESM2 checkpoint/tokenizer", lambda: esm2_checkpoint(esm2))
    check("proteinmpnn_model", "ProteinMPNN checkpoint/source", lambda: inverse_checkpoint(features))
    if (scores.model_dir, scores.revision) != (features.model_dir, features.revision):
        check("proteinmpnn_scoring_model", "ProteinMPNN scoring checkpoint/source", lambda: inverse_checkpoint(scores))
    else:
        statuses["proteinmpnn_scoring_model"] = statuses["proteinmpnn_model"]
    check("esm_if1_model", "ESM-IF1 checkpoint/source", lambda: inverse_checkpoint(esm_if1))
    check("prosst_model", "ProSST checkpoint/source/quantizer", lambda: prosst_checkpoint(prosst))
    _, msa_model = check("msa_transformer_model", "MSA Transformer checkpoint", lambda: msa_checkpoint(msa))
    shared_failures = failures
    shared_statuses, shared_errors = statuses.copy(), errors.copy()
    ready = 0
    for index, record in enumerate(catalog):
        assay = record.assay_id
        before = failures
        statuses = {"array_index": index, "assay_id": assay, **shared_statuses, "notes": ""}
        errors = shared_errors.copy()
        print(f"\nArray index {index}: {assay}", flush=True)
        _, inputs = check("inputs", "complete single-mutant inputs", lambda: prepare_assay(esm2.assay_index, assay, singles_only=True))
        _, structure = check("structure", "ProteinMPNN structure", lambda: load_inverse_structure(
            features.structure_manifest_dir / f"{assay}.json", record.reference_sequence, require_oxygen=True))
        _, if1_structure = check("esm_if1_structure", "ESM-IF1 structure", lambda: load_inverse_structure(
            esm_if1.structure_manifest_dir / f"{assay}.json", record.reference_sequence, require_oxygen=False))
        _, prosst_structure = check("prosst_structure", "ProSST structure", lambda: load_structure(
            prosst.structure_manifest_dir / f"{assay}.json", record.reference_sequence, require_oxygen=True))
        _, msa_context = check("msa_inputs", "MSA alignment/weights/mapping", lambda: msa_inputs(msa, inputs))
        for key, label, config, kind in (
            ("esm2_embeddings", "ESM2 embeddings", esm2, "features"),
            ("esm2_scores", "ESM2 scores", esm2, "scores"),
            ("proteinmpnn_features", "ProteinMPNN features", features, "features"),
            ("proteinmpnn_scores", "ProteinMPNN scores", scores, "scores"),
        ):
            if assay in PROTEINMPNN_SCORE_SKIP_REASONS and label == "ProteinMPNN scores":
                statuses[key] = "SKIP"
                statuses["notes"] = "ProteinMPNN scores intentionally omitted: " + PROTEINMPNN_SCORE_SKIP_REASONS[assay]
                print("  SKIP   " + statuses["notes"], flush=True)
                continue
            check(key, label, lambda c=config, k=kind: validate_output(c, assay, inputs, k, structure))
        for key, label, config, kind, backbone in (
            ("esm_if1_features", "ESM-IF1 features", esm_if1, "features", if1_structure),
            ("esm_if1_scores", "ESM-IF1 scores", esm_if1, "scores", if1_structure),
            ("prosst_embeddings", "ProSST embeddings", prosst, "features", prosst_structure),
            ("prosst_scores", "ProSST scores", prosst, "scores", prosst_structure),
            ("msa_transformer_embeddings", "MSA Transformer embeddings", msa, "features", None),
            ("msa_transformer_scores", "MSA Transformer scores", msa, "scores", None),
        ):
            check(key, label, lambda c=config, k=kind, s=backbone: validate_output(
                c, assay, inputs, k, s, msa_context=msa_context, msa_model=msa_model))
        passed = failures == before
        ready += int(passed)
        statuses["outputs_status"] = "READY" if passed else "FAILED"
        statuses["status"] = "READY" if passed and not shared_failures else "FAILED"
        statuses["errors"] = " | ".join(errors)
        if writer is not None:
            writer.writerow(statuses)
            summary_stream.flush()
        print(f"{index}\t{assay}\t{'READY' if passed else 'FAILED'}", flush=True)
    print(f"\nAssay outputs ready: {ready} / {len(catalog)}; failed assays: {len(catalog) - ready}; "
          f"shared asset failures: {shared_failures}", flush=True)
    return int(failures != 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esm2-config", type=Path, default=os.environ.get("ESM2_CONFIG", CONFIG_DIR / "esm2_650m_centered.json"),
                        help="ESM2 extraction config used by prepare_esm2.sbatch")
    parser.add_argument("--prosst-config", type=Path, default=os.environ.get("PROSST_CONFIG", CONFIG_DIR / "prosst_2048_singles.json"),
                        help="ProSST extraction config used by prepare_prosst.sbatch")
    parser.add_argument("--msa-transformer-config", type=Path, default=os.environ.get("MSA_TRANSFORMER_CONFIG", CONFIG_DIR / "msa_transformer.json"),
                        help="MSA extraction config used by prepare_msa_transformer.sbatch")
    parser.add_argument("--summary-csv", type=Path,
                        help="New CSV summary path; defaults to results/preparation-check-<job-id>-<timestamp>.csv")
    args = parser.parse_args(argv)
    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
        esm2 = ExtractionConfig.load(args.esm2_config)
        features = InverseFoldingConfig.load(CONFIG_DIR / "proteinmpnn_v48_020_kermut.json")
        scores = InverseFoldingConfig.load(CONFIG_DIR / "proteinmpnn_v48_020.json")
        # The ESM-IF1 batch job overrides the standalone preset's output root.
        esm_if1 = replace(InverseFoldingConfig.load(CONFIG_DIR / "esm_if1_142m.json"),
                          output_root=PROJECT_ROOT / "data/features/pretrained/singles")
        prosst = ProSSTConfig.load(args.prosst_config)
        msa = MSAConfig.load(args.msa_transformer_config)
        print(f"ESM2 preparation config: {args.esm2_config}", flush=True)
        print(f"ProteinMPNN preparation configs: {CONFIG_DIR / 'proteinmpnn_v48_020_kermut.json'}, "
              f"{CONFIG_DIR / 'proteinmpnn_v48_020.json'}", flush=True)
        print(f"ESM-IF1 preparation config: {CONFIG_DIR / 'esm_if1_142m.json'} (singles output root)", flush=True)
        print(f"ProSST preparation config: {args.prosst_config}", flush=True)
        print(f"MSA Transformer preparation config: {args.msa_transformer_config}", flush=True)
        print(f"Catalog: {esm2.assay_index}", flush=True)
        print(f"Output roots: ESM2={esm2.output_root}; ProteinMPNN features={features.output_root}; scores={scores.output_root}", flush=True)
        print(f"Output roots: ESM-IF1={esm_if1.output_root}; ProSST={prosst.output_root}; MSA Transformer={msa.output_root}", flush=True)
        summary_path = args.summary_csv or (
            PROJECT_ROOT / "results" /
            f"preparation-check-{os.environ.get('SLURM_JOB_ID', 'manual')}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.csv"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        # Preserve earlier audits. Flush each completed assay so interrupted jobs
        # retain a partial summary; absent rows have not been checked yet.
        with summary_path.open("x", newline="", encoding="utf-8") as handle:
            print(f"CSV summary: {summary_path.resolve()}", flush=True)
            return audit(esm2, features, scores, esm_if1, prosst, msa, summary_stream=handle)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError) as error:
        print(f"error: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
