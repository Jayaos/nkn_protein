"""Read-only audit of the ESM2/ProteinMPNN preparation jobs' assets."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import torch

from data_loading.catalog import AssayCatalog
from .checkpoints import local_checkpoint as esm2_checkpoint
from .config import ExtractionConfig, PROJECT_ROOT
from .inputs import prepare_assay
from .inverse_checkpoints import local_checkpoint as proteinmpnn_checkpoint
from .inverse_config import InverseFoldingConfig
from .inverse_storage import validate_inverse_cache
from .score_runner import default_cache_id
from .score_storage import validate_score_cache
from .storage import validate_sequence_cache
from .structure_fragments import load_inverse_structure


BRCA2 = "BRCA2_HUMAN_Erwood_2022_HEK293T"
CONFIG_DIR = PROJECT_ROOT / "configs/extraction"
SUMMARY_COLUMNS = (
    "array_index", "assay_id", "inputs", "structure", "esm2_embeddings", "esm2_scores",
    "proteinmpnn_features", "proteinmpnn_scores", "esm2_model", "proteinmpnn_model",
    "proteinmpnn_scoring_model", "outputs_status", "status", "errors", "notes",
)


def validate_output(config, assay, inputs, kind, structure=None):
    """Validate contents and compare them with the current preparation inputs."""
    family = getattr(config, "model_family", "esm2")
    root = config.output_root / assay
    if kind == "scores":
        report = validate_score_cache(root, family, default_cache_id(config, num_orders=1))
    elif family == "esm2":
        report = validate_sequence_cache(root, config.cache_id, model_family=family)
    else:
        report = validate_inverse_cache(root, family, config.cache_id)
    if inputs is None:
        raise ValueError("Cannot verify coverage: current assay input validation failed")
    if report["input_signature"] != inputs.input_signature:
        raise ValueError("Cache differs from current catalog/assay or complete single-mutant coverage")
    saved = json.loads((Path(report["path"]) / "config.json").read_text(encoding="utf-8"))
    expected = {key: getattr(config, key) for key in ("model_id", "revision", "inference_dtype")}
    if kind == "scores":
        if family == "proteinmpnn":
            expected.update(seed=config.seed, num_orders=1)
        else:
            # The score cache ID also identifies full versus centered windows.
            expected["scoring_window"] = config.scoring_window
    elif family == "esm2":
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
    if family == "proteinmpnn":
        if structure is None:
            raise ValueError("Cannot verify cache structure: registered structure validation failed")
        if report["structure_input_signature"] != structure.input_signature:
            raise ValueError("Cache uses a different registered structure")
    return report


def audit(esm2, features, scores, summary_stream=None):
    """Check every stage independently, continuing after missing/corrupt assets."""
    if any(config.assay_index != esm2.assay_index for config in (features, scores)):
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
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError, EOFError) as error:
            failures += 1
            statuses[key] = "FAILED"
            errors.append(f"{label}: {error}")
            print(f"  FAILED {label}: {error}", flush=True)
            return False, None

    print("Shared model assets (offline validation):", flush=True)
    check("esm2_model", "ESM2 checkpoint/tokenizer", lambda: esm2_checkpoint(esm2))
    check("proteinmpnn_model", "ProteinMPNN checkpoint/source", lambda: proteinmpnn_checkpoint(features))
    if (scores.model_dir, scores.revision) != (features.model_dir, features.revision):
        check("proteinmpnn_scoring_model", "ProteinMPNN scoring checkpoint/source", lambda: proteinmpnn_checkpoint(scores))
    else:
        statuses["proteinmpnn_scoring_model"] = statuses["proteinmpnn_model"]
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
        for key, label, config, kind in (
            ("esm2_embeddings", "ESM2 embeddings", esm2, "features"),
            ("esm2_scores", "ESM2 scores", esm2, "scores"),
            ("proteinmpnn_features", "ProteinMPNN features", features, "features"),
            ("proteinmpnn_scores", "ProteinMPNN scores", scores, "scores"),
        ):
            if assay == BRCA2 and label == "ProteinMPNN scores":
                statuses[key] = "SKIP"
                statuses["notes"] = "ProteinMPNN scores intentionally omitted by preparation for BRCA2 fragments"
                print("  SKIP   ProteinMPNN scores: intentionally omitted by preparation for BRCA2 fragments", flush=True)
                continue
            check(key, label, lambda c=config, k=kind: validate_output(c, assay, inputs, k, structure))
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
    parser.add_argument("--summary-csv", type=Path,
                        help="New CSV summary path; defaults to results/preparation-check-<job-id>-<timestamp>.csv")
    args = parser.parse_args(argv)
    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
        esm2 = ExtractionConfig.load(args.esm2_config)
        features = InverseFoldingConfig.load(CONFIG_DIR / "proteinmpnn_v48_020_kermut.json")
        scores = InverseFoldingConfig.load(CONFIG_DIR / "proteinmpnn_v48_020.json")
        print(f"ESM2 preparation config: {args.esm2_config}", flush=True)
        print(f"ProteinMPNN preparation configs: {CONFIG_DIR / 'proteinmpnn_v48_020_kermut.json'}, "
              f"{CONFIG_DIR / 'proteinmpnn_v48_020.json'}", flush=True)
        print(f"Catalog: {esm2.assay_index}", flush=True)
        print(f"Output roots: ESM2={esm2.output_root}; ProteinMPNN features={features.output_root}; scores={scores.output_root}", flush=True)
        summary_path = args.summary_csv or (
            PROJECT_ROOT / "results" /
            f"preparation-check-{os.environ.get('SLURM_JOB_ID', 'manual')}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.csv"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        # Preserve earlier audits. Flush each completed assay so interrupted jobs
        # retain a partial summary; absent rows have not been checked yet.
        with summary_path.open("x", newline="", encoding="utf-8") as handle:
            print(f"CSV summary: {summary_path.resolve()}", flush=True)
            return audit(esm2, features, scores, summary_stream=handle)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError) as error:
        print(f"error: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
