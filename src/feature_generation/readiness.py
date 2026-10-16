"""Read-only audit of the ESM2/ProteinMPNN preparation jobs' assets."""

import argparse
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


def audit(esm2, features, scores):
    """Check every stage independently, continuing after missing/corrupt assets."""
    if any(config.assay_index != esm2.assay_index for config in (features, scores)):
        raise ValueError("Preparation configs must select the same assay catalog")
    if features.structure_manifest_dir != scores.structure_manifest_dir:
        raise ValueError("ProteinMPNN configs must select the same structure manifests")
    catalog = AssayCatalog(esm2.assay_index)
    failures = 0

    def check(label, action):
        nonlocal failures
        try:
            value = action()
            print(f"  READY  {label}", flush=True)
            return True, value
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError, EOFError) as error:
            failures += 1
            print(f"  FAILED {label}: {error}", flush=True)
            return False, None

    print("Shared model assets (offline validation):", flush=True)
    check("ESM2 checkpoint/tokenizer", lambda: esm2_checkpoint(esm2))
    check("ProteinMPNN checkpoint/source", lambda: proteinmpnn_checkpoint(features))
    if (scores.model_dir, scores.revision) != (features.model_dir, features.revision):
        check("ProteinMPNN scoring checkpoint/source", lambda: proteinmpnn_checkpoint(scores))
    shared_failures = failures
    ready = 0
    for index, record in enumerate(catalog):
        assay = record.assay_id
        before = failures
        print(f"\nArray index {index}: {assay}", flush=True)
        _, inputs = check("complete single-mutant inputs", lambda: prepare_assay(esm2.assay_index, assay, singles_only=True))
        _, structure = check("ProteinMPNN structure", lambda: load_inverse_structure(
            features.structure_manifest_dir / f"{assay}.json", record.reference_sequence, require_oxygen=True))
        for label, config, kind in (
            ("ESM2 embeddings", esm2, "features"),
            ("ESM2 scores", esm2, "scores"),
            ("ProteinMPNN features", features, "features"),
            ("ProteinMPNN scores", scores, "scores"),
        ):
            if assay == BRCA2 and label == "ProteinMPNN scores":
                print("  SKIP   ProteinMPNN scores: intentionally omitted by preparation for BRCA2 fragments", flush=True)
                continue
            check(label, lambda c=config, k=kind: validate_output(c, assay, inputs, k, structure))
        passed = failures == before
        ready += int(passed)
        print(f"{index}\t{assay}\t{'READY' if passed else 'FAILED'}", flush=True)
    print(f"\nAssay outputs ready: {ready} / {len(catalog)}; failed assays: {len(catalog) - ready}; "
          f"shared asset failures: {shared_failures}", flush=True)
    return int(failures != 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esm2-config", type=Path, default=os.environ.get("ESM2_CONFIG", CONFIG_DIR / "esm2_650m_centered.json"),
                        help="ESM2 extraction config used by prepare_esm2.sbatch")
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
        return audit(esm2, features, scores)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError) as error:
        print(f"error: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
