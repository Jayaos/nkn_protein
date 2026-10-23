"""Prepare MSA Transformer caches from completed generated alignment/weight pairs."""

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=os.environ.get(
        "MSA_TRANSFORMER_CONFIG", str(ROOT / "configs/extraction/msa_transformer.json")))
    parser.add_argument("--assay", action="append", help="Repeat for an ordered subset; default: entire catalog")
    parser.add_argument("--task-index", type=int, default=os.environ.get("SLURM_ARRAY_TASK_ID"),
                        help="Zero-based index in the selected catalog; defaults to SLURM_ARRAY_TASK_ID")
    parser.add_argument("--count", action="store_true", help="Print selected catalog size without importing PyTorch")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--validate-inputs-only", action="store_true", help="Check all selected pairs and mappings without loading a model or writing caches")
    args = parser.parse_args(argv)
    try:
        from list_assays import list_assays
        assays = list_assays(args.config)
        if args.assay:
            if len(set(args.assay)) != len(args.assay) or any(name not in assays for name in args.assay):
                raise ValueError("Requested assays must be unique members of the catalog")
            assays = args.assay
        if args.count:
            print(len(assays))
            return 0
        if args.task_index is not None:
            if not 0 <= args.task_index < len(assays):
                raise ValueError(f"Task index outside 0..{len(assays) - 1}")
            assays = [assays[args.task_index]]
        from feature_generation.inputs import prepare_assay
        from feature_generation.msa_config import MSAConfig
        from feature_generation.msa_runner import MSAInference, load_contexts, prepare, validate_score_coverage
        from generated_msa import validate_pair
        config = MSAConfig.load(args.config)
        if args.device:
            config = replace(config, device=args.device)
        print(f"Config: {args.config}\nSelected assays: {len(assays)}\nOutput: {config.output_root}", flush=True)
        if args.validate_inputs_only:
            failed = False
            for assay in assays:
                try:
                    inputs = prepare_assay(config.assay_index, assay, singles_only=True)
                    contexts, _, directory = load_contexts(config, inputs, validate_pair)
                    context = contexts[0]
                    validate_score_coverage(inputs, context)
                    print(f"{assay}: valid; {len(inputs.rows)} singles; depth={len(context.indices)}; "
                          f"seeds={list(config.seeds)}; MSA coverage={len(context.columns)}/{inputs.length}; {directory}", flush=True)
                except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, csv.Error) as error:
                    print(f"{assay}: failed: {error}", file=sys.stderr, flush=True)
                    failed = True
            return int(failed)
        from setup_msa_transformer import validate_runtime
        runtime_config = json.loads(config.runtime_config.read_text(encoding="utf-8"))
        report, model, alphabet = validate_runtime(config.runtime_dir, runtime_config, config.device, return_model=True)
        inference = MSAInference(model, alphabet, config)
        results = prepare(config, assays, report, inference, validate_pair, resume=args.resume,
                          progress=lambda message: print(message, flush=True))
        failed = sum(result["status"] == "failed" for result in results)
        print(f"Finished: {len(results) - failed}/{len(results)} assays ready; {failed} failed", flush=True)
        return int(failed > 0)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
