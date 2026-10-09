"""Command-line entry points for cached-feature experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import torch

from config import ExperimentConfig
from data_loading import development_folds, load_feature_bundle, load_folds, validate_assay_coverage
from .runner import prepare_output, run_cv, run_extrapolation, run_fit, run_predict


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m experiments.cli", description="Exact protein GP experiments from cached representations")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "fit", "cv", "predict"):
        sub = commands.add_parser(command)
        sub.add_argument("--features", required=True, type=Path, help="Assay cache directory containing assay.json")
        sub.add_argument("--config", type=Path, help="JSON experiment configuration; otherwise use documented defaults")
        sub.add_argument("--output", required=True, type=Path, help="New or empty output directory")
        sub.add_argument("--device", help="Override configured device, e.g. cpu or cuda")
        sub.add_argument("--quiet", action="store_true", help="Suppress optimizer progress")
        sub.add_argument("--assay-index", type=Path, help="Optionally validate against a processed ProteinGym assay index")
        coverage = sub.add_mutually_exclusive_group()
        coverage.add_argument("--full-assay", action="store_true", help="Require complete indexed assay coverage (default with --assay-index)")
        coverage.add_argument("--allow-subset", action="store_true", help="Explicitly allow a cached subset of the indexed assay")
        sub.add_argument("--singles-only", action="store_true", help="Select variants with exactly one substitution")
        if command in ("fit", "cv"):
            splits = sub.add_mutually_exclusive_group(required=command == "cv")
            splits.add_argument("--folds", type=Path, help="ID-aligned CSV containing supplied fold assignments")
            splits.add_argument("--development-splits", action="store_true", help="Construct development splits; these are not benchmark folds")
            if command == "fit":
                splits.add_argument("--single-to-double", action="store_true", help="Train all single mutants and evaluate double mutants in one fixed split")
            sub.add_argument("--scheme", choices=("random", "modulo", "contiguous"), default="random")
            sub.add_argument("--split-seed", type=int, default=0)
        if command == "fit":
            sub.add_argument("--test-fold", type=int, help="Held-out fold for split fitting (default 0)")
        if command == "predict":
            sub.add_argument("--checkpoint", required=True, type=Path)
    return parser


def _load_config(args) -> ExperimentConfig:
    path = args.config
    if path is None and args.command == "predict":
        adjacent = args.checkpoint.parent / "resolved_config.json"
        if adjacent.is_file():
            path = adjacent
    config = ExperimentConfig.load(path) if path is not None else ExperimentConfig()
    return replace(config, device=args.device) if args.device is not None else config


def _load_batch(args, config):
    if (args.full_assay or args.allow_subset) and args.assay_index is None:
        raise ValueError("--full-assay and --allow-subset require --assay-index")
    batch = load_feature_bundle(
        args.features,
        feature_selection=config.feature_selection,
        score_sources=config.score_sources,
        pooling=config.pooling,
        device=config.device,
    )
    if args.singles_only:
        counts = torch.bincount(batch.owners, minlength=batch.num_variants)
        batch = batch.select(torch.nonzero(counts == 1, as_tuple=False).flatten())
        if batch.num_variants == 0:
            raise ValueError("No single-substitution variants remain")
    if args.assay_index is not None:
        validate_assay_coverage(batch, args.assay_index, singles_only=args.singles_only, allow_subset=args.allow_subset)
    return batch


def _splits(args, batch, config):
    if args.folds is not None:
        return (
            load_folds(args.folds, batch.variant_ids, args.scheme, config.evaluation.n_folds),
            "provided",
        )
    if args.development_splits:
        return (
            development_folds(batch, args.scheme, config.evaluation.n_folds, args.split_seed),
            "development",
        )
    return None, "none"


def _progress(event):
    if event.get("event") == "fold_start":
        print(f"Fold {event['fold']}: {event['n_train']} training, {event['n_test']} held-out variants", file=sys.stderr)
    elif "step" in event and (event["step"] == 0 or event["step"] % 100 == 0):
        fold = f"fold {event['fold']} " if "fold" in event else ""
        print(f"{fold}seed {event['seed']} step {event['step']}: NLL/variant={event['nll_per_variant']:.6g}", file=sys.stderr)


def _input_manifest(args, batch):
    metadata = json.loads((args.features / "assay.json").read_text(encoding="utf-8"))
    return {
        "feature_directory": str(args.features.resolve()),
        "assay_id": batch.assay_id,
        "n_variants": batch.num_variants,
        "synthetic": metadata.get("synthetic"),
        "assay_metadata": metadata,
        "assay_index": str(args.assay_index.resolve()) if args.assay_index else None,
        "coverage": "subset_allowed" if args.allow_subset else "full" if args.assay_index else "not_checked_against_index",
        "singles_only": args.singles_only,
        "fold_source": str(args.folds.resolve()) if getattr(args, "folds", None) else None,
        "development_split_seed": args.split_seed if getattr(args, "development_splits", False) else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = _load_config(args)
        if getattr(args, "single_to_double", False) and (args.singles_only or args.test_fold is not None):
            raise ValueError("--single-to-double cannot be combined with --singles-only or --test-fold")
        torch.set_num_threads(config.num_threads)
        batch = _load_batch(args, config)
        progress = None if args.quiet else _progress
        if args.command == "validate":
            output = prepare_output(args.output, config)
            result = {
                "valid": True,
                "assay_id": batch.assay_id,
                "n_variants": batch.num_variants,
                "n_mutation_occurrences": batch.owners.numel(),
                "n_score_sources": batch.num_scores,
                "has_targets": batch.targets is not None,
                "global_feature_dimension": batch.global_features.shape[1],
                "local_feature_dimensions": {name: values.shape[1] for name, values in batch.local_features.items()},
            }
            (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        elif args.command == "predict":
            result = run_predict(batch, args.checkpoint, config, args.output)
        elif getattr(args, "single_to_double", False):
            result = run_extrapolation(batch, config, args.output, progress=progress)
        else:
            folds, source = _splits(args, batch, config)
            if args.command == "cv":
                result = run_cv(batch, folds, config, args.output, scheme=args.scheme, split_source=source, progress=progress)
            else:
                if args.test_fold is not None and folds is None:
                    raise ValueError("--test-fold requires --folds or --development-splits")
                result = run_fit(
                    batch, config, args.output, folds=folds,
                    test_fold=0 if args.test_fold is None else args.test_fold,
                    scheme=args.scheme, split_source=source, progress=progress,
                )
        (args.output / "input_manifest.json").write_text(
            json.dumps(_input_manifest(args, batch), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        summary = {name: value for name, value in result.items() if name not in ("fold_metrics", "metrics", "variance_summary")}
        summary["output_directory"] = str(args.output.resolve())
        print(json.dumps(summary, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
