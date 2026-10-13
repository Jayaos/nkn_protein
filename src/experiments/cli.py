"""Command-line entry points for cached-feature experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys

import torch

from config import ExperimentConfig
from data_loading import load_feature_bundle, load_folds, validate_assay_coverage
from .runner import prepare_output, run_cv, run_extrapolation, run_predict
from .settings import CV_SETTINGS, SETTING_SCHEMES, select_setting_batch


def _parser(prog="python -m experiments.cli") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Exact protein GP experiments from cached representations")
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
        coverage.add_argument("--full-assay", action="store_true", help="Require complete indexed assay coverage (all singles with --singles-only; default with --assay-index)")
        coverage.add_argument("--allow-subset", action="store_true", help="Explicitly allow a cached subset of the indexed assay")
        if command in ("validate", "cv", "predict"):
            sub.add_argument("--singles-only", action="store_true", help="Select and validate only single substitutions; CV requires a singles_* setting")
        else:
            sub.set_defaults(singles_only=False)
        if command in ("fit", "cv"):
            sub.add_argument(
                "--setting", required=True, choices=CV_SETTINGS if command == "cv" else ("single_to_double",),
                help="Kermut domain setting for all CV folds" if command == "cv" else "Fixed single-to-double extrapolation setting",
            )
            if command == "cv":
                sub.add_argument("--folds", required=True, type=Path, help="ID-aligned CSV containing supplied fold assignments")
                sub.add_argument(
                    "--fold-workers", type=int, default=1,
                    help="Concurrent CPU fold processes (default: 1); each uses config.num_threads threads",
                )
            else:
                sub.add_argument(
                    "--sample-ratio", type=float, default=0.8,
                    help="Fraction of singles used for training, in (0, 1] (default 0.8; 1 uses all singles)",
                )
                sub.add_argument(
                    "--split-seed", type=int, default=2024,
                    help="Seed for sampling training singles (default 2024; no split effect when --sample-ratio is 1)",
                )
        if command == "predict":
            sub.add_argument("--checkpoint", required=True, type=Path)
    return parser


def _load_config(args, config_type=ExperimentConfig):
    path = args.config
    if path is None and args.command == "predict":
        adjacent = args.checkpoint.parent / "resolved_config.json"
        if adjacent.is_file():
            path = adjacent
    config = config_type.load(path) if path is not None else config_type()
    return replace(config, device=args.device) if args.device is not None else config


def _load_batch(args, config, feature_loader=None):
    if (args.full_assay or args.allow_subset) and args.assay_index is None:
        raise ValueError("--full-assay and --allow-subset require --assay-index")
    batch = feature_loader(args.features, config) if feature_loader is not None else load_feature_bundle(
        args.features,
        feature_selection=config.feature_selection,
        score_sources=config.score_sources,
        pooling=config.pooling,
        device=config.device,
        enabled_branches=config.model.enabled_branches,
        inverse_folding_policy=config.inverse_folding_policy,
        structure_manifest_dir=config.structure_manifest_dir,
    )
    if args.singles_only:
        counts = torch.bincount(batch.owners, minlength=batch.num_variants)
        batch = batch.select(torch.nonzero(counts == 1, as_tuple=False).flatten())
        if batch.num_variants == 0:
            raise ValueError("No single-substitution variants remain")
    if args.assay_index is not None:
        validate_assay_coverage(batch, args.assay_index, singles_only=args.singles_only, allow_subset=args.allow_subset)
    return batch


def _progress(event):
    if event.get("event") == "fold_start":
        print(f"Fold {event['fold']}: {event['n_train']} training, {event['n_test']} held-out variants", file=sys.stderr)
    elif event.get("event") == "fold_complete":
        print(f"Fold {event['fold']}: completed", file=sys.stderr)
    elif "step" in event and (event["step"] == 0 or event["step"] % 100 == 0):
        fold = f"fold {event['fold']} " if "fold" in event else ""
        print(f"{fold}seed {event['seed']} step {event['step']}: NLL/variant={event['nll_per_variant']:.6g}", file=sys.stderr)


def _input_manifest(args, batch, excluded_variant_ids=()):
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
        "setting": getattr(args, "setting", None),
        "excluded_variant_ids": list(excluded_variant_ids),
        "fold_source": str(args.folds.resolve()) if getattr(args, "folds", None) else None,
        "fold_workers": getattr(args, "fold_workers", None),
        "extrapolation_sample_ratio": args.sample_ratio if getattr(args, "setting", None) == "single_to_double" else None,
        "extrapolation_split_seed": args.split_seed if getattr(args, "setting", None) == "single_to_double" else None,
    }


def _validate_setting_arguments(args):
    if args.command not in ("fit", "cv"):
        return
    if args.singles_only and not args.setting.startswith("singles_"):
        raise ValueError("--singles-only requires a singles_* CV setting")
    args.scheme = SETTING_SCHEMES[args.setting]
    if args.command == "cv" and args.fold_workers < 1:
        raise ValueError("--fold-workers must be a positive integer")
    if args.command == "fit":
        if not math.isfinite(args.sample_ratio) or not 0 < args.sample_ratio <= 1:
            raise ValueError("--sample-ratio must be finite and greater than 0 and at most 1")
        if not 0 <= args.split_seed < 2**32:
            raise ValueError("--split-seed must be between 0 and 2**32 - 1")


def main(
    argv: list[str] | None = None, *, config_type=ExperimentConfig,
    feature_loader=None, cv_runner=None, extrapolation_runner=None,
    prediction_runner=None, prog="python -m experiments.cli",
) -> int:
    """Run shared protocols with optional baseline model and feature adapters."""
    parser = _parser(prog)
    args = parser.parse_args(argv)
    try:
        _validate_setting_arguments(args)
        config = _load_config(args, config_type)
        if args.command == "cv":
            if args.fold_workers > config.evaluation.n_folds:
                raise ValueError("--fold-workers cannot exceed the number of folds")
            if args.fold_workers > 1:
                if torch.device(config.device).type != "cpu":
                    raise ValueError("Parallel folds require device cpu")
                allocation = os.environ.get("SLURM_CPUS_PER_TASK")
                if allocation is not None and args.fold_workers * config.num_threads > int(allocation):
                    raise ValueError("Fold workers times num_threads exceeds SLURM_CPUS_PER_TASK")
        torch.set_num_threads(config.num_threads)
        batch = _load_batch(args, config, feature_loader)
        excluded_variant_ids = ()
        if args.command in ("fit", "cv"):
            selected = select_setting_batch(batch, args.setting)
            selected_ids = set(selected.variant_ids)
            excluded_variant_ids = tuple(value for value in batch.variant_ids if value not in selected_ids)
            # Keep the full batch for extrapolation so its partition manifest
            # records unused singles and excluded WT/higher-order variants.
            if args.setting != "single_to_double":
                batch = selected
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
                "enabled_branches": list(config.model.enabled_branches),
                "global_feature_dimension": None if batch.global_features is None else batch.global_features.shape[1],
                "local_feature_dimensions": {name: values.shape[1] for name, values in batch.local_features.items()},
            }
            (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        elif args.command == "predict":
            result = (prediction_runner or run_predict)(batch, args.checkpoint, config, args.output)
        elif args.command == "fit":
            result = (extrapolation_runner or run_extrapolation)(
                batch, config, args.output, sample_ratio=args.sample_ratio,
                split_seed=args.split_seed, progress=progress,
            )
        else:
            folds = load_folds(
                args.folds, batch.variant_ids, args.scheme, config.evaluation.n_folds,
                allowed_extra_ids=excluded_variant_ids,
            )
            result = (cv_runner or run_cv)(
                batch, folds, config, args.output, scheme=args.scheme, setting=args.setting,
                split_source="provided", progress=progress,
                **({"fold_workers": args.fold_workers} if args.fold_workers > 1 else {}),
            )
        (args.output / "input_manifest.json").write_text(
            json.dumps(_input_manifest(args, batch, excluded_variant_ids), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        summary = {
            name: value for name, value in result.items()
            if name not in ("fold_metrics", "metrics", "variance_summary", "setting_metrics")
            and not name.endswith("variant_ids")
        }
        summary["output_directory"] = str(args.output.resolve())
        print(json.dumps(summary, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
