"""GPU timing run for the 10,094-single-variant human OCT1 assay."""

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import torch

from config import ExperimentConfig
from data_loading import load_feature_bundle, load_folds, validate_assay_coverage
from experiments.runner import _validate_fold_assignments, run_cv
from experiments.settings import select_setting_batch
from experiments.synthetic_gpu import DEFAULT_CONFIG, SETTINGS, validate_device_parity


ASSAY = "S22A1_HUMAN_Yee_2023_activity"
ROOT = Path(__file__).resolve().parents[2]


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", nargs="?", default="singles", choices=("singles", *SETTINGS))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--min-steps", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--features", type=Path,
                        default=ROOT / "data/features/pretrained/singles" / ASSAY)
    parser.add_argument("--folds", type=Path, default=ROOT
                        / "data/DMS_ProteinGym_substitutions/cv_folds_singles_substitutions"
                        / f"{ASSAY}.csv")
    parser.add_argument("--assay-index", type=Path, default=ROOT
                        / "data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    output_created = False
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use CUDA-enabled PyTorch in a GPU allocation")
        if args.output.exists():
            raise ValueError(f"Output directory already exists: {args.output}")
        if args.log_every < 1:
            raise ValueError("log-every must be positive")
        base = ExperimentConfig.load(args.config).for_assay(ASSAY)
        updates = {name: getattr(args, name) for name in ("max_steps", "min_steps")
                   if getattr(args, name) is not None}
        config = replace(base, device="cuda", num_threads=args.num_threads,
                         training=replace(base.training, **updates))
        if config.evaluation.n_folds != 5:
            raise ValueError("Published ProteinGym protocols require five folds")
        if config.training.min_steps > config.training.max_steps:
            raise ValueError("min-steps exceeds max-steps; supply both for a shorter timing run")
        if config.training.max_train_variants is not None:
            raise ValueError("This run requires all single variants; max_train_variants must be null")
        metadata = json.loads((args.features / "assay.json").read_text(encoding="utf-8"))
        if metadata.get("assay_id") != ASSAY or metadata.get("synthetic") is not False:
            raise ValueError(f"Expected real prepared assets for {ASSAY}")
        torch.set_num_threads(config.num_threads)
        print(f"Assay: {ASSAY}; features: {args.features}", flush=True)
        print("[1/3] Validating prepared assets and complete single-variant coverage on CPU", flush=True)
        batch = load_feature_bundle(
            args.features, feature_selection=config.feature_selection,
            score_sources=config.score_sources, pooling=config.pooling, device="cpu",
            enabled_branches=config.model.enabled_branches,
            inverse_folding_policy=config.inverse_folding_policy,
            structure_manifest_dir=config.structure_manifest_dir,
        )
        if batch.assay_id != ASSAY:
            raise ValueError(f"Loaded feature batch does not identify {ASSAY}")
        batch = select_setting_batch(batch, "singles_random")
        validate_assay_coverage(batch, args.assay_index, singles_only=True)
        if batch.num_variants <= 10000:
            raise ValueError("The selected assay must contain more than 10,000 single variants")
        settings = SETTINGS if args.selection == "singles" else (args.selection,)
        args.output.mkdir(parents=True, exist_ok=False)
        output_created = True
        _write_json(args.output / "config.json", config.to_dict())
        _write_json(args.output / "execution_plan.json", {
            "assay_id": ASSAY, "device": "cuda", "parallel": False,
            "setting_workers": 1, "fold_workers": 1, "threads_per_worker": config.num_threads,
            "settings": {setting: {"fits": 5, "fold_workers": 1} for setting in settings},
        })
        print("[2/3] Snapshotting and validating published folds", flush=True)
        fold_dir = args.output / "folds"
        fold_dir.mkdir()
        snapshot = fold_dir / "singles.csv"
        source_bytes = args.folds.read_bytes()
        snapshot.write_bytes(source_bytes)
        with snapshot.open(newline="", encoding="utf-8-sig") as handle:
            columns = csv.DictReader(handle).fieldnames or []
        splits, fold_counts = {}, {}
        for setting in settings:
            scheme = setting.removeprefix("singles_")
            if f"fold_{scheme}_5" not in columns or scheme in columns:
                raise ValueError(f"Expected unambiguous published fold_{scheme}_5 assignments")
            splits[setting] = _validate_fold_assignments(
                batch, load_folds(snapshot, batch.variant_ids, scheme, 5), config, scheme,
            )
            fold_counts[setting] = torch.bincount(splits[setting], minlength=5).tolist()
            print(f"{setting}: {batch.num_variants} singles; held-out counts {fold_counts[setting]}", flush=True)
        _write_json(args.output / "fold_manifest.json", {
            "assay_id": ASSAY, "source": str(args.folds.resolve()), "snapshot": "folds/singles.csv",
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "n_variants": batch.num_variants, "held_out_counts": fold_counts,
        })
        print(f"PyTorch: {torch.__version__}; GPU: {torch.cuda.get_device_name()}; "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB; "
              f"float64; sequential folds; threads: {config.num_threads}", flush=True)
        print(f"Training: min_steps={config.training.min_steps}; max_steps={config.training.max_steps}; "
              f"stopping={config.training.stopping_criterion}; seeds={config.training.seeds}", flush=True)
        parity = validate_device_parity(config)
        _write_json(args.output / "device_parity.json", parity)
        print("Synthetic CPU/CUDA numerical preflight passed", flush=True)
        summary = {"assay_id": ASSAY, "n_variants": batch.num_variants,
                   "gpu": torch.cuda.get_device_name(), "settings": {}}
        for setting in settings:
            print(f"[3/3] Running five-fold NKN/GP CV on CUDA: {setting}", flush=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with (args.output / f"{setting}.progress.jsonl").open("x", encoding="utf-8") as handle:
                def progress(event):
                    elapsed = time.perf_counter() - started
                    handle.write(json.dumps({**event, "elapsed_seconds": elapsed}) + "\n")
                    handle.flush()
                    if event.get("event") == "fold_start":
                        print(f"Fold {event['fold']}: {event['n_train']} training, "
                              f"{event.get('n_validation', 0)} validation, "
                              f"{event['n_test']} held-out variants", flush=True)
                    elif "step" in event and event["step"] % args.log_every == 0:
                        print(f"fold {event['fold']} seed {event['seed']} step {event['step']}: "
                              f"NLL/variant={event['nll_per_variant']:.6g}; "
                              f"setting elapsed={elapsed:.2f}s", flush=True)
                run_cv(
                    batch, splits[setting], config, args.output / "cv" / setting,
                    scheme=setting.removeprefix("singles_"), setting=setting,
                    split_source="provided", fold_workers=1, progress=progress,
                )
            torch.cuda.synchronize()
            timing = {
                "cv_seconds": time.perf_counter() - started,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            summary["settings"][setting] = timing
            _write_json(args.output / "benchmark.json", summary)
            print(f"Completed {setting}: {timing} (includes fitting, evaluation, and output)", flush=True)
        print(f"Completed. Results: {args.output}", flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, AssertionError, KeyError, TypeError) as error:
        if output_created:
            _write_json(args.output / "failure.json", {
                "error": str(error), "traceback": traceback.format_exc(),
            })
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
