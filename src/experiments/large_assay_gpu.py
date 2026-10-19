"""GPU runs using prepared assets and published folds for selected human assays."""

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
from experiments.cuda_memory import CudaMemoryLogger
from experiments.runner import _validate_fold_assignments, run_cv
from experiments.settings import select_setting_batch
from gp.inference import add_inference_arguments, inference_metadata, resolve_training_config

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (ROOT / "configs"
                  / "esm2_proteinmpnn_nkn_two_scores_additive_four_branches_noise_initial_1p0.json")
SETTINGS = ("singles_random", "singles_modulo", "singles_contiguous")


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", nargs="?", default="singles", choices=("singles", *SETTINGS))
    parser.add_argument("--assay", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--min-steps", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--cuda-memory-every", type=int, default=0,
                        help="Report CUDA memory every N steps and at fold boundaries (0 disables)")
    parser.add_argument("--features", type=Path,
                        help="Prepared feature directory; defaults to data/features/pretrained/singles/ASSAY")
    parser.add_argument("--folds", type=Path,
                        help="Published fold CSV; defaults to cv_folds_singles_substitutions/ASSAY.csv")
    parser.add_argument("--assay-index", type=Path, default=ROOT
                        / "data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv")
    add_inference_arguments(parser)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    assay = args.assay
    if args.features is None:
        args.features = ROOT / "data/features/pretrained/singles" / assay
    if args.folds is None:
        args.folds = (ROOT / "data/DMS_ProteinGym_substitutions/cv_folds_singles_substitutions"
                      / f"{assay}.csv")
    output_created = False
    memory_logger = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use CUDA-enabled PyTorch in a GPU allocation")
        if args.output.exists():
            raise ValueError(f"Output directory already exists: {args.output}")
        if args.log_every < 1:
            raise ValueError("log-every must be positive")
        if args.cuda_memory_every < 0:
            raise ValueError("cuda-memory-every must be nonnegative")
        base = ExperimentConfig.load(args.config).for_assay(assay)
        config = replace(base, device="cuda", num_threads=args.num_threads,
                         training=resolve_training_config(base.training, args))
        if config.evaluation.n_folds != 5:
            raise ValueError("Published ProteinGym fold columns require nominal n_folds=5")
        if config.training.min_steps > config.training.max_steps:
            raise ValueError("min-steps exceeds max-steps; supply both for a shorter timing run")
        if config.training.max_train_variants is not None:
            raise ValueError("This run requires all single variants; max_train_variants must be null")
        metadata = json.loads((args.features / "assay.json").read_text(encoding="utf-8"))
        if metadata.get("assay_id") != assay or metadata.get("synthetic") is not False:
            raise ValueError(f"Expected real prepared assets for {assay}")
        torch.set_num_threads(config.num_threads)
        inference = inference_metadata(config.training)
        print(f"Inference: {json.dumps(inference)}", flush=True)
        print(f"Assay: {assay}; features: {args.features}", flush=True)
        print(f"Folds: {args.folds}; kernel branches: {', '.join(config.model.enabled_branches)}; "
              f"score sources: {config.score_sources}", flush=True)
        print("[1/3] Validating prepared assets and complete single-variant coverage on CPU", flush=True)
        batch = load_feature_bundle(
            args.features, feature_selection=config.feature_selection,
            score_sources=config.score_sources, pooling=config.pooling, device="cpu",
            enabled_branches=config.model.enabled_branches,
            inverse_folding_policy=config.inverse_folding_policy,
            structure_manifest_dir=config.structure_manifest_dir,
        )
        if batch.assay_id != assay:
            raise ValueError(f"Loaded feature batch does not identify {assay}")
        batch = select_setting_batch(batch, "singles_random")
        validate_assay_coverage(batch, args.assay_index, singles_only=True)
        settings = SETTINGS if args.selection == "singles" else (args.selection,)
        args.output.mkdir(parents=True, exist_ok=False)
        output_created = True
        if args.cuda_memory_every:
            memory_logger = CudaMemoryLogger(args.output, args.cuda_memory_every, torch.cuda)
            print(f"CUDA memory: every {args.cuda_memory_every} steps and at fold boundaries; "
                  f"allocator={memory_logger.allocator}; "
                  f"PYTORCH_CUDA_ALLOC_CONF={memory_logger.allocator_config or '(unset)'}", flush=True)
            memory_logger.record("run_start")
        _write_json(args.output / "config.json", config.to_dict())
        print("[2/3] Snapshotting and validating published folds", flush=True)
        fold_dir = args.output / "folds"
        fold_dir.mkdir()
        snapshot = fold_dir / "singles.csv"
        source_bytes = args.folds.read_bytes()
        snapshot.write_bytes(source_bytes)
        with snapshot.open(newline="", encoding="utf-8-sig") as handle:
            columns = csv.DictReader(handle).fieldnames or []
        splits, fold_counts, fold_ids = {}, {}, {}
        for setting in settings:
            scheme = setting.removeprefix("singles_")
            if f"fold_{scheme}_5" not in columns or scheme in columns:
                raise ValueError(f"Expected unambiguous published fold_{scheme}_5 assignments")
            splits[setting] = _validate_fold_assignments(
                batch, load_folds(snapshot, batch.variant_ids, scheme, 5), config, scheme,
            )
            fold_counts[setting] = torch.bincount(splits[setting], minlength=5).tolist()
            fold_ids[setting] = splits[setting].unique(sorted=True).tolist()
            print(f"{setting}: {batch.num_variants} singles; populated folds {fold_ids[setting]}; "
                  f"held-out counts {fold_counts[setting]}", flush=True)
        _write_json(args.output / "execution_plan.json", {
            "assay_id": assay, "device": "cuda", "parallel": False,
            "gp_computation_backend": config.training.gp_computation_backend, "inference": inference,
            "setting_workers": 1, "fold_workers": 1, "threads_per_worker": config.num_threads,
            "settings": {setting: {"fits": len(fold_ids[setting]), "fold_workers": 1,
                                   "fold_ids": fold_ids[setting]} for setting in settings},
        })
        _write_json(args.output / "fold_manifest.json", {
            "assay_id": assay, "source": str(args.folds.resolve()), "snapshot": "folds/singles.csv",
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "n_variants": batch.num_variants, "held_out_counts": fold_counts, "fold_ids": fold_ids,
        })
        print(f"PyTorch: {torch.__version__}; GPU: {torch.cuda.get_device_name()}; "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB; "
              f"float64; sequential folds; threads: {config.num_threads}", flush=True)
        print(f"Training: min_steps={config.training.min_steps}; max_steps={config.training.max_steps}; "
              f"stopping={config.training.stopping_criterion}; seeds={config.training.seeds}", flush=True)
        summary = {"assay_id": assay, "n_variants": batch.num_variants,
                   "gp_computation_backend": config.training.gp_computation_backend, "inference": inference,
                   "gpu": torch.cuda.get_device_name(), "settings": {}}
        for setting in settings:
            print(f"[3/3] Running {len(fold_ids[setting])}-fold NKN/GP CV on CUDA: {setting}", flush=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            if memory_logger is not None:
                memory_logger.progress(setting, {"event": "setting_start"})
            started = time.perf_counter()
            with (args.output / f"{setting}.progress.jsonl").open("x", encoding="utf-8") as handle:
                def progress(event):
                    elapsed = time.perf_counter() - started
                    handle.write(json.dumps({**event, "elapsed_seconds": elapsed}) + "\n")
                    handle.flush()
                    if memory_logger is not None:
                        memory_logger.progress(setting, event)
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
            if memory_logger is not None:
                memory_logger.progress(setting, {"event": "setting_complete"})
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
        if memory_logger is not None:
            memory_logger.failure()
        if output_created:
            _write_json(args.output / "failure.json", {
                "error": str(error), "traceback": traceback.format_exc(),
            })
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
