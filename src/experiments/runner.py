"""Fit, predict, and cross-validate with reproducible local artifacts."""

from __future__ import annotations

import csv
import importlib
import json
import math
import multiprocessing
import os
import queue
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, CancelledError, ProcessPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from config import ExperimentConfig, STRUCTURE_DISTANCE_DISABLED_ASSAYS
from data_loading import FeatureBatch
from data_loading.splits import _validate_folds
from evaluation import aggregate_folds, evaluate_fold, spearman
from gp.linalg import NumericalError
from gp.training import FittedGP, fit_gp

from .settings import SETTING_SCHEMES


_MIXED_DEPTHS = {"mixed_random_singles": 1, "mixed_random_doubles": 2}
_THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
_FOLD_WORKER_STATE = None


def _resolve_assay_config(batch, config):
    """Keep programmatic runs consistent with CLI assay restrictions."""
    if not hasattr(config, "for_assay"):
        return config
    resolved = config.for_assay(batch.assay_id)
    if isinstance(config, ExperimentConfig) and batch.assay_id in STRUCTURE_DISTANCE_DISABLED_ASSAYS and "structure_distance" in batch.local_features:
        raise ValueError(
            f"{batch.assay_id} disables structure_distance; reload features using "
            "config.for_assay(batch.assay_id).model.enabled_branches"
        )
    return resolved


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty prediction table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_output(output_dir: str | Path, config: ExperimentConfig) -> Path:
    """Create an empty output directory without replacing an existing run."""
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "resolved_config.json", config.to_dict())
    return output


def _mutation_counts(batch: FeatureBatch) -> Tensor:
    return torch.bincount(batch.owners, minlength=batch.num_variants)


def _uses_validation(config):
    return isinstance(config, ExperimentConfig) and config.training.stopping_criterion == "validation_spearman"


def _validate_fold_assignments(batch: FeatureBatch, folds: Tensor, config: ExperimentConfig, scheme: str) -> Tensor:
    if scheme not in ("random", "modulo", "contiguous", "random_multiples"):
        raise ValueError(f"Unknown splitting scheme: {scheme}")
    if not isinstance(folds, Tensor) or folds.ndim != 1 or len(folds) != batch.num_variants:
        raise ValueError("One fold assignment is required for every variant")
    if folds.dtype == torch.bool or folds.is_floating_point() or folds.is_complex():
        raise ValueError("Fold assignments must be integer identifiers")
    result = folds.detach().to(device="cpu", dtype=torch.long)
    _validate_folds(result, config.evaluation.n_folds)
    observed = sorted(set(result.tolist()))
    if any(batch.num_variants - int((result == fold).sum()) < 2 for fold in observed):
        raise ValueError("Every training partition must contain at least two variants")
    if scheme in ("modulo", "contiguous"):
        if not bool((_mutation_counts(batch) == 1).all()):
            raise ValueError("Positional evaluation currently supports single substitutions only; multi-mutant rules are deferred")
        positions = torch.empty(batch.num_variants, dtype=torch.long)
        positions[batch.owners.cpu()] = batch.mutation_positions.cpu()
        for position in positions.unique():
            if result[positions == position].unique().numel() != 1:
                raise ValueError("A mutation position occurs in multiple positional folds")
    if _uses_validation(config):
        if (config.evaluation.n_folds != 5 or observed != list(range(5))
                or scheme == "random_multiples" or not bool((_mutation_counts(batch) == 1).all())):
            raise ValueError("validation_spearman requires five populated folds for five-fold single-substitution CV "
                             "(three train, one validation, one test)")
        for fold in observed:
            validation_fold = (fold + 1) % 5
            if int(((result != fold) & (result != validation_fold)).sum()) < 2:
                raise ValueError("Every three-fold training partition must contain at least two variants")
            labels = None if batch.targets is None else batch.targets[result.to(batch.targets.device) == validation_fold]
            if labels is None or len(labels) < 2 or not torch.isfinite(labels).all() or labels.unique().numel() < 2:
                raise ValueError(f"Validation fold {validation_fold} requires at least two finite, nonconstant targets")
    return result


def _require_targets(batch: FeatureBatch) -> None:
    if batch.targets is None:
        raise ValueError("Training and cross-validation require observed targets")
    if batch.num_variants < 2:
        raise ValueError("Training requires at least two variants")


def _validate_setting(batch: FeatureBatch, setting: str | None, scheme: str) -> None:
    if setting is None:
        return
    if setting not in SETTING_SCHEMES or SETTING_SCHEMES[setting] != scheme:
        raise ValueError(f"Setting {setting!r} does not use splitting scheme {scheme!r}")
    counts = _mutation_counts(batch)
    if setting in _MIXED_DEPTHS:
        if not bool(((counts == 1) | (counts == 2)).all()):
            raise ValueError("Mixed settings require a batch containing only single and double mutants")
        if not bool((counts == 1).any()) or not bool((counts == 2).any()):
            raise ValueError("Mixed settings require both single and double mutants")
    elif not bool((counts == 1).all()):
        raise ValueError("Single-mutant settings require a batch containing only single mutants")


def _save_fit(fitted: FittedGP, output: Path, n_train: int) -> dict[str, Any]:
    fitted.save(output / "checkpoint.pt")
    _write_json(output / "history.json", fitted.history)
    _write_json(output / "restart_summaries.json", fitted.restart_summaries)
    summary = fitted.fit_summary(n_train) if hasattr(fitted, "fit_summary") else {
        "n_train": n_train,
        "selected_seed": fitted.selected_seed,
        "best_nll_per_variant": fitted.best_nll_per_variant,
    }
    _write_json(output / "fit_summary.json", summary)
    return summary


def _fit_with_failure_artifacts(training: FeatureBatch, config: ExperimentConfig, output: Path,
                                progress, fit_function=None, held_out=None, validation=None):
    diagnostic_handles = []
    try:
        kwargs = {}
        streams = []
        if isinstance(config, ExperimentConfig) and config.training.held_out_interval:
            kwargs["held_out"] = held_out
            streams.append(("held_out_spearman", ("seed", "step", "nll", "nll_per_variant",
                                                 "held_out_spearman", "n_held_out", "held_out_error")))
        if _uses_validation(config):
            kwargs["validation"] = validation
            streams.append(("validation_spearman", ("seed", "step", "nll", "nll_per_variant",
                                                   "validation_spearman", "best_validation_spearman", "n_validation")))
        writers = []
        for metric, columns in streams:
            handle = (output / f"{metric}.csv").open("w", newline="", encoding="utf-8")
            diagnostic_handles.append(handle)
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            handle.flush()
            writers.append((metric, columns, writer, handle))

        def callback(event):
            for metric, columns, writer, handle in writers:
                if metric in event:
                    writer.writerow({name: event[name] for name in columns})
                    handle.flush()
            if progress is not None:
                progress(event)

        return (fit_function or fit_gp)(training, config.model, config.training,
                                       progress=callback if writers else progress, **kwargs)
    except NumericalError as error:
        _write_json(output / "history.json", getattr(error, "history", []))
        _write_json(output / "restart_summaries.json", getattr(error, "restart_summaries", []))
        _write_json(output / "failure.json", {"phase": "training", "error": str(error)})
        raise
    finally:
        for handle in diagnostic_handles:
            handle.close()


def _prediction_rows(
    batch: FeatureBatch,
    prediction: Any,
    fitted: FittedGP,
    fold: int | None = None,
) -> list[dict[str, Any]]:
    columns = {
        "mean": prediction.mean,
        "latent_variance": prediction.latent_covariance.diagonal(),
        "observation_variance": prediction.observation_covariance.diagonal(),
        "mean_standardized": prediction.mean_standardized,
        "latent_variance_standardized": prediction.latent_covariance_standardized.diagonal(),
        "observation_variance_standardized": prediction.observation_covariance_standardized.diagonal(),
    }
    values = {name: tensor.detach().cpu().tolist() for name, tensor in columns.items()}
    counts = _mutation_counts(batch).cpu().tolist()
    targets = batch.targets.detach().cpu().tolist() if batch.targets is not None else [None] * batch.num_variants
    standardized_targets = (
        fitted.transform.standardize_targets(batch.targets).detach().cpu().tolist()
        if batch.targets is not None else [None] * batch.num_variants
    )
    rows = []
    for index, variant_id in enumerate(batch.variant_ids):
        row = {
            "assay_id": batch.assay_id,
            "variant_id": variant_id,
            "fold": fold,
            "target": targets[index],
            "target_standardized": standardized_targets[index],
            "n_mutations": counts[index],
        }
        row.update({name: value[index] for name, value in values.items()})
        rows.append(row)
    return rows


def _variance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for name, selected in (
        ("all", rows),
        ("single_mutants", [row for row in rows if row["n_mutations"] == 1]),
        ("double_mutants", [row for row in rows if row["n_mutations"] == 2]),
    ):
        result[name] = {
            "n_variants": len(selected),
            "mean_latent_variance_standardized": (
                sum(row["latent_variance_standardized"] for row in selected) / len(selected)
                if selected else None
            ),
            "mean_observation_variance_standardized": (
                sum(row["observation_variance_standardized"] for row in selected) / len(selected)
                if selected else None
            ),
        }
    return result


def _evaluate(
    batch: FeatureBatch,
    prediction: Any,
    config: ExperimentConfig,
    indices: Tensor | None = None,
) -> dict[str, Any]:
    settings = config.evaluation
    targets = batch.targets
    mean = prediction.mean
    covariance = prediction.observation_covariance
    if indices is not None:
        indices = indices.to(device=mean.device)
        targets = targets.index_select(0, indices)
        mean = mean.index_select(0, indices)
        covariance = covariance.index_select(0, indices).index_select(1, indices)
    if mean.numel() == 0:
        return {
            "interval_ece": None, "pair_ece": None, "spearman": None,
            "n_variants": 0, "n_pairs": 0, "coverage": [], "bin_stats": [],
        }
    metrics = evaluate_fold(
        targets,
        mean,
        covariance,
        settings.coverage_levels,
        settings.n_bins,
        settings.pair_block_size,
    )
    metrics["spearman"] = spearman(targets, mean)
    return metrics


def _evaluate_settings(
    batch: FeatureBatch,
    prediction: Any,
    config: ExperimentConfig,
    rows: list[dict[str, Any]],
    setting: str | None,
    fold: int,
) -> dict[str, Any]:
    """Evaluate both mixed domains from the same fold posterior."""
    if setting not in _MIXED_DEPTHS:
        metrics = _evaluate(batch, prediction, config)
        metrics.update({"fold": fold, "variance_summary": _variance_summary(rows)})
        if setting is not None:
            metrics["setting"] = setting
        return metrics
    counts = _mutation_counts(batch)
    setting_metrics = {}
    for name, depth in _MIXED_DEPTHS.items():
        indices = torch.nonzero(counts == depth, as_tuple=False).flatten()
        metrics = _evaluate(batch, prediction, config, indices)
        selected_rows = [row for row in rows if row["n_mutations"] == depth]
        metrics.update({
            "setting": name, "fold": fold,
            "variance_summary": _variance_summary(selected_rows),
        })
        setting_metrics[name] = metrics
    result = dict(setting_metrics[setting])
    result.update({"setting_metrics": setting_metrics, "n_predicted_variants": len(rows)})
    return result


def _aggregate_evaluations(
    fold_metrics: list[dict[str, Any]], rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Keep empty depth folds explicit while averaging only defined errors."""
    evaluated = [fold for fold in fold_metrics if fold["n_variants"] > 0]
    if evaluated:
        result = aggregate_folds(evaluated)
    else:
        result = {
            "avg_spearman": None, "std_spearman": None,
            "n_valid_spearman_folds": 0, "interval_ece": None,
            "pair_ece": None, "n_valid_pair_folds": 0,
            "n_variants": 0, "n_pairs": 0,
        }
    result.update({
        "n_folds": len(fold_metrics),
        "n_valid_interval_folds": len(evaluated),
        "fold_n_variants": [fold["n_variants"] for fold in fold_metrics],
        "fold_n_pairs": [fold["n_pairs"] for fold in fold_metrics],
        "fold_metrics": fold_metrics,
        "variance_summary": _variance_summary(rows),
        "spearman_oof": spearman(
            torch.tensor([row["target"] for row in rows], dtype=torch.float64),
            torch.tensor([row["mean"] for row in rows], dtype=torch.float64),
        ),
        "spearman_oof_standardized": spearman(
            torch.tensor([row["target_standardized"] for row in rows], dtype=torch.float64),
            torch.tensor([row["mean_standardized"] for row in rows], dtype=torch.float64),
        ),
    })
    return result


def _write_fold_assignments(output: Path, batch: FeatureBatch, folds: Tensor) -> None:
    _write_csv(
        output / "fold_assignments.csv",
        [{"variant_id": variant_id, "fold": fold} for variant_id, fold in zip(batch.variant_ids, folds.tolist())],
    )


def _run_fold(
    batch, folds, config, output, setting, fold, progress=None, fit_function=None,
    cancelled=None,
):
    """Fit and evaluate one fold without exporting its model or covariances."""
    training_mask = folds != fold
    validation, validation_fold = None, None
    if _uses_validation(config):
        validation_fold = (fold + 1) % config.evaluation.n_folds
        training_mask &= folds != validation_fold
        validation = batch.select(torch.nonzero(folds == validation_fold, as_tuple=False).flatten())
    training = batch.select(torch.nonzero(training_mask, as_tuple=False).flatten())
    test = batch.select(torch.nonzero(folds == fold, as_tuple=False).flatten())
    fold_output = prepare_output(output / f"fold_{fold}", config)
    if validation is not None:
        _write_csv(fold_output / "partition_assignments.csv", [
            {"variant_id": variant_id, "fold": assigned,
             "partition": "test" if assigned == fold else "validation" if assigned == validation_fold else "train"}
            for variant_id, assigned in zip(batch.variant_ids, folds.tolist())
        ])
        _write_json(fold_output / "partition_summary.json", {
            "test_fold": fold, "validation_fold": validation_fold,
            "training_folds": [i for i in range(config.evaluation.n_folds) if i not in (fold, validation_fold)],
            "n_train": training.num_variants, "n_validation": validation.num_variants, "n_test": test.num_variants,
        })
    phase = "training"
    try:
        callback = None
        if progress is not None:
            callback = lambda event: progress({**event, "fold": fold})
            callback({"event": "fold_start", "n_train": training.num_variants, "n_test": test.num_variants,
                      "n_validation": 0 if validation is None else validation.num_variants})
        held_out = test
        if setting in _MIXED_DEPTHS and isinstance(config, ExperimentConfig) and config.training.held_out_interval:
            held_out = test.select(torch.nonzero(_mutation_counts(test) == _MIXED_DEPTHS[setting],
                                                as_tuple=False).flatten())
        fitted = _fit_with_failure_artifacts(training, config, fold_output, callback, fit_function,
                                             held_out=held_out, validation=validation)
        if callback is not None:
            callback({"event": "fold_training_complete"})
        if cancelled is not None and cancelled.is_set():
            raise CancelledError("Fold cancelled after another fold failed or the parent stopped")
        phase = "checkpoint"
        _save_fit(fitted, fold_output, training.num_variants)
        phase = "prediction"
        if callback is not None:
            callback({"event": "fold_prediction_start"})
        prediction = fitted.predict(test)
        if callback is not None:
            callback({"event": "fold_prediction_complete"})
        rows = _prediction_rows(test, prediction, fitted, fold)
        _write_csv(fold_output / "predictions.csv", rows)
        phase = "evaluation"
        metrics = _evaluate_settings(test, prediction, config, rows, setting, fold)
        _write_json(fold_output / "metrics.json", metrics)
        if callback is not None:
            callback({"event": "fold_evaluation_complete"})
        return rows, metrics
    except CancelledError as error:
        _write_json(fold_output / "cancelled.json", {"phase": phase, "error": str(error)})
        raise
    except Exception as error:
        # Numerical training failures already include their detailed history.
        if not (fold_output / "failure.json").exists():
            _write_json(fold_output / "failure.json", {
                "phase": phase, "error": str(error), "traceback": traceback.format_exc(),
            })
        raise


@contextmanager
def _worker_thread_environment(num_threads):
    """Limit native pools before spawned children import NumPy and PyTorch."""
    previous = {name: os.environ.get(name) for name in _THREAD_ENVIRONMENT}
    try:
        for name in _THREAD_ENVIRONMENT:
            os.environ[name] = str(num_threads)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _validate_parallel_fit_function(fit_function):
    """Fail before writing artifacts when a callback cannot survive spawn."""
    if fit_function is None:
        return
    message = (
        "A custom fit_function for parallel folds must be an importable module-level callable; "
        "use fold_workers=1 for local callbacks"
    )
    module_name = getattr(fit_function, "__module__", None)
    name = getattr(fit_function, "__name__", None)
    if (
        not callable(fit_function)
        or not isinstance(module_name, str)
        or module_name in ("__main__", "__mp_main__")
        or not isinstance(name, str)
        or not name.isidentifier()
        or getattr(fit_function, "__qualname__", None) != name
    ):
        raise ValueError(message)
    try:
        module = importlib.import_module(module_name)
        if getattr(module, name, None) is not fit_function:
            raise ValueError("Callback differs from the module-level object")
        multiprocessing.reduction.ForkingPickler.dumps(fit_function)
    except Exception as error:
        raise ValueError(message) from error


def _initialize_fold_worker(batch, folds, config, output, setting, events, cancelled, fit_function):
    global _FOLD_WORKER_STATE
    torch.set_num_threads(config.num_threads)
    torch.set_num_interop_threads(1)
    _FOLD_WORKER_STATE = (batch, folds, config, output, setting, events, cancelled, fit_function)


def _run_fold_worker(fold):
    batch, folds, config, output, setting, events, cancelled, fit_function = _FOLD_WORKER_STATE
    if cancelled.is_set():
        raise CancelledError("Fold cancelled before starting")
    fold_output = output / f"fold_{fold}"
    progress_handle = None

    def progress(event):
        nonlocal progress_handle
        if cancelled.is_set():
            raise CancelledError("Fold cancelled after another fold failed or the parent stopped")
        if progress_handle is None:
            progress_handle = (fold_output / "progress.jsonl").open("a", encoding="utf-8")
        progress_handle.write(json.dumps(event, allow_nan=False) + "\n")
        if "event" in event or event.get("step", 0) % 100 == 0:
            progress_handle.flush()
        if event.get("event") == "fold_start":
            _write_json(fold_output / "execution.json", {
                "pid": os.getpid(), "num_threads": torch.get_num_threads(),
                "num_interop_threads": torch.get_num_interop_threads(),
                "thread_environment": {name: os.environ.get(name) for name in _THREAD_ENVIRONMENT},
            })
        events.put(event)

    try:
        return _run_fold(
            batch, folds, config, output, setting, fold, progress,
            fit_function=fit_function, cancelled=cancelled,
        )
    finally:
        if progress_handle is not None:
            progress_handle.close()


def _drain_fold_events(events, progress):
    while True:
        try:
            event = events.get_nowait()
        except queue.Empty:
            return
        if progress is not None:
            progress(event)


def _run_parallel_folds(batch, folds, config, output, setting, fold_workers, progress, fit_function=None):
    context = multiprocessing.get_context("spawn")
    events, cancelled = context.Queue(), context.Event()
    received = queue.Queue()
    receiver_finished = threading.Event()

    def receive_events():
        # Keep the IPC pipe drained during executor shutdown as well as training.
        # User callbacks are dispatched separately on the parent's main thread.
        while True:
            try:
                event = events.get()
            except (EOFError, OSError, ValueError):
                return
            if event is None:
                receiver_finished.set()
                return
            received.put(event)

    receiver = threading.Thread(target=receive_events, daemon=True)
    receiver.start()
    results = {}
    futures = {}
    failed_fold = None
    try:
        with _worker_thread_environment(config.num_threads):
            with ProcessPoolExecutor(
                max_workers=fold_workers, mp_context=context,
                initializer=_initialize_fold_worker,
                initargs=(batch, folds, config, output, setting, events, cancelled, fit_function),
            ) as executor:
                try:
                    futures = {executor.submit(_run_fold_worker, fold): fold for fold in folds.unique(sorted=True).tolist()}
                    pending = set(futures)
                    while pending:
                        finished, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                        _drain_fold_events(received, progress)
                        for future in finished:
                            failed_fold = futures[future]
                            results[failed_fold] = future.result()
                            if progress is not None:
                                progress({"event": "fold_complete", "fold": failed_fold})
                            failed_fold = None
                except BaseException:
                    cancelled.set()
                    for future in futures:
                        future.cancel()
                    raise
    except BaseException as error:
        _write_json(output / "failure.json", {
            "phase": "parallel_cv", "fold": failed_fold,
            "error": str(error) or type(error).__name__, "traceback": traceback.format_exc(),
        })
        raise
    finally:
        events.put(None)
        # A worker killed externally can leave its queue pipe corrupt. Do not
        # turn a BrokenProcessPool error into an unbounded receiver join.
        receiver.join(timeout=5)
        if not receiver_finished.is_set():
            events.cancel_join_thread()
        events.close()
        if receiver_finished.is_set():
            events.join_thread()
    _drain_fold_events(received, progress)
    return [results[fold] for fold in folds.unique(sorted=True).tolist()]


def run_cv(
    batch: FeatureBatch,
    folds: Tensor,
    config: ExperimentConfig,
    output_dir: str | Path,
    *,
    scheme: str = "random",
    setting: str | None = None,
    split_source: str = "provided",
    progress: Callable[[dict], None] | None = None,
    fit_function: Callable | None = None,
    fold_workers: int = 1,
) -> dict[str, Any]:
    """Cross-validate, optionally using CPU processes with num_threads per fold.

    Parallel custom fit functions must be importable module-level callables.
    Progress callbacks run in the parent and may remain local closures.
    Only populated published folds are evaluated; their original IDs are retained.
    """
    config = _resolve_assay_config(batch, config)
    if type(fold_workers) is not int or not 1 <= fold_workers <= config.evaluation.n_folds:
        raise ValueError("fold_workers must be an integer between 1 and n_folds")
    if fold_workers > 1:
        if torch.device(config.device).type != "cpu":
            raise ValueError("Parallel folds require device='cpu'")
        _validate_parallel_fit_function(fit_function)
    _require_targets(batch)
    _validate_setting(batch, setting, scheme)
    torch.set_num_threads(config.num_threads)
    batch = batch.to(config.device)
    folds = _validate_fold_assignments(batch, folds, config, scheme)
    fold_ids = folds.unique(sorted=True).tolist()
    fold_workers = min(fold_workers, len(fold_ids))
    output = prepare_output(output_dir, config)
    _write_fold_assignments(output, batch, folds)
    if fold_workers > 1:
        _write_json(output / "execution.json", {
            "fold_workers": fold_workers, "threads_per_fold": config.num_threads,
            "start_method": "spawn",
        })
        fold_results = _run_parallel_folds(
            batch, folds, config, output, setting, fold_workers, progress, fit_function,
        )
    else:
        fold_results = (
            _run_fold(batch, folds, config, output, setting, fold, progress, fit_function)
            for fold in fold_ids
        )
    by_variant: dict[str, dict[str, Any]] = {}
    fold_metrics = []
    for result_index, (rows, metrics) in enumerate(fold_results):
        fold_metrics.append(metrics)
        by_variant.update({row["variant_id"]: row for row in rows})
        if fold_workers == 1 and progress is not None:
            # _run_fold has returned: its model, predictions, and local tensors
            # are now out of scope. Observe natural release without forcing GC.
            progress({"event": "fold_returned", "fold": fold_ids[result_index]})

    ordered_rows = [by_variant[variant_id] for variant_id in batch.variant_ids]
    _write_csv(output / "predictions.csv", ordered_rows)
    metadata = {"assay_id": batch.assay_id, "scheme": scheme, "split_source": split_source,
                "fold_ids": fold_ids}
    if setting in _MIXED_DEPTHS:
        setting_metrics = {}
        for name, depth in _MIXED_DEPTHS.items():
            depth_folds = [fold["setting_metrics"][name] for fold in fold_metrics]
            depth_rows = [row for row in ordered_rows if row["n_mutations"] == depth]
            setting_metrics[name] = {
                **_aggregate_evaluations(depth_folds, depth_rows), **metadata, "setting": name,
            }
        metrics = dict(setting_metrics[setting])
        metrics.update({
            "setting_metrics": setting_metrics, "n_predicted_variants": len(ordered_rows),
        })
    else:
        metrics = {**_aggregate_evaluations(fold_metrics, ordered_rows), **metadata}
        if setting is not None:
            metrics["setting"] = setting
    _write_json(output / "metrics.json", metrics)
    return metrics


def run_predict(
    batch: FeatureBatch,
    checkpoint: str | Path,
    config: ExperimentConfig,
    output_dir: str | Path,
    *,
    load_function: Callable | None = None,
) -> dict[str, Any]:
    """Predict a feature batch; experimental labels are optional and unused."""
    config = _resolve_assay_config(batch, config)
    if batch.num_variants == 0:
        raise ValueError("Prediction requires at least one variant")
    torch.set_num_threads(config.num_threads)
    batch = batch.to(config.device)
    fitted = (load_function or FittedGP.load)(checkpoint, device=config.device)
    if config.model.enabled_branches != fitted.model_config.enabled_branches:
        raise ValueError("Prediction enabled branches differ from the fitted checkpoint")
    if getattr(config.model, "use_structure_distance", None) != getattr(fitted.model_config, "use_structure_distance", None):
        raise ValueError("Prediction structure-distance setting differs from the fitted checkpoint")
    prediction = fitted.predict(batch)
    output = prepare_output(output_dir, config)
    rows = _prediction_rows(batch, prediction, fitted)
    _write_csv(output / "predictions.csv", rows)
    summary = {
        "assay_id": batch.assay_id,
        "mode": "predict",
        "n_variants": batch.num_variants,
        "checkpoint": str(Path(checkpoint).resolve()),
        "variance_summary": _variance_summary(rows),
    }
    _write_json(output / "run_summary.json", summary)
    return summary


def run_extrapolation(
    batch: FeatureBatch,
    config: ExperimentConfig,
    output_dir: str | Path,
    *,
    sample_ratio: float = 0.8,
    split_seed: int = 2024,
    progress: Callable[[dict], None] | None = None,
    fit_function: Callable | None = None,
) -> dict[str, Any]:
    """Train on a configurable sample of singles and evaluate every double.

    Below a ratio of one, sample size and RNG match pandas
    ``sample(frac=sample_ratio, random_state=split_seed)`` given the same incoming
    single-mutant order. A ratio of one uses all singles in their original order.
    Unused singles, wild type, and higher-order variants are recorded without
    entering either partition.
    """
    config = _resolve_assay_config(batch, config)
    _require_targets(batch)
    if _uses_validation(config):
        raise ValueError("validation_spearman requires five-fold singles CV; single_to_double is not supported")
    if (
        isinstance(sample_ratio, bool) or not isinstance(sample_ratio, (int, float))
        or not math.isfinite(sample_ratio) or not 0 < sample_ratio <= 1
    ):
        raise ValueError("sample_ratio must be finite and greater than 0 and at most 1")
    sample_ratio = float(sample_ratio)
    if isinstance(split_seed, bool) or not isinstance(split_seed, int) or not 0 <= split_seed < 2**32:
        raise ValueError("split_seed must be an integer between 0 and 2**32 - 1")
    torch.set_num_threads(config.num_threads)
    batch = batch.to(config.device)
    counts = _mutation_counts(batch)
    single_indices = torch.nonzero(counts == 1, as_tuple=False).flatten().cpu()
    test_indices = torch.nonzero(counts == 2, as_tuple=False).flatten().cpu()
    n_train = round(sample_ratio * len(single_indices))
    if n_train < 2 or len(test_indices) < 1:
        raise ValueError("Single-to-double extrapolation requires at least two sampled training singles and one double mutant")
    if sample_ratio == 1.0:
        train_indices = single_indices
        sampling_method = "all_singles"
    else:
        sampled = np.random.RandomState(split_seed).choice(len(single_indices), size=n_train, replace=False)
        train_indices = single_indices[torch.as_tensor(sampled, dtype=torch.long)]
        sampling_method = "numpy_randomstate_choice_without_replacement"
    training = batch.select(train_indices)
    test = batch.select(test_indices)
    output = prepare_output(output_dir, config)
    partitions = ["unused_single" if count == 1 else "test_double" if count == 2 else "excluded" for count in counts.cpu().tolist()]
    for index in train_indices.tolist():
        partitions[index] = "train_single"
    _write_csv(
        output / "partition_assignments.csv",
        [{"variant_id": variant_id, "partition": partition} for variant_id, partition in zip(batch.variant_ids, partitions)],
    )
    fitted = _fit_with_failure_artifacts(training, config, output, progress, fit_function, held_out=test)
    summary = _save_fit(fitted, output, training.num_variants)
    summary.update(
        {
            "assay_id": batch.assay_id,
            "mode": "single_to_double",
            "setting": "single_to_double",
            "n_test": test.num_variants,
            "n_available_singles": len(single_indices),
            "n_unused_singles": partitions.count("unused_single"),
            "n_excluded": partitions.count("excluded"),
            "sample_ratio": sample_ratio,
            "training_single_fraction": sample_ratio,
            "split_seed": split_seed,
            "sampling_method": sampling_method,
            "train_variant_ids": list(training.variant_ids),
            "test_variant_ids": list(test.variant_ids),
            "unused_single_variant_ids": [
                variant_id for variant_id, partition in zip(batch.variant_ids, partitions)
                if partition == "unused_single"
            ],
            "excluded_variant_ids": [
                variant_id for variant_id, partition in zip(batch.variant_ids, partitions)
                if partition == "excluded"
            ],
        }
    )
    prediction = fitted.predict(test)
    rows = _prediction_rows(test, prediction, fitted)
    _write_csv(output / "predictions.csv", rows)
    metrics = _evaluate(test, prediction, config)
    metrics.update(
        {
            "evaluation_protocol": "single_to_double",
            "setting": "single_to_double",
            "aggregation": "single_fixed_split",
            "sample_ratio": sample_ratio,
            "training_single_fraction": sample_ratio,
            "split_seed": split_seed,
            "variance_summary": _variance_summary(rows),
        }
    )
    _write_json(output / "metrics.json", metrics)
    summary["metrics"] = metrics
    _write_json(output / "run_summary.json", summary)
    return summary
