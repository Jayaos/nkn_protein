"""Fit, predict, and cross-validate with reproducible local artifacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from config import ExperimentConfig
from data_loading import FeatureBatch
from evaluation import aggregate_folds, evaluate_fold, spearman
from gp.linalg import NumericalError
from gp.training import FittedGP, fit_gp


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


def _validate_fold_assignments(batch: FeatureBatch, folds: Tensor, config: ExperimentConfig, scheme: str) -> Tensor:
    if scheme not in ("random", "modulo", "contiguous"):
        raise ValueError(f"Unknown splitting scheme: {scheme}")
    if not isinstance(folds, Tensor) or folds.ndim != 1 or len(folds) != batch.num_variants:
        raise ValueError("One fold assignment is required for every variant")
    if folds.dtype == torch.bool or folds.is_floating_point() or folds.is_complex():
        raise ValueError("Fold assignments must be integer identifiers")
    result = folds.detach().to(device="cpu", dtype=torch.long)
    expected = set(range(config.evaluation.n_folds))
    if set(result.tolist()) != expected:
        raise ValueError(f"Every configured fold must be present, with identifiers {sorted(expected)}")
    if any(batch.num_variants - int((result == fold).sum()) < 2 for fold in expected):
        raise ValueError("Every training partition must contain at least two variants")
    if scheme in ("modulo", "contiguous"):
        if not bool((_mutation_counts(batch) == 1).all()):
            raise ValueError("Positional evaluation currently supports single substitutions only; multi-mutant rules are deferred")
        positions = torch.empty(batch.num_variants, dtype=torch.long)
        positions[batch.owners.cpu()] = batch.mutation_positions.cpu()
        for position in positions.unique():
            if result[positions == position].unique().numel() != 1:
                raise ValueError("A mutation position occurs in multiple positional folds")
    return result


def _require_targets(batch: FeatureBatch) -> None:
    if batch.targets is None:
        raise ValueError("Training and cross-validation require observed targets")
    if batch.num_variants < 2:
        raise ValueError("Training requires at least two variants")


def _save_fit(fitted: FittedGP, output: Path, n_train: int) -> dict[str, Any]:
    fitted.save(output / "checkpoint.pt")
    _write_json(output / "history.json", fitted.history)
    _write_json(output / "restart_summaries.json", fitted.restart_summaries)
    summary = {
        "n_train": n_train,
        "selected_seed": fitted.selected_seed,
        "best_nll_per_variant": fitted.best_nll_per_variant,
    }
    _write_json(output / "fit_summary.json", summary)
    return summary


def _fit_with_failure_artifacts(training: FeatureBatch, config: ExperimentConfig, output: Path, progress):
    try:
        return fit_gp(training, config.model, config.training, progress=progress)
    except NumericalError as error:
        _write_json(output / "history.json", getattr(error, "history", []))
        _write_json(output / "restart_summaries.json", getattr(error, "restart_summaries", []))
        _write_json(output / "failure.json", {"phase": "training", "error": str(error)})
        raise


def _prediction_rows(batch: FeatureBatch, prediction: Any, fold: int | None = None) -> list[dict[str, Any]]:
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
    rows = []
    for index, variant_id in enumerate(batch.variant_ids):
        row = {
            "assay_id": batch.assay_id,
            "variant_id": variant_id,
            "fold": fold,
            "target": targets[index],
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


def _evaluate(batch: FeatureBatch, prediction: Any, config: ExperimentConfig) -> dict[str, Any]:
    settings = config.evaluation
    metrics = evaluate_fold(
        batch.targets,
        prediction.mean,
        prediction.observation_covariance,
        settings.coverage_levels,
        settings.n_bins,
        settings.pair_block_size,
    )
    metrics["spearman"] = spearman(batch.targets, prediction.mean)
    return metrics


def _write_fold_assignments(output: Path, batch: FeatureBatch, folds: Tensor) -> None:
    _write_csv(
        output / "fold_assignments.csv",
        [{"variant_id": variant_id, "fold": fold} for variant_id, fold in zip(batch.variant_ids, folds.tolist())],
    )


def run_fit(
    batch: FeatureBatch,
    config: ExperimentConfig,
    output_dir: str | Path,
    *,
    folds: Tensor | None = None,
    test_fold: int = 0,
    scheme: str = "random",
    split_source: str = "provided",
    progress: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Fit all variants, or train on the complement of one explicit test fold."""
    _require_targets(batch)
    torch.set_num_threads(config.num_threads)
    batch = batch.to(config.device)
    if folds is None:
        training = batch
        test = None
    else:
        folds = _validate_fold_assignments(batch, folds, config, scheme)
        if isinstance(test_fold, bool) or not isinstance(test_fold, int) or not 0 <= test_fold < config.evaluation.n_folds:
            raise ValueError("test_fold must name one configured fold")
        training = batch.select(torch.nonzero(folds != test_fold, as_tuple=False).flatten())
        test = batch.select(torch.nonzero(folds == test_fold, as_tuple=False).flatten())
    output = prepare_output(output_dir, config)
    if folds is not None:
        _write_fold_assignments(output, batch, folds)
    fitted = _fit_with_failure_artifacts(training, config, output, progress)
    summary = _save_fit(fitted, output, training.num_variants)
    summary.update({"assay_id": batch.assay_id, "mode": "fit", "n_test": 0 if test is None else test.num_variants})
    if test is not None:
        prediction = fitted.predict(test)
        rows = _prediction_rows(test, prediction, test_fold)
        _write_csv(output / "predictions.csv", rows)
        metrics = _evaluate(test, prediction, config)
        metrics["variance_summary"] = _variance_summary(rows)
        metrics.update({"scheme": scheme, "split_source": split_source, "test_fold": test_fold})
        _write_json(output / "metrics.json", metrics)
        summary["metrics"] = metrics
    _write_json(output / "run_summary.json", summary)
    return summary


def run_cv(
    batch: FeatureBatch,
    folds: Tensor,
    config: ExperimentConfig,
    output_dir: str | Path,
    *,
    scheme: str = "random",
    split_source: str = "provided",
    progress: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Cross-validate one assay/scheme; average calibration errors within folds."""
    _require_targets(batch)
    torch.set_num_threads(config.num_threads)
    batch = batch.to(config.device)
    folds = _validate_fold_assignments(batch, folds, config, scheme)
    output = prepare_output(output_dir, config)
    _write_fold_assignments(output, batch, folds)
    by_variant: dict[str, dict[str, Any]] = {}
    fold_metrics = []
    for fold in range(config.evaluation.n_folds):
        training = batch.select(torch.nonzero(folds != fold, as_tuple=False).flatten())
        test = batch.select(torch.nonzero(folds == fold, as_tuple=False).flatten())
        fold_output = prepare_output(output / f"fold_{fold}", config)
        callback = None
        if progress is not None:
            callback = lambda event, fold=fold: progress({**event, "fold": fold})
            progress({"event": "fold_start", "fold": fold, "n_train": training.num_variants, "n_test": test.num_variants})
        fitted = _fit_with_failure_artifacts(training, config, fold_output, callback)
        _save_fit(fitted, fold_output, training.num_variants)
        prediction = fitted.predict(test)
        rows = _prediction_rows(test, prediction, fold)
        _write_csv(fold_output / "predictions.csv", rows)
        metrics = _evaluate(test, prediction, config)
        metrics["fold"] = fold
        metrics["variance_summary"] = _variance_summary(rows)
        _write_json(fold_output / "metrics.json", metrics)
        fold_metrics.append(metrics)
        by_variant.update({row["variant_id"]: row for row in rows})
        # Only scalar per-variant outputs survive this fold; no covariance is pooled.
        del prediction, fitted

    ordered_rows = [by_variant[variant_id] for variant_id in batch.variant_ids]
    _write_csv(output / "predictions.csv", ordered_rows)
    oof_means = torch.tensor([row["mean"] for row in ordered_rows], dtype=torch.float64)
    metrics = aggregate_folds(fold_metrics)
    metrics.update(
        {
            "assay_id": batch.assay_id,
            "scheme": scheme,
            "split_source": split_source,
            "spearman_oof": spearman(batch.targets, oof_means),
            "variance_summary": _variance_summary(ordered_rows),
            "fold_metrics": fold_metrics,
        }
    )
    _write_json(output / "metrics.json", metrics)
    return metrics


def run_predict(
    batch: FeatureBatch,
    checkpoint: str | Path,
    config: ExperimentConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Predict a feature batch; experimental labels are optional and unused."""
    if batch.num_variants == 0:
        raise ValueError("Prediction requires at least one variant")
    torch.set_num_threads(config.num_threads)
    batch = batch.to(config.device)
    fitted = FittedGP.load(checkpoint, device=config.device)
    if config.model.enabled_branches != fitted.model_config.enabled_branches:
        raise ValueError("Prediction enabled branches differ from the fitted checkpoint")
    prediction = fitted.predict(batch)
    output = prepare_output(output_dir, config)
    rows = _prediction_rows(batch, prediction)
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
    progress: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Train on all single mutants and evaluate all doubles in one fixed split.

    Wild-type and higher-order variants are excluded from both partitions and
    counted in the output. This protocol does not average across artificial CV
    folds or assign multi-mutants to positional folds.
    """
    _require_targets(batch)
    torch.set_num_threads(config.num_threads)
    batch = batch.to(config.device)
    counts = _mutation_counts(batch)
    training = batch.select(torch.nonzero(counts == 1, as_tuple=False).flatten())
    test = batch.select(torch.nonzero(counts == 2, as_tuple=False).flatten())
    if training.num_variants < 2 or test.num_variants < 1:
        raise ValueError("Single-to-double extrapolation requires at least two singles and one double mutant")
    output = prepare_output(output_dir, config)
    partitions = ["train_single" if count == 1 else "test_double" if count == 2 else "excluded" for count in counts.cpu().tolist()]
    _write_csv(
        output / "partition_assignments.csv",
        [{"variant_id": variant_id, "partition": partition} for variant_id, partition in zip(batch.variant_ids, partitions)],
    )
    fitted = _fit_with_failure_artifacts(training, config, output, progress)
    summary = _save_fit(fitted, output, training.num_variants)
    summary.update(
        {
            "assay_id": batch.assay_id,
            "mode": "single_to_double",
            "n_test": test.num_variants,
            "n_excluded": batch.num_variants - training.num_variants - test.num_variants,
        }
    )
    prediction = fitted.predict(test)
    rows = _prediction_rows(test, prediction)
    _write_csv(output / "predictions.csv", rows)
    metrics = _evaluate(test, prediction, config)
    metrics.update(
        {
            "evaluation_protocol": "single_to_double",
            "aggregation": "single_fixed_split",
            "variance_summary": _variance_summary(rows),
        }
    )
    _write_json(output / "metrics.json", metrics)
    summary["metrics"] = metrics
    _write_json(output / "run_summary.json", summary)
    return summary
