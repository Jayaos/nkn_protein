"""Evaluation of held-out predictions without pooling calibration across folds.

The covariance accepted here is the observation-predictive covariance in the
same units as the means and targets. Observation noise is already included.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor


def _vector(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 1 or value.is_complex():
        raise ValueError(f"{name} must be a one-dimensional real tensor")
    result = value.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def mse(targets: Tensor, means: Tensor) -> float | None:
    """Mean squared prediction error in the supplied units, or None if empty."""
    predicted = _vector(means, "means")
    observed = _vector(targets, "targets").to(device=predicted.device)
    if observed.shape != predicted.shape:
        raise ValueError("targets and means must have the same length")
    if observed.numel() == 0:
        return None
    result = float((observed - predicted).square().mean().item())
    if not math.isfinite(result):
        raise ValueError("mean squared error must be finite")
    return result


def _confidence_bin_indices(confidences: Tensor, n_bins: int) -> Tensor:
    # right=True puts confidence equal to an interior edge in the bin on its
    # right. Excluding the outer edges also includes confidence exactly one.
    interior_edges = (
        0.5
        + torch.arange(1, n_bins, device=confidences.device, dtype=torch.float64)
        / (2 * n_bins)
    )
    return torch.bucketize(confidences, interior_edges, right=True)


def error_calibration(
    targets: Tensor, means: Tensor, observation_variance: Tensor, n_bins: int = 5,
) -> dict[str, Any]:
    """Kermut Appendix J.2 ENCE and sample CV of predictive standard deviations.

    Variance quantiles use linear interpolation, right-closed intervals, and
    dropped duplicate edges, matching pandas.qcut(..., duplicates='drop').
    Constant variances form one bin and have exactly zero CV for two or more
    variants; empty bins never contribute to ENCE.
    RMV/RMSE use the supplied target scale. Scalars are invariant to a common
    change of target units. A singleton has defined ENCE but undefined CV.
    """
    _positive_integer(n_bins, "error_n_bins")
    predicted = _vector(means, "means")
    observed = _vector(targets, "targets").to(device=predicted.device)
    variance = _vector(observation_variance, "observation_variance").to(device=predicted.device).contiguous()
    if observed.shape != predicted.shape or variance.shape != predicted.shape:
        raise ValueError("targets, means, and observation_variance must have the same length")
    if not bool((variance > 0).all()):
        raise ValueError("observation_variance must be positive")
    result = {"ence": None, "uncertainty_cv": None, "error_n_bins": n_bins,
              "n_error_bins": 0, "error_bin_stats": []}
    if predicted.numel() == 0:
        return result
    squared_errors = (observed - predicted).square()
    if not bool(torch.isfinite(squared_errors).all()):
        raise ValueError("squared prediction errors must be finite")
    edges = torch.quantile(variance, torch.linspace(0, 1, n_bins + 1, device=variance.device,
                                                  dtype=torch.float64)).unique_consecutive()
    assignments = torch.bucketize(variance, edges[1:-1], right=False)
    bins = []
    for index in assignments.unique(sorted=True).tolist():
        selected = assignments == index
        rmv = float(variance[selected].mean().sqrt().item())
        rmse = float(squared_errors[selected].mean().sqrt().item())
        relative_error = abs(rmv - rmse) / rmv
        if not math.isfinite(relative_error):
            raise ValueError("normalized calibration error must be finite")
        bins.append({"bin": index, "count": int(selected.sum().item()),
                     "rmv": rmv, "rmse": rmse, "relative_error": relative_error})
    sigma = variance.sqrt()
    if sigma.numel() < 2:
        cv = None
    elif edges.numel() == 1:
        # Equal variances imply zero spread. A reduction over identical,
        # inexact floating-point standard deviations can otherwise leave a
        # tiny positive residual. Only exact equality takes this branch.
        cv = 0.0
    else:
        cv = float((sigma.std(unbiased=True) / sigma.mean()).item())
    if cv is not None and not math.isfinite(cv):
        raise ValueError("uncertainty coefficient of variation must be finite")
    result.update(ence=sum(row["relative_error"] for row in bins) / len(bins),
                  uncertainty_cv=cv, n_error_bins=len(bins), error_bin_stats=bins)
    return result


def evaluate_fold(
    targets: Tensor,
    mean: Tensor,
    observation_covariance: Tensor,
    coverage_levels: Sequence[float],
    n_bins: int,
    pair_block_size: int = 512,
    error_n_bins: int = 5,
) -> dict[str, Any]:
    """Evaluate one held-out fold and return JSON-serializable statistics.

    All unordered pairs within this fold are evaluated, except exact ties in
    either targets or predictive means. Pair calculations are blocked to avoid
    allocating a full pair-index list; the supplied covariance is still dense.
    Invalid evaluated difference variances raise an error rather than being
    replaced with arbitrary positive values.
    """
    _positive_integer(n_bins, "n_bins")
    _positive_integer(pair_block_size, "pair_block_size")
    _positive_integer(error_n_bins, "error_n_bins")
    means = _vector(mean, "mean")
    observed = _vector(targets, "targets").to(device=means.device)
    n = means.numel()
    if n == 0 or observed.shape != means.shape:
        raise ValueError("targets and mean must have the same nonzero length")

    if (
        not isinstance(observation_covariance, Tensor)
        or observation_covariance.shape != (n, n)
        or observation_covariance.is_complex()
    ):
        raise ValueError("observation_covariance must be a real [N, N] tensor")
    covariance = observation_covariance.detach().to(
        device=means.device, dtype=torch.float64
    )
    if not bool(torch.isfinite(covariance).all()):
        raise ValueError("observation_covariance must contain only finite values")
    if not torch.allclose(covariance, covariance.T, rtol=1e-8, atol=1e-10):
        raise ValueError("observation_covariance must be symmetric")
    diagonal = covariance.diagonal()
    if not bool((diagonal > 0).all()):
        raise ValueError("observation_covariance must have positive diagonal values")

    levels = list(coverage_levels)
    if any(isinstance(q, bool) for q in levels):
        raise ValueError("coverage_levels must contain numbers, not booleans")
    levels = [float(q) for q in levels]
    if not levels or any(not math.isfinite(q) or not 0 <= q <= 1 for q in levels):
        raise ValueError("coverage_levels must be nonempty, finite, and between 0 and 1 inclusive")
    absolute_errors = (observed - means).abs()
    standard_deviations = diagonal.sqrt()
    coverage: list[dict[str, float]] = []
    for q in levels:
        # sqrt(2) erfinv(q) equals Phi^-1((1+q)/2) without the addition
        # rounding to one for representable coverage levels very close to one.
        # q=0 has zero width (exact matches count); q=1 has infinite width.
        # Strictly positive variances make both endpoint intervals well-defined.
        z = math.sqrt(2.0) * torch.erfinv(means.new_tensor(q))
        empirical = float(
            (absolute_errors <= z * standard_deviations)
            .to(dtype=torch.float64)
            .mean()
            .item()
        )
        coverage.append({"nominal": q, "empirical": empirical})
    interval_ece = sum(abs(row["empirical"] - row["nominal"]) for row in coverage)
    interval_ece /= len(coverage)

    counts = torch.zeros(n_bins, device=means.device, dtype=torch.int64)
    confidence_sums = torch.zeros(n_bins, device=means.device, dtype=torch.float64)
    correct_sums = torch.zeros_like(confidence_sums)
    for row_start in range(0, n, pair_block_size):
        row_stop = min(row_start + pair_block_size, n)
        rows = slice(row_start, row_stop)
        row_indices = torch.arange(row_start, row_stop, device=means.device)
        for col_start in range(row_start, n, pair_block_size):
            col_stop = min(col_start + pair_block_size, n)
            columns = slice(col_start, col_stop)
            col_indices = torch.arange(col_start, col_stop, device=means.device)
            mean_difference = means[rows, None] - means[None, columns]
            target_difference = observed[rows, None] - observed[None, columns]
            eligible = (
                (row_indices[:, None] < col_indices[None, :])
                & (mean_difference != 0)
                & (target_difference != 0)
            )
            if not bool(eligible.any()):
                continue
            difference_variance = (
                diagonal[rows, None]
                + diagonal[None, columns]
                - 2 * covariance[rows, columns]
            )[eligible]
            if not bool(
                (torch.isfinite(difference_variance) & (difference_variance > 0)).all()
            ):
                raise ValueError("eligible pair difference variances must be finite and positive")

            pair_mean_difference = mean_difference[eligible]
            pair_target_difference = target_difference[eligible]
            z = pair_mean_difference.abs() / difference_variance.sqrt()
            confidences = 0.5 * (1 + torch.erf(z / math.sqrt(2.0)))
            # Comparing signs avoids overflow in the product of differences.
            correct = ((pair_mean_difference > 0) == (pair_target_difference > 0)).to(
                dtype=torch.float64
            )
            bins = _confidence_bin_indices(confidences, n_bins)
            counts += torch.bincount(bins, minlength=n_bins)
            confidence_sums += torch.bincount(bins, weights=confidences, minlength=n_bins)
            correct_sums += torch.bincount(bins, weights=correct, minlength=n_bins)

    count_values = counts.cpu().tolist()
    confidence_values = confidence_sums.cpu().tolist()
    correct_values = correct_sums.cpu().tolist()
    n_pairs = sum(count_values)
    bin_stats: list[dict[str, Any]] = []
    error_sum = 0.0
    for index, count in enumerate(count_values):
        confidence = confidence_values[index] / count if count else None
        accuracy = correct_values[index] / count if count else None
        bin_stats.append(
            {
                "bin": index,
                "lower": 0.5 + index / (2 * n_bins),
                "upper": 0.5 + (index + 1) / (2 * n_bins),
                "count": count,
                "confidence": confidence,
                "accuracy": accuracy,
            }
        )
        if count:
            error_sum += abs(correct_values[index] - confidence_values[index])

    return {
        **error_calibration(observed, means, diagonal, error_n_bins),
        "mse": mse(observed, means),
        "interval_ece": interval_ece,
        "pair_ece": error_sum / n_pairs if n_pairs else None,
        "n_variants": n,
        "n_pairs": n_pairs,
        "coverage": coverage,
        "bin_stats": bin_stats,
    }


def aggregate_folds(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Equally average folds; Spearman SD uses ddof=0 over defined values.

    Missing/undefined Spearman and undefined PairECE values are omitted from
    their respective summaries, with contributing-fold counts reported.
    ENCE and uncertainty CV also average defined folds equally, with sample
    standard deviations (ddof=1) and separate valid-fold counts.
    """
    if not folds:
        raise ValueError("at least one evaluated fold is required")
    interval_values = [float(fold["interval_ece"]) for fold in folds]
    pair_values = [float(fold["pair_ece"]) for fold in folds if fold["pair_ece"] is not None]
    spearman_values = [float(fold["spearman"]) for fold in folds if fold.get("spearman") is not None]
    if any(not math.isfinite(value) for value in interval_values + pair_values + spearman_values):
        raise ValueError("fold metric values must be finite or, for PairECE/Spearman, None")
    avg_spearman = sum(spearman_values) / len(spearman_values) if spearman_values else None
    std_spearman = (
        math.sqrt(sum((value - avg_spearman) ** 2 for value in spearman_values) / len(spearman_values))
        if spearman_values else None
    )
    fold_n_variants = [int(fold["n_variants"]) for fold in folds]
    fold_n_pairs = [int(fold["n_pairs"]) for fold in folds]
    error_metrics = {}
    for name in ("ence", "uncertainty_cv"):
        values = [float(fold[name]) for fold in folds if fold.get(name) is not None]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"{name} values must be finite and nonnegative or None")
        average = sum(values) / len(values) if values else None
        error_metrics.update({
            name: average,
            f"std_{name}": (math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))
                            if len(values) > 1 else None),
            f"n_valid_{name}_folds": len(values),
        })
    return {
        **error_metrics,
        "avg_spearman": avg_spearman,
        "std_spearman": std_spearman,
        "n_valid_spearman_folds": len(spearman_values),
        "interval_ece": sum(interval_values) / len(interval_values),
        "pair_ece": sum(pair_values) / len(pair_values) if pair_values else None,
        "n_folds": len(folds),
        "n_valid_pair_folds": len(pair_values),
        "n_variants": sum(fold_n_variants),
        "n_pairs": sum(fold_n_pairs),
        "fold_n_variants": fold_n_variants,
        "fold_n_pairs": fold_n_pairs,
    }


def _average_ranks(values: Tensor) -> Tensor:
    ordered, indices = torch.sort(values)
    _, counts = torch.unique_consecutive(ordered, return_counts=True)
    stops = counts.cumsum(0)
    starts = stops - counts
    average = (starts.to(dtype=torch.float64) + stops - 1) / 2 + 1
    ordered_ranks = torch.repeat_interleave(average, counts)
    ranks = torch.empty_like(values)
    ranks[indices] = ordered_ranks
    return ranks


def spearman(targets: Tensor, means: Tensor) -> float | None:
    """Spearman correlation using average ranks, or None when undefined."""
    predicted = _vector(means, "means")
    observed = _vector(targets, "targets").to(device=predicted.device)
    if observed.shape != predicted.shape:
        raise ValueError("targets and means must have the same length")
    if observed.numel() < 2:
        return None
    target_ranks = _average_ranks(observed)
    prediction_ranks = _average_ranks(predicted)
    target_ranks -= target_ranks.mean()
    prediction_ranks -= prediction_ranks.mean()
    denominator = target_ranks.norm() * prediction_ranks.norm()
    if float(denominator.item()) == 0:
        return None
    correlation = torch.dot(target_ranks, prediction_ranks) / denominator
    return float(correlation.clamp(-1, 1).item())
