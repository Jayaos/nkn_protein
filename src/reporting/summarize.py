"""Print prediction and calibration statistics across assays from summary.csv.

Usage: python src/reporting/summarize.py results/EXPERIMENT/aggregate/summary.csv

Defaults: training-standardized out-of-fold Spearman and MSE, interval ECE,
ENCE, uncertainty CV, and PairECE. Each complete CV row contributes its saved metric with
equal assay weight. Standard deviation is across assays (population SD,
ddof=0), not the saved within-assay fold SD. Runs sharing a prefix are grouped
after removing numeric job/task suffixes and a trailing _gpu marker, combining
general and GPU runs before computing statistics. For each metric independently,
use the latest valid job per assay and split across both; missing metric columns yield N/A.
The average row combines all three splits within each assay, then summarizes
those per-assay averages. Only assays with three valid values contribute.
"""

import argparse
from collections import Counter
import csv
import math
from pathlib import Path
import re
from statistics import fmean, pstdev


SETTINGS = ("random", "modulo", "contiguous")
SETTING_NAMES = {f"singles_{name}": name for name in SETTINGS}
DEFAULT_METRICS = (
    "spearman_oof_standardized", "mse_oof_standardized", "interval_ece", "ence", "uncertainty_cv",
    "pair_ece",
)
METRIC_LABELS = {
    "spearman_oof_standardized": "Spearman (pooled OOF, training-standardized)",
    "mse_oof_standardized": "MSE (training-standardized)",
    "interval_ece": "ECE (interval)",
    "ence": "ENCE",
    "uncertainty_cv": "Uncertainty CV",
    "avg_spearman": "Spearman (fold mean)",
    "mse_oof": "MSE (original target units)",
    "pair_ece": "PairECE",
}


def _run_group(run_id: str) -> tuple[str, int]:
    """Remove Slurm job/task suffixes and merge trailing GPU run markers."""
    # A manual run's trailing task index is not a job ID.
    match = None if "_manual_" in run_id else re.search(r"_(\d+)(?:_\d+)?$", run_id)
    prefix, job_id = (run_id[:match.start()], int(match[1])) if match else (run_id, -1)
    return prefix.removesuffix("_gpu"), job_id


def summarize_csv(path: str | Path, metric: str = "spearman_oof_standardized") -> list[dict]:
    """Summarize one saved metric per run prefix and singles split.

    Default to pooled Spearman in training-standardized units. Missing columns
    and invalid values are skipped independently for each metric, without
    substituting fold-mean Spearman or metrics in original target units.
    """
    if metric not in METRIC_LABELS:
        raise ValueError(f"Unknown metric: {metric}")
    groups: dict[str, dict[str, dict[str, tuple[int, str, float]]]] = {}
    group_assays: dict[str, set[str]] = {}
    skipped: Counter = Counter()
    seen = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"assay_id", "run_id", "setting"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        for row in reader:
            setting = SETTING_NAMES.get(row.get("setting"))
            if setting is None or row.get("stage", "cv") != "cv":
                continue
            run_id, assay_id = row.get("run_id"), row.get("assay_id")
            if not run_id or not assay_id:
                raise ValueError(f"Missing run_id or assay_id on line {reader.line_num}")
            identity = (run_id, assay_id, setting)
            if identity in seen:
                raise ValueError(
                    f"Duplicate assay/setting for run_id {run_id!r}: "
                    f"{assay_id}/{setting} (line {reader.line_num})"
                )
            seen.add(identity)
            group_id, job_id = _run_group(run_id)
            group_assays.setdefault(group_id, set()).add(assay_id)
            values = groups.setdefault(group_id, {name: {} for name in SETTINGS})
            if row.get("status", "complete") != "complete":
                skipped[group_id, setting] += 1
                continue
            try:
                value = float(row.get(metric))
            except (TypeError, ValueError):
                skipped[group_id, setting] += 1
                continue
            if not math.isfinite(value):
                skipped[group_id, setting] += 1
                continue
            candidate = (job_id, run_id, value)
            previous = values[setting].get(assay_id)
            if previous is not None:
                skipped[group_id, setting] += 1
                if candidate[:2] <= previous[:2]:
                    continue
            values[setting][assay_id] = candidate

    summary = []
    for group_id, settings in sorted(groups.items()):
        for setting, assays in settings.items():
            values = [score for _, _, score in assays.values()]
            summary.append({
                "run_id": group_id, "setting": setting, "n_assays": len(values),
                "mean": fmean(values) if values else None,
                "std": pstdev(values) if values else None,
                "skipped": skipped[group_id, setting],
            })
        complete_assays = set.intersection(*(set(assays) for assays in settings.values()))
        averages = [
            fmean(settings[setting][assay][2] for setting in SETTINGS)
            for assay in sorted(complete_assays)
        ]
        summary.append({
            "run_id": group_id, "setting": "average", "n_assays": len(averages),
            "mean": fmean(averages) if averages else None,
            "std": pstdev(averages) if averages else None,
            "skipped": len(group_assays[group_id]) - len(averages),
        })
    return summary


def print_summary(rows: list[dict], metric: str = "spearman_oof_standardized") -> None:
    print(f"\n{METRIC_LABELS[metric]}: mean and population std (ddof=0) across assays of {metric}.")
    if not rows:
        print("No random, modulo, or contiguous singles CV rows found.")
        return
    current_run = None
    for row in rows:
        if row["run_id"] != current_run:
            current_run = row["run_id"]
            print(f"\nrun prefix: {current_run}")
            print(f"{'Setting':<12} {'Assays':>6} {'Mean':>10} {'Std':>10} {'Skipped':>8}")
        mean = f"{row['mean']:.6f}" if row["mean"] is not None else "N/A"
        std = f"{row['std']:.6f}" if row["std"] is not None else "N/A"
        print(f"{row['setting']:<12} {row['n_assays']:>6} {mean:>10} {std:>10} {row['skipped']:>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", type=Path, help="Path to an aggregated summary.csv")
    parser.add_argument(
        "--metrics", nargs="+", choices=tuple(METRIC_LABELS), default=DEFAULT_METRICS,
        help="Saved CSV columns to summarize (default: %(default)s)",
    )
    args = parser.parse_args()
    try:
        summaries = [(metric, summarize_csv(args.summary_csv, metric))
                     for metric in dict.fromkeys(args.metrics)]
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        parser.error(str(error))
    print("Grouped by run prefix; numeric job/task suffixes and trailing _gpu are ignored; "
          "general and GPU runs are combined with equal assay weight.")
    print("Each metric uses its latest valid job per assay and split independently.")
    print("Skipped: incomplete/invalid rows, missing/non-finite/non-numeric values, or superseded jobs.")
    print("Missing columns yield N/A. Average requires all three settings per assay, separately for each metric.")
    print("Average Skipped counts assays missing any valid setting.")
    for metric, rows in summaries:
        print_summary(rows, metric)


if __name__ == "__main__":
    main()
