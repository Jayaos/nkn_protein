"""Print Spearman statistics across assays from an aggregated summary.csv.

Usage: python src/reporting/summarize.py results/EXPERIMENT/aggregate/summary.csv

Each complete CV row contributes its avg_spearman with equal weight. Standard
deviation is across assays (population SD, ddof=0), not the saved within-assay
fold std_spearman. Runs sharing a prefix are grouped after removing numeric
job/task suffixes. For repeated assays, use the latest valid job per split.
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


def _run_group(run_id: str) -> tuple[str, int]:
    """Remove a numeric Slurm job ID and optional array task index."""
    # A manual run's trailing task index is not a job ID.
    match = None if "_manual_" in run_id else re.search(r"_(\d+)(?:_\d+)?$", run_id)
    return (run_id[:match.start()], int(match[1])) if match else (run_id, -1)


def summarize_csv(path: str | Path) -> list[dict]:
    """Summarize finite avg_spearman values per run prefix and singles split."""
    groups: dict[str, dict[str, dict[str, tuple[int, str, float]]]] = {}
    skipped: Counter = Counter()
    seen = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"assay_id", "run_id", "setting", "avg_spearman"}
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
            values = groups.setdefault(group_id, {name: {} for name in SETTINGS})
            if row.get("status", "complete") != "complete":
                skipped[group_id, setting] += 1
                continue
            try:
                value = float(row["avg_spearman"])
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
    return summary


def print_summary(rows: list[dict]) -> None:
    print("Spearman across assays: mean and population std (ddof=0) of avg_spearman.")
    print("Grouped by run prefix; numeric job/task suffixes are ignored.")
    print("Skipped: incomplete/invalid rows, missing/non-finite/non-numeric scores, or superseded jobs.")
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
    args = parser.parse_args()
    try:
        rows = summarize_csv(args.summary_csv)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        parser.error(str(error))
    print_summary(rows)


if __name__ == "__main__":
    main()
