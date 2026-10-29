"""Print Spearman statistics across assays from an aggregated summary.csv.

Usage: python src/reporting/summarize.py results/EXPERIMENT/aggregate/summary.csv

Each complete CV row contributes its avg_spearman with equal weight. Standard
deviation is across assays (population SD, ddof=0), not the saved within-assay
fold std_spearman. Exact run IDs, including job/task suffixes, stay separate.
"""

import argparse
from collections import Counter
import csv
import math
from pathlib import Path
from statistics import fmean, pstdev


SETTINGS = ("random", "modulo", "contiguous")
SETTING_NAMES = {f"singles_{name}": name for name in SETTINGS}


def summarize_csv(path: str | Path) -> list[dict]:
    """Summarize finite avg_spearman values per exact run ID and singles split."""
    groups: dict[str, dict[str, list[float]]] = {}
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
            values = groups.setdefault(run_id, {name: [] for name in SETTINGS})
            if row.get("status", "complete") != "complete":
                skipped[run_id, setting] += 1
                continue
            try:
                value = float(row["avg_spearman"])
            except (TypeError, ValueError):
                skipped[run_id, setting] += 1
                continue
            if not math.isfinite(value):
                skipped[run_id, setting] += 1
                continue
            values[setting].append(value)

    return [
        {
            "run_id": run_id, "setting": setting, "n_assays": len(values),
            "mean": fmean(values) if values else None,
            "std": pstdev(values) if values else None,
            "skipped": skipped[run_id, setting],
        }
        for run_id, settings in sorted(groups.items())
        for setting, values in settings.items()
    ]


def print_summary(rows: list[dict]) -> None:
    print("Spearman across assays: mean and population std (ddof=0) of avg_spearman.")
    print("Skipped: incomplete/invalid rows or missing/non-finite/non-numeric scores.")
    if not rows:
        print("No random, modulo, or contiguous singles CV rows found.")
        return
    current_run = None
    for row in rows:
        if row["run_id"] != current_run:
            current_run = row["run_id"]
            print(f"\nrun_id: {current_run}")
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
