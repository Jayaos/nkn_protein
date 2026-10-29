"""Run with PYTHONPATH=src python -m reporting --output results/aggregate."""

import argparse
from collections import Counter

from .aggregate import aggregate_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate saved NKN and Kermut experiment metrics.")
    parser.add_argument("--results", default="results", help="Results root (default: results)")
    parser.add_argument("--output", default="results/aggregate", help="Destination for report CSVs")
    parser.add_argument(
        "--all-runs", action="store_true",
        help="Include every eligible run instead of only the latest per assay and model",
    )
    args = parser.parse_args()
    tables = aggregate_results(args.results, args.output, all_runs=args.all_runs)
    print(f"Settings: {len(tables['summary'])}; status: {dict(Counter(r['status'] for r in tables['summary']))}")
    print(f"Folds: {len(tables['folds'])}; issues: {len(tables['issues'])}; output: {args.output}")


if __name__ == "__main__":
    main()
