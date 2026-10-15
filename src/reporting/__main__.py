"""Run with PYTHONPATH=src python -m reporting --output results/aggregate."""

import argparse
from collections import Counter

from .aggregate import aggregate_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate saved NKN and Kermut experiment metrics.")
    parser.add_argument("--results", default="results", help="Results root (default: results)")
    parser.add_argument("--output", default="results/aggregate", help="Destination for report CSVs")
    args = parser.parse_args()
    tables = aggregate_results(args.results, args.output)
    print(f"Settings: {len(tables['summary'])}; status: {dict(Counter(r['status'] for r in tables['summary']))}")
    print(f"Folds: {len(tables['folds'])}; issues: {len(tables['issues'])}; output: {args.output}")


if __name__ == "__main__":
    main()
