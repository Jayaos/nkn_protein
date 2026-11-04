"""Check all catalog assays for complete saved Kermut singles CV results.

Uses only the standard library. Reads JSON/CSV and checks checkpoint file sizes;
never loads model tensors, changes experiment files, or starts training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS = ("singles_random", "singles_modulo", "singles_contiguous")
DEFAULT_PREFIX = "kermut_prosst_esmif"
PREDICTION_VALUES = (
    "target", "target_standardized", "mean", "mean_standardized",
    "latent_variance", "observation_variance",
    "latent_variance_standardized", "observation_variance_standardized",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _nonempty(path):
    _require(path.is_file() and path.stat().st_size > 0, f"Missing or empty file: {path}")


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def _json(path, kind=dict):
    _nonempty(path)
    value = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=_invalid_constant)
    _require(isinstance(value, kind) and bool(value), f"Expected a nonempty {kind.__name__}: {path}")
    return value


def _rows(path, required):
    _nonempty(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, strict=True)
        columns = reader.fieldnames or []
        _require(len(columns) == len(set(columns)) and set(required).issubset(columns),
                 f"Missing or duplicate CSV columns: {path}")
        for row in reader:
            _require(None not in row and all(value is not None for value in row.values()),
                     f"Malformed CSV row: {path}")
            yield row


def _catalog(path):
    assays = []
    seen = set()
    for row in _rows(path, ("DMS_id",)):
        name = row["DMS_id"]
        _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) and name not in seen,
                 f"Invalid or duplicate assay ID in {path}: {name!r}")
        raw_count = row.get("DMS_number_single_mutants", "")
        count = int(raw_count) if raw_count else None
        _require(count is None or count >= 0, f"Invalid single-mutant count for {name}")
        assays.append((name, count))
        seen.add(name)
    _require(bool(assays), f"Assay catalog contains no records: {path}")
    return assays


def _no_failure(directory):
    for name in ("failure.json", "cancelled.json"):
        _require(not (directory / name).exists(), f"Failure marker present: {directory / name}")


def _assignments(path, nominal):
    assignments = {}
    for row in _rows(path, ("variant_id", "fold")):
        variant, fold = row["variant_id"], int(row["fold"])
        _require(variant and variant not in assignments and 0 <= fold < nominal,
                 f"Invalid/duplicate variant or fold in {path}: {variant!r}")
        assignments[variant] = fold
    _require(bool(assignments), f"No fold assignments: {path}")
    return assignments


def _predictions(path, assay, assignments):
    values = {}
    required = ("assay_id", "variant_id", "fold", "n_mutations", *PREDICTION_VALUES)
    for row in _rows(path, required):
        variant = row["variant_id"]
        _require(variant in assignments and variant not in values,
                 f"Unexpected or duplicate predicted variant in {path}: {variant!r}")
        _require(row["assay_id"] == assay and int(row["fold"]) == assignments[variant]
                 and int(row["n_mutations"]) == 1, f"Prediction assay/fold/mutation count mismatch: {path}")
        numbers = tuple(float(row[name]) for name in PREDICTION_VALUES)
        _require(all(math.isfinite(number) for number in numbers)
                 and all(number >= 0 for number in numbers[4:]), f"Invalid prediction values: {path}")
        values[variant] = numbers
    _require(values.keys() == assignments.keys(), f"Prediction coverage differs from fold assignments: {path}")
    return values


def _check_setting(run, assay, setting, config, manifest, catalog_count):
    directory = run / "cv" / setting
    _no_failure(directory)
    nominal = config["evaluation"]["n_folds"]
    resolved = _json(directory / "resolved_config.json")
    _require(resolved == config, f"Resolved setting config differs from run config: {directory}")
    inputs = _json(directory / "input_manifest.json")
    _require(inputs.get("assay_id") == assay and inputs.get("setting") == setting,
             f"Input manifest identity mismatch: {directory}")
    _require(inputs.get("synthetic") is False and inputs.get("coverage") == "full",
             f"Input manifest does not confirm full real-assay coverage: {directory}")
    metrics = _json(directory / "metrics.json")
    _require(metrics.get("assay_id") == assay and metrics.get("setting") == setting
             and metrics.get("scheme") == setting.removeprefix("singles_"),
             f"Aggregate metric identity mismatch: {directory}")
    assignments = _assignments(directory / "fold_assignments.csv", nominal)
    count = len(assignments)
    expected = manifest["sources"]["singles"]["settings"][setting]
    counts = expected["test_counts"]
    _require(isinstance(counts, list) and len(counts) == nominal
             and all(type(value) is int and value >= 0 for value in counts),
             f"Invalid published fold counts for {setting}")
    actual_counts = Counter(assignments.values())
    _require(counts == [actual_counts[index] for index in range(nominal)],
             f"Published fold counts disagree with saved assignments: {directory}")
    _require(expected.get("num_variants") == count == inputs.get("n_variants") == metrics.get("n_variants"),
             f"Variant counts disagree between manifest, metrics and assignments: {directory}")
    _require(catalog_count is None or count == catalog_count,
             f"Expected {catalog_count} catalog singles, found {count}: {directory}")
    fold_ids = sorted(actual_counts)
    _require(len(fold_ids) >= 2, f"Fewer than two populated CV folds: {directory}")
    _require(metrics.get("n_folds") == len(fold_ids)
             and metrics.get("fold_ids", fold_ids) == fold_ids,
             f"Aggregate fold IDs/count disagree with assignments: {directory}")
    records = metrics.get("fold_metrics")
    _require(isinstance(records, list) and all(isinstance(record, dict) for record in records)
             and [record.get("fold") for record in records] == fold_ids,
             f"Aggregate metrics are missing populated folds: {directory}")
    if "fold_n_variants" in metrics:
        _require(metrics["fold_n_variants"] == [actual_counts[fold] for fold in fold_ids],
                 f"Aggregate fold variant counts disagree: {directory}")
    predictions = _predictions(directory / "predictions.csv", assay, assignments)
    observed_dirs = {path.name for path in directory.glob("fold_*") if path.is_dir()}
    _require(observed_dirs == {f"fold_{fold}" for fold in fold_ids},
             f"Missing or unexpected populated fold directories: {directory}")
    for fold, record in zip(fold_ids, records):
        fold_dir = directory / f"fold_{fold}"
        _no_failure(fold_dir)
        _require(record.get("n_variants") == actual_counts[fold] and record.get("setting") == setting,
                 f"Embedded fold metric identity/count mismatch: {fold_dir}")
        _require(_json(fold_dir / "metrics.json") == record,
                 f"Saved and aggregate fold metrics disagree: {fold_dir}")
        _nonempty(fold_dir / "checkpoint.pt")
        _require(_json(fold_dir / "resolved_config.json") == resolved,
                 f"Fold config differs from setting config: {fold_dir}")
        fit = _json(fold_dir / "fit_summary.json")
        _require(fit.get("model") == "kermut", f"Expected a Kermut fit: {fold_dir}")
        for filename in ("history.json", "restart_summaries.json"):
            _json(fold_dir / filename, list)
        selected = {variant: value for variant, value in assignments.items() if value == fold}
        fold_values = _predictions(fold_dir / "predictions.csv", assay, selected)
        _require(fold_values == {variant: predictions[variant] for variant in selected},
                 f"Fold and aggregate predictions disagree: {fold_dir}")
    return set(assignments)


def _check_run(run, assay, settings, catalog_count):
    result = {"run_dir": str(run), "complete": False, "issues": []}
    try:
        _no_failure(run)
        config = _json(run / "config.json")
        nominal = config["evaluation"]["n_folds"]
        _require(type(nominal) is int and nominal >= 2, f"Invalid configured fold count: {run}")
        manifest = _json(run / "fold_manifest.json")
        _require(manifest.get("assay_id") == assay, f"Fold manifest assay mismatch: {run}")
        requested = manifest.get("requested_settings")
        _require(isinstance(requested, list) and set(settings).issubset(requested),
                 f"Fold manifest does not request all required settings: {run}")
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        result["issues"].append(str(error))
        return result
    covered = None
    for setting in settings:
        try:
            variants = _check_setting(run, assay, setting, config, manifest, catalog_count)
            _require(covered is None or covered == variants, "Variant IDs differ across singles settings")
            covered = variants
        except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
            result["issues"].append(f"{setting}: {error}")
    result["complete"] = not result["issues"]
    return result


def check_results(results_dir, assay_index, *, run_prefix=DEFAULT_PREFIX, settings=DEFAULT_SETTINGS):
    root, index = Path(results_dir).resolve(), Path(assay_index).resolve()
    _require(root.is_dir(), f"Results directory does not exist: {root}")
    _require(isinstance(run_prefix, str) and re.fullmatch(r"kermut(?:_[A-Za-z0-9_.-]+)?", run_prefix),
             "Run prefix must be 'kermut' or a name starting with 'kermut_' (no paths or wildcards)")
    settings = tuple(settings)
    _require(bool(settings) and len(set(settings)) == len(settings)
             and set(settings).issubset(DEFAULT_SETTINGS), "Choose unique singles CV settings")
    assays = []
    for catalog_index, (assay, count) in enumerate(_catalog(index)):
        assay_root = root / assay
        runs = [_check_run(path, assay, settings, count)
                for path in sorted(assay_root.glob(run_prefix + "_*")) if path.is_dir()]
        completed = [run["run_dir"] for run in runs if run["complete"]]
        assays.append({"assay_id": assay, "catalog_index": catalog_index,
                       "status": "complete" if completed else "incomplete" if runs else "missing",
                       "complete_runs": completed, "runs": runs})
    counts = Counter(assay["status"] for assay in assays)
    return {"results_dir": str(root), "assay_index": str(index), "run_prefix": run_prefix,
            "settings": list(settings), "assays": assays,
            "summary": {"expected": len(assays), **{status: counts[status]
                        for status in ("complete", "missing", "incomplete")}},
            "rerun_indices": [assay["catalog_index"] for assay in assays if assay["status"] != "complete"]}


def _default_catalog():
    if os.environ.get("ASSAY_INDEX"):
        return Path(os.environ["ASSAY_INDEX"])
    return next((ROOT / path for path in (
        "data/DMS_ProteinGym_substitutions/DMS_substitutions.csv", "data/DMS_substitutions.csv",
        "data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv",
    ) if (ROOT / path).is_file()), None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="Group directory containing <assay>/<kermut-run>/")
    parser.add_argument("--assay-index", type=Path, help="Catalog CSV; default ASSAY_INDEX or the local ProteinGym catalog")
    parser.add_argument("--run-prefix", default=DEFAULT_PREFIX, help=f"Matching run prefix (default: {DEFAULT_PREFIX})")
    parser.add_argument("--settings", nargs="+", choices=DEFAULT_SETTINGS, default=DEFAULT_SETTINGS)
    parser.add_argument("--report", type=Path, help="Optional new JSON report file; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        index = args.assay_index if args.assay_index is not None else _default_catalog()
        _require(index is not None, "ProteinGym catalog not found; supply --assay-index")
        if args.report is not None:
            _require(not args.report.exists(), f"Report already exists; choose a new path: {args.report}")
        report = check_results(args.results_dir, index, run_prefix=args.run_prefix, settings=args.settings)
        print(f"Results: {report['results_dir']}\nCatalog: {report['assay_index']}")
        print(f"Prefix: {args.run_prefix}; required settings: {', '.join(args.settings)}")
        for assay in report["assays"]:
            print(f"{assay['status'].upper()} [{assay['catalog_index']}] {assay['assay_id']}")
            if assay["complete_runs"]:
                for run in assay["complete_runs"]:
                    print(f"  Complete run: {run}")
                rejected = len(assay["runs"]) - len(assay["complete_runs"])
                if rejected:
                    print(f"  Other incomplete attempts: {rejected} (do not prevent assay completion)")
            else:
                for run in assay["runs"]:
                    print(f"  {run['run_dir']}")
                    for issue in run["issues"]:
                        print(f"    {issue}")
        print("Summary: " + "; ".join(f"{key}={value}" for key, value in report["summary"].items()))
        if report["rerun_indices"]:
            print("Incomplete/missing catalog indices: " + ",".join(map(str, report["rerun_indices"])))
            print("These indices follow the selected catalog; no jobs were submitted.")
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, allow_nan=False)
                handle.write("\n")
            print(f"JSON report: {args.report.resolve()}")
        return int(bool(report["rerun_indices"]))
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
