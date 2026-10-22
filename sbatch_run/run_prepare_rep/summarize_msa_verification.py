"""Summarize every catalog assay, including missing and unsuccessful runs.

With --job-id, use the array's logs to select its exact reports. Otherwise use
the latest available run per assay, including directories without a report.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_report(directory):
    path = directory / "verification_report.json"
    if not path.is_file():
        return None, "No report; task may be pending, running, failed, or terminated."
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("Report is not an object")
        return report, ""
    except (ValueError, OSError) as error:
        return None, f"Invalid report: {error}"


def run_time(directory):
    report, _ = read_report(directory)
    try:
        return datetime.fromisoformat(report["created_at"]).timestamp()
    except (TypeError, KeyError, ValueError):
        return directory.stat().st_mtime


def classify(report, assay_id, reference_hash):
    if report.get("assay_id") != assay_id:
        return "wrong_assay"
    if report.get("inputs", {}).get("reference", {}).get("sha256") != reference_hash:
        return "reference_mismatch"
    status = report.get("status")
    if status == "failed":
        return "failed"
    if status not in ("weights_reproduced", "weight_mismatch"):
        return "incomplete"
    comparison = report.get("comparisons", {}).get("specialized", {})
    if not comparison:
        return "missing_comparison"
    if comparison.get("finite_positive_weights_at_most_one") is not True:
        return "invalid_weights"
    if comparison.get("counts_match") is not True:
        return "count_mismatch"
    if comparison.get("values_match_in_order") is not True:
        return "value_mismatch"
    if status != "weights_reproduced":
        return "other_comparison_mismatch"
    return "exact_match" if comparison.get("exact_match") is True else "match_within_tolerance"


def summarize(reference, reports_root, logs_dir, job_id=None):
    reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
    with reference.open(newline="", encoding="utf-8-sig") as handle:
        catalog = list(csv.DictReader(handle))
    ids = [entry["DMS_id"] for entry in catalog]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Reference CSV must contain unique, nonempty assay IDs")
    rows = []
    for index, assay_id in enumerate(ids):
        directories = [path for path in reports_root.glob(assay_id + "_*") if path.is_dir()]
        selected = None
        log_path = None
        note = ""
        if job_id is not None:
            log_path = logs_dir / f"verify_msa_all-{job_id}_{index}.out"
            if log_path.is_file():
                for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.startswith("Verification output: "):
                        selected = Path(line.partition(": ")[2].strip())
                note = "No report directory recorded in this task's log." if selected is None else ""
            else:
                note = "Array task log is missing; check the job ID, log directory, and Slurm state."
        elif directories:
            selected = max(directories, key=lambda path: (run_time(path), path.name))
            note = "Latest available run selected; use --job-id to restrict to one array."
        else:
            note = "No run directory found for this assay."
        row = {"assay_index": index, "DMS_id": assay_id, "status": "no_report",
               "raw_rows": "", "hhfilter_rows": "", "processed_rows": "", "supplied_weights": "",
               "exact_match": "", "max_absolute_difference": "", "mismatching_weight_count": "",
               "available_run_directories": len(directories), "selected_run": str(selected) if selected else "",
               "report_path": str(selected / "verification_report.json") if selected else "",
               "log_path": str(log_path) if log_path else "", "note": note}
        if selected is not None:
            report, error = read_report(selected)
            if report is None:
                row["status"] = "invalid_report" if error.startswith("Invalid report") else "no_report"
                row["note"] = error
            else:
                try:
                    row["status"] = classify(report, assay_id, reference_hash)
                    comparison = report.get("comparisons", {}).get("specialized", {})
                    row.update(raw_rows=report.get("raw_rows", ""), hhfilter_rows=report.get("hhfilter_rows", ""),
                               processed_rows=comparison.get("recomputed_count", ""),
                               supplied_weights=comparison.get("supplied_count", ""),
                               exact_match=comparison.get("exact_match", ""),
                               max_absolute_difference=comparison.get("max_absolute_difference", ""),
                               mismatching_weight_count=len(comparison.get("mismatching_indices_zero_based", [])))
                    if report.get("error"):
                        row["note"] = report["error"]
                except (TypeError, AttributeError):
                    row["status"] = "invalid_report"
                    row["note"] = "Report fields have unexpected types."
        rows.append(row)
    counts = dict(sorted(Counter(row["status"] for row in rows).items()))
    matched = counts.get("exact_match", 0) + counts.get("match_within_tolerance", 0)
    summary = {"reference_csv": str(reference), "reference_sha256": reference_hash,
               "job_id": job_id, "selection": "array_logs" if job_id else "latest_available_per_assay",
               "expected_assays": len(rows), "matched_assays": matched,
               "assays_needing_attention": len(rows) - matched, "counts": counts,
               "all_assays_matched": matched == len(rows),
               "assays_with_multiple_run_directories": sum(row["available_run_directories"] > 1 for row in rows),
               "note": "No-report tasks are not counted as passes. Numerical agreement does not prove historical row identity."}
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", type=Path, default=ROOT / "data/DMS_ProteinGym_substitutions/DMS_substitutions.csv")
    parser.add_argument("--reports-root", type=Path, default=ROOT / "data/msa/verification")
    parser.add_argument("--logs-dir", type=Path, default=ROOT, help="Directory where the array was submitted")
    parser.add_argument("--job-id", help="Slurm array job ID; selects this array using its task logs")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.job_id and not args.job_id.isdigit():
        parser.error("--job-id must be a numeric Slurm array ID")
    rows, summary = summarize(args.reference_csv.resolve(), args.reports_root.resolve(),
                              args.logs_dir.resolve(), args.job_id)
    output = args.output_dir or args.reports_root / (f"summary_{args.job_id}" if args.job_id else "summary_latest")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "verification_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "verification_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Assays: {summary['expected_assays']} | Matched: {summary['matched_assays']} | Need attention: {summary['assays_needing_attention']}")
    for status, count in summary["counts"].items():
        print(f"  {status}: {count}")
    print(f"Assays with multiple run directories: {summary['assays_with_multiple_run_directories']}")
    if not args.job_id:
        print("Selected latest available runs, possibly from different arrays. Use --job-id for one array.")
    print(f"CSV: {output / 'verification_summary.csv'}")
    print(f"JSON: {output / 'verification_summary.json'}")


if __name__ == "__main__":
    main()
