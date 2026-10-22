"""Validate saved alignment/weight pairs and summarize a generation directory."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from verify_msa_preprocessing import read_alignment, sha256, sequence_hash


def validate_pair(directory, manifest=None):
    """Load a pair only after checking checksums, row identities and weight indices.

    Return (ordered FASTA rows, weights). Weights include the query in row zero;
    sampling must retain the query separately and normalize the other weights.
    """
    import numpy as np
    directory = Path(directory)
    if manifest is None:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "generated":
        raise ValueError("The pair does not have a completed generation manifest")
    for name in ("alignment.a2m", "weights.npy", "rows.csv", "columns.csv"):
        if sha256(directory / name) != manifest.get("artifacts", {}).get(name):
            raise ValueError(f"Generated artifact checksum mismatch: {name}")
    rows = read_alignment(directory / "alignment.a2m")
    weights = np.load(directory / "weights.npy", allow_pickle=False)
    if (weights.dtype.kind != "f" or weights.shape != (len(rows),)
            or not np.isfinite(weights).all() or not (weights > 0).all() or not (weights <= 1).all()):
        raise ValueError("Invalid generated weight vector")
    if manifest["processed_rows"] != len(rows) or manifest["processed_columns"] != len(rows[0][1]):
        raise ValueError("Manifest alignment dimensions do not match")
    with (directory / "rows.csv").open(newline="", encoding="utf-8") as handle:
        mapping = list(csv.DictReader(handle))
    if len(mapping) != len(rows):
        raise ValueError("Row manifest length mismatch")
    for i, ((header, sequence), entry) in enumerate(zip(rows, mapping)):
        if (int(entry["weight_index_zero_based"]) != i or entry["processed_header"] != header
                or entry["processed_sequence_sha256"] != sequence_hash(sequence)
                or float(entry["recomputed_weight"]) != float(weights[i])):
            raise ValueError(f"Sequence/weight correspondence mismatch at row {i}")
    return rows, weights


def summarize_batch(batch):
    batch = Path(batch)
    reference = batch / "DMS_substitutions.csv"
    reference_hash = sha256(reference)
    with reference.open(newline="", encoding="utf-8-sig") as handle:
        ids = [row["DMS_id"] for row in csv.DictReader(handle)]
    if not ids or any(not name for name in ids) or len(set(ids)) != len(ids):
        raise ValueError("Catalog must have unique, nonempty assay IDs")
    records = []
    for assay_id in ids:
        runs = list((batch / "assays").glob(assay_id + "_*"))
        row = dict(DMS_id=assay_id, status="no_report", processed_rows="", neff="",
                   has_homologs="", directory="", error="")
        if len(runs) > 1:
            row.update(status="multiple_runs", error="Select a run explicitly; no automatic choice made")
        elif runs:
            directory = runs[0]
            row["directory"] = str(directory)
            try:
                if (directory / "manifest.json").is_file():
                    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
                    if manifest["assay_id"] != assay_id or manifest["inputs"]["reference"]["sha256"] != reference_hash:
                        raise ValueError("Assay or reference snapshot mismatch")
                    validate_pair(directory, manifest)
                    row.update(status="generated", processed_rows=manifest["processed_rows"],
                               neff=manifest["neff"], has_homologs=manifest["has_homologs"])
                elif (directory / "generation_report.json").is_file():
                    report = json.loads((directory / "generation_report.json").read_text(encoding="utf-8"))
                    row.update(status="failed" if report.get("status") == "failed" else "incomplete",
                               error=report.get("error", "No completed manifest"))
            except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, csv.Error) as error:
                row.update(status="invalid", error=str(error))
        records.append(row)
    counts = dict(Counter(row["status"] for row in records))
    summary = dict(output_directory=str(batch), reference_csv=str(reference), reference_sha256=reference_hash,
                   expected_assays=len(ids), generated_assays=counts.get("generated", 0), counts=counts,
                   assays_needing_attention=len(ids) - counts.get("generated", 0),
                   all_assays_generated=bool(ids) and counts.get("generated", 0) == len(ids),
                   query_only_assays=sum(row["has_homologs"] is False for row in records))
    with (batch / "generation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (batch / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Summary: {batch / 'generation_summary.csv'}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", "--batch-dir", dest="output_dir", type=Path, required=True)
    summarize_batch(parser.parse_args().output_dir)
