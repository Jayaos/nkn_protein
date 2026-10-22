"""Reconstruct a filtered MSA and compare freshly calculated sequence weights.

Uses the pinned ProteinGym MSA_processing implementation. Every run writes to a
new directory; supplied alignments and weight files are never modified.
Numerical agreement is not proof of historical row identity (weights can repeat).
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_CONFIG = PROJECT_ROOT / "configs/setup/msa_transformer_runtime.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_hash(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def normalize(text):
    # Match ProteinGym's tr '.' '-' followed by dd conv=ucase, including headers.
    # Do not remove lowercase residues as if they were A3M insertions.
    return text.replace(".", "-").upper()


def read_alignment(path):
    rows = []
    with Path(path).open(encoding="ascii") as handle:
        for line in handle:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                rows.append((line, ""))
            elif rows:
                name, sequence = rows[-1]
                if any(char.isspace() for char in line):
                    raise ValueError(f"Whitespace inside sequence: {name}")
                rows[-1] = (name, sequence + line)
            else:
                raise ValueError("Sequence data before the first FASTA header")
    if not rows or not rows[0][1]:
        raise ValueError("Empty alignment")
    if len({name for name, _ in rows}) != len(rows):
        raise ValueError("Duplicate sequence headers would make the row mapping ambiguous")
    if any(len(sequence) != len(rows[0][1]) for _, sequence in rows):
        raise ValueError("Alignment rows have different lengths")
    return rows


def write_alignment(path, rows):
    with Path(path).open("w", encoding="ascii", newline="\n") as handle:
        for name, sequence in rows:
            handle.write(f"{name}\n{sequence}\n")


def resolve_reference(configured, override=None):
    requested = PROJECT_ROOT / (override if override is not None else configured)
    if requested.is_file():
        return requested.resolve()
    candidates = [PROJECT_ROOT / "data/DMS_ProteinGym_substitutions/DMS_substitutions.csv",
                  PROJECT_ROOT / "data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv",
                  PROJECT_ROOT / "data/DMS_substitutions.csv"]
    # An explicit CLI path must not silently select a different dataset.
    existing = [path for path in candidates if path.is_file()] if override is None else []
    if existing:
        if len({sha256(path) for path in existing}) != 1:
            raise ValueError("Multiple different reference CSVs found; select one with --reference-csv PATH: "
                             + ", ".join(str(path) for path in existing))
        print(f"Configured reference missing; using {existing[0]}", flush=True)
        return existing[0].resolve()
    checked = [requested] if override is not None else list(dict.fromkeys([requested, *candidates]))
    raise FileNotFoundError("Assay reference CSV not found. Checked:\n  "
                            + "\n  ".join(str(path) for path in checked)
                            + "\nSync the reference CSV to Phoenix or pass --reference-csv PATH.")


def load_inputs(config, reference_csv=None, check_reference_theta=True):
    reference = resolve_reference(config["reference_csv"], reference_csv)
    with reference.open(newline="", encoding="utf-8-sig") as handle:
        matches = [row for row in csv.DictReader(handle) if row["DMS_id"] == config["assay_id"]]
    if len(matches) != 1:
        raise ValueError("Expected exactly one reference row for the requested assay")
    metadata = matches[0]
    msa = PROJECT_ROOT / config["msa_folder"] / metadata["MSA_filename"]
    weight_root = PROJECT_ROOT / config["weights_folder"]
    paths = {
        "reference": reference, "msa": msa,
        "general_weights": weight_root / metadata["weight_file_name"],
        "specialized_weights": weight_root / config["specialized_weights_subfolder"] / metadata["weight_file_name"],
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    raw = read_alignment(msa)
    normalized = [(normalize(name), normalize(seq)) for name, seq in raw]
    if len({name for name, _ in normalized}) != len(raw):
        raise ValueError("Headers collide after ProteinGym's uppercase/dot normalization")
    start, end = int(metadata["MSA_start"]), int(metadata["MSA_end"])
    if not 1 <= start <= end <= len(metadata["target_seq"]):
        raise ValueError("Invalid reference MSA coordinates")
    query = normalized[0][1].replace("-", "")
    if query != metadata["target_seq"][start - 1:end].upper():
        raise ValueError("MSA query does not match the reference sequence interval")
    if check_reference_theta and float(metadata["MSA_theta"]) != config["msa_processing"]["theta"]:
        raise ValueError("Reference weighting theta differs from this verification recipe")
    return metadata, paths, raw, normalized


def compare_weights(recomputed, supplied, rtol, atol):
    calculated = [float(value) for value in recomputed]
    downloaded = [float(value) for value in supplied]
    valid = all(math.isfinite(value) and 0 < value <= 1 for value in calculated + downloaded)
    same_count = len(calculated) == len(downloaded) and bool(calculated)
    result = {
        "recomputed_count": len(calculated), "supplied_count": len(downloaded),
        "finite_positive_weights_at_most_one": valid, "counts_match": same_count,
        "rtol": rtol, "atol": atol, "values_match_in_order": False,
        "exact_match": False, "max_absolute_difference": None,
        "mismatching_indices_zero_based": [],
    }
    if valid and same_count:
        differences = [abs(a - b) for a, b in zip(calculated, downloaded)]
        mismatches = [i for i, (a, b) in enumerate(zip(calculated, downloaded))
                      if abs(a - b) > atol + rtol * abs(b)]
        result.update(values_match_in_order=not mismatches,
                      exact_match=calculated == downloaded,
                      max_absolute_difference=max(differences),
                      mismatching_indices_zero_based=mismatches)
    return result


def retained_rows(raw, normalized, filtered, processed):
    """Join by normalized full header, then verify sequence content and order."""
    raw_index = {name: (i, seq) for i, (name, seq) in enumerate(normalized)}
    filtered_index = {name: i for i, (name, _) in enumerate(filtered)}
    if filtered[0][0] != normalized[0][0] or not processed or processed[0][0] != normalized[0][0]:
        raise ValueError("The query must remain the first row at every stage")
    if any(name not in raw_index for name, _ in filtered):
        raise ValueError("HHfilter returned an unknown or changed sequence header")
    columns = [i for i, aa in enumerate(normalized[0][1]) if aa != "-"]
    mapping = []
    previous = -1
    for i, (name, sequence) in enumerate(processed):
        if name not in filtered_index or name not in raw_index:
            raise ValueError("Processed sequence is absent from the input alignment")
        stage_index = filtered_index[name]
        if stage_index <= previous:
            raise ValueError("Postprocessing changed HHfilter row order")
        previous = stage_index
        original_index, original_sequence = raw_index[name]
        projected = "".join(original_sequence[col] for col in columns)
        if sequence != projected:
            raise ValueError(f"Retained sequence changed beyond query-gap column removal: {name}")
        mapping.append({
            "weight_index_zero_based": i, "processed_row_one_based": i + 1,
            "raw_row_one_based": original_index + 1, "hhfilter_row_one_based": stage_index + 1,
            "original_header": raw[original_index][0], "processed_header": name,
            "original_sequence_sha256": sequence_hash(raw[original_index][1]),
            "processed_sequence_sha256": sequence_hash(sequence),
        })
    return mapping, columns


def check_runtime(runtime):
    if f"{sys.version_info.major}.{sys.version_info.minor}" != runtime["python_version"]:
        raise ValueError("Activate the Python 3.10 nkn_msa_transformer environment")
    installed = {name: importlib.metadata.version(name) for name in runtime["packages"]}
    if installed != runtime["packages"]:
        raise ValueError("Installed package versions differ from the runtime configuration")
    source = PROJECT_ROOT / "data/models/msa_transformer" / f"ProteinGym-{runtime['proteingym_commit']}"
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    if revision != runtime["proteingym_commit"] or dirty:
        raise ValueError("ProteinGym checkout must be clean and at the pinned commit")
    hhfilter = shutil.which("hhfilter")
    if not hhfilter:
        raise ValueError("HHfilter is not on PATH; complete runtime setup first")
    help_result = subprocess.run([hhfilter, "-h"], capture_output=True, text=True, check=True)
    if runtime["hhfilter_version"] not in help_result.stdout + help_result.stderr:
        raise ValueError("HHfilter version differs from the pinned runtime")
    sys.path[:0] = [str(source / "proteingym"), str(source)]
    from proteingym.utils.msa_utils import MSA_processing
    import inspect
    if not Path(inspect.getfile(MSA_processing)).resolve().is_relative_to(source.resolve()):
        raise ValueError("Imported a different ProteinGym implementation")
    return MSA_processing, hhfilter, source, installed


def run_verification(config_path, cpus, reference_csv=None, assay_index=None, specialized_only=False):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = json.loads(RUNTIME_CONFIG.read_text(encoding="utf-8"))
    if assay_index is not None:
        reference = resolve_reference(config["reference_csv"], reference_csv)
        with reference.open(newline="", encoding="utf-8-sig") as handle:
            catalog = list(csv.DictReader(handle))
        if not 0 <= assay_index < len(catalog):
            raise ValueError(f"Assay index {assay_index} is outside 0..{len(catalog) - 1}")
        config["assay_id"] = catalog[assay_index]["DMS_id"]
    metadata, paths, raw, normalized = load_inputs(config, reference_csv, not specialized_only)
    # This pilot uses all query residue columns. Other column recipes require
    # explicit coordinate handling before they can be verified by this script.
    if config["msa_processing"]["threshold_focus_cols_frac_gaps"] != 1.0:
        raise ValueError("This pilot requires all query residue columns")
    processor, hhfilter, source, installed = check_runtime(runtime)
    import numpy as np
    supplied = {kind: np.load(paths[f"{kind}_weights"], allow_pickle=False)
                for kind in ("general", "specialized")}
    if any(array.ndim != 1 or array.dtype.kind != "f" for array in supplied.values()):
        raise ValueError("Expected one-dimensional floating-point weight arrays")
    output_root = PROJECT_ROOT / config["output_folder"]
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=config["assay_id"] + "_", dir=output_root))
    report = {
        "status": "running", "assay_id": config["assay_id"],
        "created_at": datetime.now(timezone.utc).isoformat(), "config": config,
        "config_sha256": sha256(config_path), "runtime_config_sha256": sha256(RUNTIME_CONFIG),
        "helper_sha256": sha256(__file__), "proteingym_commit": runtime["proteingym_commit"],
        "proteingym_msa_utils_sha256": sha256(source / "proteingym/utils/msa_utils.py"),
        "proteingym_weights_sha256": sha256(source / "proteingym/utils/weights.py"),
        "packages": installed, "cpus": cpus,
        "hhfilter": hhfilter, "hhfilter_version": runtime["hhfilter_version"],
        "hhfilter_sha256": sha256(hhfilter),
        "inputs": {key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()},
        "raw_rows": len(raw), "raw_columns": len(raw[0][1]),
        "assay_index_zero_based": assay_index,
        "specialized_only": specialized_only,
        "reference_theta": float(metadata["MSA_theta"]),
        "query_matches_reference": True,
        "historical_row_identity_verified": False,
        "interpretation": "Agreement establishes numerical consistency with this recipe, not unique historical row identity.",
    }
    report_path = output / "verification_report.json"
    print(f"Verification output: {output}", flush=True)
    try:
        normalized_path = output / "normalized.a2m"
        write_alignment(normalized_path, normalized)
        filtered_path = output / "hhfiltered.a2m"
        settings = config["hhfilter"]
        command = [hhfilter, "-cov", str(settings["min_coverage"]),
                   "-id", str(settings["max_pairwise_identity"]),
                   "-qid", str(settings["min_query_identity"]),
                   "-i", str(normalized_path), "-o", str(filtered_path)]
        report["hhfilter_command"] = command
        with (output / "hhfilter.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        filtered = read_alignment(filtered_path)
        report["hhfilter_rows"] = len(filtered)
        report["comparisons"] = {}
        comparisons = [("specialized", filtered_path)] if specialized_only else [("general", normalized_path), ("specialized", filtered_path)]
        for kind, alignment in comparisons:
            print(f"Recomputing {kind} weights using pinned ProteinGym code", flush=True)
            # Fresh path is essential: MSA_processing loads existing weights otherwise.
            weight_path = output / f"{kind}_recomputed_weights.npy"
            processed = processor(MSA_location=str(alignment), use_weights=True,
                                  weights_location=str(weight_path), skip_one_hot_encodings=True,
                                  num_cpus=cpus, **config["msa_processing"])
            rows = [(name, "".join(seq)) for name, seq in processed.seq_name_to_sequence.items()]
            mapping, columns = retained_rows(raw, normalized,
                                            normalized if kind == "general" else filtered, rows)
            comparison = compare_weights(processed.weights, supplied[kind],
                                         config["comparison"]["relative_tolerance"],
                                         config["comparison"]["absolute_tolerance"])
            report["comparisons"][kind] = comparison
            write_alignment(output / f"{kind}_processed.a2m", rows)
            for i, entry in enumerate(mapping):
                entry["recomputed_weight"] = float(processed.weights[i])
                # Positional candidate, never described as a proven historical pairing.
                entry["supplied_weight_at_index"] = float(supplied[kind][i]) if comparison["counts_match"] else ""
            with (output / f"{kind}_rows.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(mapping[0]))
                writer.writeheader()
                writer.writerows(mapping)
            retained = {name for name, _ in rows}
            stage = {name for name, _ in (normalized if kind == "general" else filtered)}
            with (output / f"{kind}_row_disposition.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["raw_row_one_based", "original_header", "normalized_header", "disposition"])
                for i, (name, _) in enumerate(normalized):
                    disposition = "retained" if name in retained else "removed_by_postprocessing" if name in stage else "removed_by_hhfilter"
                    writer.writerow([i + 1, raw[i][0], name, disposition])
            if kind == "specialized":
                with (output / "columns.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["processed_column_one_based", "raw_column_one_based", "target_position_one_based"])
                    writer.writerows((i + 1, column + 1, int(metadata["MSA_start"]) + i) for i, column in enumerate(columns))
            print(f"{kind}: {json.dumps(comparison)}", flush=True)
        agreed = all(value["values_match_in_order"] for value in report["comparisons"].values())
        report["status"] = "weights_reproduced" if agreed else "weight_mismatch"
        report["artifacts"] = {path.name: sha256(path) for path in output.iterdir() if path.is_file()}
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Report: {report_path}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/setup/msa_preprocessing_brca1.json")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--assay-index", type=int, help="Zero-based row index in the reference CSV")
    parser.add_argument("--specialized-only", action="store_true",
                        help="Compare only specialized weights using the pinned scoring recipe (theta 0.2)")
    parser.add_argument("--reference-csv", type=Path,
                        help="Reference CSV override; relative paths resolve from the repository root")
    args = parser.parse_args()
    if args.cpus < 1:
        parser.error("--cpus must be positive")
    report = run_verification(args.config.resolve(), args.cpus, args.reference_csv,
                              args.assay_index, args.specialized_only)
    print(f"Result: {report['status']}. Historical row identity remains unverified.", flush=True)
    return 0 if report["status"] == "weights_reproduced" else 2


if __name__ == "__main__":
    sys.exit(main())
