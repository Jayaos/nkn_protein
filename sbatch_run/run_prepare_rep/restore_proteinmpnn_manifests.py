"""Restore missing manifest JSON snapshots from completed ProteinMPNN caches.

Uses the standard library only. Preview by default; --apply creates missing
files without replacing existing ones. No downloads, inference, or cache edits.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(path, assay):
    """Verify the saved JSON's checksum and identity before extracting its copy."""
    completion = json.loads((path.parent / "completion.json").read_text(encoding="utf-8"))
    if completion["files_sha256"]["config.json"] != sha256(path):
        raise ValueError(f"Cached config checksum mismatch: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    signature = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if (completion.get("model_family") != "proteinmpnn" or completion.get("cache_id") != path.parent.name
            or completion.get("config_signature") != signature
            or config.get("model_family") != "proteinmpnn" or config.get("feature_type") != "inverse_folding"
            or config.get("checkpoint") != path.parent.name):
        raise ValueError(f"Cached config identity mismatch: {path}")
    metadata = config["structure"]
    if (not isinstance(metadata, dict) or metadata.get("assay_id") != assay
            or not metadata.get("input_signature")
            or completion.get("structure_input_signature") != metadata["input_signature"]):
        raise ValueError(f"Cached manifest identity mismatch: {path}")
    verify_collection(config)
    return metadata


def verify_collection(value):
    """Check saved source metadata; recipe-only cache names do not identify it."""
    if isinstance(value, dict):
        archive = value.get("source_archive") or {}
        sources = (value.get("structure_source"), archive.get("source_id"))
        if any(source is not None and source != "proteingym" for source in sources):
            raise ValueError("Cached structure belongs to another collection, not ProteinGym")
        for child in value.values():
            verify_collection(child)
    elif isinstance(value, list):
        for child in value:
            verify_collection(child)


def verify_referenced_files(metadata, manifest_dir):
    """Follow saved file references, including nested fragment/mapping snapshots."""
    found = 0

    def visit(value):
        nonlocal found
        if isinstance(value, dict):
            if "structure_path" in value:
                stored = value["structure_path"]
                if not isinstance(stored, str) or not stored:
                    raise ValueError("Missing structure file path in cached manifest")
                path = (manifest_dir / stored).resolve()
                if sha256(path) != value.get("structure_sha256"):
                    raise ValueError(f"Referenced structure checksum mismatch: {path}")
                found += 1
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(metadata)
    if not found:
        raise ValueError("Cached manifest contains no structure file references")


def write_missing(path, metadata):
    """Publish a complete file atomically without overwriting a concurrent restore."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".restore-", suffix=".json", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def restore_one(root, manifest_dir, assay, apply=False):
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", assay) is None or assay.endswith("."):
        raise ValueError(f"Invalid assay directory name: {assay!r}")
    destination = manifest_dir / f"{assay}.json"
    if destination.exists() or destination.is_symlink():
        return "EXISTS", destination
    parent = root / assay / "inverse_folding/proteinmpnn"
    # Accept recipe-only names and historical ProteinGym prefixes. Saved metadata
    # verifies the collection even when the directory name does not identify it.
    candidates = [path for path in sorted(parent.glob("*/config.json"))
                  if not path.parent.name.startswith(".")
                  and (path.parent.name.startswith("proteingym--") or "--" not in path.parent.name)
                  and (path.parent / "completion.json").is_file()]
    if not candidates:
        raise ValueError(f"No completed ProteinGym ProteinMPNN feature cache in {parent}")
    metadata = snapshot(candidates[0], assay)
    for path in candidates[1:]:
        if snapshot(path, assay) != metadata:
            raise ValueError(f"Conflicting cached manifest snapshots for {assay}; restore the intended backup explicitly")
    verify_referenced_files(metadata, manifest_dir)
    if apply:
        write_missing(destination, metadata)
    return "RESTORED" if apply else "RECOVERABLE", destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/features/pretrained/singles")
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data/structures/proteingym/manifests")
    parser.add_argument("--assay", action="append", help="Restore selected assay IDs; default: all saved assay directories")
    parser.add_argument("--apply", action="store_true", help="Write missing manifests; otherwise preview only")
    args = parser.parse_args(argv)
    root, destination = args.output_root.resolve(), args.manifest_dir.resolve()
    if not root.is_dir():
        parser.error(f"Output root does not exist: {root}")
    assays = args.assay if args.assay else [path.name for path in sorted(root.iterdir())
                                         if path.is_dir() and not path.name.startswith(".")]
    if not assays or len(set(assays)) != len(assays):
        parser.error("Select at least one unique assay")
    counts = {"RESTORED": 0, "RECOVERABLE": 0, "EXISTS": 0, "FAILED": 0}
    for assay in assays:
        try:
            status, path = restore_one(root, destination, assay, args.apply)
            counts[status] += 1
            print(f"{status} {assay}: {path}", flush=True)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            counts["FAILED"] += 1
            print(f"FAILED {assay}: {error}", flush=True)
    print("Summary: " + "; ".join(f"{key}={value}" for key, value in counts.items()), flush=True)
    if not args.apply:
        print("Preview only. Add --apply to restore missing manifests.", flush=True)
    return int(counts["FAILED"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
