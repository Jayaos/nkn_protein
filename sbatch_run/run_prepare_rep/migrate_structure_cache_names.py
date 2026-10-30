"""Remove the historical ProtSSN prefix from ProSST and ESM-IF1 cache IDs.

Applies by default; --dry-run validates and previews without writing files.
Only completed caches are migrated. No models, structures, or network needed.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch

from feature_generation.inputs import safe_component, sha256_file
from feature_generation.storage import (
    _assay_lock, _atomic_json, _read_json, _signature, _validate_saved_cache, _write_json,
)


PREFIX = "protssn_proteingym--"
TARGETS = (("sequence", "prosst"), ("inverse_folding", "esm_if1"),
           ("scores", "prosst"), ("scores", "esm_if1"))
WORK_PREFIX = ".cache-name-migration-"


def _check_path(root, path):
    """Reject linked or escaping paths before reading, moving, or removing data."""
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes assay directory: {path}")
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError(f"Symbolic links are not supported during migration: {part}")


def _check_tree(root, path):
    _check_path(root, path)
    for child in path.rglob("*"):
        _check_path(root, child)


def _remove_work(root, work):
    if work.parent != root or not work.name.startswith(WORK_PREFIX):
        raise ValueError(f"Refusing to remove a non-migration directory: {work}")
    _check_tree(root, work)
    # Retain the receipt throughout potentially slow backup cleanup. A killed
    # job can retry deletion even if only part of the backup tree remains.
    for child in work.iterdir():
        if child.name == "transaction.json":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    (work / "transaction.json").unlink(missing_ok=True)
    work.rmdir()


def _updated_metadata(root, kind, family, old_id, new_id):
    metadata = _read_json(root / "assay.json")
    descriptor = metadata.get("features", {}).get(kind)
    if isinstance(descriptor, dict) and descriptor.get("model_family") == family:
        old_path = f"{kind}/{family}/{old_id}"
        if descriptor.get("checkpoint") == old_id or descriptor.get("path") == old_path:
            if descriptor.get("checkpoint") != old_id or descriptor.get("path") != old_path:
                raise ValueError(f"Inconsistent default cache descriptor in {root / 'assay.json'}")
            descriptor.update(checkpoint=new_id, path=f"{kind}/{family}/{new_id}")
    return metadata


def _update_default(root, kind, family, old_id, new_id):
    metadata = _updated_metadata(root, kind, family, old_id, new_id)
    if metadata != _read_json(root / "assay.json"):
        _atomic_json(root / "assay.json", metadata)


def _receipt(root, work):
    _check_tree(root, work)
    receipt = _read_json(work / "transaction.json")
    kind, family = receipt["kind"], receipt["family"]
    old_id, new_id = receipt["old_id"], receipt["new_id"]
    safe_component(old_id, "old cache ID")
    safe_component(new_id, "new cache ID")
    if (receipt.get("version") != 1 or (kind, family) not in TARGETS
            or old_id != PREFIX + new_id or receipt.get("phase") not in ("pending", "prepared", "committed")):
        raise ValueError(f"Invalid migration transaction: {work}")
    return receipt


def _recover(root, work):
    """Roll back an interrupted publish, or finish cleanup after a committed one."""
    _check_tree(root, work)
    if not (work / "transaction.json").exists():
        # Interruption before the first receipt, or after cleanup's final unlink.
        # No cache can have moved before a complete receipt has been published.
        if any(not child.is_file() or not child.name.startswith(".transaction.json.")
               or not child.name.endswith(".tmp") for child in work.iterdir()):
            raise ValueError(f"Missing receipt in nonempty migration directory: {work}")
        _remove_work(root, work)
        return
    receipt = _receipt(root, work)
    kind, family = receipt["kind"], receipt["family"]
    old_id, new_id = receipt["old_id"], receipt["new_id"]
    source = root / kind / family / old_id
    destination = source.with_name(new_id)
    backup = work / "old"
    replacement = work / "new" / kind / family / new_id
    for path in (source, destination):
        _check_path(root, path)
    if receipt["phase"] == "committed":
        if source.exists():
            raise ValueError(f"Original name unexpectedly exists after commit: {source}")
        _validate_saved_cache(root, kind, family, new_id)
        _update_default(root, kind, family, old_id, new_id)
    else:
        if backup.exists():
            if source.exists() or sha256_file(backup / "completion.json") != receipt["original_completion"]:
                raise ValueError(f"Migration backup conflicts with current cache: {work}")
            if destination.exists():
                if (replacement.exists() or sha256_file(destination / "completion.json")
                        != receipt.get("replacement_completion")):
                    raise ValueError(f"Migration destination conflicts with its transaction: {destination}")
                destination.rename(replacement)
            backup.rename(source)
        elif destination.exists():
            raise ValueError(f"Destination exists without a migration backup: {destination}")
        _validate_saved_cache(root, kind, family, old_id)
        _update_default(root, kind, family, new_id, old_id)
    _remove_work(root, work)


def _stage(root, source, work, kind, family, new_id):
    staging = work / "new"
    staging.mkdir()
    shutil.copyfile(root / "assay.json", staging / "assay.json")
    shutil.copyfile(root / "variants.csv", staging / "variants.csv")
    replacement = staging / kind / family / new_id
    # Feature payloads are written once with their new signature below.
    shutil.copytree(source, replacement, ignore=(
        lambda directory, names: ["features.pt"] if Path(directory) == source and kind != "scores" else []
    ))
    config = _read_json(source / "config.json")
    old_signature = _signature(config)
    config["checkpoint"] = new_id
    new_signature = _signature(config)
    _write_json(replacement / "config.json", config)
    if kind != "scores":
        payload = torch.load(source / "features.pt", map_location="cpu", weights_only=True)
        if payload.get("config_signature") != old_signature:
            raise ValueError(f"Feature payload configuration signature differs: {source}")
        payload["config_signature"] = new_signature
        torch.save(payload, replacement / "features.pt")
        del payload
    completion = _read_json(source / "completion.json")
    completion.update(cache_id=new_id, config_signature=new_signature)
    completion["files_sha256"]["config.json"] = sha256_file(replacement / "config.json")
    if kind != "scores":
        completion["files_sha256"]["features.pt"] = sha256_file(replacement / "features.pt")
    _write_json(replacement / "completion.json", completion)
    _validate_saved_cache(staging, kind, family, new_id)
    return replacement


def _migrate_leaf(root, kind, family, old_id):
    source = root / kind / family / old_id
    new_id = old_id.removeprefix(PREFIX)
    destination = source.with_name(new_id)
    work = Path(tempfile.mkdtemp(prefix=WORK_PREFIX, dir=root))
    receipt = {
        "version": 1, "kind": kind, "family": family, "old_id": old_id, "new_id": new_id,
        "phase": "pending", "original_completion": sha256_file(source / "completion.json"),
    }
    try:
        _atomic_json(work / "transaction.json", receipt)
    except BaseException:
        _remove_work(root, work)
        raise
    try:
        replacement = _stage(root, source, work, kind, family, new_id)
        receipt.update(phase="prepared", replacement_completion=sha256_file(replacement / "completion.json"))
        _atomic_json(work / "transaction.json", receipt)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to overwrite destination: {destination}")
        source.rename(work / "old")
        replacement.rename(destination)
        _update_default(root, kind, family, old_id, new_id)
        _validate_saved_cache(root, kind, family, new_id)
        receipt["phase"] = "committed"
        _atomic_json(work / "transaction.json", receipt)
    except BaseException:
        # Recovery leaves its backup intact if an unexpected conflict prevents it.
        _recover(root, work)
        raise
    _remove_work(root, work)


def migrate_assay(root: Path, *, dry_run=False):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Expected a real assay directory: {root}")
    root = root.resolve()
    counts = {"migrated": 0, "planned": 0, "recovered": 0}
    for name in ("assay.json", "variants.csv", "extraction.lock"):
        _check_path(root, root / name)
    # Dry runs perform no writes, including lock-file creation. Run with writers idle.
    with nullcontext() if dry_run else _assay_lock(root):
        for work in sorted(root.glob(WORK_PREFIX + "*")):
            if dry_run:
                raise ValueError(f"Interrupted migration found; rerun without --dry-run to recover: {work}")
            _recover(root, work)
            counts["recovered"] += 1
            print(f"RECOVERED {root.name}: {work.name}", flush=True)
        candidates = []
        for kind, family in TARGETS:
            parent = root / kind / family
            _check_path(root, parent)
            for source in sorted(parent.glob(PREFIX + "*")):
                safe_component(source.name, "old cache ID")
                new_id = safe_component(source.name.removeprefix(PREFIX), "new cache ID")
                _check_tree(root, source)
                destination = source.with_name(new_id)
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError(f"Refusing to overwrite destination: {destination}")
                _updated_metadata(root, kind, family, source.name, new_id)
                _validate_saved_cache(root, kind, family, source.name)
                candidates.append((kind, family, source.name, new_id))
        # All candidates in an assay pass preflight before any of them is changed.
        for kind, family, old_id, new_id in candidates:
            if dry_run:
                counts["planned"] += 1
                status = "WOULD_MIGRATE"
            else:
                _migrate_leaf(root, kind, family, old_id)
                counts["migrated"] += 1
                status = "MIGRATED"
            print(f"{status} {root.name}: {kind}/{family}/{old_id} -> {new_id}", flush=True)
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/features/pretrained/singles")
    parser.add_argument("--assay", action="append", help="Limit to selected assay IDs; repeatable. Default: all assays.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and preview without changing files.")
    args = parser.parse_args(argv)
    root = args.output_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Output root does not exist: {root}")
    assays = args.assay if args.assay is not None else [
        path.name for path in sorted(root.iterdir()) if path.is_dir() and not path.name.startswith(".")
    ]
    if len(set(assays)) != len(assays):
        parser.error("Select unique assay IDs")
    for assay in assays:
        try:
            safe_component(assay, "assay ID")
        except ValueError as error:
            parser.error(str(error))
    torch.set_num_threads(1)
    totals = {"migrated": 0, "planned": 0, "recovered": 0, "failed": 0}
    print(f"{'DRY RUN' if args.dry_run else 'APPLY'}: {root}; assays={len(assays)}", flush=True)
    for assay in assays:
        try:
            counts = migrate_assay(root / assay, dry_run=args.dry_run)
            for key, value in counts.items():
                totals[key] += value
            if not any(counts.values()):
                print(f"UNCHANGED {assay}: no matching prefixed caches", flush=True)
        except Exception as error:
            totals["failed"] += 1
            print(f"FAILED {assay}: {error}", flush=True)
    print("Summary: " + "; ".join(f"{key}={value}" for key, value in totals.items()), flush=True)
    return int(totals["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
