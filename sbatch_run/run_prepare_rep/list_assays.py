"""List preparation targets from the extraction config's ProteinGym catalog.

Uses only the Python standard library so array sizes can be computed on a login
node before activating the GPU environment. Task indices follow catalog order.
"""

import argparse
import csv
import json
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def list_assays(config_path: Path) -> list[str]:
    with config_path.open(encoding="utf-8-sig") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("assay_index"), str):
        raise ValueError("Extraction config must specify an assay_index path")
    index_path = Path(config["assay_index"]).expanduser()
    if not index_path.is_absolute():
        index_path = PROJECT_ROOT / index_path
    index_path = index_path.resolve()
    with index_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"DMS_id", "DMS_filename"}.issubset(reader.fieldnames or []):
            raise ValueError("Assay catalog requires DMS_id and DMS_filename columns")
        assays, filenames = [], set()
        for row in reader:
            assay, filename = row["DMS_id"], row["DMS_filename"]
            if not assay or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", assay) is None:
                raise ValueError(f"Invalid assay ID: {assay!r}")
            if assay in assays:
                raise ValueError(f"Duplicate assay ID: {assay}")
            if not filename or filename in filenames:
                raise ValueError(f"Missing or duplicate assay filename for {assay}")
            assay_path = (index_path.parent / filename).resolve()
            if not assay_path.is_relative_to(index_path.parent):
                raise ValueError(f"Assay filename leaves the catalog directory: {filename}")
            if not assay_path.is_file():
                raise ValueError(f"Assay CSV does not exist: {assay_path}")
            assays.append(assay)
            filenames.add(filename)
    if not assays:
        raise ValueError(f"No assays in catalog: {index_path}")
    return assays


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--count", action="store_true", help="Print the number of catalog assays")
    selection.add_argument("--task-index", type=int, help="Select one assay by zero-based catalog row")
    args = parser.parse_args(argv)
    try:
        assays = list_assays(args.config)
        if args.count:
            print(len(assays))
        elif args.task_index is not None:
            if not 0 <= args.task_index < len(assays):
                raise ValueError(f"Task index {args.task_index} is outside 0..{len(assays) - 1}")
            print(assays[args.task_index])
        else:
            print("\n".join(assays))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
