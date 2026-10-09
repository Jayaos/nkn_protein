"""Validated, deterministic ProteinGym inputs for offline feature extraction."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

from data_loading.catalog import AssayCatalog
from data_loading.features import read_csv


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_component(value: str, name: str = "path component") -> str:
    """Accept a single portable directory name, never a relative path."""
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) is None:
        raise ValueError(f"Invalid {name}: {value!r}")
    if value.endswith(".") or value.upper().split(".")[0] in {
        "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))
    }:
        raise ValueError(f"Invalid {name}: {value!r}")
    return value


def stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class AssayInput:
    assay_id: str
    reference_sequence: str
    rows: list[dict]
    source_assay_path: Path
    source_assay_sha256: str
    source_index_sha256: str
    coverage: dict

    @property
    def sequences(self) -> tuple[str, ...]:
        return tuple(row["mutated_sequence"] for row in self.rows)

    @property
    def variant_ids(self) -> tuple[str, ...]:
        return tuple(row["variant_id"] for row in self.rows)

    @property
    def length(self) -> int:
        return len(self.reference_sequence)

    @property
    def sequence_hashes(self) -> tuple[str, ...]:
        return tuple(hashlib.sha256(sequence.encode("utf-8")).hexdigest() for sequence in self.sequences)

    def variants_csv(self) -> str:
        columns = ["variant_id", "source_row_index", "mutant", "mutated_sequence", "DMS_score"]
        if any("DMS_score_bin" in row for row in self.rows):
            columns.append("DMS_score_bin")
        handle = io.StringIO(newline="")
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(self.rows)
        return handle.getvalue()

    @property
    def variants_sha256(self) -> str:
        return hashlib.sha256(self.variants_csv().encode("utf-8")).hexdigest()

    @property
    def input_signature(self) -> str:
        identity = {
            "assay_id": self.assay_id,
            "reference_sequence": self.reference_sequence,
            "source_assay_sha256": self.source_assay_sha256,
            "source_index_sha256": self.source_index_sha256,
            "variants_sha256": self.variants_sha256,
            "coverage": self.coverage,
        }
        return hashlib.sha256(stable_json(identity).encode("utf-8")).hexdigest()


def prepare_assay(
    index_path: str | Path,
    assay_id: str,
    singles_only: bool = False,
    limit: int | None = None,
) -> AssayInput:
    """Validate source rows and select the first N eligible rows without labels."""
    safe_component(assay_id, "assay ID")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer or None")
    index_path = Path(index_path).resolve()
    index_hash = sha256_file(index_path)
    catalog = AssayCatalog(index_path)
    record = catalog.get(assay_id)
    source_hash = sha256_file(record.path)
    variants = catalog.load_variants(assay_id, singles_only=singles_only)
    source_rows = read_csv(record.path)
    selected = variants if limit is None else variants[:limit]
    if not selected:
        raise ValueError(f"No variants selected for assay {assay_id}")
    rows = []
    for variant in selected:
        source = source_rows[variant.source_row_index]
        row = {
            "variant_id": variant.variant_id,
            "source_row_index": variant.source_row_index,
            "mutant": variant.mutant,
            "mutated_sequence": variant.sequence,
            "DMS_score": source["DMS_score"],
        }
        if "DMS_score_bin" in source:
            row["DMS_score_bin"] = source["DMS_score_bin"]
        rows.append(row)
    if sha256_file(record.path) != source_hash or sha256_file(index_path) != index_hash:
        raise ValueError("Assay inputs changed while being read; retry with stable source files")
    return AssayInput(
        assay_id=assay_id,
        reference_sequence=record.reference_sequence,
        rows=rows,
        source_assay_path=record.path,
        source_assay_sha256=source_hash,
        source_index_sha256=index_hash,
        coverage={
            "source_row_count": len(source_rows),
            "eligible_row_count": len(variants),
            "num_variants": len(rows),
            "singles_only": singles_only,
            "limit": limit,
            "selection": "source_order",
            "complete_assay": len(rows) == len(source_rows),
        },
    )
