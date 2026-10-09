"""Assay discovery and explicit validation against processed ProteinGym tables."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .features import FeatureBatch, parse_substitutions, read_csv


@dataclass(frozen=True)
class AssayRecord:
    assay_id: str
    filename: str
    reference_sequence: str
    path: Path
    metadata: dict[str, str]

    @property
    def DMS_id(self) -> str:
        return self.assay_id


@dataclass(frozen=True)
class VariantRecord:
    variant_id: str
    mutant: str
    sequence: str
    target: float
    source_row_index: int
    mutation_positions: tuple[int, ...]


class AssayCatalog:
    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path).resolve()
        rows = read_csv(self.index_path)
        if not {"DMS_id", "DMS_filename", "target_seq"}.issubset(rows[0]):
            raise ValueError("Assay index requires DMS_id, DMS_filename, and target_seq")
        self.records: dict[str, AssayRecord] = {}
        filenames: set[str] = set()
        for row in rows:
            assay_id, filename, reference = row["DMS_id"], row["DMS_filename"], row["target_seq"]
            if not assay_id or assay_id in self.records or filename in filenames:
                raise ValueError("Assay IDs and filenames must be nonempty and unique")
            path = (self.index_path.parent / filename).resolve()
            if not filename or not path.is_relative_to(self.index_path.parent):
                raise ValueError("Assay filename must stay inside the index directory")
            if not path.is_file():
                raise FileNotFoundError(path)
            parse_substitutions("WT", reference, reference)
            if row.get("seq_len") and int(row["seq_len"]) != len(reference):
                raise ValueError(f"Reference length mismatch: {assay_id}")
            self.records[assay_id] = AssayRecord(assay_id, filename, reference, path, row)
            filenames.add(filename)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records.values())

    def get(self, assay_id: str) -> AssayRecord:
        try:
            return self.records[assay_id]
        except KeyError as error:
            raise KeyError(f"Unknown assay: {assay_id}") from error

    def load_variants(self, assay_id: str, singles_only: bool = False) -> tuple[VariantRecord, ...]:
        record = self.get(assay_id)
        rows = read_csv(record.path)
        if not {"mutant", "mutated_sequence", "DMS_score"}.issubset(rows[0]):
            raise ValueError(f"Missing required assay columns: {record.path}")
        variants = []
        seen: set[str] = set()
        for i, row in enumerate(rows):
            variant_id = row.get("variant_id") or row["mutant"]
            if not variant_id or variant_id in seen:
                raise ValueError(f"Duplicate or empty variant ID in {record.path}: {variant_id!r}")
            seen.add(variant_id)
            positions = parse_substitutions(row["mutant"], row["mutated_sequence"], record.reference_sequence)
            target = float(row["DMS_score"])
            if not math.isfinite(target):
                raise ValueError(f"Non-finite target in {record.path}, row {i}")
            if not singles_only or len(positions) == 1:
                variants.append(VariantRecord(variant_id, row["mutant"], row["mutated_sequence"], target, i, positions))
        return tuple(variants)


def validate_assay_coverage(
    batch: FeatureBatch,
    index_path: str | Path,
    singles_only: bool = False,
    allow_subset: bool = False,
) -> AssayRecord:
    """Require full selected-assay coverage unless fixture subsets are explicit."""
    catalog = AssayCatalog(index_path)
    record = catalog.get(batch.assay_id)
    if batch.reference_sequence != record.reference_sequence:
        raise ValueError("Feature reference sequence does not match the assay index")
    variants = {row.variant_id: row for row in catalog.load_variants(batch.assay_id, singles_only=singles_only)}
    observed, expected = set(batch.variant_ids), set(variants)
    if not observed.issubset(expected) or (not allow_subset and observed != expected):
        raise ValueError(f"Feature coverage mismatch: {len(expected - observed)} missing, {len(observed - expected)} unknown variants; subset coverage must be explicit")
    packed_positions: list[list[int]] = [[] for _ in batch.variant_ids]
    for owner, position in zip(batch.owners.cpu().tolist(), batch.mutation_positions.cpu().tolist()):
        packed_positions[owner].append(position)
    targets = None if batch.targets is None else batch.targets.detach().cpu().tolist()
    for i, variant_id in enumerate(batch.variant_ids):
        row = variants[variant_id]
        if batch.sequences and batch.sequences[i] != row.sequence:
            raise ValueError(f"Feature sequence differs from indexed assay: {variant_id}")
        if sorted(packed_positions[i]) != sorted(row.mutation_positions):
            raise ValueError(f"Feature mutation positions differ from indexed assay: {variant_id}")
        if targets is not None and not math.isclose(targets[i], row.target, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(f"Feature target differs from indexed assay: {variant_id}")
    return record
