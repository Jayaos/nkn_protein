"""Independent, explicitly mapped structures for inverse-folding extraction.

Fragment coordinates have separate reference frames. This module never creates
combined coordinates or invents distances between independently folded pieces.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence
import zipfile

import torch

from data_loading.catalog import AssayCatalog
from .config import PROJECT_ROOT
from .inputs import safe_component, sha256_file, stable_json
from .structure_inputs import (
    ATOM_ORDER, RESIDUE_CODES, StructureInput, _atomic_json, _load_structure_metadata,
    _parse_structure, _signature, load_structure, validate_structure_metadata,
    PROTEINGYM_STRUCTURE_SHA256, PROTEINGYM_STRUCTURE_URL,
)


COVERAGE_POLICY = "explicit_reference_positions"
STITCH_POLICY = "concatenate_disjoint_reference_ranges"
BRCA2_FRAGMENTS = (
    ("BRCA2_HUMAN_1-1000.pdb", 1, 1000),
    ("BRCA2_HUMAN_1001-2085.pdb", 1001, 2085),
    ("BRCA2_HUMAN_2086-2832.pdb", 2086, 2832),
)


@dataclass(frozen=True)
class StructureFragments:
    reference_sequence: str
    fragments: tuple[StructureInput, ...]
    reference_ranges: tuple[tuple[int, int], ...]
    positions: torch.Tensor
    metadata: dict
    input_signature: str
    structure_sha256: str


def _ranges(values, reference_length):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("Structure fragments require explicit nonempty reference ranges")
    previous_end = 0
    result = []
    for value in values:
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or any(type(item) is not int for item in value)):
            raise ValueError("Fragment ranges must be pairs of one-based integer positions")
        start, end = value
        if not previous_end < start <= end <= reference_length:
            raise ValueError("Fragment ranges must be sorted, disjoint, and inside the reference sequence")
        result.append((start, end))
        previous_end = end
    return tuple(result)


def _composite_digest(fragments):
    identity = [{"reference_start": row["reference_start"], "reference_end": row["reference_end"],
                 "structure_sha256": row["structure"]["structure_sha256"]} for row in fragments]
    return hashlib.sha256(stable_json(identity).encode("utf-8")).hexdigest()


def _bundle_signature(metadata):
    identity = {key: metadata[key] for key in (
        "schema_version", "kind", "reference_sequence", "coverage_policy", "stitch_policy",
        "residue_mapping", "atom_order", "structure_sha256",
    )}
    identity["fragments"] = [
        {"reference_start": row["reference_start"], "reference_end": row["reference_end"],
         "input_signature": row["structure"]["input_signature"]}
        for row in metadata["fragments"]
    ]
    return hashlib.sha256(stable_json(identity).encode("utf-8")).hexdigest()


def _global_mapping(fragments):
    return [
        {**residue, "reference_position": row["reference_start"] + residue["reference_position"] - 1,
         "fragment_index": index, "local_reference_position": residue["reference_position"]}
        for index, row in enumerate(fragments) for residue in row["structure"]["residue_mapping"]
    ]


def validate_inverse_structure_metadata(metadata: dict, reference_sequence: str) -> None:
    """Validate legacy structure or fragment metadata without reading PDB files."""
    if isinstance(metadata, dict) and metadata.get("schema_version") == 1:
        validate_structure_metadata(metadata, reference_sequence)
        return
    if (not isinstance(metadata, dict) or metadata.get("schema_version") != 2
            or metadata.get("kind") != "reference_fragments"):
        raise ValueError("Unsupported inverse-folding structure manifest schema")
    if not reference_sequence or any(aa not in RESIDUE_CODES.values() for aa in reference_sequence):
        raise ValueError("Structure reference must contain canonical amino acids")
    if metadata.get("reference_sequence") != reference_sequence:
        raise ValueError("Structure reference sequence does not match assay reference")
    if (metadata.get("coverage_policy") != COVERAGE_POLICY or metadata.get("stitch_policy") != STITCH_POLICY
            or metadata.get("atom_order") != list(ATOM_ORDER)):
        raise ValueError("Unsupported fragment coverage, stitching, or atom policy")
    fragments = metadata.get("fragments")
    if not isinstance(fragments, list) or not fragments or any(not isinstance(row, dict) for row in fragments):
        raise ValueError("Structure fragments must be a nonempty list of mappings")
    ranges = _ranges([(row.get("reference_start"), row.get("reference_end")) for row in fragments], len(reference_sequence))
    for row, (start, end) in zip(fragments, ranges):
        local = row.get("structure")
        validate_structure_metadata(local, reference_sequence[start - 1:end])
        if local.get("assay_id") != metadata.get("assay_id"):
            raise ValueError("Structure fragment belongs to a different assay")
    if metadata.get("residue_mapping") != _global_mapping(fragments):
        raise ValueError("Fragment residue mapping disagrees with the explicit reference ranges")
    if metadata.get("structure_sha256") != _composite_digest(fragments):
        raise ValueError("Fragment composite structure SHA-256 mismatch")
    if metadata.get("input_signature") != _bundle_signature(metadata):
        raise ValueError("Fragment input signature does not match manifest contents")


def inverse_structure_positions(metadata: dict, reference_sequence: str) -> torch.Tensor:
    """Return verified global reference positions in probability row order."""
    validate_inverse_structure_metadata(metadata, reference_sequence)
    return torch.tensor([row["reference_position"] for row in metadata["residue_mapping"]], dtype=torch.long)


def load_inverse_structure(manifest_path: str | Path, reference_sequence: str,
                           require_oxygen: bool = True) -> StructureInput | StructureFragments:
    """Load separate coordinate arrays for fragments or a legacy full structure."""
    manifest_path = Path(manifest_path).resolve()
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(metadata, dict) and metadata.get("schema_version") == 1:
        return load_structure(manifest_path, reference_sequence, require_oxygen)
    validate_inverse_structure_metadata(metadata, reference_sequence)
    ranges = tuple((row["reference_start"], row["reference_end"]) for row in metadata["fragments"])
    fragments = tuple(
        _load_structure_metadata(row["structure"], manifest_path, reference_sequence[start - 1:end], require_oxygen)
        for row, (start, end) in zip(metadata["fragments"], ranges)
    )
    return StructureFragments(reference_sequence, fragments, ranges,
                              inverse_structure_positions(metadata, reference_sequence), metadata,
                              metadata["input_signature"], metadata["structure_sha256"])


def prepare_structure_fragments(index_path: str | Path, assay_id: str,
                                structure_files: Sequence[str | Path], reference_ranges: Sequence[tuple[int, int]],
                                chain_id: str | None = None, model_index: int = 1,
                                output_dir: str | Path = PROJECT_ROOT / "data/structures/manifests",
                                source_archive: dict | None = None) -> Path:
    """Register exact, disjoint reference slices without placing them in one frame.

    Uncovered reference sites remain absent. Downstream extraction must reject
    mutations at such sites instead of filling them with invented probabilities.
    """
    safe_component(assay_id, "assay ID")
    index_path = Path(index_path).resolve()
    record = AssayCatalog(index_path).get(assay_id)
    reference = record.reference_sequence
    ranges = _ranges(reference_ranges, len(reference))
    if not isinstance(structure_files, (list, tuple)) or len(structure_files) != len(ranges):
        raise ValueError("Provide exactly one structure file per fragment reference range")
    files = tuple(Path(path).resolve() for path in structure_files)
    index_hash = sha256_file(index_path)
    destination = Path(output_dir).resolve() / f"{assay_id}.json"
    fragments = []
    for structure_file, (start, end) in zip(files, ranges):
        sequence = reference[start - 1:end]
        digest = sha256_file(structure_file)
        _, mapping, selected_chain, parser_version = _parse_structure(structure_file, sequence, chain_id, model_index)
        if sha256_file(structure_file) != digest:
            raise ValueError("Structure fragment changed while being read")
        local = {
            "schema_version": 1, "assay_id": assay_id, "reference_sequence": sequence,
            "structure_path": Path(os.path.relpath(structure_file, destination.parent)).as_posix(),
            "structure_sha256": digest, "chain_id": selected_chain, "model_index": model_index,
            "residue_mapping": mapping, "source_index_sha256": index_hash,
            "parser": "biopython", "parser_version": parser_version,
            "atom_order": list(ATOM_ORDER), "mapping_policy": "exact_full_chain_sequence",
            "altloc_policy": "reject_backbone_alternates",
        }
        local["input_signature"] = _signature(local)
        fragments.append({"reference_start": start, "reference_end": end, "structure": local})
    if sha256_file(index_path) != index_hash:
        raise ValueError("Assay index changed while reading structure fragments")
    metadata = {
        "schema_version": 2, "kind": "reference_fragments", "assay_id": assay_id,
        "reference_sequence": reference, "fragments": fragments,
        "coverage_policy": COVERAGE_POLICY, "stitch_policy": STITCH_POLICY,
        "atom_order": list(ATOM_ORDER), "residue_mapping": _global_mapping(fragments),
        "source_index_sha256": index_hash,
        "catalog_structure": {key: record.metadata.get(key) for key in ("pdb_file", "pdb_range")},
        "structure_sha256": _composite_digest(fragments),
    }
    if source_archive is not None:
        metadata["source_archive"] = source_archive
    metadata["input_signature"] = _bundle_signature(metadata)
    validate_inverse_structure_metadata(metadata, reference)
    if destination.exists():
        existing = load_inverse_structure(destination, reference, require_oxygen=False)
        if existing.input_signature != metadata["input_signature"]:
            raise ValueError(f"A different structure is already registered at {destination}; choose another manifest directory")
        return destination
    _atomic_json(destination, metadata)
    return destination


def download_proteingym_fragments(index_path: str | Path, assay_id: str,
                                 chain_id: str | None = None, model_index: int = 1,
                                 raw_dir: str | Path = PROJECT_ROOT / "data/structures/raw",
                                 manifest_dir: str | Path = PROJECT_ROOT / "data/structures/manifests") -> Path:
    """Acquire ProteinGym's known BRCA2 pieces with exact reference-slice checks."""
    from .downloads import download_verified

    safe_component(assay_id, "assay ID")
    record = AssayCatalog(index_path).get(assay_id)
    if assay_id != "BRCA2_HUMAN_Erwood_2022_HEK293T" or record.metadata.get("pdb_file") != "BRCA2_HUMAN.pdb":
        raise ValueError("No explicit ProteinGym fragment recipe for this assay")
    ranges = tuple((start, end) for _, start, end in BRCA2_FRAGMENTS)
    _ranges(ranges, len(record.reference_sequence))
    manifest = Path(manifest_dir).resolve() / f"{assay_id}.json"
    if manifest.exists():
        existing = load_inverse_structure(manifest, record.reference_sequence, require_oxygen=False)
        source = existing.metadata.get("source_archive", {})
        if (not isinstance(existing, StructureFragments) or existing.reference_ranges != ranges
                or source.get("sha256") != PROTEINGYM_STRUCTURE_SHA256
                or any(fragment.model_index != model_index or (chain_id is not None and fragment.chain_id != chain_id)
                       for fragment in existing.fragments)):
            raise ValueError("The existing fragment manifest uses a different source, model, chain, or reference ranges")
        return manifest
    raw_dir = Path(raw_dir).resolve()
    archive_path = raw_dir.parent / "downloads" / "ProteinGym_v1.1_AF2_structures.zip"
    download_verified(PROTEINGYM_STRUCTURE_URL, archive_path, PROTEINGYM_STRUCTURE_SHA256)
    raw_dir.mkdir(parents=True, exist_ok=True)
    files, source_members = [], []
    with zipfile.ZipFile(archive_path) as archive:
        for filename, start, end in BRCA2_FRAGMENTS:
            members = [member for member in archive.infolist() if not member.is_dir()
                       and member.filename.replace("\\", "/").rsplit("/", 1)[-1] == filename]
            if len(members) != 1:
                raise ValueError(f"Expected one ProteinGym archive member named {filename!r}; found {len(members)}")
            member = members[0]
            if not 0 < member.file_size <= 64 * 1024 * 1024:
                raise ValueError("ProteinGym structure member is empty or exceeds 64 MiB")
            target = raw_dir / filename
            with tempfile.NamedTemporaryFile(dir=raw_dir, suffix=target.suffix, delete=False) as handle:
                temporary = Path(handle.name)
                try:
                    with archive.open(member) as source:
                        while True:
                            block = source.read(1024 * 1024)
                            if not block:
                                break
                            handle.write(block)
                    handle.close()
                    _parse_structure(temporary, record.reference_sequence[start - 1:end], chain_id, model_index)
                    if target.exists():
                        if sha256_file(target) != sha256_file(temporary):
                            raise ValueError(f"Existing raw structure differs from the pinned ProteinGym archive: {target}")
                    else:
                        os.replace(temporary, target)
                finally:
                    handle.close()
                    temporary.unlink(missing_ok=True)
            files.append(target)
            source_members.append(member.filename)
    return prepare_structure_fragments(
        index_path, assay_id, files, ranges, chain_id, model_index, manifest_dir,
        source_archive={"url": PROTEINGYM_STRUCTURE_URL, "sha256": PROTEINGYM_STRUCTURE_SHA256,
                        "members": source_members},
    )
