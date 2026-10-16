"""Explicit Kermut mappings between original PDB inputs and assay references.

ProteinMPNN features are inferred on the unmodified source PDB sequence and
coordinates. Only the resulting rows are selected/mapped, as in Kermut.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import torch

from data_loading.catalog import AssayCatalog
from .inputs import safe_component, sha256_file, stable_json
from .structure_inputs import (
    ATOM_ORDER, RESIDUE_CODES, StructureInput, _atomic_json, _load_structure_metadata,
    _parse_structure, _signature, validate_structure_metadata,
)


DOMAIN_RANGES = {
    "A0A140D2T1_ZIKV_Sourisseau_2019": (3423, 291, 794, "A0A140D2T1_ZIKV.pdb"),
    "POLG_HCVJF_Qi_2014": (3033, 1982, 2225, "POLG_HCVJF.pdb"),
}
P53_ASSAYS = frozenset({
    "P53_HUMAN_Giacomelli_2018_Null_Etoposide",
    "P53_HUMAN_Giacomelli_2018_Null_Nutlin",
    "P53_HUMAN_Giacomelli_2018_WT_Nutlin",
})
CAS9_ASSAY = "CAS9_STRP1_Spencer_2017_positive"
MAPPED_ASSAYS = frozenset({*DOMAIN_RANGES, *P53_ASSAYS, CAS9_ASSAY})


def mapping_recipe(assay_id, reference, source_sequence):
    """Reject every difference except the six explicitly documented cases."""
    if assay_id in DOMAIN_RANGES:
        length, start, end, filename = DOMAIN_RANGES[assay_id]
        if len(reference) != length or source_sequence != reference[start - 1:end]:
            raise ValueError("Kermut domain must exactly match its registered reference interval")
        return "domain", list(range(start, end + 1)), list(range(1, len(source_sequence) + 1)), filename
    if assay_id == CAS9_ASSAY:
        if len(reference) != 1368 or len(source_sequence) != 1390 or source_sequence[:1368] != reference:
            raise ValueError("Kermut CAS9 requires the exact 1368-residue prefix of the 1390-residue PDB")
        return "prefix_after_inference", list(range(1, 1369)), list(range(1, 1369)), "CAS9_STRP1.pdb"
    if assay_id in P53_ASSAYS:
        if (len(reference) != 393 or len(source_sequence) != 393 or source_sequence[71] != "P"
                or reference[71] != "R" or source_sequence[:71] + "R" + source_sequence[72:] != reference):
            raise ValueError("Kermut P53 permits only the source P72 / assay R72 mismatch")
        return "p53_P72_R72_matching_only", list(range(1, 394)), list(range(1, 394)), "P53_HUMAN.pdb"
    raise ValueError("No explicit Kermut structure mapping for this assay")


def _mapped_rows(source, reference, positions, source_positions):
    return [dict(source["residue_mapping"][s - 1], reference_position=p,
                 source_position=s, amino_acid=reference[p - 1]) for p, s in zip(positions, source_positions)]


def _mapped_signature(metadata):
    identity = {key: metadata[key] for key in (
        "schema_version", "kind", "assay_id", "reference_sequence", "mapping_policy",
        "reference_positions", "source_positions", "residue_mapping", "structure_sha256", "atom_order",
    )}
    identity["source_input_signature"] = metadata["source"]["input_signature"]
    return hashlib.sha256(stable_json(identity).encode("utf-8")).hexdigest()


def validate_mapped_metadata(metadata, reference, require_full=False):
    if not reference or any(aa not in RESIDUE_CODES.values() for aa in reference):
        raise ValueError("Mapped structure reference must contain canonical amino acids")
    if metadata.get("schema_version") != 3 or metadata.get("kind") != "kermut_reference_mapping":
        raise ValueError("Unsupported mapped structure schema")
    if metadata.get("reference_sequence") != reference:
        raise ValueError("Mapped structure reference differs from assay")
    source = metadata["source"]
    if source.get("schema_version") != 1 or source.get("assay_id") != metadata["assay_id"]:
        raise ValueError("Mapped source must be an exact original PDB chain for this assay")
    validate_structure_metadata(source, source["reference_sequence"])
    policy, positions, source_positions, _ = mapping_recipe(metadata["assay_id"], reference, source["reference_sequence"])
    if (metadata.get("mapping_policy") != policy or metadata.get("reference_positions") != positions
            or metadata.get("source_positions") != source_positions
            or metadata.get("residue_mapping") != _mapped_rows(source, reference, positions, source_positions)
            or metadata.get("structure_sha256") != source["structure_sha256"]
            or metadata.get("atom_order") != list(ATOM_ORDER)):
        raise ValueError("Mapped structure does not match its explicit Kermut recipe")
    if metadata.get("input_signature") != _mapped_signature(metadata):
        raise ValueError("Mapped structure signature mismatch")
    if require_full and positions != list(range(1, len(reference) + 1)):
        raise ValueError("Domain-only structures cannot supply full-reference ProteinMPNN scores; use ESM2 scores")


@dataclass(frozen=True)
class MappedStructure:
    reference_sequence: str
    source: StructureInput
    positions: torch.Tensor
    source_indices: torch.Tensor
    metadata: dict

    @property
    def coords(self):
        return self.source.coords[self.source_indices]

    @property
    def input_signature(self):
        return self.metadata["input_signature"]

    @property
    def structure_sha256(self):
        return self.source.structure_sha256

    @property
    def model_index(self):
        return self.source.model_index

    @property
    def chain_id(self):
        return self.source.chain_id

    def full_reference_structure(self):
        """Full-reference scoring uses assay sequence and mapped backbone only."""
        validate_mapped_metadata(self.metadata, self.reference_sequence, require_full=True)
        return StructureInput(self.reference_sequence, self.coords, self.positions, self.chain_id,
                              self.metadata["residue_mapping"], self.structure_sha256,
                              self.input_signature, self.metadata, self.model_index)


def load_mapped_structure(manifest_path, reference, require_oxygen=True):
    path = Path(manifest_path).resolve()
    metadata = json.loads(path.read_text(encoding="utf-8"))
    validate_mapped_metadata(metadata, reference)
    source_metadata = metadata["source"]
    source = _load_structure_metadata(source_metadata, path, source_metadata["reference_sequence"], require_oxygen)
    return MappedStructure(reference, source, torch.tensor(metadata["reference_positions"], dtype=torch.long),
                           torch.tensor(metadata["source_positions"], dtype=torch.long) - 1, metadata)


def prepare_mapped_structure(index_path, assay_id, structure_file, chain_id, model_index,
                             output_dir, source_url=None, source_archive=None):
    safe_component(assay_id, "assay ID")
    record = AssayCatalog(index_path).get(assay_id)
    structure_file = Path(structure_file).resolve()
    destination = Path(output_dir).resolve() / f"{assay_id}.json"
    digest = sha256_file(structure_file)
    index_digest = sha256_file(index_path)
    _, mapping, chain, parser_version = _parse_structure(structure_file, None, chain_id, model_index)
    source_sequence = "".join(row["amino_acid"] for row in mapping)
    policy, positions, source_positions, filename = mapping_recipe(assay_id, record.reference_sequence, source_sequence)
    if record.metadata.get("pdb_file") != filename:
        raise ValueError("Kermut mapping requires the registered ProteinGym PDB filename")
    if sha256_file(structure_file) != digest or sha256_file(index_path) != index_digest:
        raise ValueError("Structure/catalog changed during mapping")
    source = {
        "schema_version": 1, "assay_id": assay_id, "reference_sequence": source_sequence,
        "structure_path": Path(os.path.relpath(structure_file, destination.parent)).as_posix(),
        "structure_sha256": digest, "chain_id": chain, "model_index": model_index,
        "residue_mapping": mapping, "parser": "biopython", "parser_version": parser_version,
        "atom_order": list(ATOM_ORDER), "mapping_policy": "exact_full_chain_sequence",
        "altloc_policy": "reject_backbone_alternates",
    }
    source["input_signature"] = _signature(source)
    metadata = {
        "schema_version": 3, "kind": "kermut_reference_mapping", "assay_id": assay_id,
        "reference_sequence": record.reference_sequence, "source": source,
        "mapping_policy": policy, "reference_positions": positions, "source_positions": source_positions,
        "residue_mapping": _mapped_rows(source, record.reference_sequence, positions, source_positions),
        "structure_sha256": digest, "atom_order": list(ATOM_ORDER),
        "source_index_sha256": index_digest, "source_url": source_url,
        "source_archive": source_archive,
    }
    metadata["input_signature"] = _mapped_signature(metadata)
    validate_mapped_metadata(metadata, record.reference_sequence)
    if destination.exists():
        existing = load_mapped_structure(destination, record.reference_sequence, require_oxygen=False)
        if existing.input_signature != metadata["input_signature"]:
            raise ValueError("A different mapped structure is already registered; choose another manifest directory")
    else:
        _atomic_json(destination, metadata)
    return destination
