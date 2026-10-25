"""Strict, offline structure inputs shared by inverse-folding backends.

Manifests reference structure files relative to the manifest directory. Their
identity records exact full-chain sequence mapping, original author residue
numbers, model selection, file checksum, and the supported backbone atoms.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import torch

from data_loading.catalog import AssayCatalog
from .config import PROJECT_ROOT
from .inputs import safe_component, sha256_file, stable_json

ATOM_ORDER = ("N", "CA", "C", "O")
RESIDUE_CODES = dict(zip(
    ("ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
     "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"),
    "ACDEFGHIKLMNPQRSTVWY",
))
IDENTITY_KEYS = ("schema_version", "reference_sequence", "structure_sha256", "chain_id", "model_index", "residue_mapping")


@dataclass(frozen=True)
class StructureInput:
    reference_sequence: str
    coords: torch.Tensor
    positions: torch.Tensor
    chain_id: str
    residue_mapping: list[dict]
    structure_sha256: str
    input_signature: str
    metadata: dict
    model_index: int = 1


def _signature(metadata: dict) -> str:
    identity = {key: metadata[key] for key in IDENTITY_KEYS}
    return hashlib.sha256(stable_json(identity).encode("utf-8")).hexdigest()


def validate_structure_metadata(metadata: dict, reference_sequence: str) -> None:
    """Validate a copied manifest without needing the source structure file."""
    if isinstance(metadata, dict) and metadata.get("schema_version") == 3:
        from .structure_mappings import validate_mapped_metadata
        validate_mapped_metadata(metadata, reference_sequence, require_full=True)
        return
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("Unsupported structure manifest schema")
    if not reference_sequence or any(aa not in RESIDUE_CODES.values() for aa in reference_sequence):
        raise ValueError("Structure reference must contain canonical amino acids")
    if metadata.get("reference_sequence") != reference_sequence:
        raise ValueError("Structure reference sequence does not match assay reference")
    if type(metadata.get("model_index")) is not int or metadata["model_index"] < 1:
        raise ValueError("Structure model_index must be a positive, one-based integer")
    if not isinstance(metadata.get("chain_id"), str) or not metadata["chain_id"]:
        raise ValueError("Structure chain_id must be nonempty")
    digest = metadata.get("structure_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid structure SHA-256")
    mapping = metadata.get("residue_mapping")
    if not isinstance(mapping, list) or len(mapping) != len(reference_sequence):
        raise ValueError("Structure mapping must cover every reference position")
    seen = set()
    for position, (amino_acid, row) in enumerate(zip(reference_sequence, mapping), 1):
        if not isinstance(row, dict) or type(row.get("reference_position")) is not int or row["reference_position"] != position:
            raise ValueError("Structure mapping positions must be exactly 1..L in order")
        if row.get("chain_id") != metadata["chain_id"] or row.get("amino_acid") != amino_acid or RESIDUE_CODES.get(row.get("residue_name")) != amino_acid:
            raise ValueError("Structure residue mapping disagrees with chain/reference sequence")
        if type(row.get("residue_number")) is not int or not isinstance(row.get("insertion_code"), str) or len(row["insertion_code"]) > 1:
            raise ValueError("Invalid structure author residue identifier")
        key = (row["residue_number"], row["insertion_code"])
        if key in seen:
            raise ValueError("Duplicate author residue identifier in structure mapping")
        seen.add(key)
        atoms = row.get("backbone_atoms")
        if atoms not in [list(ATOM_ORDER), list(ATOM_ORDER[:3])]:
            raise ValueError("Every structure residue must have N, CA, C backbone atoms")
    if metadata.get("atom_order") != list(ATOM_ORDER) or metadata.get("mapping_policy") != "exact_full_chain_sequence" or metadata.get("altloc_policy") != "reject_backbone_alternates":
        raise ValueError("Unsupported structure atom/mapping/alternate-conformation policy")
    if metadata.get("input_signature") != _signature(metadata):
        raise ValueError("Structure input signature does not match manifest contents")


def _parse_structure(path: Path, reference_sequence: str | None, chain_id: str | None, model_index: int):
    try:
        import Bio
        from Bio.PDB import MMCIFParser, PDBParser
        from Bio.PDB.Polypeptide import is_aa
        from Bio.PDB.PDBExceptions import PDBConstructionException, PDBException
    except ImportError as error:
        raise ImportError("Install biopython from env.yaml or requirements.txt to read structures") from error
    if type(model_index) is not int or model_index < 1:
        raise ValueError("model_index must be a positive, one-based integer")
    suffix = path.suffix.lower()
    if suffix not in (".pdb", ".ent", ".cif", ".mmcif"):
        raise ValueError("Structure must be an uncompressed PDB (.pdb/.ent) or mmCIF (.cif/.mmcif) file")
    parser = PDBParser(PERMISSIVE=False, QUIET=True) if suffix in (".pdb", ".ent") else MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True)
    try:
        structure = parser.get_structure("reference", str(path))
    except (PDBConstructionException, PDBException, ValueError, KeyError, IndexError) as error:
        raise ValueError(f"Cannot parse structure {path.name}: {error}") from error
    models = list(structure.get_models())
    if model_index > len(models):
        raise ValueError(f"Requested model {model_index}; structure has {len(models)} models")
    model = models[model_index - 1]
    chains = list(model.get_chains())
    if chain_id is None:
        # Sequence matching must not silently choose between equivalent chains.
        candidates = [chain for chain in chains if any(residue.id[0] == " " for residue in chain)]
        if len(candidates) != 1:
            raise ValueError(f"Specify chain_id: structure has {len(candidates)} protein/polymer chains")
        chain = candidates[0]
    else:
        if not isinstance(chain_id, str) or not chain_id:
            raise ValueError("chain_id must be a nonempty string")
        if chain_id not in model:
            raise ValueError(f"Chain {chain_id!r} does not exist in structure model {model_index}")
        chain = model[chain_id]
    residues = []
    for residue in chain:
        if residue.id[0] != " ":
            # Waters and ordinary ligands are irrelevant, but a modified residue
            # with a peptide backbone must never disappear silently.
            if is_aa(residue, standard=False) or all(name in residue for name in ("N", "CA", "C")):
                raise ValueError(f"Unsupported modified/hetero protein residue {residue.resname} {residue.id}")
            continue
        if residue.is_disordered() == 2:
            raise ValueError(f"Ambiguous alternate residue at {residue.id}; supply a single conformation")
        if residue.resname not in RESIDUE_CODES:
            raise ValueError(f"Unsupported noncanonical protein residue {residue.resname} {residue.id}")
        residues.append(residue)
    observed = "".join(RESIDUE_CODES[residue.resname] for residue in residues)
    # Only explicit mapped-assay registration calls with None, then verifies
    # the observed sequence against its narrow, assay-specific mapping recipe.
    if reference_sequence is None:
        reference_sequence = observed
    if observed != reference_sequence:
        raise ValueError(f"Selected chain must exactly match the full reference sequence: structure length {len(observed)}, reference length {len(reference_sequence)}; partial coverage, mutations, or extra residues are unsupported")
    coords = torch.full((len(residues), 4, 3), float("nan"), dtype=torch.float32)
    mapping = []
    for index, residue in enumerate(residues):
        present = []
        for atom_index, atom_name in enumerate(ATOM_ORDER):
            if atom_name not in residue:
                if atom_name != "O":
                    raise ValueError(f"Missing backbone atom {atom_name} at reference position {index + 1}")
                continue
            atom = residue[atom_name]
            if atom.is_disordered() or atom.get_altloc().strip():
                raise ValueError(f"Ambiguous alternate backbone atom at reference position {index + 1}; supply a single conformation")
            xyz = torch.as_tensor(atom.coord.copy(), dtype=torch.float32)
            if xyz.shape != (3,) or not torch.isfinite(xyz).all():
                raise ValueError(f"Non-finite backbone coordinates at reference position {index + 1}")
            coords[index, atom_index] = xyz
            present.append(atom_name)
        mapping.append({"reference_position": index + 1, "chain_id": chain.id,
                        "residue_number": int(residue.id[1]), "insertion_code": residue.id[2].strip(),
                        "residue_name": residue.resname, "amino_acid": reference_sequence[index],
                        "backbone_atoms": present})
    return coords, mapping, chain.id, Bio.__version__


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_structure(index_path: str | Path, assay_id: str, structure_file: str | Path,
                      chain_id: str | None = None, model_index: int = 1,
                      output_dir: str | Path = PROJECT_ROOT / "data/structures/proteingym/manifests",
                      source_url: str | None = None, source_archive: dict | None = None) -> Path:
    """Register an existing structure; registration allows a missing O atom.

    The selected chain must match the entire assay reference in residue order.
    PDB author numbering need not start at one or be contiguous: it is preserved
    explicitly, never interpreted as the assay coordinate system.
    """
    safe_component(assay_id, "assay ID")
    index_path, structure_file = Path(index_path).resolve(), Path(structure_file).resolve()
    from .structure_mappings import MAPPED_ASSAYS, prepare_mapped_structure
    if assay_id in MAPPED_ASSAYS:
        return prepare_mapped_structure(index_path, assay_id, structure_file, chain_id, model_index,
                                        output_dir, source_url, source_archive)
    record = AssayCatalog(index_path).get(assay_id)
    index_hash, digest = sha256_file(index_path), sha256_file(structure_file)
    _, mapping, selected_chain, parser_version = _parse_structure(structure_file, record.reference_sequence, chain_id, model_index)
    if sha256_file(structure_file) != digest or sha256_file(index_path) != index_hash:
        raise ValueError("Structure inputs changed while being read")
    destination = Path(output_dir).resolve() / f"{assay_id}.json"
    metadata = {
        "schema_version": 1, "assay_id": assay_id,
        "reference_sequence": record.reference_sequence,
        "structure_path": Path(os.path.relpath(structure_file, destination.parent)).as_posix(),
        "structure_sha256": digest, "chain_id": selected_chain, "model_index": model_index,
        "residue_mapping": mapping, "source_url": source_url,
        "source_index_sha256": index_hash,
        "catalog_structure": {key: record.metadata.get(key) for key in ("pdb_file", "pdb_range")},
        "parser": "biopython", "parser_version": parser_version,
        "atom_order": list(ATOM_ORDER), "mapping_policy": "exact_full_chain_sequence",
        "altloc_policy": "reject_backbone_alternates",
    }
    if source_archive is not None:
        metadata["source_archive"] = source_archive
    metadata["input_signature"] = _signature(metadata)
    validate_structure_metadata(metadata, record.reference_sequence)
    if destination.exists():
        existing = load_structure(destination, record.reference_sequence, require_oxygen=False)
        if existing.input_signature != metadata["input_signature"]:
            raise ValueError(f"A different structure is already registered at {destination}; choose another manifest directory")
        return destination
    _atomic_json(destination, metadata)
    return destination


def load_structure(manifest_path: str | Path, reference_sequence: str, require_oxygen: bool = True) -> StructureInput:
    """Verify checksum, exact mapping, and required coordinates before inference."""
    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if isinstance(metadata, dict) and metadata.get("schema_version") == 3:
        from .structure_mappings import load_mapped_structure
        return load_mapped_structure(manifest_path, reference_sequence, require_oxygen).full_reference_structure()
    if isinstance(metadata, dict) and metadata.get("schema_version") == 2:
        raise ValueError("Fragment collections do not define a shared coordinate frame; use load_inverse_structure for independent fragment inference")
    return _load_structure_metadata(metadata, manifest_path, reference_sequence, require_oxygen)


def _load_structure_metadata(metadata: dict, manifest_path: Path, reference_sequence: str,
                             require_oxygen: bool = True) -> StructureInput:
    """Load one schema-1 structure, including when nested in a fragment bundle."""
    validate_structure_metadata(metadata, reference_sequence)
    stored_path = metadata.get("structure_path")
    if not isinstance(stored_path, str) or not stored_path:
        raise ValueError("Structure manifest needs structure_path")
    structure_file = (manifest_path.parent / stored_path).resolve()
    digest = sha256_file(structure_file)
    if digest != metadata["structure_sha256"]:
        raise ValueError("Structure checksum mismatch; the registered file changed")
    coords, mapping, chain_id, _ = _parse_structure(structure_file, reference_sequence, metadata["chain_id"], metadata["model_index"])
    if mapping != metadata["residue_mapping"] or sha256_file(structure_file) != digest:
        raise ValueError("Structure contents differ from the registered residue mapping")
    if require_oxygen and not torch.isfinite(coords[:, 3]).all():
        raise ValueError("This model requires an O backbone atom at every reference position")
    return StructureInput(reference_sequence, coords, torch.arange(1, len(reference_sequence) + 1, dtype=torch.int64),
                          chain_id, mapping, digest, metadata["input_signature"], metadata, metadata["model_index"])


def download_structure(index_path: str | Path, assay_id: str, url: str,
                       chain_id: str | None = None, model_index: int = 1,
                       raw_dir: str | Path = PROJECT_ROOT / "data/structures/proteingym/raw",
                       manifest_dir: str | Path = PROJECT_ROOT / "data/structures/proteingym/manifests") -> Path:
    """Download a user-selected direct structure URL and register its checksum.

    Archives are unsupported; downloads are bounded to 64 MiB. A registered
    structure is verified and reused without a network request.
    """
    safe_component(assay_id, "assay ID")
    parsed_url = urlparse(url)
    suffix = Path(parsed_url.path).suffix.lower()
    if parsed_url.scheme not in ("https", "http") or not parsed_url.netloc or suffix not in (".pdb", ".ent", ".cif", ".mmcif"):
        raise ValueError("Provide a direct HTTP(S) URL ending in .pdb, .ent, .cif, or .mmcif")
    manifest_path = Path(manifest_dir).resolve() / f"{assay_id}.json"
    record = AssayCatalog(index_path).get(assay_id)
    if manifest_path.exists():
        existing = load_structure(manifest_path, record.reference_sequence, require_oxygen=False)
        if existing.metadata.get("source_url") != url or existing.model_index != model_index or (chain_id is not None and chain_id != existing.chain_id):
            raise ValueError("The existing structure manifest uses a different source, model, or chain")
        return manifest_path
    raw_dir = Path(raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    target = raw_dir / f"{assay_id}-{url_hash}{suffix}"
    if not target.exists():
        with tempfile.NamedTemporaryFile(dir=raw_dir, suffix=suffix, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                request = Request(url, headers={"User-Agent": "nkn-protein-structure-pipeline/1"})
                with urlopen(request, timeout=60) as response:
                    total = 0
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        total += len(block)
                        if total > 64 * 1024 * 1024:
                            raise ValueError("Structure download exceeds 64 MiB")
                        handle.write(block)
                handle.close()
                # Validate before exposing a completed raw file.
                _parse_structure(temporary, record.reference_sequence, chain_id, model_index)
                os.replace(temporary, target)
            finally:
                handle.close()
                temporary.unlink(missing_ok=True)
    return prepare_structure(index_path, assay_id, target, chain_id, model_index, manifest_dir, source_url=url)

# Published by ProteinGym's own resource downloader (v1.1), not inferred from a
# mutable file name. Only the requested catalog member is written to raw_dir.
PROTEINGYM_STRUCTURE_URL = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.1/ProteinGym_AF2_structures.zip"
PROTEINGYM_STRUCTURE_SHA256 = "c78f5ff60cf59104fe19b8318c5647587aad033ee832e051d0efec8e137c423a"


def download_proteingym_structure(index_path: str | Path, assay_id: str,
                                 chain_id: str | None = None, model_index: int = 1,
                                 raw_dir: str | Path = PROJECT_ROOT / "data/structures/proteingym/raw",
                                 manifest_dir: str | Path = PROJECT_ROOT / "data/structures/proteingym/manifests") -> Path:
    """Extract the assay's named PDB from the checksum-pinned ProteinGym archive.

    The archive is cached in raw_dir's sibling downloads directory. Member names
    are matched uniquely by basename and never used as destination paths.
    """
    from .downloads import download_verified
    from .structure_mappings import MAPPED_ASSAYS, mapping_recipe
    safe_component(assay_id, "assay ID")
    record = AssayCatalog(index_path).get(assay_id)
    filename = record.metadata.get("pdb_file")
    if not filename or Path(filename).name != filename or Path(filename).suffix.lower() not in (".pdb", ".ent", ".cif", ".mmcif"):
        raise ValueError("Assay catalog needs a single pdb_file basename for ProteinGym acquisition")
    safe_component(filename, "structure filename")
    if assay_id == "BRCA2_HUMAN_Erwood_2022_HEK293T" and filename == "BRCA2_HUMAN.pdb":
        from .structure_fragments import download_proteingym_fragments
        return download_proteingym_fragments(index_path, assay_id, chain_id, model_index, raw_dir, manifest_dir)
    manifest_path = Path(manifest_dir).resolve() / f"{assay_id}.json"
    if manifest_path.exists():
        from .structure_fragments import load_inverse_structure
        existing = load_inverse_structure(manifest_path, record.reference_sequence, require_oxygen=False)
        source_archive = existing.metadata.get("source_archive", {})
        if source_archive.get("sha256") != PROTEINGYM_STRUCTURE_SHA256 or existing.model_index != model_index or (chain_id is not None and existing.chain_id != chain_id):
            raise ValueError("The existing structure manifest uses a different source, model, or chain")
        return manifest_path
    raw_dir = Path(raw_dir).resolve()
    archive_path = raw_dir.parent / "downloads" / "ProteinGym_v1.1_AF2_structures.zip"
    download_verified(PROTEINGYM_STRUCTURE_URL, archive_path, PROTEINGYM_STRUCTURE_SHA256)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / filename
    with zipfile.ZipFile(archive_path) as archive:
        members = [member for member in archive.infolist() if not member.is_dir() and member.filename.replace("\\", "/").rsplit("/", 1)[-1] == filename]
        if len(members) != 1:
            raise ValueError(f"Expected one ProteinGym archive member named {filename!r}; found {len(members)}")
        member = members[0]
        if not 0 < member.file_size <= 64 * 1024 * 1024:
            raise ValueError("ProteinGym structure member is empty or exceeds 64 MiB")
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
                if assay_id in MAPPED_ASSAYS:
                    _, rows, _, _ = _parse_structure(temporary, None, chain_id, model_index)
                    mapping_recipe(assay_id, record.reference_sequence, "".join(row["amino_acid"] for row in rows))
                else:
                    _parse_structure(temporary, record.reference_sequence, chain_id, model_index)
                if target.exists():
                    if sha256_file(target) != sha256_file(temporary):
                        raise ValueError(f"Existing raw structure differs from the pinned ProteinGym archive: {target}")
                else:
                    os.replace(temporary, target)
            finally:
                handle.close()
                temporary.unlink(missing_ok=True)
        source_member = member.filename
    return prepare_structure(
        index_path, assay_id, target, chain_id, model_index, manifest_dir,
        source_url=f"{PROTEINGYM_STRUCTURE_URL}#{source_member}",
        source_archive={"url": PROTEINGYM_STRUCTURE_URL, "sha256": PROTEINGYM_STRUCTURE_SHA256, "member": source_member},
    )
