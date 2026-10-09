"""Pinned ProSST structure quantization and reusable, verified residue-token caches.

A validated reference chain is written as one canonical PDB model. The official
radius graph, local subgraphs, GVP encoder and 2048-way codebook are used without
changing their numerical definitions. Quantization is fixed to CPU float32 and
one compute thread; language-model inference may independently use CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import random
import sys
import types
import tempfile

import torch

from .inputs import sha256_file, stable_json
from .structure_inputs import ATOM_ORDER, RESIDUE_CODES, StructureInput, validate_structure_metadata

LOCK = json.loads(Path(__file__).with_name("prosst_quantizer_artifacts.json").read_text(encoding="utf-8"))
DEPENDENCIES = ("torch", "numpy", "scipy", "biopython", "torch-geometric", "torch-scatter", "scikit-learn", "joblib")
_GENERATED_MODULES: dict[str, object] = {}
IDENTITY_KEYS = ("schema_version", "reference_sequence", "structure_input_signature", "canonical_pdb_sha256", "quantizer", "recipe", "runtime")


@dataclass(frozen=True)
class StructureTokens:
    tokens: torch.Tensor
    metadata: dict
    input_signature: str
    path: Path

    @property
    def cache_dir(self) -> Path:
        return self.path


def _digest(value: dict) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _quantizer_identity() -> dict:
    return {"upstream_repository": LOCK["upstream_repository"], "revision": LOCK["revision"],
            "artifact_lock_sha256": _digest(LOCK), "structure_vocab_size": 2048,
            "weights_sha256": LOCK["artifacts"][LOCK["weights_filename"]]["sha256"],
            "codebook_sha256": LOCK["artifacts"][LOCK["codebook_filename"]]["sha256"]}


def _recipe(batch_size: int) -> dict:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("ProSST quantizer_batch_size must be a positive integer")
    return {"implementation": "native_gvp_radius_graph_v1", "max_distance_angstrom": 10,
            "subgraph_policy": "native_nearest_50_within_10_then_first_40",
            "pooling": "scatter_mean_then_l2_normalize", "batch_size": batch_size,
            "device": "cpu", "dtype": "float32", "num_threads": 1,
            "canonical_structure": "single_model_chain_A_reference_order_pdb_3_decimal_places",
            "node_scalar_features": "zeros", "seed": 0}


def _runtime_versions() -> dict:
    try:
        result = {name: version(name) for name in DEPENDENCIES}
    except PackageNotFoundError as error:
        raise ImportError("Install ProSST dependencies from env.yaml or requirements.txt before quantizing structures") from error
    # The published KMeans pickle was trained/serialized with this exact release.
    if result["scikit-learn"] != "1.2.2":
        raise RuntimeError("The official ProSST codebook requires scikit-learn==1.2.2; update from requirements.txt")
    result.update(python=platform.python_version(), platform=platform.system(), machine=platform.machine())
    return result


def canonical_pdb_bytes(structure: StructureInput) -> bytes:
    """Export only the validated chain/model in reference order, including O."""
    validate_structure_metadata(structure.metadata, structure.reference_sequence)
    if structure.input_signature != structure.metadata["input_signature"] or structure.residue_mapping != structure.metadata["residue_mapping"]:
        raise ValueError("ProSST structure identity disagrees with the validated manifest")
    length = len(structure.reference_sequence)
    if length > 9999:
        raise ValueError("Canonical ProSST PDB supports at most 9999 reference residues")
    coords = structure.coords.detach().cpu()
    if coords.shape != (length, 4, 3) or not torch.isfinite(coords).all():
        raise ValueError("ProSST requires finite N, CA, C, O coordinates at every reference position")
    if not torch.equal(structure.positions.cpu(), torch.arange(1, length + 1, dtype=torch.int64)):
        raise ValueError("ProSST structure positions must be exactly 1..L in reference order")
    three = {aa: name for name, aa in RESIDUE_CODES.items()}
    lines = ["MODEL        1"]
    serial = 0
    for index, aa in enumerate(structure.reference_sequence, 1):
        for atom_index, atom in enumerate(ATOM_ORDER):
            serial += 1
            xyz = [f"{float(value):8.3f}" for value in coords[index - 1, atom_index]]
            if any(len(value) != 8 for value in xyz):
                raise ValueError("Structure coordinates exceed the canonical PDB coordinate field width")
            lines.append(f"ATOM  {serial:5d} {atom:^4s} {three[aa]} A{index:4d}    {''.join(xyz)}{1.0:6.2f}{0.0:6.2f}          {atom[0]:>2s}  ")
    lines.extend(["TER", "ENDMDL", "END"])
    return ("\n".join(lines) + "\n").encode("ascii")


def _input_metadata(config, structure: StructureInput, canonical: bytes) -> dict:
    metadata = {"schema_version": 1, "reference_sequence": structure.reference_sequence,
                "structure_input_signature": structure.input_signature,
                "canonical_pdb_sha256": hashlib.sha256(canonical).hexdigest(),
                "quantizer": _quantizer_identity(), "recipe": _recipe(config.quantizer_batch_size),
                "runtime": _runtime_versions()}
    metadata["input_signature"] = _digest({key: metadata[key] for key in IDENTITY_KEYS})
    return metadata


def token_ids_sha256(tokens: torch.Tensor) -> str:
    """Hash canonical little-endian int64 token IDs independently of serialization."""
    return hashlib.sha256(tokens.detach().cpu().numpy().astype("<i8", copy=False).tobytes()).hexdigest()


def _is_sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_structure_token_metadata(metadata: dict, structure_metadata: dict | None = None) -> None:
    """Validate copied provenance without importing native code or needing files."""
    required = set(IDENTITY_KEYS) | {"input_signature", "tokens_sha256", "token_ids_sha256", "num_residues"}
    if not isinstance(metadata, dict) or set(metadata) != required or metadata.get("schema_version") != 1:
        raise ValueError("Invalid ProSST structure-token metadata schema")
    sequence = metadata["reference_sequence"]
    if not isinstance(sequence, str) or not sequence or any(aa not in RESIDUE_CODES.values() for aa in sequence):
        raise ValueError("ProSST token reference must contain canonical amino acids")
    if type(metadata["num_residues"]) is not int or metadata["num_residues"] != len(sequence):
        raise ValueError("ProSST token count does not match reference sequence")
    if metadata["quantizer"] != _quantizer_identity():
        raise ValueError("ProSST token quantizer identity does not match pinned artifacts")
    recipe = metadata["recipe"]
    if not isinstance(recipe, dict) or recipe != _recipe(recipe.get("batch_size")):
        raise ValueError("Unsupported ProSST structure-token extraction recipe")
    runtime = metadata["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != set(DEPENDENCIES) | {"python", "platform", "machine"} or not all(isinstance(v, str) and v for v in runtime.values()):
        raise ValueError("Incomplete ProSST quantizer runtime provenance")
    if runtime["scikit-learn"] != "1.2.2":
        raise ValueError("Unsupported ProSST codebook scikit-learn runtime")
    for key in ("structure_input_signature", "canonical_pdb_sha256", "input_signature", "tokens_sha256", "token_ids_sha256"):
        if not _is_sha256(metadata[key]):
            raise ValueError(f"Invalid ProSST token checksum: {key}")
    if metadata["input_signature"] != _digest({key: metadata[key] for key in IDENTITY_KEYS}):
        raise ValueError("ProSST structure-token input signature does not match metadata")
    if structure_metadata is not None:
        validate_structure_metadata(structure_metadata, sequence)
        if metadata["structure_input_signature"] != structure_metadata["input_signature"]:
            raise ValueError("ProSST structure tokens belong to a different structure mapping")


def _validate_tokens(tokens, length: int) -> None:
    if not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.int64 or tokens.shape != (length,):
        raise ValueError("ProSST structure tokens must be an int64 tensor with one ID per reference residue")
    if not bool(((tokens >= 0) & (tokens < 2048)).all()):
        raise ValueError("ProSST structure tokens must be raw cluster IDs in 0..2047")


def validate_structure_token_cache(path: str | Path, expected: dict | None = None) -> StructureTokens:
    path = Path(path)
    names = {item.name for item in path.iterdir()} if path.is_dir() else set()
    if names != {"canonical.pdb", "tokens.pt", "metadata.json", "completion.json"}:
        raise ValueError(f"Incomplete or unexpected ProSST structure-token cache: {path}")
    completion = json.loads((path / "completion.json").read_text(encoding="utf-8"))
    required = {"canonical.pdb", "tokens.pt", "metadata.json"}
    if not isinstance(completion, dict) or set(completion) != {"schema_version", "files_sha256"} or completion["schema_version"] != 1 or not isinstance(completion["files_sha256"], dict) or set(completion["files_sha256"]) != required:
        raise ValueError("Invalid ProSST structure-token completion manifest")
    for name, digest in completion["files_sha256"].items():
        if not _is_sha256(digest) or sha256_file(path / name) != digest:
            raise ValueError(f"ProSST structure-token cache checksum mismatch: {name}")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    validate_structure_token_metadata(metadata)
    if expected is not None and any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing ProSST structure-token cache has a different input identity")
    if metadata["canonical_pdb_sha256"] != completion["files_sha256"]["canonical.pdb"] or metadata["tokens_sha256"] != completion["files_sha256"]["tokens.pt"]:
        raise ValueError("ProSST structure-token metadata disagrees with cached file checksums")
    tokens = torch.load(path / "tokens.pt", map_location="cpu", weights_only=True)
    _validate_tokens(tokens, metadata["num_residues"])
    if token_ids_sha256(tokens) != metadata["token_ids_sha256"]:
        raise ValueError("ProSST structure-token IDs disagree with their canonical checksum")
    return StructureTokens(tokens, metadata, metadata["input_signature"], path)


def import_pinned_quantizer(source: Path):
    """Import the verified namespace package without replacing other ProSST code."""
    source = source.resolve()
    for name, module in list(sys.modules.items()):
        if name == "prosst" or name.startswith("prosst."):
            if _GENERATED_MODULES.get(name) is module:
                continue
            filename = getattr(module, "__file__", None)
            locations = [filename] if filename else list(getattr(module, "__path__", []))
            if not locations or any(not Path(item).resolve().is_relative_to(source) for item in locations):
                raise RuntimeError("Another prosst package is imported; use a fresh CLI process or restart the notebook kernel")
    if "prosst" not in sys.modules:
        # Upstream has no top-level __init__.py. Bind that namespace explicitly,
        # so an unrelated installed regular package cannot win import resolution.
        namespace = types.ModuleType("prosst")
        namespace.__path__ = [str(source / "prosst")]
        namespace.__package__ = "prosst"
        namespace.__spec__ = importlib.machinery.ModuleSpec("prosst", loader=None, is_package=True)
        namespace.__spec__.submodule_search_locations = namespace.__path__
        sys.modules["prosst"] = namespace
    original_path, original_bytecode = sys.path[:], sys.dont_write_bytecode
    sys.path.insert(0, str(source))
    sys.dont_write_bytecode = True
    try:
        graph = importlib.import_module("prosst.structure.build_graph")
        subgraph = importlib.import_module("prosst.structure.build_subgraph")
        encoder = importlib.import_module("prosst.structure.encoder")
    except ImportError as error:
        raise ImportError("Install ProSST quantizer dependencies from env.yaml or requirements.txt") from error
    finally:
        # Upstream build_graph appends the current directory during import.
        sys.path[:] = original_path
        sys.dont_write_bytecode = original_bytecode
    return graph.generate_graph, subgraph.generate_pos_subgraph, encoder.AutoGraphEncoder


def _native_quantize(checkpoint: Path, pdb: Path, sequence: str, batch_size: int) -> torch.Tensor:
    import joblib
    import numpy as np
    from threadpoolctl import threadpool_limits
    from torch_geometric.data import Batch
    from torch_scatter import scatter_mean
    import torch.nn.functional as F

    generate_graph, generate_subgraph, encoder_type = import_pinned_quantizer(checkpoint / "source")
    previous_threads, previous_dtype = torch.get_num_threads(), torch.get_default_dtype()
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float32)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]), torch.inference_mode(), threadpool_limits(limits=1):
            torch.random.default_generator.manual_seed(0)
            model = encoder_type(node_in_dim=(20, 3), node_h_dim=(256, 32), edge_in_dim=(32, 1), edge_h_dim=(64, 2), num_layers=6)
            # PyG compiles a propagation helper in a temporary directory. Only
            # remember the actual module object produced by this verified class;
            # later calls may reuse it without confusing it with another package.
            generated_name = "prosst.structure.encoder.layer_GVPConv_propagate"
            if generated_name in sys.modules:
                _GENERATED_MODULES[generated_name] = sys.modules[generated_name]
            model.load_state_dict(torch.load(checkpoint / LOCK["weights_filename"], map_location="cpu", weights_only=True), strict=True)
            model.requires_grad_(False).eval().float().cpu()
            # Only the separately checksum-verified, official artifact is unpickled.
            codebook = joblib.load(checkpoint / LOCK["codebook_filename"])
            if tuple(codebook.cluster_centers_.shape) != (2048, 256):
                raise ValueError("Unexpected ProSST KMeans codebook shape")
            graph = generate_graph(str(pdb), max_distance=10)
            if graph.aa_seq != sequence or graph.node_s.shape[0] != len(sequence):
                raise ValueError("Native ProSST graph residues disagree with the full reference sequence")
            subgraphs = generate_subgraph(graph, subgraph_depth=None, max_distance=10, pure_subgraph=True)
            if sorted(subgraphs) != list(range(len(sequence))):
                raise ValueError("Native ProSST subgraphs do not cover reference positions in order")
            labels = []
            for start in range(0, len(sequence), batch_size):
                batch = Batch.from_data_list([subgraphs[i] for i in range(start, min(start + batch_size, len(sequence)))])
                batch.node_s = torch.zeros_like(batch.node_s)
                embeddings = model.get_embedding((batch.node_s, batch.node_v), batch.edge_index, (batch.edge_s, batch.edge_v))
                normalized = F.normalize(scatter_mean(embeddings, batch.batch, dim=0).cpu(), p=2, dim=1)
                if not torch.isfinite(normalized).all():
                    raise ValueError("ProSST quantizer returned non-finite graph embeddings")
                # Match upstream's Tensor argument: scikit-learn 1.2.2 converts
                # it to the float64 dtype used by the serialized cluster centers.
                labels.extend(codebook.predict(normalized).tolist())
            tokens = torch.tensor(labels, dtype=torch.int64)
    finally:
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
        torch.set_num_threads(previous_threads)
        torch.set_default_dtype(previous_dtype)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    _validate_tokens(tokens, len(sequence))
    return tokens


def prepare_structure_tokens(config, structure: StructureInput) -> StructureTokens:
    """Reuse verified tokens or quantize once, publishing the cache atomically."""
    from .prosst_checkpoints import local_checkpoint

    canonical = canonical_pdb_bytes(structure)
    expected = _input_metadata(config, structure, canonical)
    recipe_id = "prosst2048-" + _digest({"quantizer": expected["quantizer"], "recipe": expected["recipe"], "runtime": expected["runtime"]})[:16]
    parent = Path(config.structure_token_dir) / recipe_id
    destination = parent / expected["input_signature"]
    if destination.exists():
        return validate_structure_token_cache(destination, expected)
    checkpoint = local_checkpoint(config)
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".building-", dir=parent) as temporary:
        staging = Path(temporary) / "cache"
        staging.mkdir()
        (staging / "canonical.pdb").write_bytes(canonical)
        tokens = _native_quantize(checkpoint, staging / "canonical.pdb", structure.reference_sequence, config.quantizer_batch_size)
        _validate_tokens(tokens, len(structure.reference_sequence))
        torch.save(tokens, staging / "tokens.pt")
        metadata = dict(expected, tokens_sha256=sha256_file(staging / "tokens.pt"),
                        token_ids_sha256=token_ids_sha256(tokens), num_residues=len(structure.reference_sequence))
        validate_structure_token_metadata(metadata, structure.metadata)
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        completion = {"schema_version": 1, "files_sha256": {name: sha256_file(staging / name) for name in ("canonical.pdb", "tokens.pt", "metadata.json")}}
        (staging / "completion.json").write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        validate_structure_token_cache(staging, expected)
        try:
            # Renaming a directory never replaces a nonempty completed cache.
            os.rename(staging, destination)
        except FileExistsError:
            return validate_structure_token_cache(destination, expected)
    return validate_structure_token_cache(destination, expected)
