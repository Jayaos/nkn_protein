"""Atomic inverse-folding caches sharing the assay's immutable variant table."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import tempfile

import torch
from torch import Tensor

from data_loading.features import ALPHABET, parse_substitutions, read_csv
from .inputs import AssayInput, safe_component, sha256_file, stable_json
from .storage import (
    _assay_lock, _assay_metadata, _atomic_json, _ensure_assay,
    _inside, _read_json, _write_json,
)
from .structure_inputs import validate_structure_metadata

FAMILIES = ("proteinmpnn", "esm_if1")


def _signature(value: dict) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _resume_configs_match(saved: dict, requested: dict, model_family: str) -> bool:
    """Compare inference identity after validating the original saved artifact.

    ProteinMPNN's max_length only controls admission of a full reference; it
    does not change model inputs or inference for an admitted structure. Keep
    the on-disk config and all of its integrity signatures unchanged. Other
    families, and all other top-level or nested fields, retain exact equality.
    """
    if model_family == "proteinmpnn":
        saved = {key: value for key, value in saved.items() if key != "max_length"}
        requested = {key: value for key, value in requested.items() if key != "max_length"}
    return saved == requested


def _leaf(root: Path, model_family: str, cache_id: str) -> Path:
    if model_family not in FAMILIES:
        raise ValueError(f"Unsupported inverse-folding model family: {model_family!r}")
    safe_component(cache_id, "cache ID")
    return _inside(root, f"inverse_folding/{model_family}/{cache_id}")


def make_leaf_config(cache_id: str, model_family: str, model_metadata: dict, structure_metadata: dict) -> dict:
    config = {
        **model_metadata,
        "schema_version": 2,
        "layout_version": 1,
        "feature_type": "inverse_folding",
        "model_family": model_family,
        "checkpoint": cache_id,
        "synthetic": False,
        "output_dtype": "float32",
        "alphabet": ALPHABET,
        "temperature": 1.0,
        "normalization": model_metadata.get("normalization", "canonical_amino_acids_only"),
        "structure": structure_metadata,
    }
    stable_json(config)
    _validate_config(config, model_family, cache_id)
    return config


def _validate_config(config: dict, model_family: str, cache_id: str) -> None:
    if model_family not in FAMILIES:
        raise ValueError("Unsupported inverse-folding model family")
    safe_component(cache_id, "cache ID")
    expected = {
        "schema_version": 2, "layout_version": 1, "feature_type": "inverse_folding",
        "model_family": model_family, "checkpoint": cache_id, "synthetic": False,
        "output_dtype": "float32", "alphabet": ALPHABET, "temperature": 1.0,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Feature configuration does not match its inverse-folding directory or format")
    expected_mode = "backbone_only" if model_family == "proteinmpnn" else "reference_prefix"
    if model_family == "proteinmpnn" and config.get("conditioning_mode") == "sequence_conditional":
        recipe = {
            "normalization": "native_21_amino_acid_tokens",
            "noncanonical_tokens": "excluded_without_renormalization",
            "order_aggregation": "geometric_mean_of_probabilities",
            "inference_method": "conditional_probs",
        }
        if any(config.get(key) != value for key, value in recipe.items()):
            raise ValueError("Sequence-conditional ProteinMPNN requires the declared Kermut probability recipe")
        if type(config.get("num_orders")) is not int or config["num_orders"] < 1:
            raise ValueError("Sequence-conditional ProteinMPNN requires positive integer num_orders")
        if type(config.get("seed")) is not int or not 0 <= config["seed"] < 2**32:
            raise ValueError("Sequence-conditional ProteinMPNN requires a valid integer seed")
    else:
        if config.get("conditioning_mode") != expected_mode:
            raise ValueError(f"Unsupported conditioning mode for {model_family}; expected {expected_mode}")
        if config.get("normalization") != "canonical_amino_acids_only":
            raise ValueError("This inverse-folding conditioning mode requires canonical normalization")
    structure = config.get("structure")
    if not isinstance(structure, dict):
        raise ValueError("Missing structure metadata")
    validate_structure_metadata(structure, structure.get("reference_sequence"))
    if model_family == "proteinmpnn" and any("O" not in row["backbone_atoms"] for row in structure["residue_mapping"]):
        raise ValueError("ProteinMPNN structure metadata requires oxygen at every residue")


def validate_distributions(
    probabilities: object, log_probs: object, length: int,
    *, normalization: str = "canonical_amino_acids_only",
) -> None:
    """Validate normalized distributions or explicitly declared Kermut weights unchanged."""
    for name, value in (("probabilities", probabilities), ("log_probs", log_probs)):
        if (
            not isinstance(value, Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or value.shape != (length, len(ALPHABET))
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"{name} must be a finite CPU float32 tensor of shape [reference residues, 20]")
    if (probabilities <= 0).any():
        raise ValueError("Inverse-folding probabilities must be positive for finite log probabilities")
    mass = probabilities.sum(-1)
    if normalization == "canonical_amino_acids_only":
        if not torch.allclose(mass, torch.ones(length), atol=1e-5, rtol=1e-5):
            raise ValueError("Inverse-folding probability rows must sum to one")
    elif normalization == "native_21_amino_acid_tokens":
        # Geometric averaging and removing X can only reduce total mass.
        if (mass > 1 + 2e-5).any():
            raise ValueError("Kermut inverse-folding probability rows must have mass at most one")
    else:
        raise ValueError("Unsupported inverse-folding normalization")
    if not torch.allclose(log_probs, probabilities.log(), atol=1e-5, rtol=1e-5):
        raise ValueError("Inverse-folding log probabilities are inconsistent")


def _validate_payload(store: object, metadata: dict, config: dict) -> None:
    if not isinstance(store, dict):
        raise ValueError("features.pt must contain a tensor dictionary")
    expected = {
        "schema_version": 2,
        "synthetic": False,
        "assay_id": metadata["assay_id"],
        "reference_sequence": metadata["reference_sequence"],
        "input_signature": metadata["input_signature"],
        "structure_input_signature": config["structure"]["input_signature"],
        "config_signature": _signature(config),
        "alphabet": ALPHABET,
    }
    if any(store.get(key) != value for key, value in expected.items()):
        raise ValueError("Inverse-folding payload metadata mismatch")
    length = len(metadata["reference_sequence"])
    positions = store.get("positions")
    if (
        not isinstance(positions, Tensor)
        or positions.dtype != torch.long
        or not torch.equal(positions, torch.arange(1, length + 1))
    ):
        raise ValueError("Expected complete one-based reference positions")
    validate_distributions(
        store.get("probabilities"), store.get("log_probs"), length,
        normalization=config["normalization"],
    )


def _register_default(root: Path, model_family: str, cache_id: str) -> None:
    path = _inside(root, "assay.json")
    metadata = _read_json(path)
    features = metadata.setdefault("features", {})
    if not isinstance(features, dict):
        raise ValueError("Assay features must be a mapping")
    if "inverse_folding" not in features:
        features["inverse_folding"] = {
            "model_family": model_family,
            "checkpoint": cache_id,
            "path": f"inverse_folding/{model_family}/{cache_id}",
        }
        _atomic_json(path, metadata)


def check_existing(
    root: str | Path,
    model_family: str,
    cache_id: str,
    inputs: AssayInput,
    config: dict,
    resume: bool = False,
) -> bool:
    root = Path(root).resolve()
    _validate_config(config, model_family, cache_id)
    with _assay_lock(root):
        _ensure_assay(root, inputs)
        leaf = _leaf(root, model_family, cache_id)
        if not leaf.exists():
            return False
        if not resume:
            raise FileExistsError(f"Feature cache already exists: {leaf}; use --resume to validate and reuse it")
        validate_inverse_cache(root, model_family, cache_id)
        if not _resume_configs_match(_read_json(_inside(leaf, "config.json")), config, model_family):
            raise ValueError("Existing checkpoint, conditioning, or structure differs; use a new cache ID")
        _register_default(root, model_family, cache_id)
        return True


def write_inverse_cache(
    root: str | Path,
    model_family: str,
    cache_id: str,
    inputs: AssayInput,
    config: dict,
    probabilities: Tensor,
    log_probs: Tensor,
    provenance: dict,
) -> Path:
    root = Path(root).resolve()
    _validate_config(config, model_family, cache_id)
    validate_structure_metadata(config["structure"], inputs.reference_sequence)
    validate_distributions(probabilities, log_probs, inputs.length, normalization=config["normalization"])
    payload = {
        "schema_version": 2,
        "synthetic": False,
        "assay_id": inputs.assay_id,
        "reference_sequence": inputs.reference_sequence,
        "input_signature": inputs.input_signature,
        "structure_input_signature": config["structure"]["input_signature"],
        "config_signature": _signature(config),
        "alphabet": ALPHABET,
        "positions": torch.arange(1, inputs.length + 1),
        "probabilities": probabilities.detach().contiguous(),
        "log_probs": log_probs.detach().contiguous(),
    }
    _validate_payload(payload, _assay_metadata(inputs), config)
    with _assay_lock(root):
        _ensure_assay(root, inputs)
        leaf = _leaf(root, model_family, cache_id)
        if leaf.exists():
            raise FileExistsError(f"Refusing to overwrite existing feature cache: {leaf}")
        leaf.parent.mkdir(parents=True, exist_ok=True)
        pending = Path(tempfile.mkdtemp(prefix=f".{cache_id}.pending-", dir=leaf.parent))
        torch.save(payload, pending / "features.pt")
        _write_json(pending / "config.json", config)
        _write_json(pending / "provenance.json", provenance)
        files = {name: sha256_file(pending / name) for name in ("features.pt", "config.json", "provenance.json")}
        files["variants.csv"] = inputs.variants_sha256
        _write_json(pending / "completion.json", {
            "schema_version": 1,
            "model_family": model_family,
            "cache_id": cache_id,
            "input_signature": inputs.input_signature,
            "structure_input_signature": config["structure"]["input_signature"],
            "config_signature": _signature(config),
            "files_sha256": files,
        })
        pending.rename(leaf)
        _register_default(root, model_family, cache_id)
    return leaf


def _validate_assay(root: Path, metadata: dict) -> list[dict]:
    if metadata.get("schema_version") != 2 or metadata.get("layout_version") != 1 or metadata.get("synthetic") is not False:
        raise ValueError("Expected a nonsynthetic schema-v2, layout-v1 assay cache")
    if metadata.get("variants_file") != "variants.csv":
        raise ValueError("Expected variants_file='variants.csv'")
    safe_component(metadata.get("assay_id"), "assay ID")
    reference = metadata.get("reference_sequence")
    if not isinstance(reference, str):
        raise ValueError("Missing reference sequence")
    parse_substitutions("WT", reference, reference)
    if metadata.get("reference_length") != len(reference):
        raise ValueError("Reference length mismatch")
    variants = _inside(root, "variants.csv")
    checksum = sha256_file(variants)
    if checksum != metadata.get("variants_sha256"):
        raise ValueError("Assay variant checksum mismatch")
    rows = read_csv(variants)
    required = {"variant_id", "source_row_index", "mutant", "mutated_sequence", "DMS_score"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Missing required variant columns or rows")
    ids = [row["variant_id"] for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids) or metadata.get("num_variants") != len(rows):
        raise ValueError("Variant IDs must be unique, nonempty, and match the declared count")
    for row in rows:
        parse_substitutions(row["mutant"], row["mutated_sequence"], reference)
        if not math.isfinite(float(row["DMS_score"])):
            raise ValueError("Variant targets must be finite")
    coverage = metadata.get("coverage")
    if not isinstance(coverage, dict) or coverage.get("num_variants") != len(rows):
        raise ValueError("Variant coverage does not match the saved rows")
    source_indices = [int(row["source_row_index"]) for row in rows]
    if source_indices != sorted(set(source_indices)) or min(source_indices) < 0 or max(source_indices) >= coverage.get("source_row_count", 0):
        raise ValueError("Source row indices must be unique, ordered, and within the source assay")
    identity = {
        "assay_id": metadata["assay_id"],
        "reference_sequence": reference,
        "source_assay_sha256": metadata.get("source_assay_sha256"),
        "source_index_sha256": metadata.get("source_index_sha256"),
        "variants_sha256": checksum,
        "coverage": coverage,
    }
    if _signature(identity) != metadata.get("input_signature"):
        raise ValueError("Assay input signature mismatch")
    return rows


def _select_cache(root: Path, metadata: dict, model_family: str | None, cache_id: str | None) -> tuple[str, str]:
    if model_family is not None and model_family not in FAMILIES:
        raise ValueError("Unsupported inverse-folding model family")
    if cache_id is not None:
        safe_component(cache_id, "cache ID")
    if model_family is not None and cache_id is not None:
        return model_family, cache_id
    descriptor = metadata.get("features", {}).get("inverse_folding", {})
    if not isinstance(descriptor, dict):
        raise ValueError("Invalid inverse-folding default descriptor")
    family, checkpoint = descriptor.get("model_family"), descriptor.get("checkpoint")
    if (
        family in FAMILIES and isinstance(checkpoint, str)
        and (model_family is None or model_family == family)
        and (cache_id is None or cache_id == checkpoint)
    ):
        safe_component(checkpoint, "default inverse-folding cache ID")
        if descriptor.get("path") != f"inverse_folding/{family}/{checkpoint}":
            raise ValueError("Inconsistent inverse-folding default descriptor")
        return family, checkpoint
    # A selected family with a single recipe is unambiguous even if another
    # family was registered first. Hidden interrupted staging is never selected.
    candidates = []
    for family in (FAMILIES if model_family is None else (model_family,)):
        base = _inside(root, f"inverse_folding/{family}")
        if base.is_dir():
            for path in base.iterdir():
                if path.is_dir() and not path.name.startswith(".") and (cache_id is None or cache_id == path.name):
                    safe_component(path.name, "cache ID")
                    candidates.append((family, path.name))
    if len(candidates) != 1:
        raise ValueError("Select an inverse-folding model family and cache ID; no unique cache was found")
    return candidates[0]


def validate_inverse_cache(root: str | Path, model_family: str | None = None, cache_id: str | None = None) -> dict:
    """Validate one inverse-folding leaf without requiring any other modality."""
    root = Path(root).resolve()
    metadata = _read_json(_inside(root, "assay.json"))
    rows = _validate_assay(root, metadata)
    model_family, cache_id = _select_cache(root, metadata, model_family, cache_id)
    leaf = _leaf(root, model_family, cache_id)
    if not _inside(leaf, "completion.json").is_file():
        raise ValueError(f"Feature cache is incomplete: missing completion.json in {leaf}")
    completion = _read_json(_inside(leaf, "completion.json"))
    expected = {
        "schema_version": 1, "model_family": model_family, "cache_id": cache_id,
        "input_signature": metadata["input_signature"],
    }
    if any(completion.get(key) != value for key, value in expected.items()):
        raise ValueError("Feature completion metadata mismatch")
    files = completion.get("files_sha256")
    if not isinstance(files, dict) or set(files) != {"features.pt", "config.json", "provenance.json", "variants.csv"}:
        raise ValueError("Incomplete feature checksum manifest")
    for name, checksum in files.items():
        path = _inside(root if name == "variants.csv" else leaf, name)
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Feature checksum mismatch or missing file: {name}")
    if files["variants.csv"] != metadata["variants_sha256"]:
        raise ValueError("Assay variant checksum mismatch")
    config = _read_json(_inside(leaf, "config.json"))
    _validate_config(config, model_family, cache_id)
    validate_structure_metadata(config["structure"], metadata["reference_sequence"])
    if completion.get("config_signature") != _signature(config) or completion.get("structure_input_signature") != config["structure"]["input_signature"]:
        raise ValueError("Feature checkpoint, configuration, or structure identity mismatch")
    provenance = _read_json(_inside(leaf, "provenance.json"))
    if provenance.get("uses_dms_labels") is not False or provenance.get("structure_input_signature") != config["structure"]["input_signature"]:
        raise ValueError("Inverse-folding provenance mismatch")
    store = torch.load(_inside(leaf, "features.pt"), map_location="cpu", weights_only=True)
    _validate_payload(store, metadata, config)
    return {
        "assay_id": metadata["assay_id"],
        "model_family": model_family,
        "cache_id": cache_id,
        "num_variants": len(rows),
        "reference_length": len(metadata["reference_sequence"]),
        "shape": [len(metadata["reference_sequence"]), len(ALPHABET)],
        "alphabet": ALPHABET,
        "conditioning_mode": config["conditioning_mode"],
        "path": str(leaf),
        "input_signature": metadata["input_signature"],
        "structure_input_signature": config["structure"]["input_signature"],
    }
