"""Atomic schema-v2 sequence caches and standalone integrity validation."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import pickle
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor

from data_loading.features import ALPHABET, parse_substitutions, pooled_sequence_recipe, read_csv
from data_loading.sequence_windows import (
    COMPACT_SITE_FIELDS, compact_sequence_recipe, tensor_isfinite,
    validate_compact_sequence_payload, validate_global_window_payload,
)

from .inputs import AssayInput, safe_component, sha256_file, stable_json


def _inside(root: Path, relative: str) -> Path:
    parts = relative.split("/")
    for part in parts:
        safe_component(part)
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Cache path escapes the assay directory: {relative}")
    return path


SEQUENCE_FAMILIES = ("esm2", "prosst", "msa_transformer")


def _family(value: str) -> str:
    if value not in SEQUENCE_FAMILIES:
        raise ValueError(f"Unsupported sequence model family: {value!r}")
    return value


def _leaf(root: Path, cache_id: str, model_family: str = "esm2") -> Path:
    _family(model_family)
    safe_component(cache_id, "cache ID")
    return _inside(root, f"sequence/{model_family}/{cache_id}")


def _signature(value: dict) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _prosst_identity(config: dict) -> dict:
    return {
        "model_family": "prosst",
        "config_signature": _signature(config),
        "structure_input_signature": config["structure"]["input_signature"],
        "structure_token_input_signature": config["structure_tokens"]["input_signature"],
    }


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _atomic_bytes(path: Path, content: bytes) -> None:
    descriptor, pending = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
    finally:
        if os.path.exists(pending):
            os.unlink(pending)


def _atomic_json(path: Path, value: dict) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))


@contextmanager
def _assay_lock(root: Path) -> Iterator[None]:
    """OS-backed nonblocking lock; a crashed process cannot leave a stale lock."""
    root.mkdir(parents=True, exist_ok=True)
    path = _inside(root, "extraction.lock")
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another process is writing this assay cache: {root}") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _assay_metadata(inputs: AssayInput) -> dict:
    return {
        "schema_version": 2,
        "layout_version": 1,
        "synthetic": False,
        "assay_id": inputs.assay_id,
        "reference_sequence": inputs.reference_sequence,
        "reference_length": inputs.length,
        "variants_file": "variants.csv",
        "variants_sha256": inputs.variants_sha256,
        "num_variants": len(inputs.rows),
        "amino_acid_alphabet": ALPHABET,
        "residue_indexing": "one_based_reference_position; subtract one for tensor indexing",
        "source_row_indexing": "zero_based_data_rows_excluding_header",
        "source_assay": inputs.source_assay_path.name,
        "source_assay_sha256": inputs.source_assay_sha256,
        "source_index_sha256": inputs.source_index_sha256,
        "coverage": inputs.coverage,
        "input_signature": inputs.input_signature,
    }


def _ensure_assay(root: Path, inputs: AssayInput) -> None:
    expected = _assay_metadata(inputs)
    metadata_path = _inside(root, "assay.json")
    variants_path = _inside(root, "variants.csv")
    if metadata_path.exists():
        metadata = _read_json(metadata_path)
        if metadata.get("synthetic") is True:
            raise ValueError("Cannot write pretrained features into a synthetic fixture; choose another output root")
        mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatches:
            raise ValueError(f"Assay cache inputs or coverage differ ({', '.join(mismatches)}); choose another output root")
        if not variants_path.is_file() or sha256_file(variants_path) != inputs.variants_sha256:
            raise ValueError("Existing variants.csv does not match the immutable assay inputs")
    else:
        # A crash between these two writes leaves a recoverable, matching CSV.
        if variants_path.exists() and sha256_file(variants_path) != inputs.variants_sha256:
            raise ValueError("Existing variants.csv does not match the requested assay inputs")
        if not variants_path.exists():
            _atomic_bytes(variants_path, inputs.variants_csv().encode("utf-8"))
        _atomic_json(metadata_path, {**expected, "features": {}})


def ensure_assay(root: str | Path, inputs: AssayInput) -> None:
    """Create shared assay files or require that their input identity is unchanged."""
    root = Path(root).resolve()
    with _assay_lock(root):
        _ensure_assay(root, inputs)


def _validate_saved_cache(root: Path, feature_type: str, model_family: str, cache_id: str) -> None:
    if feature_type == "sequence":
        validate_sequence_cache(root, cache_id, model_family=model_family)
    elif feature_type == "inverse_folding":
        from .inverse_storage import validate_inverse_cache

        validate_inverse_cache(root, model_family, cache_id)
    elif feature_type == "scores":
        from .score_storage import validate_score_cache

        validate_score_cache(root, model_family, cache_id)
    else:
        raise ValueError("Unsupported cache type")


def _cache_is_complete(root: Path, feature_type: str, model_family: str, cache_id: str) -> bool:
    """An invalid saved leaf needs regeneration; validators themselves stay strict."""
    try:
        _validate_saved_cache(root, feature_type, model_family, cache_id)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError, TypeError, EOFError, pickle.UnpicklingError):
        return False
    return True


def _check_cache_directory(root: Path, path: Path) -> None:
    # Refuse redirected directories, including Windows junctions. The resolved
    # assay root may legitimately live on another volume.
    resolved = path.resolve()
    if path.is_symlink() or resolved != path or resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"Refusing cache changes through a redirected cache directory: {path}")
    if path.exists() and not path.is_dir():
        raise ValueError(f"Expected a cache directory: {path}")


def _cache_family_directory(root: Path, feature_type: str, model_family: str) -> Path:
    if feature_type not in ("sequence", "inverse_folding", "scores"):
        raise ValueError("Unsupported cache type for cleanup")
    safe_component(model_family, "model family")
    parent = root / feature_type
    target = parent / model_family
    for path in (parent, target):
        _check_cache_directory(root, path)
    return target


def _update_cache_default(
    root: Path, feature_type: str, model_family: str, cache_id: str, *, replace_family: bool,
) -> None:
    if feature_type == "scores":
        return
    path = _inside(root, "assay.json")
    metadata = _read_json(path)
    features = metadata.setdefault("features", {})
    if not isinstance(features, dict):
        raise ValueError("Assay features must be a mapping")
    default = features.get(feature_type)
    if default is None or (replace_family and isinstance(default, dict) and default.get("model_family") == model_family):
        descriptor = {
            "model_family": model_family, "checkpoint": cache_id,
            "path": f"{feature_type}/{model_family}/{cache_id}",
        }
        if default != descriptor:
            features[feature_type] = descriptor
            _atomic_json(path, metadata)


def _cleanup_other_caches(root: Path, feature_type: str, model_family: str, cache_id: str) -> None:
    """Remove sibling recipes only, under the assay lock after successful validation."""
    safe_component(cache_id, "cache ID")
    family = _cache_family_directory(root, feature_type, model_family)
    current = family / cache_id
    _check_cache_directory(root, current)
    candidates = [path for path in family.iterdir()
                  if path != current and not path.name.startswith(".") and path.is_dir()]
    # Check every deletion target before changing defaults or removing anything.
    for path in candidates:
        _check_cache_directory(root, path)
    _update_cache_default(root, feature_type, model_family, cache_id, replace_family=True)
    for path in candidates:
        _check_cache_directory(root, path)
        shutil.rmtree(path)


@contextmanager
def _cache_write_root(
    root: Path, inputs: AssayInput, feature_type: str, model_family: str,
    cache_id: str, cleanup_and_save: bool, resume: bool,
) -> Iterator[Path]:
    """Repair invalid current leaves with resume; cleanup only prunes sibling recipes.

    The caller holds the assay lock. Failed staging preserves the current leaf
    and siblings. A complete current leaf is never overwritten, even if another
    writer completed it while this process was running inference.
    """
    _ensure_assay(root, inputs)
    safe_component(cache_id, "cache ID")
    target = _inside(root, f"{feature_type}/{model_family}/{cache_id}")
    repairing = target.exists()
    if repairing:
        if not resume or _cache_is_complete(root, feature_type, model_family, cache_id):
            raise FileExistsError(f"Refusing to overwrite existing cache: {target}; use --resume to validate and reuse it")
    if not cleanup_and_save and not repairing:
        yield root
        return
    family = _cache_family_directory(root, feature_type, model_family)
    target = family / cache_id
    _check_cache_directory(root, target)
    work = Path(tempfile.mkdtemp(prefix=f".{model_family}.replacement-", dir=root))
    _check_cache_directory(root, work)
    staging, backup = work / "new", work / "old"
    try:
        staging.mkdir()
        _ensure_assay(staging, inputs)
        yield staging
        _validate_saved_cache(staging, feature_type, model_family, cache_id)
        replacement = staging / feature_type / model_family / cache_id
        family.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.rename(backup)
        try:
            replacement.rename(target)
            _update_cache_default(root, feature_type, model_family, cache_id, replace_family=cleanup_and_save)
        except BaseException:
            # Keep the backup on disk if restoration itself fails.
            if target.exists():
                target.rename(replacement)
            if backup.exists():
                backup.rename(target)
            raise
        if backup.exists():
            _check_cache_directory(root, backup)
            shutil.rmtree(backup)
        if cleanup_and_save:
            _cleanup_other_caches(root, feature_type, model_family, cache_id)
    finally:
        if not backup.exists():
            _check_cache_directory(root, work)
            shutil.rmtree(work)


def make_leaf_config(cache_id: str, model_metadata: dict, *, model_family: str = "esm2") -> dict:
    safe_component(cache_id, "cache ID")
    _family(model_family)
    if model_metadata.get("model_family", model_family) != model_family:
        raise ValueError("Model metadata family does not match the requested sequence family")
    config = {
        **model_metadata,
        "schema_version": 2,
        "layout_version": 1,
        "feature_type": "sequence",
        "model_family": model_family,
        "checkpoint": cache_id,
        "synthetic": False,
        "output_dtype": "float32",
    }
    stable_json(config)  # Fail immediately on nonserializable or nonfinite metadata.
    _validate_config(config, cache_id, model_family)
    return config


def _validate_config(config: dict, cache_id: str, model_family: str | None = None) -> str:
    model_family = _family(config.get("model_family") if model_family is None else model_family)
    safe_component(cache_id, "cache ID")
    expected = {
        "schema_version": 2, "layout_version": 1, "feature_type": "sequence",
        "model_family": model_family, "checkpoint": cache_id, "synthetic": False,
        "output_dtype": "float32",
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Feature configuration does not match its sequence model family, cache directory, or format")
    size = config.get("hidden_size")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("Feature configuration requires a positive integer hidden_size")
    compact_sequence_recipe(config)
    pooled_sequence_recipe(config)
    if model_family == "msa_transformer":
        from .msa_config import validate_msa_metadata
        validate_msa_metadata(config)
    if model_family == "prosst":
        from .structure_inputs import validate_structure_metadata
        from .prosst_structure import validate_structure_token_metadata

        structure, tokens = config.get("structure"), config.get("structure_tokens")
        if not isinstance(structure, dict) or not isinstance(tokens, dict):
            raise ValueError("ProSST sequence features require structure and structure-token metadata")
        validate_structure_metadata(structure, structure.get("reference_sequence"))
        if any("O" not in row["backbone_atoms"] for row in structure["residue_mapping"]):
            raise ValueError("ProSST requires complete N, CA, C, O backbone metadata")
        validate_structure_token_metadata(tokens, structure)
        if config.get("structure_input_signature") != structure["input_signature"] or config.get("structure_token_input_signature") != tokens["input_signature"]:
            raise ValueError("ProSST structure or structure-token identity mismatch")
        if config.get("conditioning_mode") != "sequence_and_fixed_reference_structure":
            raise ValueError("ProSST features must declare fixed reference structure conditioning")
    return model_family


def _register_default(root: Path, cache_id: str, model_family: str = "esm2") -> None:
    path = _inside(root, "assay.json")
    metadata = _read_json(path)
    features = metadata.setdefault("features", {})
    if not isinstance(features, dict):
        raise ValueError("Assay features must be a mapping")
    # The first completed sequence cache is the default; later recipes are explicit selections.
    if "sequence" not in features:
        features["sequence"] = {
            "model_family": model_family, "checkpoint": cache_id, "path": f"sequence/{model_family}/{cache_id}",
        }
        _atomic_json(path, metadata)


def check_existing(
    root: str | Path,
    cache_id: str,
    inputs: AssayInput,
    config: dict,
    resume: bool = False,
    *,
    model_family: str | None = None,
    cleanup_and_save: bool = False,
) -> bool:
    """Return True only for a completely validated matching cache with --resume."""
    root = Path(root).resolve()
    model_family = _validate_config(config, cache_id, model_family)
    with _assay_lock(root):
        _ensure_assay(root, inputs)
        leaf = _leaf(root, cache_id, model_family)
        if not leaf.exists():
            return False
        if not resume:
            raise FileExistsError(f"Feature cache already exists: {leaf}; use --resume to validate and reuse it")
        if not _cache_is_complete(root, "sequence", model_family, cache_id):
            return False
        if _read_json(_inside(leaf, "config.json")) != config:
            raise ValueError("Existing feature configuration differs; use a new cache ID")
        _register_default(root, cache_id, model_family)
        if cleanup_and_save:
            _cleanup_other_caches(root, "sequence", model_family, cache_id)
        return True


def _validate_payload(store: object, metadata: dict, rows: list[dict], config: dict) -> Tensor:
    if not isinstance(store, dict):
        raise ValueError("features.pt must contain a tensor dictionary")
    for key, value in {
        "schema_version": 2, "synthetic": False, "assay_id": metadata["assay_id"],
        "reference_sequence": metadata["reference_sequence"], "input_signature": metadata["input_signature"],
    }.items():
        if store.get(key) != value:
            raise ValueError(f"Feature payload metadata mismatch: {key}")
    if config["model_family"] == "prosst":
        from .structure_inputs import validate_structure_metadata

        validate_structure_metadata(config["structure"], metadata["reference_sequence"])
        if any(store.get(key) != value for key, value in _prosst_identity(config).items()):
            raise ValueError("ProSST feature payload structure, token, or configuration identity mismatch")
    length = len(metadata["reference_sequence"])
    positions = store.get("positions")
    if not isinstance(positions, Tensor) or positions.dtype != torch.long or not torch.equal(positions, torch.arange(1, length + 1)):
        raise ValueError("Expected complete one-based reference positions")
    compact_recipe = compact_sequence_recipe(config, length)
    ids = [row["variant_id"] for row in rows]
    cached_ids = store.get("variant_ids")
    if compact_recipe is not None:
        if not isinstance(cached_ids, list) or any(not isinstance(value, str) for value in cached_ids) or len(cached_ids) != len(ids) or set(cached_ids) != set(ids):
            raise ValueError("Compact sequence cache must cover exactly the variant table with unique IDs")
        rows_by_id = {row["variant_id"]: row for row in rows}
        ordered_rows = [rows_by_id[variant_id] for variant_id in cached_ids]
    else:
        if cached_ids != ids:
            raise ValueError("Sequence cache IDs must match variants.csv exactly and in order")
        ordered_rows = rows
        if any(key in store for key in COMPACT_SITE_FIELDS) or "cache_layout" in store:
            raise ValueError("Compact site tensors require an explicit centered-site recipe")
    hashes = [hashlib.sha256(row["mutated_sequence"].encode("utf-8")).hexdigest() for row in ordered_rows]
    if store.get("sequence_hashes") != hashes:
        raise ValueError("Sequence hash mismatch")
    if compact_recipe is not None:
        mutations = [parse_substitutions(row["mutant"], row["mutated_sequence"], metadata["reference_sequence"]) for row in ordered_rows]
        return validate_compact_sequence_payload(store, config, length, mutations)[0]
    embeddings = store.get("embeddings")
    pooled_recipe = pooled_sequence_recipe(config, length)
    validate_global_window_payload(store, pooled_recipe)
    if pooled_recipe is not None:
        if any(type(store.get(key)) is not type(value) or store.get(key) != value
               for key, value in pooled_recipe.items()):
            raise ValueError("Pooled sequence payload does not match its pooling/truncation recipe or represented length")
        expected_shape = (len(rows), config["hidden_size"])
    else:
        expected_shape = (len(rows), length, config["hidden_size"])
    if not isinstance(embeddings, Tensor) or embeddings.dtype != torch.float32 or tuple(embeddings.shape) != expected_shape:
        layout = "variants, hidden_size" if pooled_recipe else "variants, reference residues, hidden_size"
        raise ValueError(f"Sequence embeddings must have float32 shape [{layout}]")
    if not tensor_isfinite(embeddings):
        raise ValueError("Sequence embeddings must contain only finite values")
    return embeddings


def write_sequence_cache(
    root: str | Path,
    cache_id: str,
    inputs: AssayInput,
    config: dict,
    embeddings: Tensor,
    provenance: dict,
    *,
    model_family: str | None = None,
    site_data: dict | None = None,
    cleanup_and_save: bool = False,
    resume: bool = False,
) -> Path:
    """Commit one complete leaf; failed staging never replaces existing outputs."""
    root = Path(root).resolve()
    model_family = _validate_config(config, cache_id, model_family)
    if not isinstance(embeddings, Tensor) or not embeddings.is_floating_point():
        raise ValueError("Embeddings must be a floating-point tensor")
    payload = {
        "schema_version": 2,
        "synthetic": False,
        "assay_id": inputs.assay_id,
        "reference_sequence": inputs.reference_sequence,
        "input_signature": inputs.input_signature,
        "positions": torch.arange(1, inputs.length + 1),
        "variant_ids": list(inputs.variant_ids),
        "sequence_hashes": list(inputs.sequence_hashes),
        "embeddings": embeddings.detach().to(device="cpu", dtype=torch.float32).contiguous(),
    }
    compact_recipe = compact_sequence_recipe(config, inputs.length)
    if compact_recipe is not None:
        if not isinstance(site_data, dict) or set(site_data) != set(COMPACT_SITE_FIELDS):
            raise ValueError("Compact sequence caches require exactly the declared site_data tensors")
        payload.update(compact_recipe)
        for key in COMPACT_SITE_FIELDS:
            value = site_data[key]
            if not isinstance(value, Tensor):
                raise ValueError(f"Compact {key} must be a tensor")
            payload[key] = value.detach().to(device="cpu").contiguous()
    elif site_data is not None:
        raise ValueError("site_data requires an explicit compact centered-site recipe")
    pooled_recipe = pooled_sequence_recipe(config, inputs.length)
    if pooled_recipe is not None:
        payload.update(pooled_recipe)
    if model_family == "prosst":
        payload.update(_prosst_identity(config))
        _validate_prosst_provenance(provenance, config)
    _validate_payload(payload, _assay_metadata(inputs), inputs.rows, config)
    with _assay_lock(root), _cache_write_root(
        root, inputs, "sequence", model_family, cache_id, cleanup_and_save, resume,
    ) as write_root:
        leaf = _leaf(write_root, cache_id, model_family)
        if leaf.exists():
            raise FileExistsError(f"Refusing to overwrite existing feature cache: {leaf}")
        leaf.parent.mkdir(parents=True, exist_ok=True)
        # Pending directories are deliberately ignored by discovery/resume. Keeping
        # an interrupted one avoids recursive cleanup of potentially large outputs.
        pending = Path(tempfile.mkdtemp(prefix=f".{cache_id}.pending-", dir=leaf.parent))
        torch.save(payload, pending / "features.pt")
        _write_json(pending / "config.json", config)
        _write_json(pending / "provenance.json", provenance)
        files = {name: sha256_file(pending / name) for name in ("features.pt", "config.json", "provenance.json")}
        files["variants.csv"] = inputs.variants_sha256
        _write_json(pending / "completion.json", {
            "schema_version": 1,
            "cache_id": cache_id,
            "input_signature": inputs.input_signature,
            "files_sha256": files,
            **(_prosst_identity(config) if model_family == "prosst" else {}),
        })
        pending.rename(leaf)
        _register_default(write_root, cache_id, model_family)
    return _leaf(root, cache_id, model_family)


def _validate_prosst_provenance(provenance: dict, config: dict) -> None:
    if provenance.get("uses_dms_labels") is not False or any(
        provenance.get(key) != config[key]
        for key in ("structure_input_signature", "structure_token_input_signature")
    ):
        raise ValueError("ProSST provenance differs from the structure or structure-token identity")


def _select_sequence_cache(root: Path, metadata: dict, cache_id: str | None, model_family: str | None) -> tuple[str, str]:
    if model_family is not None:
        _family(model_family)
    if cache_id is not None:
        safe_component(cache_id, "cache ID")
    if model_family is not None and cache_id is not None:
        return model_family, cache_id
    features = metadata.get("features", {})
    if not isinstance(features, dict):
        raise ValueError("Assay features must be a mapping")
    descriptor = features.get("sequence", {})
    if not isinstance(descriptor, dict):
        raise ValueError("Invalid default sequence descriptor")
    family, checkpoint = descriptor.get("model_family"), descriptor.get("checkpoint")
    if (
        family in SEQUENCE_FAMILIES and isinstance(checkpoint, str)
        and (model_family is None or model_family == family)
        and (cache_id is None or cache_id == checkpoint)
    ):
        safe_component(checkpoint, "default sequence cache ID")
        if descriptor.get("path") != f"sequence/{family}/{checkpoint}":
            raise ValueError("Inconsistent default sequence descriptor")
        return family, checkpoint
    if model_family is None and cache_id is None:
        raise ValueError("Missing or invalid default sequence descriptor; select a model family and cache ID explicitly")
    candidates = []
    for family in SEQUENCE_FAMILIES if model_family is None else (model_family,):
        base = _inside(root, f"sequence/{family}")
        if base.is_dir():
            for path in base.iterdir():
                if path.is_dir() and not path.name.startswith(".") and (cache_id is None or cache_id == path.name):
                    safe_component(path.name, "cache ID")
                    candidates.append((family, path.name))
    if len(candidates) != 1:
        raise ValueError("Select a sequence model family and cache ID; no unique cache was found")
    return candidates[0]


def validate_sequence_cache(root: str | Path, cache_id: str | None = None, model_family: str | None = None) -> dict:
    """Validate a sequence-only cache without requiring structures or zero-shot scores."""
    root = Path(root).resolve()
    metadata = _read_json(_inside(root, "assay.json"))
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
    model_family, cache_id = _select_sequence_cache(root, metadata, cache_id, model_family)
    leaf = _leaf(root, cache_id, model_family)
    completion_path = _inside(leaf, "completion.json")
    if not completion_path.is_file():
        raise ValueError(f"Feature cache is incomplete: missing completion.json in {leaf}")
    completion = _read_json(completion_path)
    if completion.get("schema_version") != 1 or completion.get("cache_id") != cache_id or completion.get("input_signature") != metadata.get("input_signature"):
        raise ValueError("Feature completion metadata mismatch")
    files = completion.get("files_sha256")
    if not isinstance(files, dict) or set(files) != {"features.pt", "config.json", "provenance.json", "variants.csv"}:
        raise ValueError("Incomplete feature checksum manifest")
    for name, checksum in files.items():
        path = _inside(root if name == "variants.csv" else leaf, name)
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Feature checksum mismatch or missing file: {name}")
    if files["variants.csv"] != metadata.get("variants_sha256"):
        raise ValueError("Assay variant checksum mismatch")
    rows = read_csv(_inside(root, "variants.csv"))
    required = {"variant_id", "source_row_index", "mutant", "mutated_sequence", "DMS_score"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Missing required variant columns or rows")
    ids = [row.get("variant_id") for row in rows]
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
        "variants_sha256": files["variants.csv"],
        "coverage": coverage,
    }
    if hashlib.sha256(stable_json(identity).encode("utf-8")).hexdigest() != metadata.get("input_signature"):
        raise ValueError("Assay input signature mismatch")
    config = _read_json(_inside(leaf, "config.json"))
    _validate_config(config, cache_id, model_family)
    if model_family == "prosst":
        if any(completion.get(key) != value for key, value in _prosst_identity(config).items()):
            raise ValueError("ProSST completion structure, token, or configuration identity mismatch")
        _validate_prosst_provenance(_read_json(_inside(leaf, "provenance.json")), config)
    store = torch.load(
        _inside(leaf, "features.pt"), map_location="cpu", weights_only=True,
        **({"mmap": True} if compact_sequence_recipe(config) is not None
           and "mmap" in inspect.signature(torch.load).parameters else {}),
    )
    embeddings = _validate_payload(store, metadata, rows, config)
    return {
        "assay_id": metadata["assay_id"],
        "model_family": model_family,
        "cache_id": cache_id,
        "num_variants": len(rows),
        "reference_length": len(reference),
        "embedding_dim": embeddings.shape[-1],
        "path": str(leaf),
        "input_signature": metadata["input_signature"],
    }
