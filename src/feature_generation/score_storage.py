"""Atomic per-variant GP mean scores, independent of representation caches."""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
import tempfile

import torch
from torch import Tensor

from data_loading.features import read_csv
from .inputs import AssayInput, safe_component, sha256_file, stable_json
from .inverse_storage import _resume_configs_match, _validate_assay
from .storage import _assay_lock, _ensure_assay, _inside, _read_json, _write_json
from .structure_inputs import validate_structure_metadata

FAMILIES = ("proteinmpnn", "esm2")
SCORE_DESCRIPTOR = {"file": "mean_score.csv", "id_column": "variant_id", "score_column": "score"}


def _signature(value: dict) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _leaf(root: Path, model_family: str, cache_id: str) -> Path:
    if model_family not in FAMILIES:
        raise ValueError(f"Unsupported mean-score model family: {model_family!r}")
    safe_component(cache_id, "cache ID")
    return _inside(root, f"scores/{model_family}/{cache_id}")


def make_score_config(
    cache_id: str, model_family: str, model_metadata: dict, structure_metadata: dict | None = None,
) -> dict:
    if model_metadata.get("model_family", model_family) != model_family:
        raise ValueError("Model metadata family does not match the requested score family")
    config = {
        **model_metadata,
        "schema_version": 2,
        "layout_version": 1,
        "feature_type": "scores",
        "model_family": model_family,
        "checkpoint": cache_id,
        "synthetic": False,
        "mean_score": dict(SCORE_DESCRIPTOR),
        "score_direction": "higher_is_more_likely",
    }
    if model_family == "proteinmpnn":
        config["structure"] = structure_metadata
    elif structure_metadata is not None:
        raise ValueError("ESM2 mean scores do not use structure metadata")
    stable_json(config)
    _validate_config(config, model_family, cache_id)
    return config


def _validate_config(config: dict, model_family: str, cache_id: str) -> None:
    if model_family not in FAMILIES:
        raise ValueError("Unsupported mean-score model family")
    safe_component(cache_id, "cache ID")
    expected = {
        "schema_version": 2, "layout_version": 1, "feature_type": "scores",
        "model_family": model_family, "checkpoint": cache_id, "synthetic": False,
        "mean_score": SCORE_DESCRIPTOR, "score_direction": "higher_is_more_likely",
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Score configuration does not match its cache directory or format")
    if model_family == "esm2":
        recipe = {
            "score_definition": "sum_masked_marginal_log_odds",
            "conditioning_mode": "masked_wild_type",
            "inference_method": "masked_marginals",
            "normalization": "model_vocabulary",
            "output_dtype": "float32",
        }
        if any(config.get(key) != value for key, value in recipe.items()):
            raise ValueError("Unsupported ESM2 mean-score recipe")
        recipe_version = config.get("scoring_recipe_version")
        if type(recipe_version) is not int or recipe_version not in (1, 2):
            raise ValueError("Unsupported ESM2 mean-score recipe version")
        if recipe_version == 1:
            # Original artifacts may omit these optional metadata fields, but
            # a v1 artifact must never silently claim windowed WT context.
            if (
                config.get("scoring_window", "full") != "full"
                or config.get("truncation", "reject") != "reject"
                or "window_policy" in config or "window_indexing" in config
            ):
                raise ValueError("Legacy ESM2 mean scores require full-reference context without centered windows")
        else:
            centered_recipe = {
                "scoring_window": "centered", "truncation": "centered",
                "window_policy": "centered_clamped", "window_indexing": "zero_based_half_open",
                "residue_aggregation": "sum_over_mutations",
                "masking": "one_wild_type_residue_at_a_time",
                "wild_type_subtraction": True,
                "noncanonical_tokens": "retained_in_normalization",
                "alphabet": "ACDEFGHIKLMNPQRSTVWY",
                "precision_policy": "float32_log_softmax",
            }
            if any(type(config.get(key)) is not type(value) or config.get(key) != value for key, value in centered_recipe.items()):
                raise ValueError("Incomplete or inconsistent centered-window ESM2 mean-score recipe")
            max_length = config.get("max_length")
            if type(max_length) is not int or not 1 <= max_length <= 1022:
                raise ValueError("Centered ESM2 score max_length must be an integer between 1 and 1022")
        if "structure" in config:
            raise ValueError("ESM2 mean scores do not use structure metadata")
        return
    recipe = {
        "scoring_recipe_version": 1,
        "score_definition": "negative_mean_sequence_nll",
        "conditioning_mode": "sequence_autoregressive",
        "inference_method": "forward",
        "decoding_order": "shared_random_orders",
        "normalization": "native_21_amino_acid_tokens",
        "alphabet": "ACDEFGHIKLMNPQRSTVWYX",
        "output_dtype": "float32",
    }
    if any(config.get(key) != value for key, value in recipe.items()):
        raise ValueError("Unsupported ProteinMPNN mean-score recipe")
    if type(config.get("num_orders")) is not int or config["num_orders"] < 1:
        raise ValueError("Score configuration num_orders must be a positive integer")
    if type(config.get("seed")) is not int or config["seed"] < 0:
        raise ValueError("Score configuration seed must be a nonnegative integer")
    structure = config.get("structure")
    if not isinstance(structure, dict):
        raise ValueError("Missing score structure metadata")
    validate_structure_metadata(structure, structure.get("reference_sequence"))
    if any("O" not in row["backbone_atoms"] for row in structure["residue_mapping"]):
        raise ValueError("ProteinMPNN score structure metadata requires oxygen at every residue")


def _validate_scores(scores: object, num_variants: int) -> None:
    if (
        not isinstance(scores, Tensor)
        or scores.device.type != "cpu"
        or scores.dtype not in (torch.float32, torch.float64)
        or scores.shape != (num_variants,)
        or not torch.isfinite(scores).all()
    ):
        raise ValueError("Mean scores must be a finite CPU float32/float64 tensor with one value per variant")


def _validate_provenance(provenance: dict, config: dict) -> None:
    if (
        not isinstance(provenance, dict)
        or provenance.get("uses_dms_labels") is not False
        or (config["model_family"] == "proteinmpnn"
            and provenance.get("structure_input_signature") != config["structure"]["input_signature"])
    ):
        raise ValueError("Mean-score provenance mismatch")


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
    if model_family == "proteinmpnn":
        validate_structure_metadata(config["structure"], inputs.reference_sequence)
    with _assay_lock(root):
        _ensure_assay(root, inputs)
        leaf = _leaf(root, model_family, cache_id)
        if not leaf.exists():
            return False
        if not resume:
            raise FileExistsError(f"Mean-score cache already exists: {leaf}; use --resume to validate and reuse it")
        validate_score_cache(root, model_family, cache_id)
        if not _resume_configs_match(_read_json(_inside(leaf, "config.json")), config, model_family):
            raise ValueError("Existing checkpoint, scoring recipe, or structure differs; use a new cache ID")
        return True


def write_score_cache(
    root: str | Path,
    model_family: str,
    cache_id: str,
    inputs: AssayInput,
    config: dict,
    scores: Tensor,
    provenance: dict,
) -> Path:
    """Publish complete scores in immutable variant order under a distinct leaf."""
    root = Path(root).resolve()
    _validate_config(config, model_family, cache_id)
    if model_family == "proteinmpnn":
        validate_structure_metadata(config["structure"], inputs.reference_sequence)
    _validate_scores(scores, len(inputs.rows))
    _validate_provenance(provenance, config)
    stable_json(provenance)
    with _assay_lock(root):
        _ensure_assay(root, inputs)
        leaf = _leaf(root, model_family, cache_id)
        if leaf.exists():
            raise FileExistsError(f"Refusing to overwrite existing mean-score cache: {leaf}")
        leaf.parent.mkdir(parents=True, exist_ok=True)
        pending = Path(tempfile.mkdtemp(prefix=f".{cache_id}.pending-", dir=leaf.parent))
        with (pending / "mean_score.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("variant_id", "score"))
            for variant_id, score in zip(inputs.variant_ids, scores.detach().tolist()):
                writer.writerow((variant_id, format(score, ".17g")))
        _write_json(pending / "config.json", config)
        _write_json(pending / "provenance.json", provenance)
        files = {name: sha256_file(pending / name) for name in ("mean_score.csv", "config.json", "provenance.json")}
        files["variants.csv"] = inputs.variants_sha256
        completion = {
            "schema_version": 1,
            "model_family": model_family,
            "cache_id": cache_id,
            "input_signature": inputs.input_signature,
            "config_signature": _signature(config),
            "files_sha256": files,
        }
        if model_family == "proteinmpnn":
            completion["structure_input_signature"] = config["structure"]["input_signature"]
        _write_json(pending / "completion.json", completion)
        pending.rename(leaf)
    return leaf


def _select_cache(root: Path, model_family: str, cache_id: str | None) -> str:
    if model_family not in FAMILIES:
        raise ValueError("Unsupported mean-score model family")
    if cache_id is not None:
        return safe_component(cache_id, "cache ID")
    base = _inside(root, f"scores/{model_family}")
    candidates = []
    if base.is_dir():
        for path in base.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                safe_component(path.name, "cache ID")
                candidates.append(path.name)
    if len(candidates) != 1:
        raise ValueError("Select a mean-score cache ID; no unique score cache was found")
    return candidates[0]


def validate_score_cache(root: str | Path, model_family: str = "proteinmpnn", cache_id: str | None = None) -> dict:
    """Validate a score artifact and shared variants without loading representations."""
    root = Path(root).resolve()
    metadata = _read_json(_inside(root, "assay.json"))
    variants = _validate_assay(root, metadata)
    cache_id = _select_cache(root, model_family, cache_id)
    leaf = _leaf(root, model_family, cache_id)
    if not _inside(leaf, "completion.json").is_file():
        raise ValueError(f"Mean-score cache is incomplete: missing completion.json in {leaf}")
    completion = _read_json(_inside(leaf, "completion.json"))
    expected = {
        "schema_version": 1, "model_family": model_family, "cache_id": cache_id,
        "input_signature": metadata["input_signature"],
    }
    if any(completion.get(key) != value for key, value in expected.items()):
        raise ValueError("Mean-score completion metadata mismatch")
    files = completion.get("files_sha256")
    if not isinstance(files, dict) or set(files) != {"mean_score.csv", "config.json", "provenance.json", "variants.csv"}:
        raise ValueError("Incomplete mean-score checksum manifest")
    for name, checksum in files.items():
        path = _inside(root if name == "variants.csv" else leaf, name)
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Mean-score checksum mismatch or missing file: {name}")
    if files["variants.csv"] != metadata["variants_sha256"]:
        raise ValueError("Assay variant checksum mismatch")
    config = _read_json(_inside(leaf, "config.json"))
    _validate_config(config, model_family, cache_id)
    if model_family == "proteinmpnn":
        validate_structure_metadata(config["structure"], metadata["reference_sequence"])
    if (
        completion.get("config_signature") != _signature(config)
        or (model_family == "proteinmpnn"
            and completion.get("structure_input_signature") != config["structure"]["input_signature"])
    ):
        raise ValueError("Mean-score checkpoint, configuration, or structure identity mismatch")
    _validate_provenance(_read_json(_inside(leaf, "provenance.json")), config)
    rows = read_csv(_inside(leaf, "mean_score.csv"))
    if not rows or any(set(row) != {"variant_id", "score"} for row in rows):
        raise ValueError("Mean-score CSV requires variant_id and score columns and nonempty rows")
    ids = [row["variant_id"] for row in rows]
    if len(set(ids)) != len(ids) or set(ids) != {row["variant_id"] for row in variants}:
        raise ValueError("Mean-score IDs must cover exactly the immutable variant table")
    try:
        values = [float(row["score"]) for row in rows]
    except (TypeError, ValueError) as error:
        raise ValueError("Mean scores must be finite numeric values") from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Mean scores must be finite numeric values")
    report = {
        "assay_id": metadata["assay_id"],
        "model_family": model_family,
        "cache_id": cache_id,
        "feature_type": "scores",
        "num_variants": len(rows),
        "reference_length": len(metadata["reference_sequence"]),
        "score_definition": config["score_definition"],
        "score_direction": config["score_direction"],
        "mean_score": dict(SCORE_DESCRIPTOR),
        "path": str(leaf),
        "input_signature": metadata["input_signature"],
        "config_signature": _signature(config),
    }
    if model_family == "proteinmpnn":
        report["structure_input_signature"] = config["structure"]["input_signature"]
    return report
