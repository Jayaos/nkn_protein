"""Read cached representations without changing their on-disk layout."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor

from config import DEFAULT_BRANCHES, validate_enabled_branches
from .sequence_windows import (
    COMPACT_SITE_FIELDS, compact_sequence_recipe, mean_sequence_recipe, tensor_isfinite,
    validate_compact_sequence_payload, validate_global_window_payload,
)


ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
FEATURE_TYPES = ("sequence", "structure", "inverse_folding")
LOCAL_FEATURES = ("inverse_folding", "structure", "sequence_site", "structure_distance", "inverse_folding_site")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Missing or duplicate CSV columns: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV contains no records: {path}")
    return rows


def parse_substitutions(mutant: str, sequence: str, reference: str) -> tuple[int, ...]:
    """Return one-based positions, validating annotations against the full sequence.

    ``WT`` and an empty annotation identify the unchanged reference sequence.
    No raw-data, MSA, or structure offset is applied to processed ProteinGym rows.
    """
    if not reference or set(reference) - set(ALPHABET):
        raise ValueError("Reference sequence must contain canonical amino acids")
    reconstructed = list(reference)
    positions: list[int] = []
    if mutant not in ("", "WT"):
        for token in mutant.split(":"):
            match = re.fullmatch(r"([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])", token)
            if match is None:
                raise ValueError(f"Unsupported substitution annotation: {token!r}")
            wildtype, raw_position, replacement = match.groups()
            position = int(raw_position)
            if not 1 <= position <= len(reference) or position in positions:
                raise ValueError(f"Invalid or duplicate mutation position: {token}")
            if reference[position - 1] != wildtype or wildtype == replacement:
                raise ValueError(f"Invalid wild-type or unchanged substitution: {token}")
            positions.append(position)
            reconstructed[position - 1] = replacement
    if "".join(reconstructed) != sequence:
        raise ValueError(f"Mutation annotation disagrees with sequence: {mutant!r}")
    return tuple(positions)


@dataclass(frozen=True)
class FeatureBatch:
    """Variant-level features and packed, identically ordered mutation features."""

    assay_id: str
    variant_ids: tuple[str, ...]
    targets: Tensor | None
    scores: Tensor
    global_features: Tensor | None
    local_features: dict[str, Tensor]
    owners: Tensor
    mutation_positions: Tensor
    reference_sequence: str = ""
    sequences: tuple[str, ...] = ()
    score_names: tuple[str, ...] = ()
    feature_signature: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.variant_ids)
        if len(set(self.variant_ids)) != n:
            raise ValueError("Variant IDs must be unique within a batch")
        if self.scores.ndim != 2 or self.scores.shape[0] != n or self.scores.shape[1] < 1:
            raise ValueError("Scores must have shape [variants, scorers], with at least one scorer")
        if self.global_features is not None and (self.global_features.ndim != 2 or self.global_features.shape[0] != n or self.global_features.shape[1] < 1):
            raise ValueError("Global features must have shape [variants, features]")
        if self.targets is not None and self.targets.shape != (n,):
            raise ValueError("Targets must have shape [variants]")
        if self.owners.dtype != torch.long or self.owners.ndim != 1:
            raise ValueError("Owners must be a one-dimensional int64 tensor")
        if self.mutation_positions.dtype != torch.long or self.mutation_positions.shape != self.owners.shape:
            raise ValueError("Mutation positions must match owners and use int64")
        if self.owners.numel() and ((self.owners < 0).any() or (self.owners >= n).any()):
            raise ValueError("Mutation owner index is outside the batch")
        if (self.mutation_positions < 1).any():
            raise ValueError("Mutation positions must be one-based")
        if self.reference_sequence and (self.mutation_positions > len(self.reference_sequence)).any():
            raise ValueError("Mutation positions exceed the reference length")
        if set(self.local_features) - set(LOCAL_FEATURES):
            raise ValueError(f"Unknown local feature; expected a subset of {LOCAL_FEATURES}")
        if self.global_features is None and not self.local_features:
            raise ValueError("At least one local or global feature branch is required")
        for name, values in self.local_features.items():
            if values.ndim != 2 or values.shape[0] != self.owners.numel() or values.shape[1] < 1:
                raise ValueError(f"Invalid packed local feature shape: {name}")
            if name == "structure_distance" and values.shape[1] != 3:
                raise ValueError("Physical-distance features must contain three-dimensional CA coordinates")
            if name == "inverse_folding_site" and values.shape[1] != 1:
                raise ValueError("Inverse-folding site features must contain one mutant log probability per occurrence")
        if self.sequences and len(self.sequences) != n:
            raise ValueError("Sequence count must match variant count")
        if self.score_names and (len(self.score_names) != self.num_scores or len(set(self.score_names)) != self.num_scores):
            raise ValueError("Score names must identify every score column uniquely and in order")
        tensors = [self.scores, *self.local_features.values()]
        if self.global_features is not None:
            tensors.append(self.global_features)
        if self.targets is not None:
            tensors.append(self.targets)
        if any(not x.is_floating_point() or not tensor_isfinite(x) for x in tensors):
            raise ValueError("Features, scores, and targets must be finite floating-point tensors")
        devices = {x.device for x in [*tensors, self.owners, self.mutation_positions]}
        if len(devices) != 1:
            raise ValueError("All batch tensors must reside on the same device")

    @property
    def device(self) -> torch.device:
        return self.scores.device

    @property
    def num_variants(self) -> int:
        return len(self.variant_ids)

    @property
    def num_scores(self) -> int:
        return self.scores.shape[1]

    def select(self, indices: Sequence[int] | Tensor) -> FeatureBatch:
        raw = torch.as_tensor(indices, device=self.owners.device)
        if raw.dtype == torch.bool:
            if raw.shape != (self.num_variants,):
                raise ValueError("Boolean selection must match variant count")
            index = torch.nonzero(raw, as_tuple=False).flatten()
        else:
            if raw.ndim != 1 or (raw.numel() and raw.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
                raise ValueError("Selection must contain integer indices")
            index = raw.to(dtype=torch.long)
        if index.numel() and ((index < 0).any() or (index >= self.num_variants).any()):
            raise IndexError("Variant selection is outside the batch")
        if index.unique().numel() != index.numel():
            raise ValueError("Variant selection must not contain duplicates")
        remap = torch.full((self.num_variants,), -1, dtype=torch.long, device=self.owners.device)
        remap[index] = torch.arange(len(index), device=index.device)
        mapped = remap[self.owners]
        occurrence = torch.nonzero(mapped >= 0, as_tuple=False).flatten()
        occurrence = occurrence[torch.argsort(mapped[occurrence], stable=True)]
        python_index = index.cpu().tolist()
        return replace(
            self,
            variant_ids=tuple(self.variant_ids[i] for i in python_index),
            targets=None if self.targets is None else self.targets[index],
            scores=self.scores[index],
            global_features=None if self.global_features is None else self.global_features[index],
            local_features={key: value[occurrence] for key, value in self.local_features.items()},
            owners=mapped[occurrence],
            mutation_positions=self.mutation_positions[occurrence],
            sequences=tuple(self.sequences[i] for i in python_index) if self.sequences else (),
        )

    def to(self, device: str | torch.device) -> FeatureBatch:
        return replace(
            self,
            targets=None if self.targets is None else self.targets.to(device=device, dtype=torch.float64),
            scores=self.scores.to(device=device, dtype=torch.float64),
            global_features=None if self.global_features is None else self.global_features.to(device=device, dtype=torch.float64),
            local_features={key: value.to(device=device, dtype=torch.float64) for key, value in self.local_features.items()},
            owners=self.owners.to(device=device, dtype=torch.long),
            mutation_positions=self.mutation_positions.to(device=device, dtype=torch.long),
        )

    def with_scores(self, scores: Tensor) -> FeatureBatch:
        return replace(self, scores=scores)


def _inside(root: Path, relative: str, parts: int | None = None) -> Path:
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError(f"Invalid relative cache path: {relative!r}")
    pieces = relative.split("/")
    if any(piece in ("", ".", "..") for piece in pieces) or (parts is not None and len(pieces) != parts):
        raise ValueError(f"Invalid relative cache path: {relative!r}")
    path = root.joinpath(*pieces).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Cache path escapes the assay directory")
    return path


def _load_leaf(root: Path, source: str, assay_id: str, schema: int) -> tuple[dict, dict]:
    leaf = _inside(root, source, parts=3)
    name, family, checkpoint = source.split("/")
    config = json.loads(_inside(root, f"{source}/config.json").read_text(encoding="utf-8"))
    if any(config.get(k) != v for k, v in (("feature_type", name), ("model_family", family), ("checkpoint", checkpoint))):
        raise ValueError(f"Feature configuration does not match its directory: {source}")
    compact = compact_sequence_recipe(config) if name == "sequence" else None
    store = torch.load(
        _inside(root, f"{source}/features.pt"), map_location="cpu", weights_only=True,
        **({"mmap": True} if compact is not None else {}),
    )
    if not isinstance(store, Mapping) or store.get("assay_id") != assay_id or store.get("schema_version") != schema:
        raise ValueError(f"Feature cache metadata mismatch: {leaf}")
    if "synthetic" in config and store.get("synthetic") != config["synthetic"]:
        raise ValueError(f"Feature cache provenance mismatch: {leaf}")
    return dict(store), config


def _float_tensor(value: object, name: str, ndim: int, convert: bool = True) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point() or value.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-dimensional floating-point tensor")
    if min(value.shape) < 1 or not tensor_isfinite(value):
        raise ValueError(f"{name} must be nonempty and finite")
    return value.to(dtype=torch.float64) if convert else value


def _config_fingerprint(config: Mapping) -> str:
    """Fingerprint model/preprocessing identity independently of variant coverage."""
    coverage_fields = {"num_variants", "variants_sha256", "variant_ids", "sequence_hashes"}

    def clean(value):
        if isinstance(value, Mapping):
            return {key: clean(item) for key, item in value.items() if key not in coverage_fields}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return hashlib.sha256(json.dumps(clean(config), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def pooled_sequence_recipe(config: Mapping, reference_length: int | None = None) -> dict | None:
    """Validate an explicit ESM2 global-only mean recipe, preserving old caches."""
    if compact_sequence_recipe(config, reference_length) is not None:
        return None
    return mean_sequence_recipe(config, reference_length)


def kermut_probability_recipe(config: Mapping) -> dict | None:
    """Recognize explicitly declared, unnormalized ProteinMPNN probabilities.

    Other inverse-folding recipes retain the normalized-distribution contract.
    A partial declaration must not turn arbitrary weights into Kermut inputs.
    """
    expected = {
        "model_family": "proteinmpnn",
        "conditioning_mode": "sequence_conditional",
        "inference_method": "conditional_probs",
        "normalization": "native_21_amino_acid_tokens",
        "noncanonical_tokens": "excluded_without_renormalization",
        "order_aggregation": "geometric_mean_of_probabilities",
    }
    declared = any(config.get(name) == expected[name] for name in (
        "conditioning_mode", "normalization", "noncanonical_tokens",
    ))
    if not declared:
        return None
    if any(config.get(name) != value for name, value in expected.items()):
        raise ValueError("Incomplete or inconsistent Kermut ProteinMPNN probability recipe")
    num_orders, seed = config.get("num_orders"), config.get("seed")
    if type(num_orders) is not int or num_orders < 1:
        raise ValueError("Kermut ProteinMPNN num_orders must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Kermut ProteinMPNN seed must be an integer in [0, 2**32)")
    return {**expected, "num_orders": num_orders, "seed": seed}


def _structure_distance_features(manifest_dir, assay_id, reference, root, stores, configs, sources):
    """Load mapped CA atoms and check any cached structural provenance."""
    from feature_generation.structure_inputs import load_structure, validate_structure_metadata

    manifest_root = Path(manifest_dir)
    if not manifest_root.is_absolute():
        manifest_root = Path(__file__).resolve().parents[2] / manifest_root
    manifest_root = manifest_root.resolve()
    manifest = (manifest_root / f"{assay_id}.json").resolve()
    if manifest.parent != manifest_root:
        raise ValueError("Assay structure manifest path escapes structure_manifest_dir")
    structure = load_structure(manifest, reference, require_oxygen=False)
    if structure.metadata.get("assay_id") != assay_id:
        raise ValueError("Structure manifest belongs to a different assay")
    # Sequence representations can also be structure-conditioned (e.g. ProSST).
    # Check every selected cache that records structural identity.
    for name, store in stores.items():
        config = configs[name]
        cached_structure = config.get("structure")
        if cached_structure is not None:
            validate_structure_metadata(cached_structure, reference)
            if cached_structure.get("assay_id") != assay_id:
                raise ValueError(f"{name} structure provenance belongs to a different assay")
            if cached_structure["input_signature"] != structure.input_signature:
                raise ValueError(f"Selected structure differs from the {name} feature structure")
        recorded_provenance = [config, store]
        if "provenance" in store:
            recorded_provenance.append(store["provenance"])
        provenance_path = _inside(root, f"{sources[name]}/provenance.json")
        if provenance_path.exists():
            recorded_provenance.append(json.loads(provenance_path.read_text(encoding="utf-8")))
        for recorded in recorded_provenance:
            if not isinstance(recorded, Mapping):
                raise ValueError(f"{name} structure provenance must be an object")
            for key, expected in (("structure_sha256", structure.structure_sha256),
                                  ("structure_input_signature", structure.input_signature)):
                if key in recorded and recorded[key] != expected:
                    raise ValueError(f"{name} provenance disagrees with selected structure: {key}")
    signature = {
        "structure_distance.input_signature": structure.input_signature,
        "structure_distance.structure_sha256": structure.structure_sha256,
        "structure_distance.coordinates": "reference_CA_angstrom",
    }
    return structure.coords[:, 1].double(), signature


def load_feature_bundle(
    feature_dir: str | Path,
    feature_selection: Mapping[str, str] | None = None,
    score_sources: Sequence[str] | None = None,
    pooling: str = "mean",
    device: str | torch.device = "cpu",
    enabled_branches: Sequence[str] | None = None,
    *,
    inverse_folding_policy: str = "normalized",
    structure_manifest_dir: str | Path = "data/structures/manifests",
) -> FeatureBatch:
    """Load schema-v2 caches; selection and score sources never depend on labels.

    Feature selections map feature types to ``family/checkpoint``. Score sources
    use ``feature_type/family/checkpoint`` and default to the distinct leaves
    needed by enabled branches. Explicit score sources are independent of branch
    selection and require only score files, not their representation tensors.
    New score leaves use scores/family/recipe and a mean_score descriptor; legacy
    zero_shot descriptors in representation leaves remain supported.
    Omitting enabled_branches retains all four branches.
    Opt into ``inverse_folding_policy="sequence_conditional"`` to preserve
    declared geometric-mean ProteinMPNN probabilities without rescaling.
    The ``"kermut"`` policy remains an alias for the Kermut consumer.
    The optional inverse-folding-site branch selects each substituted amino
    acid's log probability from that same cache, without renormalization or
    another model evaluation. It does not require the distribution branch.
    The optional physical-distance branch reads CA coordinates directly from
    verified structure manifests; relative manifest directories resolve from
    the project root. It adds no representation cache or mean score source.
    A physical-distance-only selection therefore needs explicit score_sources.
    """
    if inverse_folding_policy not in ("normalized", "sequence_conditional", "kermut"):
        raise ValueError("inverse_folding_policy must be 'normalized', 'sequence_conditional', or 'kermut'")
    branches = DEFAULT_BRANCHES if enabled_branches is None else validate_enabled_branches(enabled_branches)
    needed_types = {"sequence" if name.startswith("sequence_") else
                    "inverse_folding" if name == "inverse_folding_site" else name
                    for name in branches if name != "structure_distance"}
    root = Path(feature_dir).resolve()
    metadata = json.loads((root / "assay.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 2 or metadata.get("layout_version") != 1:
        raise ValueError("Expected feature schema version 2 and layout version 1")
    if metadata.get("variants_file") != "variants.csv":
        raise ValueError("Expected variants_file='variants.csv'")
    if pooling != "mean":
        raise ValueError("Only global residue-mean pooling is currently supported, including for prepooled sequence caches")
    assay_id, reference = metadata["assay_id"], metadata["reference_sequence"]
    if not reference or set(reference) - set(ALPHABET):
        raise ValueError("Reference sequence must contain canonical amino acids")
    if metadata.get("reference_length", len(reference)) != len(reference):
        raise ValueError("Reference length does not match sequence")
    rows = read_csv(_inside(root, "variants.csv"))
    required = {"variant_id", "mutant", "mutated_sequence"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Missing variant columns: {required - rows[0].keys()}")
    ids = tuple(row["variant_id"] for row in rows)
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Variant IDs must be nonempty and unique")
    sequences = tuple(row["mutated_sequence"] for row in rows)
    mutations = [parse_substitutions(row["mutant"], row["mutated_sequence"], reference) for row in rows]
    label_values = [row.get("DMS_score", "") for row in rows]
    if any(label_values) and not all(label_values):
        raise ValueError("Targets must be present for every variant or absent for every variant")
    targets = torch.tensor([float(value) for value in label_values], dtype=torch.float64) if all(label_values) else None
    if targets is not None and not torch.isfinite(targets).all():
        raise ValueError("Targets must be finite")
    selection = dict(feature_selection or {})
    if set(selection) - set(FEATURE_TYPES):
        raise ValueError("Unknown feature selection type")
    stores, configs, sources = {}, {}, {}
    defaults = metadata.get("features", {})
    for name in FEATURE_TYPES:
        if name not in needed_types:
            continue
        if name in selection:
            suffix = selection[name]
        else:
            default = defaults.get(name)
            if not isinstance(default, Mapping):
                raise ValueError(f"Missing default feature descriptor: {name}")
            suffix = f"{default.get('model_family')}/{default.get('checkpoint')}"
            if default.get("path") != f"{name}/{suffix}":
                raise ValueError(f"Default feature path is inconsistent: {name}")
        _inside(root, suffix, parts=2)
        source = f"{name}/{suffix}"
        stores[name], configs[name] = _load_leaf(root, source, assay_id, 2)
        sources[name] = source
        store = stores[name]
        if store.get("reference_sequence") != reference:
            raise ValueError(f"Reference sequence mismatch: {name}")
        positions = store.get("positions")
        expected_positions = torch.arange(1, len(reference) + 1)
        structure_metadata = configs[name].get("structure", {})
        if name == "inverse_folding" and isinstance(structure_metadata, Mapping) and structure_metadata.get("schema_version") == 2:
            from feature_generation.structure_fragments import inverse_structure_positions
            from feature_generation.inverse_storage import _validate_config

            _validate_config(configs[name], configs[name]["model_family"], configs[name]["checkpoint"])
            if structure_metadata.get("assay_id") != assay_id:
                raise ValueError("Inverse-folding structure belongs to a different assay")
            expected_positions = inverse_structure_positions(structure_metadata, reference)
            if store.get("structure_input_signature") != structure_metadata["input_signature"]:
                raise ValueError("Inverse-folding fragment structure identity mismatch")
        if not isinstance(positions, Tensor) or positions.dtype != torch.long or not torch.equal(positions, expected_positions):
            raise ValueError(f"Expected complete one-based reference positions: {name}")
        if "synthetic" in metadata and store.get("synthetic") != metadata["synthetic"]:
            raise ValueError(f"Assay and feature provenance disagree: {name}")
    owners = torch.tensor([i for i, positions in enumerate(mutations) for _ in positions], dtype=torch.long)
    positions = torch.tensor([position for values in mutations for position in values], dtype=torch.long)
    global_features = None
    local_features = {}
    probability_recipe = None
    if "sequence" in stores:
        sequence_store = stores["sequence"]
        sequence_config = configs["sequence"]
        compact_recipe = compact_sequence_recipe(sequence_config, len(reference))
        pooled_recipe = pooled_sequence_recipe(sequence_config, len(reference))
        validate_global_window_payload(sequence_store, compact_recipe or pooled_recipe)
        embedding = _float_tensor(
            sequence_store.get("embeddings"), "Sequence embeddings", 2 if pooled_recipe or compact_recipe else 3, convert=False,
        )
        if compact_recipe is None and (any(key in sequence_store for key in COMPACT_SITE_FIELDS) or "cache_layout" in sequence_store):
            raise ValueError("Compact site tensors require an explicit centered-site recipe")
        if pooled_recipe is not None:
            if any(type(sequence_store.get(key)) is not type(value) or sequence_store.get(key) != value
                   for key, value in pooled_recipe.items()):
                raise ValueError("Pooled sequence payload does not match its pooling/truncation recipe or represented length")
            if embedding.dtype != torch.float32 or embedding.shape[1] != sequence_config["hidden_size"]:
                raise ValueError("Pooled sequence embeddings must have float32 shape [variants, hidden_size]")
            if "sequence_site" in branches:
                raise ValueError("sequence_site requires per-residue embeddings; the selected sequence cache is mean-pooled")
        cached_ids = sequence_store.get("variant_ids", [])
        hashes = sequence_store.get("sequence_hashes", [])
        if compact_recipe is not None and (
            not isinstance(cached_ids, list) or any(not isinstance(value, str) for value in cached_ids)
            or not isinstance(hashes, list) or any(not isinstance(value, str) for value in hashes)
        ):
            raise ValueError("Compact sequence cache IDs and hashes must be string lists")
        if len(cached_ids) != len(set(cached_ids)) or set(cached_ids) != set(ids) or len(cached_ids) != embedding.shape[0]:
            raise ValueError("Sequence cache must cover exactly the variant table, with unique IDs")
        if len(hashes) != len(cached_ids) or (pooled_recipe is None and compact_recipe is None and embedding.shape[1] != len(reference)):
            raise ValueError("Sequence cache residue axis or hash count is invalid")
        lookup = {variant_id: i for i, variant_id in enumerate(cached_ids)}
        order = [lookup[variant_id] for variant_id in ids]
        for sequence, i in zip(sequences, order):
            if hashlib.sha256(sequence.encode("utf-8")).hexdigest() != hashes[i]:
                raise ValueError("Sequence hash mismatch")
        cache_order = torch.tensor(order, dtype=torch.long)
        site_embedding = None
        site_order = None
        if compact_recipe is not None:
            if "input_signature" in metadata and sequence_store.get("input_signature") != metadata["input_signature"]:
                raise ValueError("Compact sequence input signature mismatch")
            mutations_by_id = dict(zip(ids, mutations))
            cached_mutations = [mutations_by_id[variant_id] for variant_id in cached_ids]
            embedding, site_embedding = validate_compact_sequence_payload(
                sequence_store, sequence_config, len(reference), cached_mutations,
            )
            if cached_ids != list(ids):
                offsets = [0]
                for mutation in cached_mutations:
                    offsets.append(offsets[-1] + len(mutation))
                site_order = torch.tensor([
                    occurrence for index in order for occurrence in range(offsets[index], offsets[index + 1])
                ], dtype=torch.long)
        if "sequence_global" in branches:
            if compact_recipe is not None:
                global_features = (embedding if cached_ids == list(ids) else embedding[cache_order]).double()
            else:
                global_features = (embedding.double() if pooled_recipe else embedding.mean(dim=1, dtype=torch.float64))[cache_order]
        if "sequence_site" in branches:
            if compact_recipe is not None:
                local_features["sequence_site"] = (site_embedding if site_order is None else site_embedding[site_order]).double()
            else:
                local_features["sequence_site"] = embedding[cache_order[owners], positions - 1].double()
    if "structure" in stores:
        structural = _float_tensor(stores["structure"].get("embeddings"), "Structural embeddings", 2)
        if structural.shape[0] != len(reference):
            raise ValueError("Structural features must cover the reference residues")
        local_features["structure"] = structural[positions - 1]
    if "inverse_folding" in stores:
        inverse_store = stores["inverse_folding"]
        inverse = _float_tensor(inverse_store.get("probabilities"), "Inverse-folding probabilities", 2)
        inverse_positions = inverse_store["positions"]
        if inverse.shape != (len(inverse_positions), 20):
            raise ValueError("Probability features must match their declared reference positions")
        if inverse_store.get("alphabet") != ALPHABET:
            raise ValueError("Inverse-folding amino-acid alphabet mismatch")
        probability_recipe = kermut_probability_recipe(configs["inverse_folding"])
        if probability_recipe is not None:
            if inverse_folding_policy not in ("sequence_conditional", "kermut"):
                raise ValueError("Sequence-conditioned ProteinMPNN probabilities require an explicit sequence_conditional or Kermut loader policy")
            if (inverse <= 0).any() or (inverse.sum(-1) > 1 + 2e-5).any():
                raise ValueError("Kermut ProteinMPNN probabilities must be positive with row mass at most one")
        elif (inverse < 0).any() or not torch.allclose(inverse.sum(-1), torch.ones(len(inverse_positions), dtype=torch.float64), atol=1e-5, rtol=1e-5):
            raise ValueError("Inverse-folding distributions must be nonnegative and sum to one")
        if "log_probs" in inverse_store:
            log_probs = inverse_store["log_probs"]
            if not isinstance(log_probs, Tensor) or log_probs.shape != inverse.shape or not torch.allclose(log_probs.double(), inverse.log(), atol=1e-5, rtol=1e-5):
                raise ValueError("Inverse-folding log probabilities are inconsistent")
        inverse_lookup = torch.full((len(reference) + 1,), -1, dtype=torch.long)
        inverse_lookup[inverse_positions] = torch.arange(len(inverse_positions))
        inverse_rows = inverse_lookup[positions]
        if (inverse_rows < 0).any():
            missing = positions[inverse_rows < 0].unique().tolist()
            raise ValueError(f"Inverse-folding structures do not cover mutation positions: {missing}")
        packed_inverse = inverse[inverse_rows]
        if "inverse_folding" in branches:
            local_features["inverse_folding"] = packed_inverse
        if "inverse_folding_site" in branches:
            # Match Kermut's mutation-probability factor: select the substituted
            # amino acid, not the WT residue or a sequence-level mean score.
            # Keep the declared conditional/geometric-mean weights unchanged.
            amino_acids = torch.tensor([
                ALPHABET.index(row["mutated_sequence"][position - 1])
                for row, mutation_positions in zip(rows, mutations)
                for position in mutation_positions
            ], dtype=torch.long)
            selected = packed_inverse.gather(1, amino_acids[:, None])
            if (selected <= 0).any():
                raise ValueError("Inverse-folding site features require positive probabilities for substituted amino acids")
            local_features["inverse_folding_site"] = selected.log()
    geometry_signature = {}
    if "structure_distance" in branches:
        coordinates, geometry_signature = _structure_distance_features(
            structure_manifest_dir, assay_id, reference, root, stores, configs, sources,
        )
        local_features["structure_distance"] = coordinates[positions - 1]
    selected_scores = list(score_sources) if score_sources is not None else list(sources.values())
    if not selected_scores or len(set(selected_scores)) != len(selected_scores):
        raise ValueError("Select at least one unique mean score source")
    columns = []
    signature = {"schema_version": "2", "layout_version": "1", "pooling": pooling, **sources, **geometry_signature}
    if probability_recipe is not None:
        signature["inverse_folding.probability_recipe"] = json.dumps(probability_recipe, sort_keys=True, separators=(",", ":"))
    if "inverse_folding_site" in branches:
        signature["inverse_folding_site.feature"] = "mutant_log_probability"
    if branches != DEFAULT_BRANCHES:
        signature["enabled_branches"] = json.dumps(branches, separators=(",", ":"))
    for name, config in configs.items():
        signature[f"{name}.config_sha256"] = _config_fingerprint(config)
    for source in selected_scores:
        leaf = _inside(root, source, parts=3)
        config = json.loads(_inside(root, f"{source}/config.json").read_text(encoding="utf-8"))
        source_type, family, checkpoint = source.split("/")
        if any(config.get(key) != value for key, value in (("feature_type", source_type), ("model_family", family), ("checkpoint", checkpoint))):
            raise ValueError(f"Score configuration does not match its directory: {source}")
        if source_type == "scores":
            from feature_generation.score_storage import validate_score_cache

            validate_score_cache(root, model_family=family, cache_id=checkpoint)
        signature[f"score:{source}.config_sha256"] = _config_fingerprint(config)
        descriptor = config.get("mean_score", config.get("zero_shot"))
        if "mean_score" in config and "zero_shot" in config and config["mean_score"] != config["zero_shot"]:
            raise ValueError(f"Conflicting mean_score and legacy zero_shot descriptors: {leaf}")
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"Missing mean score descriptor: {leaf}")
        if not {"file", "id_column", "score_column"}.issubset(descriptor):
            raise ValueError(f"Incomplete mean score descriptor: {leaf}")
        score_path = _inside(root, f"{source}/{descriptor['file']}")
        score_rows = read_csv(score_path)
        id_column, score_column = descriptor["id_column"], descriptor["score_column"]
        if not {id_column, score_column}.issubset(score_rows[0]):
            raise ValueError(f"Missing declared score columns: {score_path}")
        table = {row[id_column]: float(row[score_column]) for row in score_rows}
        if len(table) != len(score_rows) or set(table) != set(ids):
            raise ValueError(f"Score IDs must cover exactly the variant table: {score_path}")
        if not all(math.isfinite(value) for value in table.values()):
            raise ValueError(f"Mean scores must be finite: {score_path}")
        columns.append(torch.tensor([table[variant_id] for variant_id in ids], dtype=torch.float64))
    batch = FeatureBatch(
        assay_id=assay_id,
        variant_ids=ids,
        targets=targets,
        scores=torch.stack(columns, dim=1),
        global_features=global_features,
        local_features={name: local_features[name] for name in LOCAL_FEATURES if name in local_features},
        owners=owners,
        mutation_positions=positions,
        reference_sequence=reference,
        sequences=sequences,
        score_names=tuple(selected_scores),
        feature_signature=signature,
    )
    return batch.to(device)
