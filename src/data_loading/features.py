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


ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
FEATURE_TYPES = ("sequence", "structure", "inverse_folding")
LOCAL_FEATURES = ("inverse_folding", "structure", "sequence_site")


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
    global_features: Tensor
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
        if self.global_features.ndim != 2 or self.global_features.shape[0] != n or self.global_features.shape[1] < 1:
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
        if set(self.local_features) != set(LOCAL_FEATURES):
            raise ValueError(f"Local features must have keys {LOCAL_FEATURES}")
        for name, values in self.local_features.items():
            if values.ndim != 2 or values.shape[0] != self.owners.numel() or values.shape[1] < 1:
                raise ValueError(f"Invalid packed local feature shape: {name}")
        if self.sequences and len(self.sequences) != n:
            raise ValueError("Sequence count must match variant count")
        if self.score_names and (len(self.score_names) != self.num_scores or len(set(self.score_names)) != self.num_scores):
            raise ValueError("Score names must identify every score column uniquely and in order")
        tensors = [self.scores, self.global_features, *self.local_features.values()]
        if self.targets is not None:
            tensors.append(self.targets)
        if any(not x.is_floating_point() or not torch.isfinite(x).all() for x in tensors):
            raise ValueError("Features, scores, and targets must be finite floating-point tensors")
        devices = {x.device for x in [*tensors, self.owners, self.mutation_positions]}
        if len(devices) != 1:
            raise ValueError("All batch tensors must reside on the same device")

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
            global_features=self.global_features[index],
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
            global_features=self.global_features.to(device=device, dtype=torch.float64),
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
    store = torch.load(_inside(root, f"{source}/features.pt"), map_location="cpu", weights_only=True)
    if not isinstance(store, Mapping) or store.get("assay_id") != assay_id or store.get("schema_version") != schema:
        raise ValueError(f"Feature cache metadata mismatch: {leaf}")
    if "synthetic" in config and store.get("synthetic") != config["synthetic"]:
        raise ValueError(f"Feature cache provenance mismatch: {leaf}")
    return dict(store), config


def _float_tensor(value: object, name: str, ndim: int, convert: bool = True) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point() or value.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-dimensional floating-point tensor")
    if min(value.shape) < 1 or not torch.isfinite(value).all():
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


def load_feature_bundle(
    feature_dir: str | Path,
    feature_selection: Mapping[str, str] | None = None,
    score_sources: Sequence[str] | None = None,
    pooling: str = "mean",
    device: str | torch.device = "cpu",
) -> FeatureBatch:
    """Load schema-v2 caches; selection and score sources never depend on labels.

    Feature selections map feature types to ``family/checkpoint``. Score sources
    use ``feature_type/family/checkpoint`` and default to the selected three
    feature leaves. Global pooling currently supports mean pooling over residues.
    """
    root = Path(feature_dir).resolve()
    metadata = json.loads((root / "assay.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 2 or metadata.get("layout_version") != 1:
        raise ValueError("Expected feature schema version 2 and layout version 1")
    if metadata.get("variants_file") != "variants.csv":
        raise ValueError("Expected variants_file='variants.csv'")
    if pooling != "mean":
        raise ValueError("Only global residue-mean pooling is currently supported")
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
        if name not in defaults or not isinstance(defaults[name], Mapping):
            raise ValueError(f"Missing default feature descriptor: {name}")
        default = defaults[name]
        suffix = selection.get(name, f"{default.get('model_family')}/{default.get('checkpoint')}")
        _inside(root, suffix, parts=2)
        source = f"{name}/{suffix}"
        if name not in selection and default.get("path") != source:
            raise ValueError(f"Default feature path is inconsistent: {name}")
        stores[name], configs[name] = _load_leaf(root, source, assay_id, 2)
        sources[name] = source
        store = stores[name]
        if store.get("reference_sequence") != reference:
            raise ValueError(f"Reference sequence mismatch: {name}")
        positions = store.get("positions")
        if not isinstance(positions, Tensor) or positions.dtype != torch.long or not torch.equal(positions, torch.arange(1, len(reference) + 1)):
            raise ValueError(f"Expected complete one-based reference positions: {name}")
        if "synthetic" in metadata and store.get("synthetic") != metadata["synthetic"]:
            raise ValueError(f"Assay and feature provenance disagree: {name}")
    sequence_store = stores["sequence"]
    embedding = _float_tensor(sequence_store.get("embeddings"), "Sequence embeddings", 3, convert=False)
    cached_ids = sequence_store.get("variant_ids", [])
    hashes = sequence_store.get("sequence_hashes", [])
    if len(cached_ids) != len(set(cached_ids)) or set(cached_ids) != set(ids) or len(cached_ids) != embedding.shape[0]:
        raise ValueError("Sequence cache must cover exactly the variant table, with unique IDs")
    if len(hashes) != len(cached_ids) or embedding.shape[1] != len(reference):
        raise ValueError("Sequence cache residue axis or hash count is invalid")
    lookup = {variant_id: i for i, variant_id in enumerate(cached_ids)}
    order = [lookup[variant_id] for variant_id in ids]
    for sequence, i in zip(sequences, order):
        if hashlib.sha256(sequence.encode("utf-8")).hexdigest() != hashes[i]:
            raise ValueError("Sequence hash mismatch")
    structural = _float_tensor(stores["structure"].get("embeddings"), "Structural embeddings", 2)
    inverse = _float_tensor(stores["inverse_folding"].get("probabilities"), "Inverse-folding probabilities", 2)
    if structural.shape[0] != len(reference) or inverse.shape != (len(reference), 20):
        raise ValueError("Structural and probability features must cover the reference residues")
    if stores["inverse_folding"].get("alphabet") != ALPHABET:
        raise ValueError("Inverse-folding amino-acid alphabet mismatch")
    if (inverse < 0).any() or not torch.allclose(inverse.sum(-1), torch.ones(len(reference), dtype=torch.float64), atol=1e-5, rtol=1e-5):
        raise ValueError("Inverse-folding distributions must be nonnegative and sum to one")
    if "log_probs" in stores["inverse_folding"]:
        log_probs = stores["inverse_folding"]["log_probs"]
        if not isinstance(log_probs, Tensor) or log_probs.shape != inverse.shape or not torch.allclose(log_probs.double(), inverse.log(), atol=1e-5, rtol=1e-5):
            raise ValueError("Inverse-folding log probabilities are inconsistent")
    selected_scores = list(score_sources) if score_sources is not None else [sources[name] for name in FEATURE_TYPES]
    if not selected_scores or len(set(selected_scores)) != len(selected_scores):
        raise ValueError("Select at least one unique zero-shot score source")
    columns = []
    signature = {"schema_version": "2", "layout_version": "1", "pooling": pooling, **sources}
    for name, config in configs.items():
        signature[f"{name}.config_sha256"] = _config_fingerprint(config)
    for source in selected_scores:
        leaf = _inside(root, source, parts=3)
        config = json.loads(_inside(root, f"{source}/config.json").read_text(encoding="utf-8"))
        source_type, family, checkpoint = source.split("/")
        if any(config.get(key) != value for key, value in (("feature_type", source_type), ("model_family", family), ("checkpoint", checkpoint))):
            raise ValueError(f"Score configuration does not match its directory: {source}")
        signature[f"score:{source}.config_sha256"] = _config_fingerprint(config)
        descriptor = config.get("zero_shot")
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"Missing zero-shot descriptor: {leaf}")
        score_path = _inside(root, f"{source}/{descriptor['file']}")
        score_rows = read_csv(score_path)
        id_column, score_column = descriptor["id_column"], descriptor["score_column"]
        if not {id_column, score_column}.issubset(score_rows[0]):
            raise ValueError(f"Missing declared score columns: {score_path}")
        table = {row[id_column]: float(row[score_column]) for row in score_rows}
        if len(table) != len(score_rows) or set(table) != set(ids):
            raise ValueError(f"Score IDs must cover exactly the variant table: {score_path}")
        if not all(math.isfinite(value) for value in table.values()):
            raise ValueError(f"Zero-shot scores must be finite: {score_path}")
        columns.append(torch.tensor([table[variant_id] for variant_id in ids], dtype=torch.float64))
    owners = torch.tensor([i for i, positions in enumerate(mutations) for _ in positions], dtype=torch.long)
    positions = torch.tensor([position for values in mutations for position in values], dtype=torch.long)
    cache_order = torch.tensor(order, dtype=torch.long)
    batch = FeatureBatch(
        assay_id=assay_id,
        variant_ids=ids,
        targets=targets,
        scores=torch.stack(columns, dim=1),
        global_features=embedding.mean(dim=1, dtype=torch.float64)[cache_order],
        local_features={"inverse_folding": inverse[positions - 1], "structure": structural[positions - 1], "sequence_site": embedding[cache_order[owners], positions - 1].double()},
        owners=owners,
        mutation_positions=positions,
        reference_sequence=reference,
        sequences=sequences,
        score_names=tuple(selected_scores),
        feature_signature=signature,
    )
    return batch.to(device)
