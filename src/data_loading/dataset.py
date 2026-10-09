"""Load the synthetic, substitution-only assay fixture and its cached features."""

import csv
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

import torch
from torch.utils.data import Dataset


ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
FEATURE_TYPES = ("sequence", "structure", "inverse_folding")


class ProteinFitnessDataset(Dataset):
    """One variant per item; structural features use one-based assay positions.

    This loader intentionally accepts only explicitly synthetic feature bundles.
    Each sequence embedding has shape [reference length, sequence dimension],
    with one row per residue and no special-token or padding rows.
    Variable mutation counts require a custom collate function for batched loading.
    ``as_feature_matrices`` optionally pools residue features for kernel demos.
    ``feature_dir`` is an assay directory containing ``assay.json`` and
    ``variants.csv``. Override its default caches with ``feature_selection``,
    for example ``{"sequence": "esm2/t6_8M_UR50"}``.
    """

    def __init__(self, feature_dir, feature_selection=None):
        self.feature_dir = Path(feature_dir).resolve()
        self.assay_metadata = json.loads((self.feature_dir / "assay.json").read_text(encoding="utf-8"))
        if self.assay_metadata.get("synthetic") is not True:
            raise ValueError("Expected an explicitly synthetic feature bundle")
        if self.assay_metadata.get("schema_version") != 2:
            raise ValueError("Expected schema version 2 with per-residue sequence embeddings; regenerate older fixtures")
        if self.assay_metadata.get("layout_version") != 1:
            raise ValueError("Expected organized assay layout version 1")
        if self.assay_metadata.get("variants_file") != "variants.csv":
            raise ValueError("Expected variants_file='variants.csv' in assay.json")
        self.assay_id = self.assay_metadata["assay_id"]
        self.reference_sequence = self.assay_metadata["reference_sequence"]
        if not self.reference_sequence or set(self.reference_sequence) - set(ALPHABET):
            raise ValueError("Reference sequence must contain canonical amino acids")
        with (self.feature_dir / "variants.csv").open(newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError("variants.csv contains no variants")
        required = {"variant_id", "mutant", "mutated_sequence", "DMS_score", "source_row_index"}
        if not required.issubset(self.rows[0]):
            raise ValueError(f"variants.csv is missing columns: {required - self.rows[0].keys()}")
        self.variant_ids = [row["variant_id"] for row in self.rows]
        if len(set(self.variant_ids)) != len(self.variant_ids):
            raise ValueError("Duplicate variant IDs in variants.csv")
        self.targets = torch.tensor([float(row["DMS_score"]) for row in self.rows], dtype=torch.float32)
        if not torch.isfinite(self.targets).all():
            raise ValueError("Targets must be finite")

        defaults = self.assay_metadata.get("features")
        if not isinstance(defaults, Mapping) or not set(FEATURE_TYPES).issubset(defaults):
            raise ValueError("assay.json must specify default features for each feature type")
        if feature_selection is None:
            feature_selection = {}
        if not isinstance(feature_selection, Mapping):
            raise ValueError("feature_selection must map feature types to 'model_family/checkpoint'")
        unknown = set(feature_selection) - set(FEATURE_TYPES)
        if unknown:
            raise ValueError(f"Unknown feature types in feature_selection: {sorted(unknown, key=str)}")

        self.available_feature_sets = {
            name: sorted(
                f"{path.parent.parent.name}/{path.parent.name}"
                for path in (self.feature_dir / name).glob("*/*/features.pt")
                if path.is_file() and (path.parent / "config.json").is_file()
                and path.resolve().is_relative_to(self.feature_dir)
                and (path.parent / "config.json").resolve().is_relative_to(self.feature_dir)
            )
            for name in FEATURE_TYPES
        }
        self.feature_selection = {}
        self.feature_configs = {}
        stores = {}
        for name in FEATURE_TYPES:
            default = defaults[name]
            if not isinstance(default, Mapping):
                raise ValueError(f"Invalid {name} default in assay.json")
            family, checkpoint = default.get("model_family"), default.get("checkpoint")
            default_selection = f"{family}/{checkpoint}"
            self._selection_parts(default_selection)
            if not isinstance(family, str) or not isinstance(checkpoint, str):
                raise ValueError(f"Missing {name} model_family or checkpoint in assay.json")
            if default.get("path") != f"{name}/{default_selection}":
                raise ValueError(f"Invalid {name} default path in assay.json")

            selection = feature_selection.get(name, default_selection)
            family, checkpoint = self._selection_parts(selection)
            leaf = self.feature_dir / name / family / checkpoint
            for filename in ("features.pt", "config.json"):
                path = (leaf / filename).resolve()
                if not path.is_relative_to(self.feature_dir):
                    raise ValueError(f"{name} feature path must stay inside the assay directory")
                if not path.is_file():
                    available = ", ".join(self.available_feature_sets[name]) or "none"
                    raise FileNotFoundError(
                        f"Missing {name} cache file: {path}. Available {name} feature sets: {available}"
                    )
            config = json.loads((leaf / "config.json").read_text(encoding="utf-8"))
            expected_config = {"feature_type": name, "model_family": family, "checkpoint": checkpoint}
            if not isinstance(config, Mapping) or any(config.get(key) != value for key, value in expected_config.items()):
                raise ValueError(f"{name} config.json does not match its feature type, model family, or checkpoint")
            store = torch.load(leaf / "features.pt", map_location="cpu", weights_only=True)
            if store.get("synthetic") is not True or store.get("assay_id") != self.assay_id:
                raise ValueError(f"{name} metadata does not match the synthetic assay")
            if config.get("synthetic") is not True:
                raise ValueError(f"{name} config.json must identify the cache as synthetic")
            if store.get("schema_version") != self.assay_metadata["schema_version"]:
                raise ValueError(f"{name} schema version does not match assay.json")
            self.feature_selection[name] = selection
            self.feature_configs[name] = config
            stores[name] = store
        seq, struct, inv = (stores[name] for name in FEATURE_TYPES)
        embeddings = self._tensor(seq["embeddings"], "sequence embeddings", ndim=3)
        length = len(self.reference_sequence)
        if embeddings.shape[1] != length:
            raise ValueError("Sequence residue axis must match reference length, without special tokens or padding")
        ids, hashes = seq["variant_ids"], seq["sequence_hashes"]
        if len(ids) != len(set(ids)) or len(ids) != len(hashes) or len(ids) != embeddings.shape[0]:
            raise ValueError("Invalid sequence variant IDs, hashes, or row count")
        if set(ids) != set(self.variant_ids):
            raise ValueError("Sequence feature IDs do not match variants.csv")
        index = {variant_id: i for i, variant_id in enumerate(ids)}
        order = [index[variant_id] for variant_id in self.variant_ids]
        for row, i in zip(self.rows, order):
            digest = hashlib.sha256(row["mutated_sequence"].encode("utf-8")).hexdigest()
            if digest != hashes[i]:
                raise ValueError(f"Sequence hash mismatch for {row['variant_id']}")
        self.sequence_embeddings = embeddings[order]

        for name, store in (("sequence", seq), ("structure", struct), ("inverse_folding", inv)):
            if store["reference_sequence"] != self.reference_sequence:
                raise ValueError(f"{name} reference sequence mismatch")
            positions = store["positions"]
            if not isinstance(positions, torch.Tensor) or positions.dtype != torch.int64:
                raise ValueError(f"{name} positions must be an int64 tensor")
            if not torch.equal(positions, torch.arange(1, length + 1)):
                raise ValueError(f"{name} positions must be one-based reference positions")
        self.structure_embeddings = self._tensor(struct["embeddings"], "structure embeddings")
        if self.structure_embeddings.shape[0] != length:
            raise ValueError("Structure embedding count does not match reference length")
        if inv["alphabet"] != ALPHABET:
            raise ValueError(f"Expected inverse-folding alphabet {ALPHABET}")
        self.probabilities = self._tensor(inv["probabilities"], "inverse-folding probabilities")
        self.log_probs = self._tensor(inv["log_probs"], "inverse-folding log probabilities")
        if self.probabilities.shape != (length, 20) or self.log_probs.shape != (length, 20):
            raise ValueError("Inverse-folding tensors must have shape [reference length, 20]")
        if (self.probabilities <= 0).any() or not torch.allclose(
            self.probabilities.sum(dim=1), torch.ones(length), atol=1e-5, rtol=1e-5
        ):
            raise ValueError("Inverse-folding probabilities must be positive and sum to one")
        if not torch.allclose(self.log_probs, self.probabilities.log(), atol=1e-5, rtol=1e-5):
            raise ValueError("Inverse-folding log probabilities are inconsistent")
        self.mutations = [self._parse_mutations(row) for row in self.rows]

    @staticmethod
    def _selection_parts(selection):
        if not isinstance(selection, str):
            raise ValueError("Feature selections must be 'model_family/checkpoint' strings")
        parts = selection.split("/")
        if len(parts) != 2 or any(
            part in ("", ".", "..") or "\\" in part or ":" in part
            for part in parts
        ):
            raise ValueError(f"Invalid feature selection {selection!r}; expected 'model_family/checkpoint'")
        return parts

    @staticmethod
    def _tensor(value, name, ndim=2):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ValueError(f"{name} must be a float32 tensor")
        if value.ndim != ndim or min(value.shape) == 0 or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be a nonempty finite {ndim}-dimensional tensor")
        return value

    def _parse_mutations(self, row):
        if row["variant_id"] != row["mutant"]:
            raise ValueError("Synthetic variant_id must equal mutant")
        positions, wildtype, mutant = [], [], []
        reconstructed = list(self.reference_sequence)
        for token in row["mutant"].split(":"):
            match = re.fullmatch(r"([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])", token)
            if match is None:
                raise ValueError(f"Unsupported substitution notation: {token!r}")
            wt, position, mut = match.groups()
            position = int(position)
            if not 1 <= position <= len(reconstructed) or position in positions:
                raise ValueError(f"Invalid or duplicate mutation position: {token}")
            if self.reference_sequence[position - 1] != wt:
                raise ValueError(f"Reference amino acid mismatch: {token}")
            positions.append(position)
            wildtype.append(ALPHABET.index(wt))
            mutant.append(ALPHABET.index(mut))
            reconstructed[position - 1] = mut
        if "".join(reconstructed) != row["mutated_sequence"]:
            raise ValueError(f"Mutation annotation disagrees with sequence: {row['variant_id']}")
        return tuple(torch.tensor(values, dtype=torch.int64) for values in (positions, wildtype, mutant))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        positions, wt, mut = self.mutations[index]
        log_probs = self.log_probs[positions - 1]
        site_indices = torch.arange(len(positions))
        return {
            "variant_id": self.variant_ids[index],
            "mutated_sequence": self.rows[index]["mutated_sequence"],
            "target": self.targets[index],
            "sequence_embedding": self.sequence_embeddings[index],
            "mutation_sequence_embeddings": self.sequence_embeddings[index, positions - 1],
            "mutation_positions": positions,
            "wildtype_aa": wt,
            "mutant_aa": mut,
            "structure_embeddings": self.structure_embeddings[positions - 1],
            "inverse_folding_log_probs": log_probs,
            "inverse_folding_probs": self.probabilities[positions - 1],
            "inverse_folding_log_odds": log_probs[site_indices, mut] - log_probs[site_indices, wt],
            "is_synthetic": True,
        }

    def as_feature_matrices(self, sequence_pooling="mean"):
        """Build [variants, features] inputs without changing the residue caches.

        ``sequence_pooling='mean'`` averages all sequence residues;
        ``'mutation_mean'`` averages only each variant's mutated positions.
        Fit any subsequent scaling on training folds only.
        """
        if sequence_pooling not in ("mean", "mutation_mean"):
            raise ValueError("sequence_pooling must be 'mean' or 'mutation_mean'")
        if sequence_pooling == "mean":
            sequence = self.sequence_embeddings.mean(dim=1)
        else:
            sequence = torch.stack([
                self.sequence_embeddings[i, positions - 1].mean(dim=0)
                for i, (positions, _, _) in enumerate(self.mutations)
            ])
        structural, inverse = [], []
        for sample in self:
            wt, mut = sample["wildtype_aa"], sample["mutant_aa"]
            identities = torch.cat([torch.nn.functional.one_hot(aa, 20) for aa in (wt, mut)], dim=1).float()
            structural.append(torch.cat((sample["structure_embeddings"], identities), dim=1).mean(dim=0))
            lp = sample["inverse_folding_log_probs"]
            sites = torch.arange(len(wt))
            mutation_scores = torch.stack((lp[sites, wt], lp[sites, mut], sample["inverse_folding_log_odds"]), dim=1)
            inverse.append(torch.cat((lp, mutation_scores), dim=1).mean(dim=0))
        return {
            "sequence": sequence,
            "structure": torch.stack(structural),
            "inverse_folding": torch.stack(inverse),
            "targets": self.targets,
            "variant_ids": list(self.variant_ids),
        }
