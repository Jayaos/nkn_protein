"""Read frozen project caches and verified structural coordinates for Kermut."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import torch

from data_loading import FeatureBatch
from data_loading import load_feature_bundle as _load_feature_bundle
from data_loading.features import ALPHABET, kermut_probability_recipe
from feature_generation.structure_fragments import (
    StructureFragments, load_inverse_structure as load_structure,
    validate_inverse_structure_metadata as validate_structure_metadata,
)

from .config import KermutConfig


def validate_batch(batch: FeatureBatch, *, use_structure_distance=True) -> None:
    """Reject incompatible sequence, mutation, probability, or coordinate inputs.

    Local inputs describe the reference structure and therefore must agree for
    every occurrence of the same reference position. WT rows are supported.
    """
    if batch.num_variants < 1:
        raise ValueError("Kermut requires at least one variant")
    if batch.global_features is None:
        raise ValueError("Kermut requires global sequence embeddings")
    if not batch.reference_sequence or set(batch.reference_sequence) - set(ALPHABET):
        raise ValueError("Kermut requires a canonical reference sequence")
    if len(batch.sequences) != batch.num_variants:
        raise ValueError("Kermut requires the full sequence for each variant")
    probabilities = batch.local_features.get("inverse_folding")
    coordinates = batch.local_features.get("structure")
    if probabilities is None or probabilities.shape != (batch.owners.numel(), 20):
        raise ValueError("Kermut requires packed 20-amino-acid inverse-folding probabilities")
    if use_structure_distance and (coordinates is None or coordinates.shape != (batch.owners.numel(), 3)):
        raise ValueError("Kermut requires packed three-dimensional CA coordinates")
    recipe = json.loads(batch.feature_signature.get("inverse_folding.probability_recipe", "{}"))
    if not isinstance(recipe, dict):
        raise ValueError("Invalid inverse-folding probability recipe signature")
    if kermut_probability_recipe(recipe) is not None:
        if (probabilities <= 0).any() or (probabilities.sum(-1) > 1 + 2e-5).any():
            raise ValueError("Kermut ProteinMPNN probabilities must be positive with row mass at most one")
    elif (probabilities < 0).any() or not torch.allclose(
        probabilities.sum(-1), torch.ones_like(probabilities[:, 0]), atol=1e-5, rtol=1e-5
    ):
        raise ValueError("Inverse-folding probabilities must be nonnegative and sum to one")
    owners = batch.owners.detach().cpu().tolist()
    positions = batch.mutation_positions.detach().cpu().tolist()
    mutation_lists = [[] for _ in batch.variant_ids]
    for owner, position in zip(owners, positions):
        mutation_lists[owner].append(position)
    reference = batch.reference_sequence
    for sequence, mutation_list in zip(batch.sequences, mutation_lists):
        if len(sequence) != len(reference) or set(sequence) - set(ALPHABET):
            raise ValueError("Variant sequences must match reference length and use canonical amino acids")
        expected = [i + 1 for i, (ref, aa) in enumerate(zip(reference, sequence)) if ref != aa]
        if sorted(mutation_list) != expected:
            raise ValueError("Packed mutation positions disagree with the variant sequence")
    if positions:
        residues = torch.tensor(
            [ALPHABET.index(batch.sequences[owner][position - 1]) for owner, position in zip(owners, positions)],
            device=probabilities.device,
        )
        actual = probabilities[torch.arange(len(residues), device=probabilities.device), residues]
        if (actual <= 0).any():
            raise ValueError("The probability of each substituted amino acid must be positive")
    for position in batch.mutation_positions.unique():
        rows = batch.mutation_positions == position
        reference_values = [(probabilities, "probabilities")]
        if use_structure_distance:
            reference_values.append((coordinates, "coordinates"))
        for values, name in reference_values:
            selected = values[rows]
            if not torch.equal(selected, selected[0].expand_as(selected)):
                raise ValueError(f"Reference {name} differ across variants at the same mutation position")


def _validate_structure_provenance(root, batch, structure):
    source = batch.feature_signature["inverse_folding"]
    leaf = root / source
    metadata = json.loads((leaf / "config.json").read_text(encoding="utf-8"))
    cached_structure = metadata.get("structure")
    if cached_structure is not None:
        validate_structure_metadata(cached_structure, batch.reference_sequence)
        if cached_structure.get("assay_id") != batch.assay_id:
            raise ValueError("Inverse-folding structure provenance belongs to a different assay")
        if cached_structure["input_signature"] != structure.input_signature:
            raise ValueError("Selected structure differs from the inverse-folding feature structure")
    # Older/synthetic caches may omit structural provenance. When present,
    # check the tensor file's recorded identity as well as the config manifest.
    store = torch.load(leaf / "features.pt", map_location="cpu", weights_only=True)
    provenance = store.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("Inverse-folding provenance must be an object")
    for key, expected in (("structure_sha256", structure.structure_sha256),
                          ("structure_input_signature", structure.input_signature)):
        for recorded in (store, provenance):
            if key in recorded and recorded[key] != expected:
                raise ValueError(f"Inverse-folding tensor provenance disagrees with selected structure: {key}")


def load_feature_bundle(feature_dir: str | Path, config: KermutConfig) -> FeatureBatch:
    """Load selected cached representations/scores and aligned reference CA atoms.

    Relative structure manifest directories resolve from the project root;
    the feature directory follows the command's normal path semantics.
    """
    root = Path(feature_dir).resolve()
    metadata = json.loads((root / "assay.json").read_text(encoding="utf-8"))
    config = config.for_assay(metadata.get("assay_id"))
    batch = _load_feature_bundle(
        root, feature_selection=config.feature_selection, score_sources=config.score_sources,
        pooling=config.pooling, device=config.device, enabled_branches=config.model.enabled_branches,
        inverse_folding_policy="kermut",
    )
    dtype = getattr(torch, config.training.dtype)
    batch = replace(
        batch,
        targets=None if batch.targets is None else batch.targets.to(dtype=dtype),
        scores=batch.scores.to(dtype=dtype), global_features=batch.global_features.to(dtype=dtype),
        local_features={key: values.to(dtype=dtype) for key, values in batch.local_features.items()},
    )
    if not config.model.use_structure_distance:
        batch = replace(batch, feature_signature={**batch.feature_signature, "structure.distance_enabled": "false"})
        validate_batch(batch, use_structure_distance=False)
        return batch
    manifest_root = config.resolved_structure_manifest_dir
    manifest = (manifest_root / f"{batch.assay_id}.json").resolve()
    if manifest.parent != manifest_root:
        raise ValueError("Assay structure manifest path escapes structure_manifest_dir")
    structure = load_structure(manifest, batch.reference_sequence, require_oxygen=False)
    if isinstance(structure, StructureFragments):
        raise ValueError("Separate structure fragments do not define a shared coordinate frame")
    if structure.metadata.get("assay_id") != batch.assay_id:
        raise ValueError("Structure manifest belongs to a different assay")
    _validate_structure_provenance(root, batch, structure)
    coordinates = torch.full((len(batch.reference_sequence), 3), float("nan"), device=batch.device, dtype=dtype)
    coordinates[structure.positions.to(batch.device) - 1] = structure.coords[:, 1].to(device=batch.device, dtype=dtype)
    signature = {
        **batch.feature_signature,
        "structure.input_signature": structure.input_signature,
        "structure.structure_sha256": structure.structure_sha256,
        "structure.coordinates": "reference_CA_angstrom",
    }
    batch = replace(
        batch,
        local_features={
            **batch.local_features,
            "structure": coordinates[batch.mutation_positions - 1],
        },
        feature_signature=signature,
    )
    validate_batch(batch)
    return batch
