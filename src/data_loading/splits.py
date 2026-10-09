"""Official fold ingestion and explicitly developmental split construction."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .features import FeatureBatch, read_csv


def _validate_folds(folds: Tensor, n_folds: int) -> Tensor:
    if n_folds < 2:
        raise ValueError("At least two folds are required")
    if set(folds.cpu().tolist()) != set(range(n_folds)):
        raise ValueError(f"Fold assignments must contain every fold numbered 0 through {n_folds - 1}")
    return folds


def load_folds(path: str | Path, variant_ids: Sequence[str], scheme: str, n_folds: int = 5) -> Tensor:
    """Read an ID-aligned scheme column; never infer official folds from row order."""
    rows = read_csv(path)
    id_column = "variant_id" if "variant_id" in rows[0] else "mutant"
    if id_column not in rows[0]:
        raise ValueError("Fold CSV requires variant_id or mutant")
    aliases = {"random": "fold_random_5", "modulo": "fold_modulo_5", "contiguous": "fold_contiguous_5"}
    candidates = (scheme, aliases.get(scheme, scheme), "fold")
    column = next((candidate for candidate in candidates if candidate in rows[0]), None)
    if column is None:
        raise ValueError(f"Fold CSV does not contain the requested scheme: {scheme}")
    if len(set(variant_ids)) != len(variant_ids):
        raise ValueError("Requested variant IDs must be unique")
    table: dict[str, int] = {}
    for row in rows:
        variant_id = row[id_column]
        if not variant_id or variant_id in table:
            raise ValueError("Fold CSV contains empty or duplicate variant IDs")
        try:
            table[variant_id] = int(row[column])
        except (TypeError, ValueError) as error:
            raise ValueError("Fold identifiers must be integers") from error
    if set(table) != set(variant_ids):
        raise ValueError("Fold CSV IDs must cover exactly the selected variants")
    return _validate_folds(torch.tensor([table[value] for value in variant_ids], dtype=torch.long), n_folds)


def development_folds(batch: FeatureBatch, scheme: str, n_folds: int = 5, seed: int = 0) -> Tensor:
    """Construct development splits; these are not ProteinGym benchmark folds.

    Positional splits require exactly one mutation per variant. Modulo cycles
    sorted observed positions; contiguous divides those positions into groups.
    """
    if n_folds < 2 or batch.num_variants < n_folds:
        raise ValueError("Development splits require at least two nonempty folds")
    scheme = scheme.removeprefix("development_")
    if scheme == "random":
        order = torch.randperm(batch.num_variants, generator=torch.Generator().manual_seed(seed))
        folds = torch.empty(batch.num_variants, dtype=torch.long)
        folds[order] = torch.arange(batch.num_variants) % n_folds
    elif scheme in ("modulo", "contiguous"):
        counts = torch.bincount(batch.owners.cpu(), minlength=batch.num_variants)
        if not torch.equal(counts, torch.ones(batch.num_variants, dtype=torch.long)):
            raise ValueError("Positional development splits support single substitutions only")
        positions = torch.empty(batch.num_variants, dtype=torch.long)
        positions[batch.owners.cpu()] = batch.mutation_positions.cpu()
        unique, ranks = torch.unique(positions, sorted=True, return_inverse=True)
        if len(unique) < n_folds:
            raise ValueError("Too few distinct mutation positions for nonempty positional folds")
        if scheme == "modulo":
            assignments = torch.arange(len(unique)) % n_folds
        else:
            assignments = torch.empty(len(unique), dtype=torch.long)
            for fold, group in enumerate(torch.tensor_split(torch.arange(len(unique)), n_folds)):
                assignments[group] = fold
        folds = assignments[ranks]
    else:
        raise ValueError(f"Unknown development split scheme: {scheme}")
    return _validate_folds(folds, n_folds)
