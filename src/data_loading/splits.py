"""ID-aligned ingestion and validation of supplied fold assignments."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .features import read_csv


def _validate_folds(folds: Tensor, n_folds: int) -> Tensor:
    if n_folds < 2:
        raise ValueError("At least two folds are required")
    if set(folds.cpu().tolist()) != set(range(n_folds)):
        raise ValueError(f"Fold assignments must contain every fold numbered 0 through {n_folds - 1}")
    return folds


def load_folds(
    path: str | Path,
    variant_ids: Sequence[str],
    scheme: str,
    n_folds: int = 5,
    *,
    allowed_extra_ids: Sequence[str] = (),
) -> Tensor:
    """Read ID-aligned published folds without silently accepting missing variants.

    Extra rows are allowed only for explicitly excluded, known cache IDs (for
    example higher-order mutants removed by a 1M/2M setting).
    """
    rows = read_csv(path)
    id_column = "variant_id" if "variant_id" in rows[0] else "mutant"
    if id_column not in rows[0]:
        raise ValueError("Fold CSV requires variant_id or mutant")
    aliases = {
        "random": "fold_random_5", "modulo": "fold_modulo_5",
        "contiguous": "fold_contiguous_5", "random_multiples": "fold_rand_multiples",
    }
    candidates = tuple(dict.fromkeys((aliases.get(scheme, scheme), scheme, "fold")))
    column = next((candidate for candidate in candidates if candidate in rows[0]), None)
    if column is None:
        raise ValueError(f"Fold CSV does not contain the requested scheme: {scheme}")
    # A generic 'fold' can coexist with published columns for other schemes.
    # Two explicit names for this scheme must agree, never silently override.
    explicit_columns = [name for name in candidates if name != "fold" and name in rows[0]]
    if len(set(variant_ids)) != len(variant_ids):
        raise ValueError("Requested variant IDs must be unique")
    table: dict[str, int] = {}
    for row in rows:
        variant_id = row[id_column]
        if not variant_id or variant_id in table:
            raise ValueError("Fold CSV contains empty or duplicate variant IDs")
        try:
            table[variant_id] = int(row[column])
            if any(int(row[name]) != table[variant_id] for name in explicit_columns):
                raise ValueError("Conflicting fold columns for the requested scheme")
        except (TypeError, ValueError) as error:
            raise ValueError(f"Fold identifiers must be integers and explicit aliases must agree: {error}") from error
    selected = set(variant_ids)
    if not selected.issubset(table) or set(table) - selected - set(allowed_extra_ids):
        raise ValueError("Fold CSV IDs must cover exactly the selected variants")
    return _validate_folds(torch.tensor([table[value] for value in variant_ids], dtype=torch.long), n_folds)
