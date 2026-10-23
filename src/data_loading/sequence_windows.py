"""Window coordinates and the compact ESM2 mutation-site cache contract."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor


COMPACT_SITE_FIELDS = (
    "site_embeddings", "site_variant_indices", "site_positions",
    "site_window_starts", "site_window_ends",
)
_COMPACT_METADATA = {
    "cache_layout": "mutation_sites_v1",
    "site_context": "mutated_sequence",
    "window_policy": "centered_clamped",
    "window_indexing": "zero_based_half_open",
}
_GLOBAL_WINDOW_METADATA = {
    "global_window": "non_overlapping",
    "global_window_aggregation": "length_weighted_mean",
}


def centered_window_bounds(position: int, reference_length: int, max_length: int) -> tuple[int, int]:
    """Map a one-based target to a clamped, zero-based half-open window.

    An even window has one more preceding than following residue at its center.
    Short references use their entire sequence.
    """
    if type(reference_length) is not int or reference_length < 1:
        raise ValueError("reference_length must be a positive integer")
    if type(max_length) is not int or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    if type(position) is not int or not 1 <= position <= reference_length:
        raise ValueError("Mutation position must be within the one-based reference")
    width = min(reference_length, max_length)
    start = max(0, min(position - 1 - width // 2, reference_length - width))
    return start, start + width


def mean_sequence_recipe(config: Mapping, reference_length: int | None = None) -> dict | None:
    """Validate a prefix mean or full mean over fixed nonoverlapping windows.

    Legacy prefix recipes omit global-window fields. New recipes must declare
    both the window policy and its aggregation so partial metadata cannot fall
    back to prefix pooling or a full-residue cache.
    """
    windowed = any(key in config for key in _GLOBAL_WINDOW_METADATA)
    if not windowed and config.get("pooling") != "mean" and config.get("truncation") != "prefix":
        return None
    if (
        config.get("model_family") not in ("esm2", "msa_transformer")
        or config.get("pooling") != "mean"
        or config.get("includes_special_tokens") is not False
    ):
        raise ValueError("Mean sequence caches require ESM2/MSA Transformer residue-mean pooling without special tokens")
    if windowed:
        if any(config.get(key) != value for key, value in _GLOBAL_WINDOW_METADATA.items()) or config.get("truncation") != "none":
            raise ValueError("Full sequence means require non_overlapping windows, length_weighted_mean aggregation, and no truncation")
    elif config.get("truncation") != "prefix":
        raise ValueError("Legacy sequence means require prefix truncation")
    max_length, size = config.get("max_length"), config.get("hidden_size")
    if type(max_length) is not int or not 1 <= max_length <= 1022:
        raise ValueError("Mean sequence max_length must be an integer between 1 and 1022")
    if type(size) is not int or size < 1:
        raise ValueError("Mean sequence configuration requires a positive integer hidden_size")
    recipe = {"pooling": "mean", "truncation": "none" if windowed else "prefix", "max_length": max_length}
    if windowed:
        recipe.update(_GLOBAL_WINDOW_METADATA)
    if reference_length is not None:
        if type(reference_length) is not int or reference_length < 1:
            raise ValueError("reference_length must be a positive integer")
        recipe["represented_length"] = reference_length if windowed else min(reference_length, max_length)
    return recipe


def validate_global_window_payload(store: Mapping, recipe: Mapping | None) -> None:
    """Reject undeclared or inconsistent global policy even for short proteins."""
    for key in _GLOBAL_WINDOW_METADATA:
        if recipe is not None and key in recipe:
            if type(store.get(key)) is not type(recipe[key]) or store.get(key) != recipe[key]:
                raise ValueError("Sequence payload does not match its global-window recipe")
        elif key in store:
            raise ValueError("Sequence payload has undeclared global-window metadata")


def compact_sequence_recipe(config: Mapping, reference_length: int | None = None) -> dict | None:
    """Recognize a centered-site ESM2 cache with a separately declared mean.

    Any partial compact declaration is invalid. The config-only string
    ``site_embeddings='centered'`` is not copied into the payload, where that
    name identifies the actual packed embedding tensor.
    """
    declared = any(key in config for key in _COMPACT_METADATA) or config.get("site_embeddings", "none") != "none"
    if not declared:
        return None
    if (
        config.get("model_family") not in ("esm2", "msa_transformer")
        or config.get("site_embeddings") != "centered"
        or any(config.get(key) != value for key, value in _COMPACT_METADATA.items())
    ):
        raise ValueError("Incomplete or inconsistent compact centered-site sequence recipe")
    mean_recipe = mean_sequence_recipe(config, reference_length)
    if mean_recipe is None:
        raise ValueError("Compact centered-site caches require a mean sequence recipe")
    return {**_COMPACT_METADATA, **mean_recipe}


def tensor_isfinite(value: Tensor, chunk_elements: int = 1024 * 1024) -> bool:
    """Check finite values without a temporary Boolean tensor the cache's size."""
    if value.ndim == 0:
        return bool(torch.isfinite(value))
    row_elements = max(1, value[0].numel()) if value.shape[0] else 1
    chunk_rows = max(1, chunk_elements // row_elements)
    return all(bool(torch.isfinite(value[start:start + chunk_rows]).all())
               for start in range(0, value.shape[0], chunk_rows))


def validate_compact_sequence_payload(
    store: Mapping,
    config: Mapping,
    reference_length: int,
    mutation_positions: Sequence[Sequence[int]],
) -> tuple[Tensor, Tensor]:
    """Validate globals and packed sites in cache-variant/annotation order.

    Callers validate full sequence identities and supply each cached variant's
    annotated mutation positions in that same cache order. WT has zero sites.
    """
    recipe = compact_sequence_recipe(config, reference_length)
    if recipe is None:
        raise ValueError("Compact site tensors require an explicit centered-site recipe")
    validate_global_window_payload(store, recipe)
    if any(type(store.get(key)) is not type(value) or store.get(key) != value for key, value in recipe.items()):
        raise ValueError("Compact sequence payload does not match its recipe or represented length")
    count, size = len(mutation_positions), config["hidden_size"]
    counts = torch.tensor([len(positions) for positions in mutation_positions], dtype=torch.long)
    num_sites = int(counts.sum())
    globals_, sites = store.get("embeddings"), store.get("site_embeddings")
    for name, value, shape in (
        ("global", globals_, (count, size)), ("site", sites, (num_sites, size)),
    ):
        if not isinstance(value, Tensor) or value.dtype != torch.float32 or tuple(value.shape) != shape:
            raise ValueError(f"Compact {name} embeddings must have float32 shape {shape}")
        if not tensor_isfinite(value):
            raise ValueError(f"Compact {name} embeddings must contain only finite values")
    for name in COMPACT_SITE_FIELDS[1:]:
        value = store.get(name)
        if not isinstance(value, Tensor) or value.dtype != torch.long or tuple(value.shape) != (num_sites,):
            raise ValueError(f"Compact {name} must be an int64 vector matching mutation occurrences")
    expected_owners = torch.repeat_interleave(torch.arange(count), counts)
    expected_positions = torch.tensor([position for positions in mutation_positions for position in positions], dtype=torch.long)
    if not torch.equal(store["site_variant_indices"], expected_owners) or not torch.equal(store["site_positions"], expected_positions):
        raise ValueError("Compact sites must cover exactly the annotated mutations in cache variant and annotation order")
    width = min(reference_length, recipe["max_length"])
    starts = (expected_positions - 1 - width // 2).clamp(min=0, max=reference_length - width)
    if not torch.equal(store["site_window_starts"], starts) or not torch.equal(store["site_window_ends"], starts + width):
        raise ValueError("Compact site window boundaries do not match the centered clamped recipe")
    return globals_, sites
