"""Shared full-coverage global means and centered mutation-site embeddings."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from data_loading.sequence_windows import centered_window_bounds


@dataclass
class _SiteResult:
    variant_index: int
    global_vector: Tensor
    site_vectors: Tensor
    remaining_windows: int
    global_sum: Tensor | None = None


class WindowedEmbeddings:
    def iter_site_embeddings(
        self, sequences: Sequence[str], mutation_positions: Sequence[Sequence[int]],
    ) -> Iterator[tuple[int, Tensor, Tensor]]:
        """Yield a global mean and centered mutation-site vectors per variant.

        Each input window is taken from the actual variant, without masking.
        Original positions are one based; the shared bounds helper returns
        zero-based, half-open slices. Identical windows within a variant,
        including global windows, are evaluated only once. Tasks stream
        into token-limited batches without allocating an assay residue tensor.
        """
        if self.config.site_embeddings != "centered":
            raise ValueError("iter_site_embeddings requires site_embeddings=centered")
        yield from self._iter_window_embeddings(sequences, mutation_positions)

    def _iter_window_embeddings(
        self, sequences: Sequence[str], mutation_positions: Sequence[Sequence[int]] | None = None,
    ) -> Iterator[tuple[int, Tensor, Tensor]]:
        """Stream global and site tasks without retaining complete residue caches."""
        length = self._validate_sequences(sequences)
        if mutation_positions is not None:
            if isinstance(mutation_positions, (str, bytes)) or len(mutation_positions) != len(sequences):
                raise ValueError("mutation_positions must contain one position collection per variant")
            for positions in mutation_positions:
                if isinstance(positions, (str, bytes)) or not isinstance(positions, Sequence):
                    raise ValueError("Mutation positions must be collections of one-based integers")
                if any(type(position) is not int or not 1 <= position <= length for position in positions):
                    raise ValueError("Mutation positions must be one-based integers within the full sequence")
                if len(set(positions)) != len(positions):
                    raise ValueError("Mutation positions must not contain duplicates")
        self.load()
        window_length = min(length, self.config.max_length)
        if self.config.global_window == "non_overlapping":
            global_windows = tuple(
                (start, min(start + window_length, length)) for start in range(0, length, window_length)
            )
        else:
            global_windows = ((0, window_length),)
        global_bounds = frozenset(global_windows)

        def tasks():
            for index, sequence in enumerate(sequences):
                positions = () if mutation_positions is None else mutation_positions[index]
                windows = {bounds: [] for bounds in global_windows}
                for row, position in enumerate(positions):
                    bounds = centered_window_bounds(position, length, self.config.max_length)
                    windows.setdefault(bounds, []).append((row, position - bounds[0] - 1))
                result = _SiteResult(
                    index, torch.empty(self.hidden_size, dtype=torch.float32),
                    torch.empty((len(positions), self.hidden_size), dtype=torch.float32), len(windows),
                    torch.zeros(self.hidden_size, dtype=torch.float64) if len(global_windows) > 1 else None,
                )
                for (start, end), sites in windows.items():
                    yield sequence[start:end], (start, end) in global_bounds, sites, result, (start, end)

        def batches():
            # A final global chunk can be shorter than the centered site
            # windows. Flush on length changes so neither padding nor stacking
            # introduces extra residues or exceeds the padded token budget.
            batch, batch_length = [], None
            for task in tasks():
                task_length = len(task[0])
                if batch and (
                    task_length != batch_length
                    or (len(batch) + 1) * (task_length + 2) > self.config.max_tokens_per_batch
                ):
                    yield batch
                    batch = []
                batch.append(task)
                batch_length = task_length
            if batch:
                yield batch

        for batch in batches():
            embeddings = self._embed_windows([task[0] for task in batch], [task[4] for task in batch])
            for (_, contributes_global, sites, result, _), values in zip(batch, embeddings):
                if contributes_global:
                    if result.global_sum is None:
                        # Preserve the existing float32 mean for one-window
                        # inputs, including all short proteins.
                        result.global_vector.copy_(values.mean(dim=0))
                    else:
                        result.global_sum.add_(values.sum(dim=0, dtype=torch.float64))
                for row, local_position in sites:
                    result.site_vectors[row].copy_(values[local_position])
                result.remaining_windows -= 1
                if result.remaining_windows == 0:
                    if result.global_sum is not None:
                        result.global_vector.copy_(result.global_sum / length)
                    if not torch.isfinite(result.global_vector).all():
                        raise ValueError("Global embeddings contain nonfinite values")
                    yield result.variant_index, result.global_vector, result.site_vectors
