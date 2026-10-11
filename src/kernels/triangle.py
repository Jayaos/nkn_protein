"""Packed geometry and index layouts for a frozen training self-covariance.

These are runtime caches, not module parameters or buffers. Full branch
geometry remains available for the ordinary dense path. As with that geometry,
feature writes through ``.data`` or external shared storage are unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass
import weakref

import torch
from torch import Tensor

from .primitives import FixedPairwiseGeometry, _TensorSnapshot


@dataclass(frozen=True)
class TriangleLayout:
    """Upper-pair indices and a differentiable gather back to a dense matrix."""

    size: int
    rows: Tensor
    columns: Tensor
    _dense_take: Tensor

    @classmethod
    def prepare(cls, size: int, device: torch.device) -> TriangleLayout:
        rows, columns = torch.triu_indices(size, size, device=device)
        pair_numbers = torch.arange(rows.numel(), device=device)
        dense_take = torch.empty((size, size), dtype=torch.long, device=device)
        dense_take[rows, columns] = pair_numbers
        dense_take[columns, rows] = pair_numbers
        return cls(size, rows, columns, dense_take.reshape(-1))

    def expand(self, packed: Tensor) -> Tensor:
        # Each diagonal occurs once. Backward sums both sensitivities for each
        # mirrored off-diagonal pair, retaining gradients of live parameters.
        return packed.index_select(0, self._dense_take).reshape(self.size, self.size)


@dataclass(frozen=True)
class _TriangleBranch:
    original_geometry: FixedPairwiseGeometry
    packed_geometry: FixedPairwiseGeometry
    layout: TriangleLayout


@dataclass(frozen=True)
class TrainingTriangle:
    """Prepared upper pairs guarded by batch and immutable tensor identities."""

    _train_ref: weakref.ReferenceType | None
    _train_id: int
    _count: int
    _owners: _TensorSnapshot
    _positions: _TensorSnapshot
    layout: TriangleLayout
    branches: dict[str, _TriangleBranch]

    @classmethod
    def prepare(cls, kernel, train) -> TrainingTriangle | None:
        """Pack existing self-geometry, or skip inputs that cannot be cached.

        The caller validates the batch and prepares the ordinary branch caches
        first. Matching those caches also rejects differentiable features and
        inference tensors. Layouts are shared when branches have equal sizes.
        """
        if torch.is_inference_mode_enabled():
            return None
        owners = _TensorSnapshot.capture(train.owners)
        positions = _TensorSnapshot.capture(train.mutation_positions)
        if owners is None or positions is None:
            return None
        geometries = {}
        sizes = {train.num_variants}
        for name in kernel.enabled_branches:
            features = kernel._features(train, name)
            geometry = kernel.branches[name]._geometry_cache
            if geometry is None or not geometry.matches(features, features):
                return None
            geometries[name] = geometry
            sizes.add(features.shape[0])

        with torch.no_grad():
            layouts = {
                size: TriangleLayout.prepare(size, train.owners.device) for size in sizes
            }
            branches = {}
            for name, geometry in geometries.items():
                layout = layouts[kernel._features(train, name).shape[0]]
                packed = FixedPairwiseGeometry(
                    geometry.left, geometry.right,
                    {key: value[layout.rows, layout.columns].contiguous()
                     for key, value in geometry.values.items()},
                )
                branches[name] = _TriangleBranch(geometry, packed, layout)

        try:
            train_ref = weakref.ref(train)
        except TypeError:
            # Some duck-typed batches (for example SimpleNamespace) cannot be
            # weak-referenced. Their identity plus all tensor snapshots remains
            # sufficient, without keeping source features alive in the cache.
            train_ref = None
        return cls(train_ref, id(train), train.num_variants, owners, positions,
                   layouts[train.num_variants], branches)

    def matches(self, kernel, left, right) -> bool:
        """Require the same immutable training self-pair and original caches."""
        if (left is not right or id(left) != self._train_id
                or left.num_variants != self._count):
            return False
        if self._train_ref is not None and self._train_ref() is not left:
            return False
        if not self._owners.matches(left.owners) or not self._positions.matches(left.mutation_positions):
            return False
        if tuple(kernel.enabled_branches) != tuple(self.branches):
            return False
        for name, prepared in self.branches.items():
            if kernel.branches[name]._geometry_cache is not prepared.original_geometry:
                return False
            features = kernel._features(left, name)
            if not prepared.original_geometry.matches(features, features):
                return False
        return True
