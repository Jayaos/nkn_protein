"""Hierarchical protein covariance with mutation-set aggregation."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .nkn import NeuralKernelNetwork, PrimitiveNKN
from .primitives import IF_KERNELS, VECTOR_KERNELS, initial_lengthscale


LOCAL_BRANCHES = ("inverse_folding", "structure", "sequence_site")
BRANCHES = (*LOCAL_BRANCHES, "sequence_global")


def aggregate_mutations(
    local_covariance: Tensor,
    left_owners: Tensor,
    right_owners: Tensor,
    left_count: int,
    right_count: int,
) -> Tensor:
    """Sum every local pair into its pair of variants, retaining autograd."""
    if local_covariance.shape != (left_owners.numel(), right_owners.numel()):
        raise ValueError("Local covariance dimensions must match mutation occurrence counts")
    by_left = local_covariance.new_zeros((left_count, right_owners.numel())).index_add(
        0, left_owners, local_covariance
    )
    return local_covariance.new_zeros((left_count, right_count)).index_add(1, right_owners, by_left)


class ProteinKernel(nn.Module):
    """Four local/global representation branches followed by a final NKN.

    ``train``, ``left`` and ``right`` follow the FeatureBatch interface. The kernel
    does not own feature generation or fit transformations using held-out inputs.
    """

    def __init__(self, config, train, lengthscale_multiplier: float = 1.0):
        super().__init__()
        if not math.isfinite(lengthscale_multiplier) or lengthscale_multiplier <= 0:
            raise ValueError("The initial lengthscale multiplier must be finite and positive")
        if set(config.branches) != set(BRANCHES):
            raise ValueError(f"Model branches must be exactly {BRANCHES}")
        self._validate_batch(train)
        self.feature_dimensions = {
            name: (train.global_features if name == "sequence_global" else train.local_features[name]).shape[1]
            for name in BRANCHES
        }
        branches = {}
        for name in BRANCHES:
            branch_config = config.branches[name]
            allowed = IF_KERNELS if name == "inverse_folding" else VECTOR_KERNELS
            if set(branch_config.copies) - allowed:
                raise ValueError(f"Invalid primitive kernels for {name}: {set(branch_config.copies) - allowed}")
            features = train.global_features if name == "sequence_global" else train.local_features[name]
            scale = initial_lengthscale(features, hellinger=name == "inverse_folding")
            branches[name] = PrimitiveNKN(
                branch_config.copies, branch_config.widths, scale * lengthscale_multiplier
            )
        self.branches = nn.ModuleDict(branches)
        self.final_nkn = NeuralKernelNetwork(len(BRANCHES), config.final_widths)
        self.to(device=train.global_features.device, dtype=torch.float64)

    @staticmethod
    def _validate_batch(batch) -> None:
        if isinstance(batch.num_variants, bool) or not isinstance(batch.num_variants, int) or batch.num_variants < 0:
            raise ValueError("FeatureBatch.num_variants must be a nonnegative integer")
        if batch.global_features.ndim != 2 or batch.global_features.shape[0] != batch.num_variants:
            raise ValueError("Global features must contain one row per variant")
        for name in LOCAL_BRANCHES:
            if name not in batch.local_features:
                raise ValueError(f"Missing local features: {name}")
        device = batch.global_features.device
        if batch.owners.ndim != 1 or batch.owners.dtype != torch.long or batch.owners.device != device:
            raise ValueError("Mutation owners must be a one-dimensional long tensor on the feature device")
        if (
            batch.mutation_positions.ndim != 1
            or batch.mutation_positions.dtype != torch.long
            or batch.mutation_positions.shape != batch.owners.shape
            or batch.mutation_positions.device != device
        ):
            raise ValueError("Mutation positions must align with the mutation owners")
        if ((batch.owners < 0) | (batch.owners >= batch.num_variants)).any():
            raise ValueError("Mutation owners must index variants in their own batch")
        for features in [batch.global_features, *(batch.local_features[name] for name in LOCAL_BRANCHES)]:
            if features.ndim != 2 or features.shape[1] == 0 or not features.is_floating_point():
                raise ValueError("Feature arrays must be floating-point matrices with nonempty feature dimensions")
            if features.device != device:
                raise ValueError("All feature arrays must use the same device")
        if any(batch.local_features[name].shape[0] != batch.owners.numel() for name in LOCAL_BRANCHES):
            raise ValueError("Local feature rows must align with mutation occurrences")

    def forward(self, left, right=None) -> Tensor:
        right = left if right is None else right
        self._validate_batch(left)
        self._validate_batch(right)
        if left.global_features.device != right.global_features.device:
            raise ValueError("Both feature batches must use the same device")
        covariance_parts = []
        for name in BRANCHES:
            left_features = left.global_features if name == "sequence_global" else left.local_features[name]
            right_features = right.global_features if name == "sequence_global" else right.local_features[name]
            if left_features.shape[1] != self.feature_dimensions[name] or right_features.shape[1] != self.feature_dimensions[name]:
                raise ValueError(f"Feature dimension for {name} differs from its training dimension")
            local_covariance = self.branches[name](left_features, right_features)
            if name != "sequence_global":
                local_covariance = aggregate_mutations(
                    local_covariance, left.owners, right.owners, left.num_variants, right.num_variants
                )
            covariance_parts.append(local_covariance)
        return self.final_nkn(torch.stack(covariance_parts, dim=-1))
