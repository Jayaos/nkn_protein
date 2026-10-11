"""Hierarchical protein covariance with mutation-set aggregation."""

from __future__ import annotations

import math
import json

import torch
from torch import Tensor, nn

from config import BRANCH_NAMES, validate_enabled_branches
from data_loading.features import kermut_probability_recipe
from .nkn import NeuralKernelNetwork, PrimitiveNKN
from .primitives import IF_KERNELS, VECTOR_KERNELS, initial_lengthscale
from .triangle import TrainingTriangle


LOCAL_BRANCHES = ("inverse_folding", "structure", "sequence_site")
BRANCHES = BRANCH_NAMES


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
    """Selected local/global representation branches followed by a final NKN.

    ``train``, ``left`` and ``right`` follow the FeatureBatch interface. The kernel
    does not own feature generation or fit transformations using held-out inputs.
    """

    def __init__(self, config, train, lengthscale_multiplier: float = 1.0):
        super().__init__()
        if not math.isfinite(lengthscale_multiplier) or lengthscale_multiplier <= 0:
            raise ValueError("The initial lengthscale multiplier must be finite and positive")
        # Older configurations without a selection retain the original four branches.
        self.enabled_branches = validate_enabled_branches(getattr(config, "enabled_branches", BRANCHES))
        missing = set(self.enabled_branches) - set(config.branches)
        if missing:
            raise ValueError(f"Missing enabled branch configurations: {sorted(missing)}")
        self.inverse_folding_recipe = (
            self._probability_recipe(train) if "inverse_folding" in self.enabled_branches else None
        )
        self.inverse_folding_policy = "sequence_conditional" if self.inverse_folding_recipe is not None else "normalized"
        self._validate_batch(train)
        self.feature_dimensions = {
            name: self._features(train, name).shape[1] for name in self.enabled_branches
        }
        branches = {}
        for name in self.enabled_branches:
            branch_config = config.branches[name]
            allowed = IF_KERNELS if name == "inverse_folding" else VECTOR_KERNELS
            if set(branch_config.copies) - allowed:
                raise ValueError(f"Invalid primitive kernels for {name}: {set(branch_config.copies) - allowed}")
            features = self._features(train, name)
            policy = self.inverse_folding_policy if name == "inverse_folding" else "normalized"
            scale = initial_lengthscale(features, hellinger=name == "inverse_folding", probability_policy=policy)
            branches[name] = PrimitiveNKN(
                branch_config.copies, branch_config.widths, scale * lengthscale_multiplier,
                probability_policy=policy,
            )
        self.branches = nn.ModuleDict(branches)
        self.final_nkn = NeuralKernelNetwork(len(self.enabled_branches), config.final_widths)
        # Runtime geometry only: excluded from parameters, buffers and checkpoints.
        self._training_triangle: TrainingTriangle | None = None
        self.to(device=train.owners.device, dtype=torch.float64)

    @staticmethod
    def _probability_recipe(batch):
        encoded = getattr(batch, "feature_signature", {}).get("inverse_folding.probability_recipe")
        if encoded is None:
            return None
        if not isinstance(encoded, str):
            raise ValueError("Invalid inverse-folding probability recipe signature")
        metadata = json.loads(encoded)
        if not isinstance(metadata, dict):
            raise ValueError("Invalid inverse-folding probability recipe signature")
        recipe = kermut_probability_recipe(metadata)
        if recipe is None:
            raise ValueError("Unrecognized inverse-folding probability recipe signature")
        return recipe

    @staticmethod
    def _features(batch, name):
        features = (
            getattr(batch, "global_features", None) if name == "sequence_global"
            else getattr(batch, "local_features", {}).get(name)
        )
        if features is None:
            raise ValueError(f"Missing features for enabled branch: {name}")
        return features

    def _validate_batch(self, batch) -> None:
        if "inverse_folding" in self.enabled_branches and self._probability_recipe(batch) != self.inverse_folding_recipe:
            raise ValueError("Inverse-folding probability recipe differs from its training recipe")
        if isinstance(batch.num_variants, bool) or not isinstance(batch.num_variants, int) or batch.num_variants < 0:
            raise ValueError("FeatureBatch.num_variants must be a nonnegative integer")
        if batch.owners.ndim != 1 or batch.owners.dtype != torch.long:
            raise ValueError("Mutation owners must be a one-dimensional long tensor")
        device = batch.owners.device
        if (
            batch.mutation_positions.ndim != 1
            or batch.mutation_positions.dtype != torch.long
            or batch.mutation_positions.shape != batch.owners.shape
            or batch.mutation_positions.device != device
        ):
            raise ValueError("Mutation positions must align with the mutation owners")
        if ((batch.owners < 0) | (batch.owners >= batch.num_variants)).any():
            raise ValueError("Mutation owners must index variants in their own batch")
        for name in self.enabled_branches:
            features = self._features(batch, name)
            if not isinstance(features, Tensor) or features.ndim != 2 or features.shape[1] == 0 or not features.is_floating_point():
                raise ValueError(f"Features for {name} must be floating-point matrices with nonempty feature dimensions")
            if features.device != device:
                raise ValueError("All enabled feature arrays and mutation owners must use the same device")
            expected_rows = batch.num_variants if name == "sequence_global" else batch.owners.numel()
            if features.shape[0] != expected_rows:
                raise ValueError(f"Feature rows for {name} must align with " + (
                    "variants" if name == "sequence_global" else "mutation occurrences"
                ))

    def prepare_training_geometry(self, train, *, use_triangle: bool = True) -> None:
        """Cache fixed geometry and symmetric NKN pairs for immutable training inputs.

        ``use_triangle=False`` retains dense NKN evaluation for numerical checks.
        Different or modified inputs continue through the ordinary dense path.
        """
        self.clear_geometry_cache()
        self._validate_batch(train)
        try:
            for name in self.enabled_branches:
                features = self._features(train, name)
                if features.shape[1] != self.feature_dimensions[name]:
                    raise ValueError(f"Feature dimension for {name} differs from its training dimension")
                self.branches[name].prepare_geometry(features)
            if use_triangle:
                self._training_triangle = TrainingTriangle.prepare(self, train)
        except Exception:
            self.clear_geometry_cache()
            raise

    def clear_geometry_cache(self) -> None:
        self._training_triangle = None
        for branch in self.branches.values():
            branch.clear_geometry_cache()

    def _apply(self, fn, recurse=True):
        # Packed constants are deliberately not buffers and must not survive a
        # device/dtype conversion, even when it happens to be a no-op.
        self.clear_geometry_cache()
        return super()._apply(fn, recurse=recurse)

    def _forward_triangle(self, left, right) -> Tensor:
        geometry = self._training_triangle
        layout = geometry.layout
        covariance_parts = []
        for name in self.enabled_branches:
            branch = self.branches[name]
            prepared = geometry.branches[name]
            primitives = torch.stack(
                [primitive.forward_from_geometry(prepared.packed_geometry)
                 for primitive in branch.primitives], dim=-1,
            )
            packed = branch.nkn(primitives)
            if name != "sequence_global":
                # The local NKN must precede mutation aggregation. Expand here
                # to preserve its summation order for multi-mutation variants.
                covariance = aggregate_mutations(
                    prepared.layout.expand(packed), left.owners, right.owners,
                    left.num_variants, right.num_variants,
                )
                packed = covariance[layout.rows, layout.columns]
            covariance_parts.append(packed)
        return layout.expand(self.final_nkn(torch.stack(covariance_parts, dim=-1)))

    def forward(self, left, right=None) -> Tensor:
        right = left if right is None else right
        self._validate_batch(left)
        self._validate_batch(right)
        if left.owners.device != right.owners.device:
            raise ValueError("Both feature batches must use the same device")
        if self._training_triangle is not None and self._training_triangle.matches(self, left, right):
            return self._forward_triangle(left, right)
        covariance_parts = []
        for name in self.enabled_branches:
            left_features = self._features(left, name)
            right_features = self._features(right, name)
            if left_features.shape[1] != self.feature_dimensions[name] or right_features.shape[1] != self.feature_dimensions[name]:
                raise ValueError(f"Feature dimension for {name} differs from its training dimension")
            local_covariance = self.branches[name](left_features, right_features)
            if name != "sequence_global":
                local_covariance = aggregate_mutations(
                    local_covariance, left.owners, right.owners, left.num_variants, right.num_variants
                )
            covariance_parts.append(local_covariance)
        return self.final_nkn(torch.stack(covariance_parts, dim=-1))
