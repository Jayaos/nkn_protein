"""Covariance functions for fixed vectors and declared amino-acid probabilities.

Normalized probabilities give unit-amplitude primitives. For raw conditional
weights, the Bhattacharyya diagonal retains each row's probability mass.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DISTANCE_KERNELS = frozenset({"hellinger_exp", "hellinger_rbf", "exp", "rbf", "matern32"})
IF_KERNELS = frozenset({"hellinger_exp", "hellinger_rbf", "bhattacharyya"})
VECTOR_KERNELS = frozenset({"cosine", "rbf", "matern32"})
STRUCTURE_DISTANCE_KERNELS = frozenset({"exp", "rbf", "matern32"})


def inverse_softplus(value: Tensor) -> Tensor:
    """Stable inverse for strictly positive, constrained parameter values."""
    if not torch.isfinite(value).all() or (value <= 0).any():
        raise ValueError("Softplus initialization values must be finite and positive")
    return value + torch.log(-torch.expm1(-value))


def _validate_pair(left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("Kernel inputs must be matrices with matching feature dimensions")
    if left.shape[1] == 0 or not left.is_floating_point() or not right.is_floating_point():
        raise ValueError("Kernel inputs must have nonempty floating-point feature dimensions")
    if left.device != right.device:
        raise ValueError("Both kernel inputs must be on the same device")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("Kernel inputs must be finite")
    return left.to(dtype=torch.float64), right.to(dtype=torch.float64)


def _validate_probability_policy(policy: str) -> None:
    if policy not in ("normalized", "sequence_conditional"):
        raise ValueError("probability_policy must be 'normalized' or 'sequence_conditional'")


def _validate_probabilities(values: Tensor, probability_policy: str = "normalized") -> None:
    if values.shape[1] != 20:
        raise ValueError("Inverse-folding probabilities must have 20 amino-acid columns")
    if probability_policy == "sequence_conditional":
        if (values <= 0).any() or (values.sum(dim=1) > 1 + 2e-5).any():
            raise ValueError("Conditional inverse-folding probabilities must be positive with row mass at most one")
        return
    if (values < 0).any() or not torch.allclose(
        values.sum(dim=1), values.new_ones(values.shape[0]), atol=1e-6, rtol=1e-6
    ):
        raise ValueError("Inverse-folding probabilities must be nonnegative and sum to one")


def initial_lengthscale(
    inputs: Tensor, *, hellinger: bool = False, probability_policy: str = "normalized",
) -> Tensor:
    """Median positive distance, keeping repeated mutation occurrences in the sample."""
    _validate_probability_policy(probability_policy)
    inputs, _ = _validate_pair(inputs, inputs)
    if hellinger:
        _validate_probabilities(inputs, probability_policy)
        inputs = inputs.sqrt() / math.sqrt(2.0)
    with torch.no_grad():
        distances = torch.pdist(inputs, p=2)
        positive = distances[distances > 0]
        if positive.numel() == 0:
            return inputs.new_tensor(1.0)
        # torch.median returns the lower middle element for even-sized collections.
        # The specified statistical median averages the two middle elements instead.
        ordered = positive.sort().values
        middle = ordered.numel() // 2
        if ordered.numel() % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2


@dataclass(frozen=True)
class _TensorSnapshot:
    source: weakref.ReferenceType[Tensor]
    version: int
    metadata: tuple

    @staticmethod
    def _metadata(values: Tensor) -> tuple:
        return values.device, values.dtype, values.shape, values.stride(), values.data_ptr()

    @classmethod
    def capture(cls, values: Tensor) -> _TensorSnapshot | None:
        if values.requires_grad:
            return None
        try:
            version = values._version
        except RuntimeError:
            # Inference tensors have no mutation counter, so reuse is unsafe.
            return None
        return cls(weakref.ref(values), version, cls._metadata(values))

    def matches(self, values: Tensor) -> bool:
        if self.source() is not values or values.requires_grad:
            return False
        try:
            return values._version == self.version and self._metadata(values) == self.metadata
        except RuntimeError:
            return False


@dataclass
class FixedPairwiseGeometry:
    """Parameter-free quantities for one explicitly prepared, frozen input pair.

    Source tensors must remain immutable. Normal PyTorch in-place updates are
    detected by their version counters; writes through ``.data`` or external
    shared storage are unsupported. No feature-gradient graph is retained.
    """

    left: _TensorSnapshot
    right: _TensorSnapshot
    values: dict[str, Tensor]

    @classmethod
    def prepare(
        cls, left: Tensor, right: Tensor, names: set[str], probability_policy: str,
    ) -> FixedPairwiseGeometry | None:
        # Constants created in inference mode cannot later be saved for the
        # learned lengthscales' backward pass, even with ordinary source inputs.
        if torch.is_inference_mode_enabled():
            return None
        left_snapshot = _TensorSnapshot.capture(left)
        right_snapshot = _TensorSnapshot.capture(right)
        if left_snapshot is None or right_snapshot is None:
            return None
        values = {}
        with torch.no_grad():
            left, right = _validate_pair(left, right)
            if names & IF_KERNELS:
                _validate_probabilities(left, probability_policy)
                _validate_probabilities(right, probability_policy)
                left_root, right_root = left.sqrt(), right.sqrt()
                if "bhattacharyya" in names:
                    values["bhattacharyya"] = left_root @ right_root.T
                if names & {"hellinger_exp", "hellinger_rbf"}:
                    distance = torch.cdist(left_root, right_root, compute_mode="donot_use_mm_for_euclid_dist")
                    values["hellinger_distance"] = distance / math.sqrt(2.0)
            if names & STRUCTURE_DISTANCE_KERNELS:
                values["euclidean_distance"] = torch.cdist(
                    left, right, compute_mode="donot_use_mm_for_euclid_dist",
                )
            if "cosine" in names:
                left_norm = torch.linalg.vector_norm(left, dim=1, keepdim=True)
                right_norm = torch.linalg.vector_norm(right, dim=1, keepdim=True)
                if (left_norm == 0).any() or (right_norm == 0).any():
                    raise ValueError("Cosine kernel inputs must have nonzero norm")
                values["cosine"] = (left / left_norm) @ (right / right_norm).T
        return cls(left_snapshot, right_snapshot, values)

    def matches(self, left: Tensor, right: Tensor) -> bool:
        return self.left.matches(left) and self.right.matches(right)


class PrimitiveKernel(nn.Module):
    """A primitive with an independent scalar lengthscale, if needed.

    Conditional probabilities are used without renormalization. Hellinger
    distances are Euclidean distances between square-root weight vectors;
    Bhattacharyya values are their dot products, with self-covariance equal to
    row mass. Both constructions remain positive semidefinite.
    """

    def __init__(
        self, name: str, initial_lengthscale: float | Tensor = 1.0,
        *, probability_policy: str = "normalized",
    ):
        super().__init__()
        if name not in IF_KERNELS | VECTOR_KERNELS | STRUCTURE_DISTANCE_KERNELS:
            raise ValueError(f"Unknown primitive kernel: {name}")
        _validate_probability_policy(probability_policy)
        self.name = name
        self.probability_policy = probability_policy
        if name in DISTANCE_KERNELS:
            value = torch.as_tensor(initial_lengthscale, dtype=torch.float64).detach().clone()
            if value.numel() != 1:
                raise ValueError("A primitive lengthscale must be scalar")
            self.raw_lengthscale = nn.Parameter(inverse_softplus(value.reshape(())))
        else:
            self.register_parameter("raw_lengthscale", None)

    @property
    def lengthscale(self) -> Tensor | None:
        return None if self.raw_lengthscale is None else F.softplus(self.raw_lengthscale)

    def forward_from_geometry(self, geometry: FixedPairwiseGeometry) -> Tensor:
        """Apply learned parameters to already validated fixed geometry."""
        if self.name in {"cosine", "bhattacharyya"}:
            return geometry.values[self.name]
        distance = geometry.values[
            "hellinger_distance" if self.name in IF_KERNELS else "euclidean_distance"
        ]
        if self.raw_lengthscale.device != distance.device:
            raise ValueError("Move the kernel and its features to the same device")
        scaled = distance / self.lengthscale
        if self.name in {"hellinger_exp", "exp"}:
            return torch.exp(-scaled)
        if self.name in {"hellinger_rbf", "rbf"}:
            return torch.exp(-0.5 * scaled.square())
        scaled = math.sqrt(3.0) * scaled
        return (1 + scaled) * torch.exp(-scaled)

    def forward(self, left: Tensor, right: Tensor | None = None) -> Tensor:
        right = left if right is None else right
        left, right = _validate_pair(left, right)
        if self.raw_lengthscale is not None and self.raw_lengthscale.device != left.device:
            raise ValueError("Move the kernel and its features to the same device")
        if self.name in IF_KERNELS:
            _validate_probabilities(left, self.probability_policy)
            _validate_probabilities(right, self.probability_policy)
            left, right = left.sqrt(), right.sqrt()
            if self.name == "bhattacharyya":
                return left @ right.T
            distance = torch.cdist(left, right, compute_mode="donot_use_mm_for_euclid_dist")
            distance = distance / math.sqrt(2.0)
            scaled = distance / self.lengthscale
            if self.name == "hellinger_exp":
                return torch.exp(-scaled)
            return torch.exp(-0.5 * scaled.square())
        if self.name == "cosine":
            left_norm = torch.linalg.vector_norm(left, dim=1, keepdim=True)
            right_norm = torch.linalg.vector_norm(right, dim=1, keepdim=True)
            if (left_norm == 0).any() or (right_norm == 0).any():
                raise ValueError("Cosine kernel inputs must have nonzero norm")
            return (left / left_norm) @ (right / right_norm).T
        # Compute distance before applying the trainable lengthscale. Zero distances
        # consequently have finite lengthscale gradients, including duplicate sites.
        distance = torch.cdist(left, right, compute_mode="donot_use_mm_for_euclid_dist")
        scaled = distance / self.lengthscale
        if self.name == "exp":
            return torch.exp(-scaled)
        if self.name == "rbf":
            return torch.exp(-0.5 * scaled.square())
        scaled = math.sqrt(3.0) * scaled
        return (1 + scaled) * torch.exp(-scaled)
