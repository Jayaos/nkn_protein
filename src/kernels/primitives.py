"""Unit-amplitude covariance functions for fixed protein representations."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DISTANCE_KERNELS = frozenset({"hellinger_exp", "hellinger_rbf", "rbf", "matern32"})
IF_KERNELS = frozenset({"hellinger_exp", "hellinger_rbf", "bhattacharyya"})
VECTOR_KERNELS = frozenset({"cosine", "rbf", "matern32"})


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


def _validate_probabilities(values: Tensor) -> None:
    if values.shape[1] != 20:
        raise ValueError("Inverse-folding probabilities must have 20 amino-acid columns")
    if (values < 0).any() or not torch.allclose(
        values.sum(dim=1), values.new_ones(values.shape[0]), atol=1e-6, rtol=1e-6
    ):
        raise ValueError("Inverse-folding probabilities must be nonnegative and sum to one")


def initial_lengthscale(inputs: Tensor, *, hellinger: bool = False) -> Tensor:
    """Median positive distance, keeping repeated mutation occurrences in the sample."""
    inputs, _ = _validate_pair(inputs, inputs)
    if hellinger:
        _validate_probabilities(inputs)
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


class PrimitiveKernel(nn.Module):
    """A fixed-amplitude primitive with an independent scalar lengthscale, if needed."""

    def __init__(self, name: str, initial_lengthscale: float | Tensor = 1.0):
        super().__init__()
        if name not in IF_KERNELS | VECTOR_KERNELS:
            raise ValueError(f"Unknown primitive kernel: {name}")
        self.name = name
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

    def forward(self, left: Tensor, right: Tensor | None = None) -> Tensor:
        right = left if right is None else right
        left, right = _validate_pair(left, right)
        if self.raw_lengthscale is not None and self.raw_lengthscale.device != left.device:
            raise ValueError("Move the kernel and its features to the same device")
        if self.name in IF_KERNELS:
            _validate_probabilities(left)
            _validate_probabilities(right)
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
        if self.name == "rbf":
            return torch.exp(-0.5 * scaled.square())
        scaled = math.sqrt(3.0) * scaled
        return (1 + scaled) * torch.exp(-scaled)
