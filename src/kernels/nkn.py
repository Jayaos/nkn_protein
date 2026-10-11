"""Positive affine and adjacent-product composition of valid covariance kernels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .primitives import FixedPairwiseGeometry, PrimitiveKernel, inverse_softplus


class NonnegativeLinear(nn.Module):
    def __init__(self, input_width: int, output_width: int):
        super().__init__()
        if input_width < 1 or output_width < 1:
            raise ValueError("NKN linear widths must be positive")
        weights = torch.exp(torch.randn(output_width, input_width, dtype=torch.float64) * 0.1)
        weights = weights / input_width
        self.raw_weight = nn.Parameter(inverse_softplus(weights))
        self.raw_bias = nn.Parameter(
            inverse_softplus(torch.full((output_width,), 0.01, dtype=torch.float64))
        )

    @property
    def weight(self) -> Tensor:
        return F.softplus(self.raw_weight)

    @property
    def bias(self) -> Tensor:
        return F.softplus(self.raw_bias)

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[-1] != self.raw_weight.shape[1]:
            raise ValueError("NKN input channel count does not match the linear layer")
        return F.linear(values, self.weight, self.bias)


class AdjacentProduct(nn.Module):
    def forward(self, values: Tensor) -> Tensor:
        if values.shape[-1] % 2:
            raise ValueError("An adjacent-product layer requires an even input width")
        return values[..., 0::2] * values[..., 1::2]


class NeuralKernelNetwork(nn.Module):
    """Transform [..., input_width] primitive values into a scalar kernel [...]."""

    def __init__(self, input_width: int, widths: Sequence[int] = (8, 4, 4, 2, 1)):
        super().__init__()
        widths = tuple(widths)
        if isinstance(input_width, bool) or not isinstance(input_width, int) or input_width < 1:
            raise ValueError("The NKN needs at least one input kernel")
        if not widths or len(widths) % 2 == 0 or widths[-1] != 1:
            raise ValueError("NKN layers must alternate linear/product and end with linear width 1")
        if any(isinstance(width, bool) or not isinstance(width, int) or width < 1 for width in widths):
            raise ValueError("NKN widths must be positive integers")
        layers = []
        previous = input_width
        for index, width in enumerate(widths):
            if index % 2 == 0:
                layers.append(NonnegativeLinear(previous, width))
            else:
                if previous % 2 or width != previous // 2:
                    raise ValueError("Each product width must equal half its even input width")
                layers.append(AdjacentProduct())
            previous = width
        self.input_width = input_width
        self.widths = widths
        self.layers = nn.ModuleList(layers)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim == 0 or values.shape[-1] != self.input_width:
            raise ValueError("Unexpected number of NKN input kernels")
        for layer in self.layers:
            values = layer(values)
        return values[..., 0]


class PrimitiveNKN(nn.Module):
    """A representation-specific NKN, before any mutation-set aggregation."""

    def __init__(
        self,
        copies: Mapping[str, int],
        widths: Sequence[int] = (8, 4, 4, 2, 1),
        initial_lengthscale: float | Tensor = 1.0,
        *,
        probability_policy: str = "normalized",
    ):
        super().__init__()
        if not copies or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in copies.values()
        ):
            raise ValueError("Primitive copy counts must be nonnegative integers")
        self.primitives = nn.ModuleList(
            PrimitiveKernel(name, initial_lengthscale, probability_policy=probability_policy)
            # Configuration serialization may reorder mapping keys. Keep channel
            # ordering stable so reloaded weights still address the same kernels.
            for name, count in sorted(copies.items())
            for _ in range(count)
        )
        self.nkn = NeuralKernelNetwork(len(self.primitives), widths)
        # This is deliberately neither a parameter nor a buffer: geometry is
        # local to a prepared feature pair, not part of a saved model.
        self._geometry_cache: FixedPairwiseGeometry | None = None

    def prepare_geometry(self, left: Tensor, right: Tensor | None = None) -> None:
        """Cache fixed geometry for one immutable pair; gradient inputs skip it."""
        self.clear_geometry_cache()
        right = left if right is None else right
        self._geometry_cache = FixedPairwiseGeometry.prepare(
            left, right, {kernel.name for kernel in self.primitives},
            self.primitives[0].probability_policy,
        )

    def clear_geometry_cache(self) -> None:
        self._geometry_cache = None

    def _apply(self, fn, recurse=True):
        self.clear_geometry_cache()
        return super()._apply(fn, recurse=recurse)

    def forward(self, left: Tensor, right: Tensor | None = None) -> Tensor:
        right = left if right is None else right
        geometry = self._geometry_cache
        if geometry is not None and geometry.matches(left, right):
            primitive_values = torch.stack(
                [kernel.forward_from_geometry(geometry) for kernel in self.primitives], dim=-1,
            )
        else:
            primitive_values = torch.stack([kernel(left, right) for kernel in self.primitives], dim=-1)
        return self.nkn(primitive_values)
