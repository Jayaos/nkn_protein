"""The same explicitly jittered factorization is used in fitting and prediction."""

import math
import torch


class NumericalError(RuntimeError):
    """A run cannot produce a finite, positive-definite training covariance."""


def factor_covariance(kernel, noise, jitter_initial=1e-6, jitter_max=1e-3):
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or not kernel.shape[0]:
        raise ValueError("Training kernel must be a nonempty square matrix")
    if kernel.dtype != torch.float64:
        raise ValueError("Exact GP calculations require float64")
    if not torch.isfinite(kernel).all() or not torch.isfinite(noise).all():
        raise NumericalError("Nonfinite training covariance or observation noise")
    if not 0 < jitter_initial <= jitter_max:
        raise ValueError("Invalid jitter range")
    identity = torch.eye(len(kernel), dtype=kernel.dtype, device=kernel.device)
    jitter = float(jitter_initial)
    while True:
        covariance = (kernel + kernel.T) * 0.5 + (noise + jitter) * identity
        factor, info = torch.linalg.cholesky_ex(covariance, check_errors=False)
        if not info.any() and torch.isfinite(factor).all():
            return factor, jitter
        if jitter >= jitter_max:
            raise NumericalError(f"Cholesky factorization failed at maximum jitter {jitter_max:g}")
        jitter = min(jitter * 10.0, jitter_max)


def negative_log_likelihood(residual, factor):
    """Return total Gaussian NLL; normalization for stopping is explicit elsewhere."""
    alpha = torch.cholesky_solve(residual.unsqueeze(-1), factor).squeeze(-1)
    return 0.5 * torch.dot(residual, alpha) + torch.log(factor.diag()).sum() + 0.5 * len(residual) * math.log(2 * math.pi)
