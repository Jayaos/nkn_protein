"""Optional float64 BBMM inference for the existing dense NKN covariance.

The kernel and observation-noise model are unchanged. CG replaces covariance
factorization and stochastic Lanczos quadrature estimates its log determinant.
This still stores the dense NKN kernel and its autograd intermediates.

BBMM always uses ``jitter_initial``. Unlike the Cholesky backend, an iterative
failure does not retry with larger jitter: failure to converge is not evidence
that changing the statistical covariance would be appropriate. ``jitter_max``
remains a Cholesky-backend setting. Numerical failures are reported explicitly.
"""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import lru_cache
import math
import warnings

import torch
from torch.autograd.function import once_differentiable

from .linalg import NumericalError


@lru_cache(maxsize=1)
def _dependencies():
    # Import lazily: existing Cholesky-only environments need no extra package.
    try:
        import linear_operator
        from linear_operator import settings
        from linear_operator.operators import AddedDiagLinearOperator, DenseLinearOperator, DiagLinearOperator
        from linear_operator.utils import linear_cg
        from linear_operator.utils.warnings import NumericalWarning
    except ImportError as exc:
        raise RuntimeError(
            "BBMM requires the optional linear_operator package (validated with "
            "linear_operator==0.5.2); install requirements-bbmm.txt in the run environment."
        ) from exc

    class BBMMAddedDiagLinearOperator(AddedDiagLinearOperator):
        def _preconditioner(self):
            # A preconditioner is an algorithmic aid, not part of the kernel.
            # Detaching it avoids differentiating a pivoted low-rank Cholesky
            # factor; InvQuadLogdet still estimates the covariance derivative.
            with torch.no_grad():
                # A PSD kernel with an entirely zero diagonal is zero. The
                # covariance then needs no low-rank preconditioner; attempting
                # its first pivot would divide by zero in linear_operator.
                if not torch.count_nonzero(self._linear_op._diagonal()).item():
                    return None, None, None
                return super()._preconditioner()

        def _probe_vectors_and_norms(self):
            if settings.num_trace_samples.value() != 0:
                return None, None
            # Exhaustive orthogonal probes remove trace sampling error for small
            # numerical validation problems. This mode is intentionally O(N^2).
            n = self.size(-1)
            return (
                torch.eye(n, dtype=self.dtype, device=self.device),
                torch.full((1, n), math.sqrt(n), dtype=self.dtype, device=self.device),
            )

        def _solve(self, rhs, preconditioner=None, num_tridiag=0):
            tolerance = settings.cg_tolerance.value()
            # The library stops on the average residual. Dividing its stopping
            # tolerance by the number of RHS columns ensures one unconverged
            # prediction or training solve cannot hide among converged probes.
            # Verify each true residual too, since recursive residuals can drift.
            internal_tolerance = tolerance / (rhs.shape[-1] if rhs.ndim == 2 else 1)
            result = linear_cg(
                self._matmul,
                rhs,
                n_tridiag=num_tridiag,
                tolerance=internal_tolerance,
                eps=min(1e-20, internal_tolerance * internal_tolerance * 1e-2),
                stop_updating_after=min(1e-12, internal_tolerance * 1e-2),
                max_iter=settings.max_cg_iterations.value(),
                max_tridiag_iter=settings.max_lanczos_quadrature_iterations.value(),
                preconditioner=preconditioner,
            )
            solution = result[0] if num_tridiag else result
            _check_residual(self, rhs, solution, tolerance)
            if num_tridiag:
                tridiagonal = result[1]
                if not torch.isfinite(tridiagonal).all():
                    raise NumericalError("BBMM produced nonfinite Lanczos coefficients")
                # linear_operator masks negative Ritz values in its SLQ helper.
                # Reject these instead of silently accepting an invalid logdet.
                eigenvalues = torch.linalg.eigvalsh(tridiagonal)
                if not torch.isfinite(eigenvalues).all() or (eigenvalues <= 0).any():
                    raise NumericalError("BBMM Lanczos quadrature found nonpositive covariance eigenvalues")
            return result

    return linear_operator, settings, DenseLinearOperator, DiagLinearOperator, BBMMAddedDiagLinearOperator, NumericalWarning


def require_bbmm():
    """Check the optional dependency and return its installed version."""
    return _dependencies()[0].__version__


@dataclass(frozen=True)
class _Options:
    cg_tolerance: float
    max_cg_iterations: int
    num_trace_samples: int
    max_lanczos_iterations: int
    preconditioner_size: int
    probe_seed: int
    jitter: float

    @classmethod
    def from_config(cls, config):
        return cls(
            config.bbmm_cg_tolerance,
            config.bbmm_max_cg_iterations,
            config.bbmm_num_trace_samples,
            config.bbmm_max_lanczos_iterations,
            config.bbmm_preconditioner_size,
            config.bbmm_probe_seed,
            float(config.jitter_initial),
        )


@contextmanager
def _settings(options):
    _, settings, _, _, _, numerical_warning = _dependencies()
    with ExitStack() as stack:
        stack.enter_context(warnings.catch_warnings())
        warnings.simplefilter("error", numerical_warning)
        for setting in (
            settings.max_cholesky_size(0),
            settings.fast_computations(covar_root_decomposition=True, log_prob=True, solves=True),
            settings.cg_tolerance(options.cg_tolerance),
            settings.max_cg_iterations(options.max_cg_iterations),
            settings.num_trace_samples(options.num_trace_samples),
            settings.max_lanczos_quadrature_iterations(
                min(options.max_lanczos_iterations, options.max_cg_iterations)
            ),
            settings.max_preconditioner_size(
                options.preconditioner_size if options.num_trace_samples else 0
            ),
            settings.min_preconditioning_size(0),
            settings.deterministic_probes(False),
            settings.skip_logdet_forward(False),
            settings.terminate_cg_by_size(False),
        ):
            stack.enter_context(setting)
        try:
            yield
        except numerical_warning as exc:
            raise NumericalError(f"BBMM failed with jitter {options.jitter:g}: {exc}") from exc


def _validate(kernel, noise, rhs):
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or not kernel.shape[0]:
        raise ValueError("Training kernel must be a nonempty square matrix")
    if kernel.dtype != torch.float64 or noise.dtype != torch.float64 or rhs.dtype != torch.float64:
        raise ValueError("BBMM GP calculations require float64")
    if noise.numel() != 1:
        raise ValueError("BBMM expects scalar observation noise")
    if rhs.ndim not in (1, 2) or rhs.shape[0] != len(kernel):
        raise ValueError("BBMM right-hand side must have one row per training observation")
    if kernel.device != noise.device or kernel.device != rhs.device:
        raise ValueError("BBMM kernel, noise, and right-hand side must share a device")
    if not all(torch.isfinite(value).all() for value in (kernel, noise, rhs)):
        raise NumericalError("Nonfinite BBMM training covariance, noise, or right-hand side")
    if noise.item() < 0:
        raise NumericalError("BBMM observation noise must be nonnegative")


def _operator(kernel, noise, options):
    _, _, dense, diagonal, added_diagonal, _ = _dependencies()
    return added_diagonal(
        dense((kernel + kernel.T) * 0.5),
        diagonal((noise + options.jitter).reshape(()).expand(len(kernel))),
    )


def _check_residual(operator, rhs, solution, tolerance):
    if not torch.isfinite(solution).all():
        raise NumericalError("BBMM CG produced a nonfinite solution")
    rhs_matrix = rhs.unsqueeze(-1) if rhs.ndim == 1 else rhs
    solution_matrix = solution.unsqueeze(-1) if solution.ndim == 1 else solution
    residual = operator.matmul(solution_matrix) - rhs_matrix
    denominator = torch.linalg.vector_norm(rhs_matrix, dim=-2)
    relative = torch.linalg.vector_norm(residual, dim=-2) / denominator.clamp_min(1e-30)
    if not torch.isfinite(relative).all() or relative.max() > tolerance * 1.05:
        raise NumericalError(
            f"BBMM CG did not converge: maximum relative residual {relative.max().item():.6g} "
            f"exceeds tolerance {tolerance:g}"
        )


@contextmanager
def _probe_rng(kernel, seed):
    devices = [kernel.device.index] if kernel.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        # Seed only the CPU and selected CUDA generator, preserving every other
        # device's state and the caller's training RNG state.
        torch.random.default_generator.manual_seed(seed)
        if kernel.is_cuda:
            torch.cuda.default_generators[kernel.device.index].manual_seed(seed)
        yield


def negative_log_likelihood(kernel, noise, residual, config):
    """Return total approximate Gaussian NLL and the fixed diagonal jitter.

    Probe randomness is local and repeatable across calls (common random
    numbers during optimization); no process-global probe cache is used.
    ``bbmm_num_trace_samples=0`` uses exhaustive probes with no preconditioner
    for small validation cases, not scalable training.
    """
    _validate(kernel, noise, residual)
    if residual.ndim != 1:
        raise ValueError("BBMM NLL residual must be a vector")
    options = _Options.from_config(config)
    with _settings(options), _probe_rng(kernel, options.probe_seed):
        operator = _operator(kernel, noise, options)
        inverse_quadratic, logdet = operator.inv_quad_logdet(
            inv_quad_rhs=residual.unsqueeze(-1), logdet=True
        )
        result = 0.5 * (inverse_quadratic + logdet + len(residual) * math.log(2 * math.pi))
        if not torch.isfinite(result):
            raise NumericalError("BBMM produced a nonfinite marginal likelihood")
    return result, options.jitter


class _CovarianceSolve(torch.autograd.Function):
    """Re-enter the configured iterative backend for first-order backward solves."""

    @staticmethod
    def forward(ctx, kernel, noise, rhs, options):
        ctx.options = options
        with _settings(options):
            solution = _operator(kernel, noise, options).solve(rhs)
        ctx.save_for_backward(kernel, noise, solution)
        return solution

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        kernel, noise, solution = ctx.saved_tensors
        # Public linear_operator.solve otherwise reads ambient global settings
        # during backward and may unexpectedly select full Cholesky.
        left = _CovarianceSolve.apply(kernel, noise, grad_output, ctx.options)
        left = left.unsqueeze(-1) if left.ndim == 1 else left
        right = solution.unsqueeze(-1) if solution.ndim == 1 else solution
        covariance_gradient = -(left @ right.T)
        covariance_gradient = (covariance_gradient + covariance_gradient.T) * 0.5
        kernel_gradient = covariance_gradient if ctx.needs_input_grad[0] else None
        noise_gradient = covariance_gradient.diag().sum().reshape_as(noise) if ctx.needs_input_grad[1] else None
        rhs_gradient = left.squeeze(-1) if solution.ndim == 1 else left
        return kernel_gradient, noise_gradient, rhs_gradient if ctx.needs_input_grad[2] else None, None


def solve(kernel, noise, rhs, config):
    """Solve the jittered training covariance against a vector or matrix RHS."""
    _validate(kernel, noise, rhs)
    options = _Options.from_config(config)
    return _CovarianceSolve.apply(kernel, noise, rhs, options), options.jitter
