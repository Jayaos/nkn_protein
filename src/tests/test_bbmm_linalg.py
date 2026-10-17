"""Numerical backend validation on synthetic SPD matrices only."""

import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gp import bbmm
from gp.linalg import NumericalError, factor_covariance, negative_log_likelihood


def options(**overrides):
    values = dict(
        bbmm_cg_tolerance=1e-10,
        bbmm_max_cg_iterations=128,
        bbmm_num_trace_samples=0,
        bbmm_max_lanczos_iterations=32,
        bbmm_preconditioner_size=0,
        bbmm_probe_seed=19,
        jitter_initial=1e-6,
        jitter_max=1e-3,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def problem(n=12, device="cpu"):
    generator = torch.Generator().manual_seed(920)
    basis = torch.randn(n, n, dtype=torch.float64, generator=generator)
    kernel = basis @ basis.T / n + torch.eye(n, dtype=torch.float64) * 0.3
    residual = torch.randn(n, dtype=torch.float64, generator=generator)
    return kernel.to(device), torch.tensor(0.4, dtype=torch.float64, device=device), residual.to(device)


class BBMMLinalgTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            bbmm.require_bbmm()
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        torch.set_num_threads(1)

    def test_exhaustive_trace_matches_cholesky_value_and_all_gradients(self):
        values = problem()
        exact = [value.clone().requires_grad_() for value in values]
        iterative = [value.clone().requires_grad_() for value in values]
        factor, jitter = factor_covariance(*exact[:2])
        expected = negative_log_likelihood(exact[2], factor)
        expected.backward()
        actual, actual_jitter = bbmm.negative_log_likelihood(*iterative, options())
        actual.backward()
        self.assertEqual(jitter, actual_jitter)
        torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-8)
        for observed, reference in zip(iterative, exact):
            torch.testing.assert_close(observed.grad, reference.grad, rtol=1e-7, atol=1e-8)

    def test_cg_vector_matrix_solve_and_backward_keep_iterative_settings(self):
        from linear_operator import settings
        from linear_operator.operators import LinearOperator

        for columns in (None, 3):
            kernel, noise, residual = problem()
            rhs = residual if columns is None else torch.stack([residual, residual.flip(0), residual * 2], -1)
            iterative = [value.clone().requires_grad_() for value in (kernel, noise, rhs)]
            exact = [value.clone().requires_grad_() for value in (kernel, noise, rhs)]
            factor, _ = factor_covariance(*exact[:2])
            matrix_rhs = exact[2].unsqueeze(-1) if columns is None else exact[2]
            expected = torch.cholesky_solve(matrix_rhs, factor)
            expected = expected.squeeze(-1) if columns is None else expected
            expected.square().sum().backward()
            with patch.object(LinearOperator, "cholesky", side_effect=AssertionError("full Cholesky forbidden")):
                actual, _ = bbmm.solve(*iterative, options())
                # Backward occurs after helper settings exit, with ambient
                # settings deliberately asking for the dense factorization.
                with settings.max_cholesky_size(100000), settings.fast_computations(solves=False):
                    actual.square().sum().backward()
            torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)
            for observed, reference in zip(iterative, exact):
                torch.testing.assert_close(observed.grad, reference.grad, rtol=1e-7, atol=1e-8)

    def test_seeded_preconditioned_nll_is_repeatable_and_preserves_rng(self):
        from linear_operator.operators import LinearOperator

        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            kernel, noise, residual = problem(24, device=device)
            config = options(bbmm_num_trace_samples=16, bbmm_preconditioner_size=5, bbmm_cg_tolerance=1e-8)
            cpu_rng = torch.random.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
            with patch.object(LinearOperator, "cholesky", side_effect=AssertionError("full Cholesky forbidden")):
                first, _ = bbmm.negative_log_likelihood(kernel, noise, residual, config)
                second, _ = bbmm.negative_log_likelihood(kernel, noise, residual, config)
            self.assertTrue(torch.equal(first, second))
            self.assertTrue(torch.equal(cpu_rng, torch.random.get_rng_state()))
            if cuda_rng is not None:
                self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))
            changed, _ = bbmm.negative_log_likelihood(
                kernel, noise, residual, options(**{**vars(config), "bbmm_probe_seed": 20})
            )
            self.assertNotEqual(first.item(), changed.item())

    def test_preconditioned_nll_has_finite_gradients_without_dense_cholesky(self):
        from linear_operator.operators import LinearOperator

        kernel, noise, residual = [value.requires_grad_() for value in problem(24)]
        config = options(bbmm_num_trace_samples=64, bbmm_preconditioner_size=5, bbmm_cg_tolerance=1e-8)
        with patch.object(LinearOperator, "cholesky", side_effect=AssertionError("full Cholesky forbidden")):
            value, _ = bbmm.negative_log_likelihood(kernel, noise, residual, config)
            value.backward()
        self.assertTrue(all(torch.isfinite(value.grad).all() for value in (kernel, noise, residual)))
        # The residual derivative is deterministic even with sampled logdet.
        factor, _ = factor_covariance(kernel.detach(), noise.detach())
        expected = torch.cholesky_solve(residual.detach().unsqueeze(-1), factor).squeeze(-1)
        torch.testing.assert_close(residual.grad, expected, rtol=1e-7, atol=1e-8)

    def test_failed_cg_and_invalid_covariance_raise_instead_of_returning_result(self):
        kernel, noise, residual = problem(30)
        bad = options(bbmm_max_cg_iterations=1, bbmm_max_lanczos_iterations=1)
        with self.assertRaisesRegex(NumericalError, "CG"):
            bbmm.solve(kernel, noise, residual, bad)
        with self.assertRaisesRegex(NumericalError, "CG"):
            bbmm.negative_log_likelihood(kernel, noise, residual, bad)
        with self.assertRaisesRegex(NumericalError, "Nonfinite"):
            bbmm.solve(kernel * math.nan, noise, residual, options())
        with self.assertRaisesRegex(ValueError, "float64"):
            bbmm.solve(kernel.float(), noise, residual, options())
        # Nonpositive Ritz values must not be silently masked by SLQ.
        with self.assertRaises(NumericalError):
            bbmm.negative_log_likelihood(-torch.eye(12, dtype=torch.float64), noise, residual[:12], options())

    def test_one_unconverged_rhs_cannot_hide_in_the_column_average(self):
        from linear_operator.operators import DenseLinearOperator

        identity = torch.eye(12, dtype=torch.float64)
        rhs = torch.ones(12, 100, dtype=torch.float64)
        approximate = rhs.clone()
        approximate[:, 0] *= 1.0001
        # Average relative residual is 1e-6, but one column fails 1e-5.
        with self.assertRaisesRegex(NumericalError, "maximum relative residual"):
            bbmm._check_residual(DenseLinearOperator(identity), rhs, approximate, 1e-5)

        generator = torch.Generator().manual_seed(77)
        basis, _ = torch.linalg.qr(torch.randn(40, 40, dtype=torch.float64, generator=generator))
        kernel = (basis * torch.logspace(-4, 0, 40, dtype=torch.float64)) @ basis.T
        noise = torch.tensor(1e-4, dtype=torch.float64)
        # Many easy eigenvector columns and one general right-hand side.
        rhs = basis[:, -1:].expand(-1, 30).clone()
        rhs[:, 0] = torch.randn(40, dtype=torch.float64, generator=generator)
        result, jitter = bbmm.solve(
            kernel, noise, rhs, options(bbmm_cg_tolerance=1e-7, bbmm_max_cg_iterations=256)
        )
        covariance = kernel + (noise + jitter) * torch.eye(40, dtype=torch.float64)
        relative = (covariance @ result - rhs).norm(dim=0) / rhs.norm(dim=0)
        self.assertLess(relative.max().item(), 1e-7)

    def test_zero_and_low_rank_psd_kernels_are_valid_with_positive_noise(self):
        noise = torch.tensor(0.4, dtype=torch.float64)
        residual = torch.zeros(12, dtype=torch.float64)
        for kernel in (
            torch.zeros(12, 12, dtype=torch.float64),
            torch.ones(12, 12, dtype=torch.float64),
            torch.eye(12, dtype=torch.float64),
        ):
            factor, _ = factor_covariance(kernel, noise)
            expected = negative_log_likelihood(residual, factor)
            config = options(bbmm_num_trace_samples=32, bbmm_preconditioner_size=32)
            actual, _ = bbmm.negative_log_likelihood(kernel, noise, residual, config)
            torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)
            solution, _ = bbmm.solve(kernel, noise, residual, config)
            torch.testing.assert_close(solution, residual, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_gpu_exhaustive_trace_matches_cpu_reference(self):
        kernel, noise, residual = problem()
        factor, _ = factor_covariance(kernel, noise)
        expected = negative_log_likelihood(residual, factor)
        values = [value.cuda().requires_grad_() for value in (kernel, noise, residual)]
        actual, _ = bbmm.negative_log_likelihood(*values, options())
        actual.backward()
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-7, atol=1e-8)
        self.assertTrue(all(torch.isfinite(value.grad).all() for value in values))


if __name__ == "__main__":
    unittest.main()
