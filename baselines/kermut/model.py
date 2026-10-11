"""Dense exact GP with Kermut's mean, covariance, and observation-noise prior."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from gp.linalg import NumericalError
from .kernel import KermutKernel


def factor_covariance(kernel, noise, jitter_initial=1e-6, jitter_max=1e-3):
    """Try the reference covariance first; add numerical jitter only if needed."""
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or not kernel.shape[0]:
        raise ValueError("Training kernel must be a nonempty square matrix")
    if kernel.dtype not in (torch.float32, torch.float64):
        raise ValueError("Kermut calculations require float32 or float64")
    if not torch.isfinite(kernel).all() or not torch.isfinite(noise).all():
        raise NumericalError("Nonfinite Kermut covariance or observation noise")
    if not 0 < jitter_initial <= jitter_max:
        raise ValueError("Invalid jitter range")
    identity = torch.eye(len(kernel), dtype=kernel.dtype, device=kernel.device)
    covariance = (kernel + kernel.T) * 0.5 + noise * identity
    jitter = 0.0
    while True:
        factor, info = torch.linalg.cholesky_ex(covariance + jitter * identity, check_errors=False)
        if not info.any() and torch.isfinite(factor).all():
            return factor, jitter
        if jitter >= jitter_max:
            raise NumericalError(f"Kermut Cholesky factorization failed at maximum jitter {jitter_max:g}")
        jitter = float(jitter_initial) if jitter == 0 else min(jitter * 10.0, jitter_max)


class ExactKermutGP(nn.Module):
    def __init__(self, train, model_config, training_config):
        super().__init__()
        self.kernel = KermutKernel(train, model_config)
        options = {"dtype": train.scores.dtype, "device": train.device}
        self.intercept = nn.Parameter(torch.zeros((), **options))
        # The original one-score mean starts at score(x), with zero intercept.
        self.coefficients = nn.Parameter(torch.ones(train.num_scores, **options))
        self.raw_noise = nn.Parameter(torch.tensor(training_config.raw_noise, **options))
        self.noise_floor = training_config.noise_floor
        self.noise_prior_scale = training_config.noise_prior_scale
        self.jitter_initial, self.jitter_max = training_config.jitter_initial, training_config.jitter_max

    @property
    def noise(self):
        """Observation variance, matching GaussianLikelihood's positive constraint."""
        return F.softplus(self.raw_noise) + self.noise_floor

    def mean(self, batch):
        return self.intercept + batch.scores @ self.coefficients

    def training_factor(self, train):
        return factor_covariance(self.kernel(train), self.noise, self.jitter_initial, self.jitter_max)

    def objective_components(self, train, standardized_targets):
        factor, jitter = self.training_factor(train)
        residual = standardized_targets - self.mean(train)
        solved = torch.cholesky_solve(residual[:, None], factor).flatten()
        nll = (0.5 * torch.dot(residual, solved) + factor.diag().log().sum()
               + 0.5 * len(residual) * math.log(2.0 * math.pi))
        prior_scale = self.noise.new_tensor(self.noise_prior_scale)
        log_prior = torch.distributions.HalfCauchy(prior_scale).log_prob(self.noise)
        return nll, log_prior, jitter

    def objective(self, train, standardized_targets):
        nll, log_prior, jitter = self.objective_components(train, standardized_targets)
        return nll - log_prior, jitter

    def posterior(self, train, standardized_targets, test):
        factor, jitter = self.training_factor(train)
        cross = self.kernel(train, test)
        residual = standardized_targets - self.mean(train)
        alpha = torch.cholesky_solve(residual[:, None], factor)
        mean = self.mean(test) + (cross.T @ alpha).flatten()
        solved = torch.linalg.solve_triangular(factor, cross, upper=False)
        latent = self.kernel(test) - solved.T @ solved
        latent = (latent + latent.T) * 0.5
        observation = latent + self.noise * torch.eye(test.num_variants, dtype=latent.dtype, device=latent.device)
        return mean, latent, observation, jitter
