"""A trainable ensemble mean and compositional-kernel exact GP."""

import torch
from torch import nn
from torch.nn import functional as F

from kernels.protein import ProteinKernel
from .linalg import factor_covariance, negative_log_likelihood


class ExactProteinGP(nn.Module):
    def __init__(self, train, model_config, training_config, lengthscale_multiplier=1.0):
        super().__init__()
        if train.num_scores < 1:
            raise ValueError("At least one mean scorer is required")
        self.kernel = ProteinKernel(model_config, train, lengthscale_multiplier=lengthscale_multiplier)
        device = train.scores.device
        self.intercept = nn.Parameter(torch.zeros((), dtype=torch.float64, device=device))
        self.coefficients = nn.Parameter(torch.full((train.num_scores,), 1.0 / train.num_scores, dtype=torch.float64, device=device))
        value = torch.tensor(training_config.noise_initial - training_config.noise_floor, dtype=torch.float64, device=device)
        self.raw_noise = nn.Parameter(value + torch.log(-torch.expm1(-value)))
        self.noise_floor = training_config.noise_floor
        self.noise_prior = training_config.noise_prior
        self.noise_prior_scale = training_config.noise_prior_scale
        self.jitter_initial, self.jitter_max = training_config.jitter_initial, training_config.jitter_max
        self.inference_backend = training_config.inference_backend
        self._inference_config = training_config
        if self.inference_backend == "bbmm":
            from .bbmm import require_bbmm
            require_bbmm()
        self.to(device=device, dtype=torch.float64)

    @property
    def noise(self):
        return F.softplus(self.raw_noise) + self.noise_floor

    def mean(self, batch):
        return self.intercept + batch.scores @ self.coefficients

    def training_factor(self, train):
        if self.inference_backend != "cholesky":
            raise ValueError("BBMM uses iterative solves, not a training Cholesky factor")
        return factor_covariance(self.kernel(train), self.noise, self.jitter_initial, self.jitter_max)

    def objective_components(self, train, standardized_targets):
        """Return raw NLL, log prior on observation variance, and numerical jitter."""
        residual = standardized_targets - self.mean(train)
        if self.inference_backend == "bbmm":
            from .bbmm import negative_log_likelihood as bbmm_nll
            nll, jitter = bbmm_nll(self.kernel(train), self.noise, residual, self._inference_config)
        else:
            factor, jitter = self.training_factor(train)
            nll = negative_log_likelihood(residual, factor)
        log_prior = nll.new_zeros(())
        if self.noise_prior == "half_cauchy":
            scale = nll.new_tensor(self.noise_prior_scale)
            log_prior = torch.distributions.HalfCauchy(scale).log_prob(self.noise)
        return nll, log_prior, jitter

    def objective(self, train, standardized_targets):
        nll, log_prior, jitter = self.objective_components(train, standardized_targets)
        return nll - log_prior, jitter

    def posterior(self, train, standardized_targets, test):
        if self.inference_backend == "bbmm":
            from .bbmm import solve
            kernel = self.kernel(train)
            cross = self.kernel(train, test)
            rhs = torch.cat(((standardized_targets - self.mean(train)).unsqueeze(-1), cross), dim=1)
            solved, jitter = solve(kernel, self.noise, rhs, self._inference_config)
            mean = self.mean(test) + cross.T @ solved[:, 0]
            latent = self.kernel(test) - cross.T @ solved[:, 1:]
        else:
            factor, jitter = self.training_factor(train)
            cross = self.kernel(train, test)
            alpha = torch.cholesky_solve((standardized_targets - self.mean(train)).unsqueeze(-1), factor)
            mean = self.mean(test) + (cross.T @ alpha).squeeze(-1)
            solved = torch.linalg.solve_triangular(factor, cross, upper=False)
            latent = self.kernel(test) - solved.T @ solved
        latent = (latent + latent.T) * 0.5
        observation = latent + self.noise * torch.eye(test.num_variants, device=latent.device, dtype=latent.dtype)
        return mean, latent, observation, jitter

    def posterior_mean(self, train, standardized_targets, test):
        """Predict means only, without allocating a held-out covariance matrix."""
        residual = (standardized_targets - self.mean(train)).unsqueeze(-1)
        if self.inference_backend == "bbmm":
            from .bbmm import solve
            alpha, _ = solve(self.kernel(train), self.noise, residual, self._inference_config)
        else:
            factor, _ = self.training_factor(train)
            alpha = torch.cholesky_solve(residual, factor)
        return self.mean(test) + (self.kernel(train, test).T @ alpha).squeeze(-1)
