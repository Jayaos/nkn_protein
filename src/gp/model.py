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
        self.jitter_initial, self.jitter_max = training_config.jitter_initial, training_config.jitter_max
        self.to(device=device, dtype=torch.float64)

    @property
    def noise(self):
        return F.softplus(self.raw_noise) + self.noise_floor

    def mean(self, batch):
        return self.intercept + batch.scores @ self.coefficients

    def training_factor(self, train):
        return factor_covariance(self.kernel(train), self.noise, self.jitter_initial, self.jitter_max)

    def objective(self, train, standardized_targets):
        factor, jitter = self.training_factor(train)
        total = negative_log_likelihood(standardized_targets - self.mean(train), factor)
        return total, jitter

    def posterior(self, train, standardized_targets, test):
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
        factor, _ = self.training_factor(train)
        alpha = torch.cholesky_solve((standardized_targets - self.mean(train)).unsqueeze(-1), factor)
        return self.mean(test) + (self.kernel(train, test).T @ alpha).squeeze(-1)
