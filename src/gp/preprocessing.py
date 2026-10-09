"""Fold-fitted standardization. Features remain frozen and unscaled by default."""

from dataclasses import dataclass
import torch


@dataclass
class FoldTransform:
    target_mean: torch.Tensor
    target_scale: torch.Tensor
    score_mean: torch.Tensor
    score_scale: torch.Tensor

    @classmethod
    def fit(cls, batch):
        y, scores = batch.targets, batch.scores
        if y is None or len(y) < 2 or not torch.isfinite(y).all():
            raise ValueError("Training needs at least two finite observed targets")
        if scores.ndim != 2 or scores.shape[1] < 1 or not torch.isfinite(scores).all():
            raise ValueError("Training needs finite scores from at least one mean scorer")
        target_scale = y.std(unbiased=False)
        if not torch.isfinite(target_scale) or target_scale <= 0:
            raise ValueError("Cannot standardize constant training targets or a nonfinite target scale")
        score_scale = scores.std(dim=0, unbiased=False)
        if not torch.isfinite(score_scale).all():
            raise ValueError("Zero-shot score scales must be finite")
        # A constant scorer becomes a zero column and retains a harmless scale of one.
        score_scale = torch.where(score_scale > 0, score_scale, torch.ones_like(score_scale))
        return cls(y.mean().detach(), target_scale.detach(), scores.mean(dim=0).detach(), score_scale.detach())

    def standardize_targets(self, values):
        return (values - self.target_mean) / self.target_scale

    def apply(self, batch):
        if batch.num_scores != self.score_mean.numel():
            raise ValueError("Prediction mean scorer count differs from training")
        return batch.with_scores((batch.scores - self.score_mean) / self.score_scale)

    def state_dict(self):
        return {name: value.detach().cpu() for name, value in vars(self).items()}

    @classmethod
    def from_state_dict(cls, state, device="cpu"):
        return cls(**{name: value.to(device=device, dtype=torch.float64) for name, value in state.items()})
