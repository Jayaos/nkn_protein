"""Kermut's fixed-length AdamW fit and self-contained prediction checkpoints."""

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path

import torch

from data_loading.features import FeatureBatch
from gp.linalg import NumericalError
from gp.training import Prediction
from .config import KermutModelConfig, KermutTrainingConfig
from .model import ExactKermutGP


def _convert_batch(batch, device, dtype, clone=False):
    def floating(value):
        if value is None:
            return None
        result = value.detach().to(device=device, dtype=dtype)
        return result.clone() if clone else result

    def integer(value):
        result = value.detach().to(device=device, dtype=torch.long)
        return result.clone() if clone else result

    return replace(batch, targets=floating(batch.targets), scores=floating(batch.scores),
                   global_features=floating(batch.global_features),
                   local_features={name: floating(value) for name, value in batch.local_features.items()},
                   owners=integer(batch.owners), mutation_positions=integer(batch.mutation_positions),
                   feature_signature=dict(batch.feature_signature))


def _batch_state(batch):
    return {"assay_id": batch.assay_id, "variant_ids": tuple(batch.variant_ids),
            "targets": None if batch.targets is None else batch.targets.detach().cpu(),
            "scores": batch.scores.detach().cpu(),
            "global_features": None if batch.global_features is None else batch.global_features.detach().cpu(),
            "local_features": {name: value.detach().cpu() for name, value in batch.local_features.items()},
            "owners": batch.owners.detach().cpu(), "mutation_positions": batch.mutation_positions.detach().cpu(),
            "reference_sequence": batch.reference_sequence, "sequences": tuple(batch.sequences),
            "score_names": tuple(batch.score_names), "feature_signature": dict(batch.feature_signature)}


@dataclass
class TargetTransform:
    """Training-only sample-standard-deviation scaling; mean scores stay raw."""

    target_mean: torch.Tensor
    target_scale: torch.Tensor

    @classmethod
    def fit(cls, batch):
        targets = batch.targets
        if targets is None or len(targets) < 2 or not torch.isfinite(targets).all():
            raise ValueError("Kermut training needs at least two finite observed targets")
        scale = targets.std(unbiased=True)
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError("Cannot standardize constant training targets or a nonfinite target scale")
        return cls(targets.mean().detach(), scale.detach())

    def standardize_targets(self, values):
        return (values - self.target_mean) / self.target_scale

    def apply(self, batch):
        return batch

    def state_dict(self):
        return {"target_mean": self.target_mean.detach().cpu(), "target_scale": self.target_scale.detach().cpu()}


@dataclass
class FittedKermut:
    model: ExactKermutGP
    train: FeatureBatch
    standardized_targets: torch.Tensor
    transform: TargetTransform
    model_config: KermutModelConfig
    training_config: KermutTrainingConfig
    selected_seed: int
    final_objective_per_variant: float
    history: list
    restart_summaries: list

    @torch.no_grad()
    def predict(self, test):
        if test.assay_id != self.train.assay_id:
            raise ValueError("A per-assay Kermut GP cannot predict a different assay")
        if test.reference_sequence != self.train.reference_sequence:
            raise ValueError("Prediction reference sequence differs from training")
        if tuple(test.score_names) != tuple(self.train.score_names):
            raise ValueError("Prediction mean scorers or their order differ from training")
        if test.feature_signature != self.train.feature_signature:
            raise ValueError("Prediction feature sources or preprocessing differ from training")
        test = _convert_batch(test, self.train.device, self.train.scores.dtype)
        from .data import validate_batch

        validate_batch(test)
        if test.num_scores != self.train.num_scores:
            raise ValueError("Prediction mean scorer count differs from training")
        if test.global_features.shape[1] != self.train.global_features.shape[1]:
            raise ValueError("Prediction sequence embedding dimension differs from training")
        # Signatures protect loaded caches. This also protects callers that
        # construct batches directly without a complete provenance signature.
        train_positions = {}
        for row, position in enumerate(self.train.mutation_positions.cpu().tolist()):
            train_positions.setdefault(position, row)
        test_positions = {}
        for row, position in enumerate(test.mutation_positions.cpu().tolist()):
            test_positions.setdefault(position, row)
        for position in train_positions.keys() & test_positions.keys():
            for name in ("inverse_folding", "structure"):
                if not torch.equal(self.train.local_features[name][train_positions[position]],
                                   test.local_features[name][test_positions[position]]):
                    raise ValueError(f"Prediction reference {name} differs from training at position {position}")
        self.model.eval()
        mean, latent, observation, jitter = self.model.posterior(self.train, self.standardized_targets, test)
        if not all(torch.isfinite(value).all() for value in (mean, latent, observation)):
            raise NumericalError("Nonfinite Kermut posterior prediction")
        scale = self.transform.target_scale
        return Prediction(self.transform.target_mean + scale * mean, scale.square() * latent,
                          scale.square() * observation, mean, latent, observation, jitter)

    def fit_summary(self, n_train=None):
        return {"model": "kermut", "n_train": self.train.num_variants if n_train is None else n_train,
                "selected_seed": self.selected_seed, "initializations": 1,
                "checkpoint_selection": "final_iterate", "optimizer_updates": self.training_config.n_steps,
                "final_objective_per_variant": self.final_objective_per_variant,
                "final_nll_per_variant": self.history[-1]["nll_per_variant"],
                "noise_variance_standardized": float(self.model.noise.detach()),
                "target_standardization": "training_sample_std", "mean_score_standardization": "none",
                "dtype": self.training_config.dtype,
                "mean_intercept_standardized_target": float(self.model.intercept.detach()),
                "mean_coefficients": {name: float(value) for name, value in zip(
                    self.train.score_names or tuple(f"score_{i}" for i in range(self.train.num_scores)),
                    self.model.coefficients.detach().cpu())},
                "kernel_parameters": {name: float(getattr(self.model.kernel, name).detach()) for name in (
                    "h_scale", "h_lengthscale", "d_lengthscale", "p_lengthscale", "alpha", "sequence_lengthscale")}}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 1, "model_type": "kermut",
                    "model_config": asdict(self.model_config), "training_config": asdict(self.training_config),
                    "parameters": {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()},
                    "train": _batch_state(self.train), "standardized_targets": self.standardized_targets.detach().cpu(),
                    "transform": self.transform.state_dict(), "selected_seed": self.selected_seed,
                    "final_objective_per_variant": self.final_objective_per_variant,
                    "history": self.history, "restart_summaries": self.restart_summaries}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format_version") != 1 or state.get("model_type") != "kermut":
            raise ValueError("Unsupported Kermut checkpoint format")
        model_config = KermutModelConfig(**state["model_config"])
        training_config = KermutTrainingConfig(**state["training_config"])
        dtype = getattr(torch, training_config.dtype)
        train = _convert_batch(FeatureBatch(**state["train"]), device, dtype)
        model = ExactKermutGP(train, model_config, training_config)
        model.load_state_dict(state["parameters"])
        model.eval()
        transform = TargetTransform(**{name: value.to(device=device, dtype=dtype)
                                      for name, value in state["transform"].items()})
        return cls(model, train, state["standardized_targets"].to(device=device, dtype=dtype), transform,
                   model_config, training_config, state["selected_seed"], state["final_objective_per_variant"],
                   state["history"], state["restart_summaries"])


def fit_gp(batch, model_config=None, training_config=None, progress=None):
    """Fit once for a fixed update count and return the final parameter state."""
    model_config = model_config or KermutModelConfig()
    cfg = training_config or KermutTrainingConfig()
    if cfg.max_train_variants is not None and batch.num_variants > cfg.max_train_variants:
        raise ValueError(f"{batch.num_variants} training variants exceed configured dense-GP limit {cfg.max_train_variants}")
    train = _convert_batch(batch, batch.device, getattr(torch, cfg.dtype), clone=True)
    transform = TargetTransform.fit(train)
    y = transform.standardize_targets(train.targets)
    # Parameter initialization is deterministic; preserving a seed in the run
    # record also supports reproducibility if initialization later changes.
    model = ExactKermutGP(train, model_config, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                                 betas=cfg.betas, eps=cfg.epsilon)
    history = []
    model.train()
    try:
        for step in range(cfg.n_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            nll, log_prior, jitter = model.objective_components(train, y)
            total = nll - log_prior
            loss = total / train.num_variants
            value = float(loss.detach())
            if not math.isfinite(value):
                raise NumericalError("Nonfinite Kermut training objective")
            row = {"seed": cfg.seed, "step": step, "nll": float(nll.detach()),
                   "nll_per_variant": float(nll.detach()) / train.num_variants,
                   "noise_log_prior": float(log_prior.detach()), "objective": float(total.detach()),
                   "objective_per_variant": value, "jitter": jitter}
            history.append(row)
            if progress is not None:
                progress(dict(row))
            if step == cfg.n_steps:
                break
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise NumericalError("Nonfinite Kermut parameter gradients")
            optimizer.step()
    except NumericalError as error:
        error.history = history
        error.restart_summaries = [{"seed": cfg.seed, "status": "failed",
                                    "checkpoint_selection": "final_iterate",
                                    "failure": str(error), "failed_step": step}]
        raise
    model.eval()
    summary = {"seed": cfg.seed, "status": "complete", "checkpoint_selection": "final_iterate",
               "final_step": cfg.n_steps, "final_objective_per_variant": value,
               "final_nll_per_variant": history[-1]["nll_per_variant"], "jitter_at_final": jitter,
               "failure": None}
    return FittedKermut(model, train, y, transform, model_config, cfg, cfg.seed, value, history, [summary])
