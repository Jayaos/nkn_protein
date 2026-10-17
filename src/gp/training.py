"""Configurable checkpoint selection and label-free prediction."""

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path

import torch

from config import ModelConfig, TrainingConfig, validate_enabled_branches
from data_loading.features import FeatureBatch
from evaluation import spearman
from .linalg import NumericalError
from .model import ExactProteinGP
from .preprocessing import FoldTransform


@dataclass
class Prediction:
    mean: torch.Tensor
    latent_covariance: torch.Tensor
    observation_covariance: torch.Tensor
    mean_standardized: torch.Tensor
    latent_covariance_standardized: torch.Tensor
    observation_covariance_standardized: torch.Tensor
    jitter: float


def _batch_state(batch):
    return {
        "assay_id": batch.assay_id, "variant_ids": tuple(batch.variant_ids),
        "targets": None if batch.targets is None else batch.targets.detach().cpu(),
        "scores": batch.scores.detach().cpu(),
        "global_features": None if batch.global_features is None else batch.global_features.detach().cpu(),
        "local_features": {k: v.detach().cpu() for k, v in batch.local_features.items()},
        "owners": batch.owners.detach().cpu(), "mutation_positions": batch.mutation_positions.detach().cpu(),
        "reference_sequence": batch.reference_sequence, "sequences": tuple(batch.sequences),
        "score_names": tuple(batch.score_names), "feature_signature": dict(batch.feature_signature),
    }


def _parameter_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _freeze_batch(batch):
    """Own detached copies so cached inputs stay constant throughout fitting."""
    frozen = lambda value: value.detach().clone()
    return replace(batch, targets=None if batch.targets is None else frozen(batch.targets),
                   scores=frozen(batch.scores),
                   global_features=None if batch.global_features is None else frozen(batch.global_features),
                   local_features={k: frozen(v) for k, v in batch.local_features.items()},
                   owners=frozen(batch.owners), mutation_positions=frozen(batch.mutation_positions),
                   feature_signature=dict(batch.feature_signature))


def _migrate_structure_distance_checkpoint(state):
    """Rename the former physical_distance branch in a loaded state, never on disk."""
    local = state["train"]["local_features"]
    if "physical_distance" in local:
        if "structure_distance" in local:
            raise ValueError("Checkpoint contains both physical_distance and structure_distance features")
        local["structure_distance"] = local.pop("physical_distance")
    signature = state["train"].get("feature_signature", {})
    for key in tuple(signature):
        if key.startswith("physical_distance."):
            canonical = "structure_distance." + key[len("physical_distance."):]
            if canonical in signature:
                raise ValueError("Checkpoint contains both distance branch signature aliases")
            signature[canonical] = signature.pop(key)
    encoded_branches = signature.get("enabled_branches")
    if isinstance(encoded_branches, str) and '"physical_distance"' in encoded_branches:
        branches = validate_enabled_branches(json.loads(encoded_branches))
        signature["enabled_branches"] = json.dumps(branches, separators=(",", ":"))
    parameters = state["parameters"]
    old_prefix = "kernel.branches.physical_distance."
    for key in tuple(parameters):
        if key.startswith(old_prefix):
            canonical = "kernel.branches.structure_distance." + key[len(old_prefix):]
            if canonical in parameters:
                raise ValueError("Checkpoint contains both distance branch parameter aliases")
            parameters[canonical] = parameters.pop(key)


@dataclass
class FittedGP:
    model: ExactProteinGP
    train: FeatureBatch
    standardized_targets: torch.Tensor
    transform: FoldTransform
    model_config: ModelConfig
    training_config: TrainingConfig
    selected_seed: int
    best_nll_per_variant: float
    history: list
    restart_summaries: list
    best_validation_spearman: float | None = None
    selected_step: int | None = None
    selected_nll_per_variant: float | None = None
    best_objective_per_variant: float | None = None
    selected_objective_per_variant: float | None = None

    def fit_summary(self, n_train):
        return {
            "n_train": n_train, "selected_seed": self.selected_seed,
            "gp_computation_backend": self.training_config.gp_computation_backend,
            "stopping_criterion": self.training_config.stopping_criterion,
            "best_nll_per_variant": self.best_nll_per_variant,
            "best_validation_spearman": self.best_validation_spearman,
            "selected_step": self.selected_step,
            "selected_nll_per_variant": self.selected_nll_per_variant,
            "best_objective_per_variant": self.best_objective_per_variant,
            "selected_objective_per_variant": self.selected_objective_per_variant,
            "noise_prior": self.training_config.noise_prior,
            "noise_prior_scale": self.training_config.noise_prior_scale,
            "noise_variance_standardized": float(self.model.noise.detach()),
        }

    @torch.no_grad()
    def predict(self, test):
        if test.assay_id != self.train.assay_id:
            raise ValueError("A per-assay GP cannot predict a different assay")
        if self.train.reference_sequence and test.reference_sequence != self.train.reference_sequence:
            raise ValueError("Prediction reference sequence differs from training")
        if tuple(test.score_names) != tuple(self.train.score_names):
            raise ValueError("Prediction mean scorers or their order differ from training")
        if test.feature_signature != self.train.feature_signature:
            raise ValueError("Prediction feature sources or preprocessing differ from training")
        test = self.transform.apply(test.to(self.train.scores.device))
        self.model.eval()
        mean, latent, observation, jitter = self.model.posterior(self.train, self.standardized_targets, test)
        if not all(torch.isfinite(x).all() for x in (mean, latent, observation)):
            raise NumericalError("Nonfinite posterior prediction")
        scale = self.transform.target_scale
        return Prediction(self.transform.target_mean + scale * mean, scale.square() * latent,
                          scale.square() * observation, mean, latent, observation, jitter)

    def save(self, path):
        from dataclasses import asdict
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 1, "model_config": asdict(self.model_config),
                    "training_config": asdict(self.training_config), "parameters": _parameter_state(self.model),
                    "train": _batch_state(self.train), "standardized_targets": self.standardized_targets.detach().cpu(),
                    "transform": self.transform.state_dict(), "selected_seed": self.selected_seed,
                    "best_nll_per_variant": self.best_nll_per_variant, "history": self.history,
                    "restart_summaries": self.restart_summaries,
                    "best_validation_spearman": self.best_validation_spearman,
                    "selected_step": self.selected_step,
                    "selected_nll_per_variant": self.selected_nll_per_variant,
                    "best_objective_per_variant": self.best_objective_per_variant,
                    "selected_objective_per_variant": self.selected_objective_per_variant}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format_version") != 1:
            raise ValueError("Unsupported GP checkpoint format")
        model_config = ModelConfig(**state["model_config"])
        training_config = TrainingConfig.from_dict(state["training_config"])
        _migrate_structure_distance_checkpoint(state)
        train = FeatureBatch(**state["train"]).to(device)
        # Initialization is overwritten; do not disturb the caller's random state.
        devices = [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            model = ExactProteinGP(train, model_config, training_config)
        model.load_state_dict(state["parameters"])
        model.eval()
        return cls(model, train, state["standardized_targets"].to(device=device, dtype=torch.float64),
                   FoldTransform.from_state_dict(state["transform"], device), model_config, training_config,
                   state["selected_seed"], state["best_nll_per_variant"], state["history"], state["restart_summaries"],
                   state.get("best_validation_spearman"), state.get("selected_step"),
                   state.get("selected_nll_per_variant"),
                   state.get("best_objective_per_variant", state["best_nll_per_variant"]),
                   state.get("selected_objective_per_variant", state.get("selected_nll_per_variant")))


def fit_gp(batch, model_config=None, training_config=None, progress=None, *, held_out=None, validation=None):
    model_config, cfg = model_config or ModelConfig(), training_config or TrainingConfig()
    if cfg.max_train_variants is not None and batch.num_variants > cfg.max_train_variants:
        raise ValueError(f"{batch.num_variants} training variants exceed configured dense-GP limit {cfg.max_train_variants}")
    batch = _freeze_batch(batch.to(batch.scores.device))
    transform = FoldTransform.fit(batch)
    train, y = transform.apply(batch), transform.standardize_targets(batch.targets)
    use_validation = cfg.stopping_criterion == "validation_spearman"
    validation_batch = None
    if use_validation:
        if validation is None or validation.targets is None:
            raise ValueError("validation_spearman requires a separate validation batch with observed targets")
        if (validation.assay_id != train.assay_id
                or validation.reference_sequence != train.reference_sequence
                or tuple(validation.score_names) != tuple(train.score_names)
                or validation.feature_signature != train.feature_signature):
            raise ValueError("Validation assay, reference, scorers, and feature sources must match training")
        if set(validation.variant_ids) & set(train.variant_ids):
            raise ValueError("Validation variants must not overlap training variants")
        if (validation.num_variants < 2 or not torch.isfinite(validation.targets).all()
                or validation.targets.unique().numel() < 2):
            raise ValueError("Validation Spearman requires at least two finite, nonconstant targets")
        validation_targets = validation.targets.detach().clone()
        validation_batch = transform.apply(_freeze_batch(replace(validation, targets=None).to(batch.scores.device)))
    diagnostic_batch = None
    if cfg.held_out_interval:
        if held_out is None or held_out.targets is None:
            raise ValueError("held_out_interval requires a held-out batch with observed targets")
        if (held_out.assay_id != train.assay_id
                or held_out.reference_sequence != train.reference_sequence
                or tuple(held_out.score_names) != tuple(train.score_names)
                or held_out.feature_signature != train.feature_signature):
            raise ValueError("Held-out assay, reference, scorers, and feature sources must match training")
        if set(held_out.variant_ids) & set(train.variant_ids):
            raise ValueError("Held-out diagnostic variants must not overlap training variants")
        if use_validation and set(held_out.variant_ids) & set(validation.variant_ids):
            raise ValueError("Held-out diagnostic variants must not overlap validation variants")
        if not torch.isfinite(held_out.targets).all():
            raise ValueError("Held-out diagnostic targets must be finite")
        diagnostic_targets = held_out.targets.detach().clone()
        diagnostic_batch = transform.apply(_freeze_batch(replace(held_out, targets=None).to(batch.scores.device)))
    history, summaries, winner = [], [], None
    device = batch.scores.device
    devices = [device.index or 0] if device.type == "cuda" else []
    for seed, multiplier in zip(cfg.seeds, cfg.lengthscale_multipliers):
        best_loss, best_state, best_step, best_jitter = math.inf, None, None, None
        best_validation, selected_loss, last_validation_improvement = None, None, 0
        best_objective, selected_objective = math.inf, None
        running_best, status, failure = [], "complete", None
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                model = ExactProteinGP(train, model_config, cfg, multiplier)
            # Features are detached, owned copies: only their pairwise geometry
            # is fixed. Learned covariances and their factors still update.
            model.kernel.prepare_training_geometry(train)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, betas=cfg.betas, eps=cfg.epsilon, weight_decay=0)
            model.train()
            # Evaluate the initial state and every state after an optimizer update.
            for step in range(cfg.max_steps + 1):
                optimizer.zero_grad(set_to_none=True)
                nll, log_prior, jitter = model.objective_components(train, y)
                total = nll - log_prior
                loss = total / train.num_variants
                value = float(loss.detach())
                nll_value = float(nll.detach()) / train.num_variants
                if not math.isfinite(value) or not math.isfinite(nll_value):
                    raise NumericalError("Nonfinite training objective or marginal likelihood")
                best_loss = min(best_loss, nll_value)
                if value < best_objective:
                    best_objective = value
                    if not use_validation:
                        best_state, best_step, best_jitter = _parameter_state(model), step, jitter
                        selected_loss, selected_objective = nll_value, value
                row = {"seed": seed, "step": step, "nll": float(nll.detach()), "nll_per_variant": nll_value,
                       "gp_computation_backend": cfg.gp_computation_backend,
                       "best_nll_per_variant": best_loss, "jitter": jitter,
                       "noise_log_prior": float(log_prior.detach()), "objective": float(total.detach()),
                       "objective_per_variant": value, "best_objective_per_variant": best_objective,
                       "noise_variance_standardized": float(model.noise.detach())}
                running_best.append(best_objective)
                early_stop = (not use_validation and step >= cfg.min_steps and step >= cfg.patience
                              and running_best[step - cfg.patience] - best_objective < cfg.tolerance)
                if use_validation and (step % cfg.validation_interval == 0 or step == cfg.max_steps):
                    with torch.no_grad():
                        mean = model.posterior_mean(train, y, validation_batch)
                        if not torch.isfinite(mean).all():
                            raise NumericalError("Nonfinite validation predictive mean")
                        score = spearman(validation_targets, mean)
                    if score is not None and (best_validation is None or score > best_validation):
                        best_validation, last_validation_improvement = score, step
                        best_state, best_step, best_jitter = _parameter_state(model), step, jitter
                        selected_loss, selected_objective = nll_value, value
                    row.update(validation_spearman=score, best_validation_spearman=best_validation,
                               n_validation=validation_batch.num_variants)
                    early_stop = (step >= cfg.min_steps and step > 0
                                  and step - last_validation_improvement >= cfg.patience)
                if diagnostic_batch is not None and (step % cfg.held_out_interval == 0
                                                      or early_stop or step == cfg.max_steps):
                    row.update(held_out_spearman=None, n_held_out=diagnostic_batch.num_variants,
                               held_out_error=None)
                    # Diagnostic labels never enter the objective, optimizer, or selection rule.
                    # Preserve the model's mode and avoid building any evaluation graph.
                    try:
                        with torch.no_grad():
                            if diagnostic_batch.num_variants:
                                mean = model.posterior_mean(train, y, diagnostic_batch)
                                if not torch.isfinite(mean).all():
                                    raise NumericalError("Nonfinite held-out predictive mean")
                                row["held_out_spearman"] = spearman(diagnostic_targets, mean)
                    except NumericalError as error:
                        row["held_out_error"] = str(error)
                history.append(row)
                if progress is not None:
                    progress(dict(row))
                if early_stop:
                    status = "early_stopped"
                    break
                if step == cfg.max_steps:
                    break
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise NumericalError("Nonfinite parameter gradients")
                optimizer.step()
            if use_validation and best_state is None:
                raise NumericalError("No defined validation Spearman was observed; no checkpoint can be selected")
        except NumericalError as error:
            status, failure = "failed", str(error)
        summary = {"seed": seed, "lengthscale_multiplier": multiplier, "status": status,
                   "gp_computation_backend": cfg.gp_computation_backend,
                   "best_step": best_step, "best_nll_per_variant": best_loss if math.isfinite(best_loss) else None,
                   "stopping_criterion": cfg.stopping_criterion,
                   "best_validation_spearman": best_validation,
                   "selected_nll_per_variant": selected_loss,
                   "best_objective_per_variant": best_objective if math.isfinite(best_objective) else None,
                   "selected_objective_per_variant": selected_objective,
                   "jitter_at_best": best_jitter, "failure": failure}
        summaries.append(summary)
        # A failed run is unsuccessful even when an earlier checkpoint was finite.
        selection_loss = -best_validation if use_validation and best_validation is not None else best_objective
        if status != "failed" and best_state is not None and (winner is None or selection_loss < winner[0]):
            winner = (selection_loss, best_loss, best_state, seed, model, best_validation, best_step,
                      selected_loss, best_objective, selected_objective)
    if winner is None:
        error = NumericalError("All initialization runs failed: " + "; ".join(f"seed {s['seed']}: {s['failure']}" for s in summaries))
        error.restart_summaries = summaries
        error.history = history
        raise error
    _, best_loss, state, seed, model, best_validation, best_step, selected_loss, best_objective, selected_objective = winner
    model.load_state_dict(state)
    model.eval()
    return FittedGP(model, train, y, transform, model_config, cfg, seed, best_loss, history, summaries,
                    best_validation, best_step, selected_loss, best_objective, selected_objective)
