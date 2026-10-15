"""Optional observation-variance priors preserve likelihood and selection semantics."""

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from config import ExperimentConfig, TrainingConfig
from experiments.cli import _load_config, _parser
from gp.model import ExactProteinGP
from gp.preprocessing import FoldTransform
from gp.training import FittedGP, fit_gp
from test_pipeline import small_config, synthetic_batch


class NoisePriorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.batch = synthetic_batch()
        self.config = small_config()
        self.training = replace(self.config.training, max_steps=3, min_steps=0,
                                patience=10, noise_prior="half_cauchy", noise_prior_scale=0.1)

    def test_config_validation_and_round_trip(self):
        self.assertEqual(TrainingConfig().noise_prior, "none")
        config = replace(self.config, training=self.training)
        self.assertEqual(ExperimentConfig.from_dict(json.loads(json.dumps(config.to_dict()))), config)
        for value in ("gaussian", "", None, True):
            with self.subTest(prior=value), self.assertRaisesRegex(ValueError, "noise_prior"):
                TrainingConfig(noise_prior=value)
        for value in (0, -1, float("nan"), float("inf"), True, "0.1", None):
            with self.subTest(scale=value), self.assertRaisesRegex(ValueError, "noise_prior_scale"):
                TrainingConfig(noise_prior_scale=value)

    def test_prior_matches_density_on_variance_and_noise_gradient(self):
        raw = self.batch.select(range(6))
        transform = FoldTransform.fit(raw)
        train, y = transform.apply(raw), transform.standardize_targets(raw.targets)
        model = ExactProteinGP(train, self.config.model, self.training)
        nll, log_prior, jitter = model.objective_components(train, y)
        covariance = model.kernel(train) + (model.noise + jitter) * torch.eye(6, dtype=torch.float64)
        reference_nll = -torch.distributions.MultivariateNormal(
            model.mean(train), covariance_matrix=covariance).log_prob(y)
        scale, variance = self.training.noise_prior_scale, model.noise
        reference_log_prior = math.log(2 / (math.pi * scale)) - torch.log1p((variance / scale) ** 2)
        torch.testing.assert_close(nll, reference_nll)
        torch.testing.assert_close(log_prior, reference_log_prior)
        objective, _ = model.objective(train, y)
        torch.testing.assert_close(objective, reference_nll - reference_log_prior)
        prior_gradient = torch.autograd.grad(-log_prior, model.raw_noise)[0]
        expected_gradient = 2 * variance / (scale ** 2 + variance ** 2) * torch.sigmoid(model.raw_noise)
        torch.testing.assert_close(prior_gradient, expected_gradient)
        objective.backward()
        analytical = model.raw_noise.grad.item()
        with torch.no_grad():
            original = model.raw_noise.item()
            model.raw_noise.fill_(original + 1e-5)
            upper = model.objective(train, y)[0].item()
            model.raw_noise.fill_(original - 1e-5)
            lower = model.objective(train, y)[0].item()
            model.raw_noise.fill_(original)
        self.assertAlmostEqual(analytical, (upper - lower) / 2e-5, delta=1e-6)

    def test_no_prior_matches_legacy_objective_and_ignores_prior_scale(self):
        cfg = replace(self.training, noise_prior="none")
        first = fit_gp(self.batch, self.config.model, cfg)
        second = fit_gp(self.batch, self.config.model, replace(cfg, noise_prior_scale=50))
        self.assertEqual(first.history, second.history)
        for name, value in first.model.state_dict().items():
            torch.testing.assert_close(value, second.model.state_dict()[name], rtol=0, atol=0)
        for row in first.history:
            self.assertEqual(row["noise_log_prior"], 0)
            self.assertEqual(row["objective"], row["nll"])
        model = first.model
        factor, _ = model.training_factor(first.train)
        residual = first.standardized_targets - model.mean(first.train)
        legacy = (0.5 * residual @ torch.cholesky_solve(residual[:, None], factor).squeeze(-1)
                  + factor.diag().log().sum() + 0.5 * len(residual) * math.log(2 * math.pi))
        torch.testing.assert_close(model.objective(first.train, first.standardized_targets)[0], legacy,
                                   rtol=0, atol=0)

    def test_training_uses_objective_for_stopping_checkpoint_and_restart_selection(self):
        # Raw NLL improves throughout. The prior-adjusted objective is best at
        # step 1 of seed 7; seed 8 has a lower NLL but a worse objective.
        values = [(6., 0.), (5., 1.), (4., -1.), (3., -3.),
                  (2., -4.), (1., -4.), (0., -6.), (-1., -8.)]

        def controlled(model, train, y):
            nll, log_prior = values.pop(0)
            return model.intercept * 0 + nll, model.intercept * 0 + log_prior, 1e-6

        cfg = replace(self.training, max_steps=10, patience=2, tolerance=1e-6,
                      seeds=(7, 8), lengthscale_multipliers=(1., 1.))
        with patch.object(ExactProteinGP, "objective_components", controlled):
            fitted = fit_gp(self.batch, self.config.model, cfg)
        self.assertEqual(values, [])
        self.assertEqual((fitted.selected_seed, fitted.selected_step), (7, 1))
        self.assertEqual(fitted.best_nll_per_variant, 3 / 8)
        self.assertEqual(fitted.selected_nll_per_variant, 5 / 8)
        self.assertEqual(fitted.best_objective_per_variant, 4 / 8)
        self.assertEqual(fitted.selected_objective_per_variant, 4 / 8)
        self.assertTrue(all(row["status"] == "early_stopped" for row in fitted.restart_summaries))

    def test_prior_training_history_checkpoint_and_predictions(self):
        cfg = replace(self.training, seeds=(7, 8), lengthscale_multipliers=(1., 1.))
        fitted = fit_gp(self.batch.select(range(6)), self.config.model, cfg)
        chosen = min(fitted.restart_summaries, key=lambda row: row["best_objective_per_variant"])
        self.assertEqual(fitted.selected_seed, chosen["seed"])
        for row in fitted.history:
            self.assertAlmostEqual(row["objective"], row["nll"] - row["noise_log_prior"])
            self.assertEqual(row["objective_per_variant"], row["objective"] / 6)
            self.assertGreater(row["noise_variance_standardized"], cfg.noise_floor)
        nll, log_prior, _ = fitted.model.objective_components(fitted.train, fitted.standardized_targets)
        self.assertAlmostEqual(fitted.selected_nll_per_variant, nll.item() / 6)
        self.assertAlmostEqual(fitted.selected_objective_per_variant, (nll - log_prior).item() / 6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            fitted.save(path)
            restored = FittedGP.load(path)
            self.assertEqual(restored.training_config, cfg)
            self.assertEqual(restored.fit_summary(6), fitted.fit_summary(6))
            test = self.batch.select([6, 7])
            torch.testing.assert_close(restored.predict(test).mean, fitted.predict(test).mean)

    def test_prior_keeps_validation_spearman_checkpoint_selection(self):
        cfg = replace(self.training, stopping_criterion="validation_spearman",
                      validation_interval=1, patience=1)
        with patch("gp.training.spearman", side_effect=[.2, .8, .3]):
            fitted = fit_gp(self.batch.select(range(4)), self.config.model, cfg,
                            validation=self.batch.select([4, 5]))
        self.assertEqual((fitted.selected_step, fitted.best_validation_spearman), (1, .8))
        self.assertEqual(fitted.selected_objective_per_variant, fitted.history[1]["objective_per_variant"])
        self.assertEqual(fitted.selected_nll_per_variant, fitted.history[1]["nll_per_variant"])

    def test_legacy_checkpoint_without_prior_fields_loads(self):
        fitted = fit_gp(self.batch, self.config.model, replace(self.training, noise_prior="none"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            fitted.save(path)
            state = torch.load(path, weights_only=True)
            for name in ("noise_prior", "noise_prior_scale"):
                state["training_config"].pop(name)
            for name in ("best_objective_per_variant", "selected_objective_per_variant"):
                state.pop(name)
            torch.save(state, path)
            restored = FittedGP.load(path)
            self.assertEqual(restored.training_config.noise_prior, "none")
            self.assertEqual(restored.best_objective_per_variant, restored.best_nll_per_variant)
            torch.testing.assert_close(restored.predict(self.batch).mean, fitted.predict(self.batch).mean)

    def test_cli_overrides_json_and_can_disable_prior(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(replace(self.config, training=self.training).to_dict()))
            base = ["fit", "--features", directory, "--output", directory,
                    "--setting", "single_to_double", "--config", str(path)]
            for prior in ("none", "half_cauchy"):
                args = _parser().parse_args(base + ["--noise-prior", prior, "--noise-prior-scale", ".2"])
                config = _load_config(args)
                self.assertEqual((config.training.noise_prior, config.training.noise_prior_scale), (prior, .2))


if __name__ == "__main__":
    unittest.main()
