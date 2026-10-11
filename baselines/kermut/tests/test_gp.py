"""Independent likelihood, conditioning, optimization, and checkpoint checks."""

from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

import torch

from baselines.kermut.config import KermutTrainingConfig
from baselines.kermut.model import ExactKermutGP
from baselines.kermut.training import FittedKermut, fit_gp
from fixtures import small_config, synthetic_batch


class ExactGPTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.batch = synthetic_batch()
        self.config = small_config(n_steps=3)
        self.train = self.batch.select(range(8))
        self.test = self.batch.select(range(10, 15))

    def test_objective_is_exact_nll_minus_half_cauchy_noise_prior(self):
        model = ExactKermutGP(self.train, self.config.model, self.config.training)
        y = (self.train.targets - self.train.targets.mean()) / self.train.targets.std(unbiased=True)
        actual, jitter = model.objective(self.train, y)
        covariance = model.kernel(self.train) + (model.noise + jitter) * torch.eye(len(y), dtype=torch.float64)
        reference_nll = -torch.distributions.MultivariateNormal(model.mean(self.train), covariance_matrix=covariance).log_prob(y)
        scale = self.config.training.noise_prior_scale
        reference_log_prior = math.log(2 / (math.pi * scale)) - torch.log1p((model.noise / scale) ** 2)
        torch.testing.assert_close(actual, reference_nll - reference_log_prior)
        components = model.objective_components(self.train, y)
        torch.testing.assert_close(components[0], reference_nll)
        torch.testing.assert_close(components[1], reference_log_prior)
        self.assertEqual(jitter, 0)

    def test_posterior_matches_independent_gaussian_conditioning(self):
        model = ExactKermutGP(self.train, self.config.model, self.config.training)
        y = (self.train.targets - self.train.targets.mean()) / self.train.targets.std(unbiased=True)
        mean, latent, observed, jitter = model.posterior(self.train, y, self.test)
        k_train = model.kernel(self.train) + (model.noise + jitter) * torch.eye(len(y), dtype=torch.float64)
        cross = model.kernel(self.test, self.train)
        expected_mean = model.mean(self.test) + cross @ torch.linalg.solve(k_train, y - model.mean(self.train))
        expected_latent = model.kernel(self.test) - cross @ torch.linalg.solve(k_train, cross.T)
        torch.testing.assert_close(mean, expected_mean)
        torch.testing.assert_close(latent, expected_latent)
        torch.testing.assert_close(observed - latent, model.noise * torch.eye(self.test.num_variants, dtype=torch.float64))
        self.assertGreater(observed[0, 1].abs().item(), 0)

    def test_training_uses_sample_sd_raw_scores_and_final_adamw_state(self):
        defaults = KermutTrainingConfig()
        self.assertEqual(defaults.n_steps, 150)
        self.assertEqual(defaults.learning_rate, 0.1)
        self.assertEqual(defaults.weight_decay, 0.01)
        self.assertEqual(defaults.noise_prior_scale, 0.1)
        fitted = fit_gp(self.train, self.config.model, self.config.training)
        torch.testing.assert_close(fitted.transform.target_mean, self.train.targets.mean())
        torch.testing.assert_close(fitted.transform.target_scale, self.train.targets.std(unbiased=True))
        torch.testing.assert_close(fitted.train.scores, self.train.scores, atol=0, rtol=0)
        torch.testing.assert_close(fitted.standardized_targets, (self.train.targets - self.train.targets.mean()) / self.train.targets.std(unbiased=True))

        # Independently replay the exact update count. This verifies the saved
        # state is the final state, not a best-iteration checkpoint or restart.
        model = ExactKermutGP(self.train, self.config.model, self.config.training)
        cfg = self.config.training
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=cfg.betas, eps=cfg.epsilon)
        for _ in range(cfg.n_steps):
            optimizer.zero_grad()
            loss = model.objective(self.train, fitted.standardized_targets)[0] / self.train.num_variants
            loss.backward()
            optimizer.step()
        for name, value in model.state_dict().items():
            torch.testing.assert_close(fitted.model.state_dict()[name], value, atol=1e-12, rtol=1e-12)
        self.assertEqual(len(fitted.restart_summaries), 1)
        self.assertEqual(fitted.history[-1]["step"], cfg.n_steps)
        self.assertAlmostEqual(fitted.final_objective_per_variant, model.objective(self.train, fitted.standardized_targets)[0].item() / self.train.num_variants)

    def test_checkpoint_preserves_label_free_predictions_and_source_guards(self):
        fitted = fit_gp(self.train, self.config.model, self.config.training)
        expected = fitted.predict(self.test)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            fitted.save(checkpoint)
            restored = FittedKermut.load(checkpoint)
            actual = restored.predict(replace(self.test, targets=None))
            torch.testing.assert_close(actual.mean, expected.mean, rtol=0, atol=0)
            torch.testing.assert_close(actual.observation_covariance, expected.observation_covariance, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "scorers|scores"):
                restored.predict(replace(self.test, score_names=tuple(reversed(self.test.score_names))))
            with self.assertRaisesRegex(ValueError, "feature sources|preprocessing"):
                restored.predict(replace(self.test, feature_signature={"source": "different"}))
        torch.testing.assert_close(expected.mean, fitted.transform.target_mean + fitted.transform.target_scale * expected.mean_standardized)
        torch.testing.assert_close(expected.observation_covariance, fitted.transform.target_scale.square() * expected.observation_covariance_standardized)

    def test_frozen_inputs_and_heldout_labels_cannot_change_model(self):
        training = replace(self.train, scores=self.train.scores.clone().requires_grad_(),
                           global_features=self.train.global_features.clone().requires_grad_(),
                           local_features={key: value.clone().requires_grad_() for key, value in self.train.local_features.items()})
        fitted = fit_gp(training, self.config.model, self.config.training)
        before = fitted.predict(self.test).mean
        changed_labels = replace(self.test, targets=self.test.targets * 100 + 55)
        torch.testing.assert_close(fitted.predict(changed_labels).mean, before, rtol=0, atol=0)
        self.assertIsNone(training.scores.grad)
        self.assertIsNone(training.global_features.grad)
        self.assertFalse(fitted.train.scores.requires_grad)
        with torch.no_grad():
            training.global_features.fill_(50)
            training.scores.fill_(100)
            training.local_features["structure"].fill_(100)
        torch.testing.assert_close(fitted.predict(self.test).mean, before, rtol=0, atol=0)

    def test_prediction_rejects_changed_reference_inputs_despite_unchanged_signature(self):
        fitted = fit_gp(self.train, self.config.model, self.config.training)
        rows = self.test.mutation_positions == 1
        coordinates = self.test.local_features["structure"].clone()
        coordinates[rows, 0] += 2
        changed_structure = replace(self.test, local_features={**self.test.local_features, "structure": coordinates})
        with self.assertRaisesRegex(ValueError, "reference|Reference"):
            fitted.predict(changed_structure)
        probabilities = self.test.local_features["inverse_folding"].clone()
        probabilities[rows] = probabilities[rows].roll(1, dims=-1)
        changed_probabilities = replace(self.test, local_features={**self.test.local_features, "inverse_folding": probabilities})
        with self.assertRaisesRegex(ValueError, "reference|Reference"):
            fitted.predict(changed_probabilities)

    def test_mean_supports_one_or_multiple_selected_scores(self):
        one_score = replace(self.train, scores=self.train.scores[:, :1], score_names=self.train.score_names[:1])
        for batch in (one_score, self.train):
            with self.subTest(n_scores=batch.num_scores):
                model = ExactKermutGP(batch, self.config.model, self.config.training)
                torch.testing.assert_close(model.mean(batch), batch.scores.sum(dim=1))
                fitted = fit_gp(batch, self.config.model, self.config.training)
                self.assertEqual(fitted.model.coefficients.numel(), batch.num_scores)
                self.assertTrue(torch.isfinite(fitted.predict(batch).mean).all())


if __name__ == "__main__":
    unittest.main()
