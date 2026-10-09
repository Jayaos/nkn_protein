"""Independent numerical references for the exact GP and fold-only training."""

from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

import torch

from config import BranchConfig, ModelConfig, TrainingConfig
from data_loading.features import load_feature_bundle
from gp.linalg import NumericalError, factor_covariance, negative_log_likelihood
from gp.model import ExactProteinGP
from gp.preprocessing import FoldTransform
from gp.training import FittedGP, fit_gp


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "data/features/A0A1I9GEU1_NEIME_Kennouche_2019"


def small_config():
    model = ModelConfig()
    model.branches = {k: BranchConfig(copies=v.copies, widths=(2, 1, 1)) for k, v in model.branches.items()}
    model.final_widths = (2, 1, 1)
    return model


class ExactGPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.batch = load_feature_bundle(FIXTURE)

    def test_likelihood_matches_multivariate_normal(self):
        kernel = torch.tensor([[1.3, .4], [.4, .7]], dtype=torch.float64)
        noise = torch.tensor(.08, dtype=torch.float64)
        residual = torch.tensor([.7, -.3], dtype=torch.float64)
        factor, jitter = factor_covariance(kernel, noise)
        self.assertEqual(jitter, 1e-6)
        reference = torch.distributions.MultivariateNormal(torch.zeros(2, dtype=torch.float64), covariance_matrix=kernel + (noise + jitter) * torch.eye(2, dtype=torch.float64))
        torch.testing.assert_close(negative_log_likelihood(residual, factor), -reference.log_prob(residual))

    def test_jitter_escalation_and_failure(self):
        matrix = torch.diag(torch.tensor([-5e-5, 1.0], dtype=torch.float64))
        _, jitter = factor_covariance(matrix, torch.tensor(0., dtype=torch.float64))
        self.assertAlmostEqual(jitter, 1e-4)
        with self.assertRaises(NumericalError):
            factor_covariance(-torch.eye(2, dtype=torch.float64), torch.tensor(0., dtype=torch.float64))

    def test_posterior_matches_independent_joint_conditioning(self):
        train, test = self.batch.select(range(6)), self.batch.select(range(6, 10))
        transform = FoldTransform.fit(train)
        normalized, heldout = transform.apply(train), transform.apply(test)
        y = transform.standardize_targets(train.targets)
        torch.manual_seed(11)
        model = ExactProteinGP(normalized, small_config(), TrainingConfig())
        mean, latent, observation, jitter = model.posterior(normalized, y, heldout)
        k = model.kernel(normalized)
        cross = model.kernel(heldout, normalized)
        noisy = k + (model.noise + jitter) * torch.eye(len(y), dtype=torch.float64)
        reference_mean = model.mean(heldout) + cross @ torch.linalg.solve(noisy, y - model.mean(normalized))
        reference_latent = model.kernel(heldout) - cross @ torch.linalg.solve(noisy, cross.T)
        torch.testing.assert_close(mean, reference_mean)
        torch.testing.assert_close(latent, reference_latent)
        torch.testing.assert_close(observation - latent, model.noise * torch.eye(test.num_variants, dtype=torch.float64))

    def test_training_gradients_match_finite_differences(self):
        batch = self.batch.select(range(5))
        transform = FoldTransform.fit(batch)
        train, y = transform.apply(batch), transform.standardize_targets(batch.targets)
        torch.manual_seed(4)
        model = ExactProteinGP(train, small_config(), TrainingConfig())
        total, _ = model.objective(train, y)
        total.backward()
        candidates = [model.coefficients, model.intercept, model.raw_noise]
        candidates.extend(p for n, p in model.named_parameters() if "raw_lengthscale" in n or "raw_weight" in n or "raw_bias" in n)
        self.assertGreaterEqual(len(candidates), 7)
        for parameter in candidates:
            index = (0,) * parameter.ndim
            analytical = parameter.grad[index].item()
            with torch.no_grad():
                original = parameter[index].item()
                parameter[index] = original + 1e-5
                upper = model.objective(train, y)[0].item()
                parameter[index] = original - 1e-5
                lower = model.objective(train, y)[0].item()
                parameter[index] = original
            self.assertAlmostEqual(analytical, (upper - lower) / 2e-5, delta=2e-5)

    def test_restart_selection_restore_rescaling_and_reload(self):
        train, test = self.batch.select(range(10)), self.batch.select(range(10, 16))
        cfg = TrainingConfig(max_steps=3)
        fitted = fit_gp(train, small_config(), cfg)
        self.assertEqual(len(fitted.restart_summaries), 3)
        self.assertEqual(fitted.best_nll_per_variant, min(row["best_nll_per_variant"] for row in fitted.restart_summaries))
        self.assertAlmostEqual(fitted.model.objective(fitted.train, fitted.standardized_targets)[0].item() / train.num_variants, fitted.best_nll_per_variant)
        prediction = fitted.predict(test)
        torch.testing.assert_close(prediction.observation_covariance, fitted.transform.target_scale.square() * prediction.observation_covariance_standardized)
        torch.testing.assert_close(prediction.mean, fitted.transform.target_mean + fitted.transform.target_scale * prediction.mean_standardized)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            fitted.save(path)
            restored = FittedGP.load(path)
            actual = restored.predict(replace(test, targets=None))
            torch.testing.assert_close(actual.mean, prediction.mean, rtol=0, atol=0)
            torch.testing.assert_close(actual.observation_covariance, prediction.observation_covariance, rtol=0, atol=0)

    def test_heldout_labels_do_not_change_training_or_predictions(self):
        train_ids, test_ids = list(range(10)), list(range(10, 16))
        changed = self.batch.targets.clone()
        changed[test_ids] = changed[test_ids] * 400 + 999
        altered = replace(self.batch, targets=changed)
        cfg = TrainingConfig(max_steps=2, seeds=(8,), lengthscale_multipliers=(1.,))
        first = fit_gp(self.batch.select(train_ids), small_config(), cfg)
        second = fit_gp(altered.select(train_ids), small_config(), cfg)
        for key, value in first.model.state_dict().items():
            torch.testing.assert_close(value, second.model.state_dict()[key], atol=0, rtol=0)
        torch.testing.assert_close(first.predict(self.batch.select(test_ids)).mean, second.predict(altered.select(test_ids)).mean, atol=0, rtol=0)

    def test_constant_scores_and_targets(self):
        batch = self.batch.select(range(5))
        constant_scores = replace(batch, scores=torch.ones_like(batch.scores))
        transform = FoldTransform.fit(constant_scores)
        self.assertEqual(transform.apply(constant_scores).scores.abs().sum().item(), 0)
        with self.assertRaisesRegex(ValueError, "constant training targets"):
            FoldTransform.fit(replace(batch, targets=torch.ones_like(batch.targets)))

    def test_frozen_input_graphs_and_caller_mutation(self):
        batch = self.batch.select(range(5))
        train = replace(batch, scores=batch.scores.clone().requires_grad_(),
                        global_features=batch.global_features.clone().requires_grad_(),
                        local_features={k: v.clone().requires_grad_() for k, v in batch.local_features.items()})
        cfg = TrainingConfig(max_steps=2, seeds=(0,), lengthscale_multipliers=(1.,))
        fitted = fit_gp(train, small_config(), cfg)
        before = fitted.predict(batch).mean.clone()
        self.assertIsNone(train.scores.grad)
        self.assertIsNone(train.global_features.grad)
        self.assertFalse(fitted.train.scores.requires_grad)
        with torch.no_grad():
            train.global_features.fill_(400)
        torch.testing.assert_close(fitted.predict(batch).mean, before, atol=0, rtol=0)

    def test_prediction_rejects_changed_source_identity(self):
        train = self.batch.select(range(5))
        cfg = TrainingConfig(max_steps=1, seeds=(0,), lengthscale_multipliers=(1.,))
        fitted = fit_gp(train, small_config(), cfg)
        test = self.batch.select(range(5, 8))
        with self.assertRaisesRegex(ValueError, "scorers or their order"):
            fitted.predict(replace(test, score_names=tuple(reversed(test.score_names))))
        with self.assertRaisesRegex(ValueError, "feature sources"):
            fitted.predict(replace(test, feature_signature={**test.feature_signature, "pooling": "changed"}))

    def test_stopping_and_capacity_guard(self):
        cfg = TrainingConfig(max_steps=15, min_steps=2, patience=2, tolerance=1e6, seeds=(0,), lengthscale_multipliers=(1.,))
        fitted = fit_gp(self.batch.select(range(5)), small_config(), cfg)
        self.assertEqual(fitted.restart_summaries[0]["status"], "early_stopped")
        self.assertEqual(fitted.history[-1]["step"], 2)
        with self.assertRaisesRegex(ValueError, "dense-GP limit"):
            fit_gp(self.batch, small_config(), replace(cfg, max_train_variants=2))


if __name__ == "__main__":
    unittest.main()
