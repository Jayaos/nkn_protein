"""Synthetic-only integration checks for the optional iterative GP backend."""

from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from config import BranchConfig, ExperimentConfig, ModelConfig, TrainingConfig
from experiments.synthetic_gpu import make_synthetic_batch, validate_backend_parity
from gp.model import ExactProteinGP
from gp.preprocessing import FoldTransform
from gp.training import FittedGP, fit_gp


BBMM_AVAILABLE = importlib.util.find_spec("linear_operator") is not None


def compact_model():
    defaults = ModelConfig()
    return ModelConfig(
        enabled_branches=("inverse_folding", "sequence_site", "sequence_global", "structure_distance"),
        branches={name: BranchConfig(copies=value.copies, widths=(1,))
                  for name, value in defaults.branches.items()},
        final_widths=(2, 1, 1),
    )


def bbmm_config(**overrides):
    values = dict(
        inference_backend="bbmm", max_steps=2, min_steps=0, patience=3,
        tolerance=0, noise_initial=1.0, seeds=(7,), lengthscale_multipliers=(1.0,),
        bbmm_cg_tolerance=1e-10, bbmm_max_cg_iterations=128,
        bbmm_num_trace_samples=0, bbmm_max_lanczos_iterations=32,
        bbmm_preconditioner_size=0, bbmm_probe_seed=9,
    )
    values.update(overrides)
    return TrainingConfig(**values)


class BBMMConfigurationTests(unittest.TestCase):
    def test_default_backend_and_json_roundtrip(self):
        self.assertEqual(TrainingConfig().inference_backend, "cholesky")
        default = TrainingConfig()
        self.assertEqual(default.bbmm_cg_tolerance, 1e-5)
        self.assertEqual(default.bbmm_max_cg_iterations, 1000)
        self.assertEqual(default.bbmm_num_trace_samples, 32)
        self.assertEqual(default.bbmm_max_lanczos_iterations, 64)
        self.assertEqual(default.bbmm_preconditioner_size, 32)
        self.assertEqual(default.bbmm_probe_seed, 0)
        config = ExperimentConfig(model=compact_model(), training=bbmm_config())
        serialized = json.loads(json.dumps(config.to_dict()))
        restored = ExperimentConfig.from_dict(serialized)
        self.assertEqual(restored.training, config.training)
        self.assertEqual(restored.model, config.model)

    def test_invalid_settings_are_rejected(self):
        invalid = {
            "inference_backend": ("unknown", "BBMM", None),
            "bbmm_cg_tolerance": (0, -1, 1, True, float("nan"), float("inf")),
            "bbmm_max_cg_iterations": (0, -1, 1.5, True),
            "bbmm_num_trace_samples": (-1, 1.5, True),
            "bbmm_max_lanczos_iterations": (0, -1, 1.5, True),
            "bbmm_preconditioner_size": (-1, 1.5, True),
            "bbmm_probe_seed": (-1, 1.5, True),
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(setting=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        TrainingConfig(**{name: value})


@unittest.skipUnless(BBMM_AVAILABLE, "Optional linear_operator dependency is unavailable")
class BBMMIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        self.batch = make_synthetic_batch(n_variants=16, feature_dim=4, seed=37)
        self.model_config = compact_model()
        self.config = bbmm_config()

    def compare_backends(self, device):
        train_raw = self.batch.select(range(12))
        transform = FoldTransform.fit(train_raw)
        train = transform.apply(train_raw).to(device)
        heldout = transform.apply(replace(self.batch.select(range(12, 16)), targets=None)).to(device)
        targets = transform.standardize_targets(train_raw.targets).to(device)
        devices = [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(123)
            reference = ExactProteinGP(train, self.model_config, replace(self.config, inference_backend="cholesky"))
            actual = ExactProteinGP(train, self.model_config, self.config)
        actual.load_state_dict(reference.state_dict())
        for model in (reference, actual):
            model.kernel.prepare_training_geometry(train)
        expected_loss, expected_jitter = reference.objective(train, targets)
        actual_loss, actual_jitter = actual.objective(train, targets)
        self.assertEqual(actual_jitter, expected_jitter)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-6, atol=1e-7)
        expected_loss.backward()
        actual_loss.backward()
        expected_parameters = dict(reference.named_parameters())
        for name, parameter in actual.named_parameters():
            with self.subTest(device=device, parameter=name):
                self.assertIsNotNone(parameter.grad)
                torch.testing.assert_close(parameter.grad, expected_parameters[name].grad,
                                           rtol=1e-5, atol=1e-7)
        with torch.no_grad():
            expected_posterior = reference.posterior(train, targets, heldout)
            actual_posterior = actual.posterior(train, targets, heldout)
            for actual_value, expected_value in zip(actual_posterior[:3], expected_posterior[:3]):
                torch.testing.assert_close(actual_value, expected_value, rtol=1e-7, atol=1e-8)
            torch.testing.assert_close(actual.posterior_mean(train, targets, heldout),
                                       expected_posterior[0], rtol=1e-7, atol=1e-8)
        self.assertTrue(torch.isfinite(actual_posterior[2]).all())

    def test_cpu_objective_gradients_and_posterior_match_cholesky(self):
        self.compare_backends("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_objective_gradients_and_posterior_match_cholesky(self):
        self.compare_backends("cuda")

    def test_preflight_uses_deterministic_validation_without_changing_run_settings(self):
        config = ExperimentConfig(
            model=self.model_config,
            training=bbmm_config(bbmm_num_trace_samples=32, bbmm_preconditioner_size=32,
                                 bbmm_cg_tolerance=1e-5),
        )
        original = config.to_dict()
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            with self.subTest(device=device):
                cpu_state = torch.random.get_rng_state().clone()
                cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                report = validate_backend_parity(config, device=device)
                self.assertEqual(report["status"], "passed")
                self.assertEqual(report["validation_settings"]["bbmm_num_trace_samples"], 0)
                self.assertEqual(report["validation_settings"]["bbmm_preconditioner_size"], 0)
                self.assertEqual(set(report["max_abs_error"]), {
                    "kernel", "objective", "gradients", "posterior_mean", "posterior_latent",
                    "posterior_observation",
                })
                self.assertTrue(all(math.isfinite(value) for value in report["max_abs_error"].values()))
                self.assertEqual(config.to_dict(), original)
                torch.testing.assert_close(torch.random.get_rng_state(), cpu_state, rtol=0, atol=0)
                for current, previous in zip(torch.cuda.get_rng_state_all(), cuda_states):
                    torch.testing.assert_close(current, previous, rtol=0, atol=0)

    def test_two_step_fit_and_checkpoint_roundtrip(self):
        train = self.batch.select(range(12))
        heldout = replace(self.batch.select(range(12, 16)), targets=None)
        fitted = fit_gp(train, self.model_config, self.config)
        self.assertEqual([row["step"] for row in fitted.history], [0, 1, 2])
        self.assertTrue(all(row["inference_backend"] == "bbmm" for row in fitted.history))
        self.assertEqual(fitted.fit_summary(train.num_variants)["inference_backend"], "bbmm")
        self.assertTrue(math.isfinite(fitted.selected_nll_per_variant))
        prediction = fitted.predict(heldout)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bbmm_checkpoint.pt"
            fitted.save(path)
            restored = FittedGP.load(path)
            self.assertEqual(restored.training_config, fitted.training_config)
            self.assertEqual(restored.training_config.inference_backend, "bbmm")
            for name, value in fitted.model.state_dict().items():
                torch.testing.assert_close(restored.model.state_dict()[name], value, rtol=0, atol=0)
            reloaded = restored.predict(heldout)
            torch.testing.assert_close(reloaded.mean, prediction.mean, rtol=1e-7, atol=1e-8)
            torch.testing.assert_close(reloaded.observation_covariance, prediction.observation_covariance,
                                       rtol=1e-7, atol=1e-8)
            if torch.cuda.is_available():
                gpu_reloaded = FittedGP.load(path, device="cuda").predict(heldout)
                torch.testing.assert_close(gpu_reloaded.mean.cpu(), prediction.mean, rtol=1e-7, atol=1e-8)
                torch.testing.assert_close(gpu_reloaded.observation_covariance.cpu(),
                                           prediction.observation_covariance, rtol=1e-7, atol=1e-8)

    def test_legacy_checkpoint_defaults_to_cholesky(self):
        train = self.batch.select(range(12))
        heldout = replace(self.batch.select(range(12, 16)), targets=None)
        fitted = fit_gp(train, self.model_config,
                        replace(self.config, inference_backend="cholesky", max_steps=1))
        expected = fitted.predict(heldout)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_checkpoint.pt"
            fitted.save(path)
            state = torch.load(path, map_location="cpu", weights_only=True)
            for name in tuple(state["training_config"]):
                if name == "inference_backend" or name.startswith("bbmm_"):
                    del state["training_config"][name]
            torch.save(state, path)
            restored = FittedGP.load(path)
            self.assertEqual(restored.training_config.inference_backend, "cholesky")
            actual = restored.predict(heldout)
            torch.testing.assert_close(actual.mean, expected.mean, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(actual.observation_covariance, expected.observation_covariance,
                                       rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
