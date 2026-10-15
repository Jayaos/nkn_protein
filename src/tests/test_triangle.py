"""Triangle evaluation must preserve dense cached GP values and gradients."""

from copy import deepcopy
from dataclasses import fields, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from config import TrainingConfig
from gp.model import ExactProteinGP
from gp.training import FittedGP, Prediction, fit_gp
from kernels import ProteinKernel
from tests.test_kernels import CONDITIONAL_RECIPE, batch, model_config
from tests.test_pipeline import small_config, synthetic_batch


DTYPE = torch.float64


def assert_close(actual, expected, *, rtol=1e-10, atol=1e-11):
    # Histories contain Python floats: retain double precision when comparing.
    torch.testing.assert_close(torch.as_tensor(actual, dtype=DTYPE),
                               torch.as_tensor(expected, dtype=DTYPE), rtol=rtol, atol=atol)


def conditional_batch():
    inputs = synthetic_batch()
    mass = torch.tensor([0.2, 0.5, 0.8, 0.6, 0.4], dtype=DTYPE)
    return replace(inputs, local_features={**inputs.local_features, "inverse_folding":
        inputs.local_features["inverse_folding"] * mass[inputs.mutation_positions - 1, None]},
        feature_signature={**inputs.feature_signature,
            "inverse_folding.probability_recipe": json.dumps(CONDITIONAL_RECIPE)})


def asymmetric_sensitivity(matrix):
    weights = torch.arange(1, matrix.numel() + 1, dtype=DTYPE).reshape(matrix.shape)
    # Distinct coefficients for both directions and the diagonal catch mirror
    # gradient errors that a simple symmetric sum can miss.
    if matrix.shape[0] == matrix.shape[1]:
        weights = weights + 17 * torch.eye(matrix.shape[0], dtype=DTYPE)
    return (matrix.square() * weights).sum()


class TriangleTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def kernels(self, inputs):
        torch.manual_seed(11)
        reference = ProteinKernel(model_config(), inputs)
        actual = deepcopy(reference)
        before = deepcopy(actual.state_dict())
        reference.prepare_training_geometry(inputs, use_triangle=False)
        actual.prepare_training_geometry(inputs)
        self.assertIsNone(reference._training_triangle)
        self.assertIsNotNone(actual._training_triangle)
        self.assertEqual(actual.state_dict().keys(), before.keys())
        for name, value in before.items():
            torch.testing.assert_close(actual.state_dict()[name], value, rtol=0, atol=0)
        return actual, reference

    def gradients_equal(self, actual, reference):
        actual_parameters = dict(actual.named_parameters())
        reference_parameters = dict(reference.named_parameters())
        self.assertEqual(actual_parameters.keys(), reference_parameters.keys())
        for name, parameter in actual_parameters.items():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertIsNotNone(reference_parameters[name].grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                assert_close(parameter.grad, reference_parameters[name].grad)

    def kernels_and_gradients_equal(self, actual, reference, *inputs):
        actual.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        result, expected = actual(*inputs), reference(*inputs)
        assert_close(result, expected)
        asymmetric_sensitivity(result).backward()
        asymmetric_sensitivity(expected).backward()
        self.gradients_equal(actual, reference)

    def test_kernel_and_gradients_with_duplicates_and_multiple_mutations(self):
        for inputs in (synthetic_batch(), conditional_batch()):
            with self.subTest(conditional="inverse_folding.probability_recipe" in inputs.feature_signature):
                actual, reference = self.kernels(inputs)
                with patch.object(actual, "_forward_triangle", wraps=actual._forward_triangle) as triangle:
                    self.kernels_and_gradients_equal(actual, reference, inputs)
                    self.kernels_and_gradients_equal(actual, reference, inputs, inputs)
                    self.assertEqual(triangle.call_count, 2)
                    for left, right in zip(actual.parameters(), reference.parameters()):
                        with torch.no_grad():
                            left.add_(0.07)
                            right.add_(0.07)
                    self.kernels_and_gradients_equal(actual, reference, inputs)
                    self.assertEqual(triangle.call_count, 3)

    def test_finite_differences_check_diagonal_and_mirrored_gradients(self):
        inputs = conditional_batch()
        actual, reference = self.kernels(inputs)
        self.kernels_and_gradients_equal(actual, reference, inputs)
        names = [name for name, value in actual.named_parameters() if "raw_lengthscale" in name][:1]
        names += [name for name, value in actual.named_parameters()
                  if name.startswith("final_nkn") and name.endswith(("raw_weight", "raw_bias"))]
        parameters = dict(actual.named_parameters())
        for name in names:
            parameter = parameters[name]
            initial = parameter.detach().clone()
            analytical = parameter.grad.flatten()[0].detach().clone()
            step = 1e-5
            with torch.no_grad():
                parameter.flatten()[0] += step
                plus = asymmetric_sensitivity(actual(inputs))
                parameter.copy_(initial)
                parameter.flatten()[0] -= step
                minus = asymmetric_sensitivity(actual(inputs))
                parameter.copy_(initial)
            with self.subTest(parameter=name):
                assert_close(analytical, (plus - minus) / (2 * step), rtol=2e-5, atol=2e-7)

    def test_all_gp_parameter_gradients_and_likelihood_match(self):
        for inputs in (synthetic_batch(), conditional_batch()):
            torch.manual_seed(31)
            reference = ExactProteinGP(inputs, small_config().model, TrainingConfig())
            actual = deepcopy(reference)
            reference.kernel.prepare_training_geometry(inputs, use_triangle=False)
            actual.kernel.prepare_training_geometry(inputs)
            actual_loss, actual_jitter = actual.objective(inputs, inputs.targets)
            reference_loss, reference_jitter = reference.objective(inputs, inputs.targets)
            assert_close(actual_loss, reference_loss)
            self.assertEqual(actual_jitter, reference_jitter)
            actual_loss.backward()
            reference_loss.backward()
            self.gradients_equal(actual, reference)

    def test_rectangular_pairs_and_different_batches_use_dense_path(self):
        inputs = synthetic_batch()
        actual, reference = self.kernels(inputs)
        heldout = inputs.select([1, 4, 7])
        with patch.object(actual, "_forward_triangle", wraps=actual._forward_triangle) as triangle:
            self.kernels_and_gradients_equal(actual, reference, inputs, heldout)
            self.kernels_and_gradients_equal(actual, reference, heldout, inputs)
            self.kernels_and_gradients_equal(actual, reference, heldout)
            self.kernels_and_gradients_equal(actual, reference, deepcopy(inputs))
            triangle.assert_not_called()
            self.kernels_and_gradients_equal(actual, reference, inputs)
            triangle.assert_called_once()

    def test_changed_features_and_mutation_ownership_use_dense_path(self):
        for change in ("global_inplace", "local_inplace", "global_replaced", "owners", "positions"):
            with self.subTest(change=change):
                inputs = synthetic_batch()
                actual, reference = self.kernels(inputs)
                if change == "global_inplace":
                    inputs.global_features[0, 0] += 3
                elif change == "local_inplace":
                    inputs.local_features["structure"][0, 0] += 3
                elif change == "global_replaced":
                    inputs = replace(inputs, global_features=inputs.global_features.flip(0).clone())
                elif change == "owners":
                    inputs.owners[5] = 6
                else:
                    inputs.mutation_positions[0] += 1
                with patch.object(actual, "_forward_triangle", wraps=actual._forward_triangle) as triangle:
                    self.kernels_and_gradients_equal(actual, reference, inputs)
                    triangle.assert_not_called()

    def test_clear_move_and_reprepare_invalidate_triangle_cache(self):
        inputs = synthetic_batch()
        actual, reference = self.kernels(inputs)
        for invalidate in (actual.clear_geometry_cache, lambda: actual.to(dtype=DTYPE),
                           lambda: actual.to(device="cpu")):
            actual.prepare_training_geometry(inputs)
            self.assertIsNotNone(actual._training_triangle)
            invalidate()
            self.assertIsNone(actual._training_triangle)
            with patch.object(actual, "_forward_triangle", wraps=actual._forward_triangle) as triangle:
                self.kernels_and_gradients_equal(actual, reference, inputs)
                triangle.assert_not_called()
                actual.prepare_training_geometry(inputs)
                self.kernels_and_gradients_equal(actual, reference, inputs)
                triangle.assert_called_once()

    def test_feature_gradients_before_or_after_preparation_use_dense_path(self):
        for before in (False, True):
            with self.subTest(before_preparation=before):
                inputs = synthetic_batch()
                expected_inputs = deepcopy(inputs)
                torch.manual_seed(11)
                actual = ProteinKernel(model_config(), inputs)
                reference = deepcopy(actual)
                inputs.global_features.requires_grad_(before)
                actual.prepare_training_geometry(inputs)
                inputs.global_features.requires_grad_()
                expected_inputs.global_features.requires_grad_()
                with patch.object(actual, "_forward_triangle", wraps=actual._forward_triangle) as triangle:
                    result, expected = actual(inputs), reference(expected_inputs)
                    triangle.assert_not_called()
                assert_close(result, expected)
                asymmetric_sensitivity(result).backward()
                asymmetric_sensitivity(expected).backward()
                self.assertIsNotNone(inputs.global_features.grad)
                assert_close(inputs.global_features.grad, expected_inputs.global_features.grad)
                self.gradients_equal(actual, reference)

    def test_preparing_in_inference_mode_preserves_later_backward(self):
        inputs = synthetic_batch()
        actual, reference = self.kernels(inputs)
        with torch.inference_mode():
            actual.prepare_training_geometry(inputs)
        self.assertIsNone(actual._training_triangle)
        self.kernels_and_gradients_equal(actual, reference, inputs)

    def test_empty_mutation_sets_and_empty_variant_batches(self):
        for global_features in ([[1., 0.], [0., 1.]], []):
            with self.subTest(variants=len(global_features)):
                inputs = batch(global_features, [], [])
                inputs.global_features = inputs.global_features.reshape(-1, 2)
                actual, reference = self.kernels(inputs)
                assert_close(actual(inputs), reference(inputs))

    def assert_fits_equal(self, actual, reference, heldout):
        self.assertEqual(actual.selected_seed, reference.selected_seed)
        self.assertEqual(len(actual.history), len(reference.history))
        self.assertEqual(len(actual.restart_summaries), len(reference.restart_summaries))
        for left, right in zip(actual.history, reference.history):
            self.assertEqual(left.keys(), right.keys())
            for name in left:
                if name in ("seed", "step", "jitter"):
                    self.assertEqual(left[name], right[name])
                else:
                    assert_close(left[name], right[name], rtol=1e-9, atol=1e-10)
        for left, right in zip(actual.restart_summaries, reference.restart_summaries):
            for name in left:
                if name.endswith("_per_variant") and left[name] is not None:
                    assert_close(left[name], right[name], rtol=1e-9, atol=1e-10)
                else:
                    self.assertEqual(left[name], right[name])
        assert_close(actual.best_nll_per_variant, reference.best_nll_per_variant, rtol=1e-9, atol=1e-10)
        self.assertEqual(actual.model.state_dict().keys(), reference.model.state_dict().keys())
        for name, value in actual.model.state_dict().items():
            assert_close(value, reference.model.state_dict()[name], rtol=1e-9, atol=1e-10)
        predicted, expected = actual.predict(heldout), reference.predict(heldout)
        for field in fields(Prediction):
            assert_close(getattr(predicted, field.name), getattr(expected, field.name), rtol=1e-9, atol=1e-10)

    def test_same_seed_training_predictions_and_checkpoint_reload(self):
        original_prepare = ProteinKernel.prepare_training_geometry

        def prepare_dense(kernel, train):
            original_prepare(kernel, train, use_triangle=False)

        for inputs in (synthetic_batch(), conditional_batch()):
            with self.subTest(conditional="inverse_folding.probability_recipe" in inputs.feature_signature):
                train, heldout = inputs.select(range(6)), inputs.select(range(6, 8))
                config = TrainingConfig(max_steps=20, min_steps=10, patience=5, tolerance=0,
                                        seeds=(3, 9), lengthscale_multipliers=(0.5, 2.0))
                with patch.object(ProteinKernel, "prepare_training_geometry", prepare_dense):
                    reference = fit_gp(train, small_config().model, config)
                actual = fit_gp(train, small_config().model, config)
                self.assertIsNotNone(actual.model.kernel._training_triangle)
                self.assert_fits_equal(actual, reference, heldout)
                self.assertEqual(len(actual.history), 2 * (config.max_steps + 1))
                self.assertTrue(all(row["status"] == "complete" for row in actual.restart_summaries))
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "triangle.pt"
                    actual.save(path)
                    state = torch.load(path, weights_only=True)
                    self.assertEqual(state["format_version"], 1)
                    self.assertEqual(state["parameters"].keys(), reference.model.state_dict().keys())
                    restored = FittedGP.load(path)
                    self.assertIsNone(restored.model.kernel._training_triangle)
                    self.assert_fits_equal(restored, reference, heldout)

    def test_same_seed_early_stopping(self):
        original_prepare = ProteinKernel.prepare_training_geometry

        def prepare_dense(kernel, train):
            original_prepare(kernel, train, use_triangle=False)

        inputs = synthetic_batch()
        train, heldout = inputs.select(range(6)), inputs.select(range(6, 8))
        config = TrainingConfig(max_steps=30, min_steps=12, patience=4, tolerance=1e6,
                                seeds=(12,), lengthscale_multipliers=(1.0,))
        with patch.object(ProteinKernel, "prepare_training_geometry", prepare_dense):
            reference = fit_gp(train, small_config().model, config)
        actual = fit_gp(train, small_config().model, config)
        self.assert_fits_equal(actual, reference, heldout)
        self.assertEqual(actual.restart_summaries[0]["status"], "early_stopped")
        self.assertEqual(actual.history[-1]["step"], config.min_steps)


if __name__ == "__main__":
    unittest.main()
