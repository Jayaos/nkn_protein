"""Fixed geometry must preserve kernels, gradients, fitting, and prediction."""

from copy import deepcopy
from dataclasses import fields, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from config import TrainingConfig
from gp.training import FittedGP, Prediction, fit_gp
from kernels import PrimitiveNKN, ProteinKernel
from tests.test_kernels import CONDITIONAL_RECIPE, batch, model_config
from tests.test_pipeline import small_config, synthetic_batch


DTYPE = torch.float64
VECTOR_COPIES = {"cosine": 1, "rbf": 2, "matern32": 2}
PROBABILITY_COPIES = {"hellinger_exp": 2, "hellinger_rbf": 2, "bhattacharyya": 1}


class GeometryAssertions(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def assert_parameters_and_gradients_equal(self, actual, expected):
        actual_parameters = dict(actual.named_parameters())
        expected_parameters = dict(expected.named_parameters())
        self.assertEqual(actual_parameters.keys(), expected_parameters.keys())
        for name, parameter in actual_parameters.items():
            with self.subTest(parameter=name):
                reference = expected_parameters[name]
                torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
                self.assertIsNotNone(parameter.grad)
                self.assertIsNotNone(reference.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                torch.testing.assert_close(parameter.grad, reference.grad, rtol=1e-12, atol=1e-12)

    def assert_kernel_and_gradients_equal(self, actual, expected, *inputs):
        actual.zero_grad(set_to_none=True)
        expected.zero_grad(set_to_none=True)
        result, reference = actual(*inputs), expected(*inputs)
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        # Nonuniform weights exercise individual pairs rather than only a sum.
        weights = torch.arange(1, result.numel() + 1, dtype=DTYPE).reshape(result.shape)
        (result.square() * weights).sum().backward()
        (reference.square() * weights).sum().backward()
        self.assert_parameters_and_gradients_equal(actual, expected)


class PrimitiveGeometryCacheTests(GeometryAssertions):
    def examples(self):
        generator = torch.Generator().manual_seed(17)
        vectors = torch.randn(5, 7, generator=generator, dtype=DTYPE)
        probabilities = torch.softmax(torch.randn(5, 20, generator=generator, dtype=DTYPE), dim=1)
        for copies, values, policy in (
            (VECTOR_COPIES, vectors, "normalized"),
            (PROBABILITY_COPIES, probabilities, "normalized"),
            (PROBABILITY_COPIES, probabilities * 0.73, "sequence_conditional"),
        ):
            # Duplicate sites must retain finite gradients and exact zero distances.
            yield copies, torch.cat([values, values[:1]]), policy

    def test_values_and_all_parameter_gradients_match_with_duplicate_inputs(self):
        for copies, inputs, policy in self.examples():
            with self.subTest(policy=policy, primitives=tuple(copies)):
                torch.manual_seed(22)
                reference = PrimitiveNKN(copies, (4, 2, 1), probability_policy=policy)
                cached = deepcopy(reference)
                cached.prepare_geometry(inputs)
                self.assert_kernel_and_gradients_equal(cached, reference, inputs)
                # Geometry remains valid when every trainable parameter changes.
                before = cached(inputs).detach().clone()
                for actual_parameter, expected_parameter in zip(cached.parameters(), reference.parameters()):
                    with torch.no_grad():
                        actual_parameter.add_(0.07)
                        expected_parameter.add_(0.07)
                self.assert_kernel_and_gradients_equal(cached, reference, inputs)
                self.assertFalse(torch.equal(before, cached(inputs)))

    def test_each_primitive_channel_matches_original(self):
        for copies, inputs, policy in self.examples():
            with self.subTest(policy=policy, primitives=tuple(copies)):
                kernel = PrimitiveNKN(copies, (4, 2, 1), probability_policy=policy)
                expected = torch.stack([primitive(inputs) for primitive in kernel.primitives], dim=-1)
                observed = []
                handle = kernel.nkn.register_forward_pre_hook(lambda module, args: observed.append(args[0]))
                try:
                    kernel.prepare_geometry(inputs)
                    kernel(inputs)
                finally:
                    handle.remove()
                self.assertEqual(len(observed), 1)
                torch.testing.assert_close(observed[0], expected, rtol=0, atol=0)

    def test_hits_skip_distances_and_prediction_pairs_preserve_training_cache(self):
        inputs = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=DTYPE)
        heldout = torch.tensor([[2., 1.], [-1., 2.]], dtype=DTYPE)
        kernel = PrimitiveNKN(VECTOR_COPIES, (4, 2, 1))
        reference = deepcopy(kernel)
        with patch("torch.cdist", wraps=torch.cdist) as distance:
            kernel.prepare_geometry(inputs)
            self.assertGreater(distance.call_count, 0)
            distance.reset_mock()
            for _ in range(3):
                kernel(inputs).sum().backward()
                kernel.zero_grad(set_to_none=True)
            self.assertEqual(distance.call_count, 0)
            cross = kernel(inputs, heldout)
            self.assertGreater(distance.call_count, 0)
            distance.reset_mock()
            kernel(inputs)
            self.assertEqual(distance.call_count, 0)
        torch.testing.assert_close(cross, reference(inputs, heldout), rtol=0, atol=0)
        torch.testing.assert_close(kernel(heldout, inputs), cross.T, rtol=1e-12, atol=1e-12)

    def test_explicit_rectangular_cache_matches_values_and_gradients(self):
        for copies, inputs, policy in self.examples():
            with self.subTest(policy=policy, primitives=tuple(copies)):
                right = inputs[:2].clone()
                reference = PrimitiveNKN(copies, (4, 2, 1), probability_policy=policy)
                cached = deepcopy(reference)
                cached.prepare_geometry(inputs, right)
                self.assert_kernel_and_gradients_equal(cached, reference, inputs, right)
                with patch("torch.cdist", wraps=torch.cdist) as distance:
                    cached(inputs, right)
                    self.assertEqual(distance.call_count, 0)

    def test_changed_tensor_identity_or_version_never_uses_stale_geometry(self):
        inputs = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=DTYPE)
        kernel = PrimitiveNKN(VECTOR_COPIES, (4, 2, 1))
        reference = deepcopy(kernel)
        kernel.prepare_geometry(inputs)
        replacement = inputs.clone()
        replacement[0, 0] += 3
        with patch("torch.cdist", wraps=torch.cdist) as distance:
            result = kernel(replacement)
            self.assertGreater(distance.call_count, 0)
        torch.testing.assert_close(result, reference(replacement), rtol=0, atol=0)
        with torch.no_grad():
            inputs[0, 1] += 2
        with patch("torch.cdist", wraps=torch.cdist) as distance:
            result = kernel(inputs)
            self.assertGreater(distance.call_count, 0)
        torch.testing.assert_close(result, reference(inputs), rtol=0, atol=0)
        inputs[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            kernel(inputs)

    def test_feature_gradients_fall_back_before_or_after_preparing(self):
        for prepare_with_gradients in (False, True):
            with self.subTest(prepare_with_gradients=prepare_with_gradients):
                inputs = torch.tensor([[1., 0.5], [-0.4, 1.2], [1.3, 1.]], dtype=DTYPE)
                kernel = PrimitiveNKN(VECTOR_COPIES, (4, 2, 1))
                reference = deepcopy(kernel)
                inputs.requires_grad_(prepare_with_gradients)
                kernel.prepare_geometry(inputs)
                inputs.requires_grad_()
                expected_inputs = inputs.detach().clone().requires_grad_()
                with patch("torch.cdist", wraps=torch.cdist) as distance:
                    result = kernel(inputs)
                    self.assertGreater(distance.call_count, 0)
                expected = reference(expected_inputs)
                result.square().sum().backward()
                expected.square().sum().backward()
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                self.assertIsNotNone(inputs.grad)
                self.assertTrue(torch.isfinite(inputs.grad).all())
                torch.testing.assert_close(inputs.grad, expected_inputs.grad, rtol=1e-12, atol=1e-12)
                self.assert_parameters_and_gradients_equal(kernel, reference)

    def test_clear_and_module_conversion_discard_cache_without_changing_state(self):
        inputs = torch.tensor([[1., 0.], [0., 1.]], dtype=DTYPE)
        kernel = PrimitiveNKN(VECTOR_COPIES, (4, 2, 1))
        original = deepcopy(kernel.state_dict())
        for clear in (kernel.clear_geometry_cache, lambda: kernel.to(dtype=DTYPE)):
            kernel.prepare_geometry(inputs)
            self.assertEqual(kernel.state_dict().keys(), original.keys())
            for name, value in original.items():
                torch.testing.assert_close(kernel.state_dict()[name], value, rtol=0, atol=0)
            clear()
            with patch("torch.cdist", wraps=torch.cdist) as distance:
                kernel(inputs)
                self.assertGreater(distance.call_count, 0)

    def test_gradient_on_only_right_input_bypasses_rectangular_cache(self):
        left = torch.tensor([[1., 0.5], [-0.4, 1.2], [1.3, 1.]], dtype=DTYPE)
        right = torch.tensor([[0.3, 1.1], [1.2, -0.7]], dtype=DTYPE)
        kernel = PrimitiveNKN(VECTOR_COPIES, (4, 2, 1))
        reference = deepcopy(kernel)
        kernel.prepare_geometry(left, right)
        right.requires_grad_()
        reference_right = right.detach().clone().requires_grad_()
        result, expected = kernel(left, right), reference(left, reference_right)
        result.square().sum().backward()
        expected.square().sum().backward()
        self.assertIsNotNone(right.grad)
        torch.testing.assert_close(right.grad, reference_right.grad, rtol=1e-12, atol=1e-12)
        self.assert_parameters_and_gradients_equal(kernel, reference)

    def test_inference_inputs_without_version_counters_use_original_path(self):
        with torch.inference_mode():
            inputs = torch.tensor([[1., 0.], [0., 1.]], dtype=DTYPE)
        kernel = PrimitiveNKN(VECTOR_COPIES, (4, 2, 1))
        reference = deepcopy(kernel)
        kernel.prepare_geometry(inputs)
        with patch("torch.cdist", wraps=torch.cdist) as distance:
            actual = kernel(inputs)
            self.assertGreater(distance.call_count, 0)
        torch.testing.assert_close(actual, reference(inputs), rtol=0, atol=0)

    def test_preparing_normal_inputs_in_inference_mode_preserves_later_backward(self):
        inputs = torch.tensor([[1., 0.5], [-0.4, 1.2], [1.3, 1.]], dtype=DTYPE)
        kernel = PrimitiveNKN(VECTOR_COPIES, (4, 2, 1))
        reference = deepcopy(kernel)
        with torch.inference_mode():
            kernel.prepare_geometry(inputs)
        with patch("torch.cdist", wraps=torch.cdist) as distance:
            actual = kernel(inputs)
            self.assertGreater(distance.call_count, 0)
        expected = reference(inputs)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assert_parameters_and_gradients_equal(kernel, reference)


class ProteinGeometryCacheTests(GeometryAssertions):
    def test_all_branches_preserve_mutation_aggregation_and_conditional_mass(self):
        for conditional in (False, True):
            with self.subTest(conditional=conditional):
                inputs = batch([[1., 0.], [0., 1.], [1., 1.]],
                               [[1., 0.], [0., 1.], [1., 0.]], [0, 1, 1])
                if conditional:
                    inputs.local_features["inverse_folding"] = torch.tensor(
                        [[0.01] * 20, [0.04] * 20, [0.01] * 20], dtype=DTYPE,
                    )
                    inputs.feature_signature = {
                        "inverse_folding.probability_recipe": json.dumps(CONDITIONAL_RECIPE),
                    }
                torch.manual_seed(11)
                reference = ProteinKernel(model_config(), inputs)
                cached = deepcopy(reference)
                cached.prepare_training_geometry(inputs, use_triangle=False)
                self.assert_kernel_and_gradients_equal(cached, reference, inputs)
                # Owners are not geometric features: changing grouping must still
                # change aggregation even while the local geometry cache is hit.
                inputs.owners[2] = 2
                self.assert_kernel_and_gradients_equal(cached, reference, inputs)

    def test_empty_mutation_sets_and_empty_variant_batches(self):
        for global_features in ([[1., 0.], [0., 1.]], []):
            with self.subTest(num_variants=len(global_features)):
                inputs = batch(global_features, [], [])
                inputs.global_features = inputs.global_features.reshape(-1, 2)
                reference = ProteinKernel(model_config(), inputs)
                cached = deepcopy(reference)
                cached.prepare_training_geometry(inputs, use_triangle=False)
                torch.testing.assert_close(cached(inputs), reference(inputs), rtol=0, atol=0)
                cached.clear_geometry_cache()
                torch.testing.assert_close(cached(inputs), reference(inputs), rtol=0, atol=0)


class FittedGeometryCacheTests(GeometryAssertions):
    def setUp(self):
        super().setUp()
        original_prepare = ProteinKernel.prepare_training_geometry

        def prepare_dense_geometry(kernel, train):
            original_prepare(kernel, train, use_triangle=False)

        # These are bit-exact geometry-cache checks. Triangle evaluation has
        # separate tests allowing roundoff from different summation orders.
        preparation = patch.object(ProteinKernel, "prepare_training_geometry", prepare_dense_geometry)
        preparation.start()
        self.addCleanup(preparation.stop)

    def assert_fits_equal(self, actual, expected, heldout):
        self.assertEqual(actual.selected_seed, expected.selected_seed)
        self.assertEqual(actual.restart_summaries, expected.restart_summaries)
        self.assertEqual(actual.history, expected.history)
        self.assertEqual(actual.best_nll_per_variant, expected.best_nll_per_variant)
        self.assertEqual(actual.model.state_dict().keys(), expected.model.state_dict().keys())
        for name, parameter in actual.model.state_dict().items():
            torch.testing.assert_close(parameter, expected.model.state_dict()[name], rtol=0, atol=0)
        actual_prediction, expected_prediction = actual.predict(heldout), expected.predict(heldout)
        for field in fields(Prediction):
            actual_value, expected_value = getattr(actual_prediction, field.name), getattr(expected_prediction, field.name)
            if isinstance(actual_value, torch.Tensor):
                torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
            else:
                self.assertEqual(actual_value, expected_value)

    def test_identical_seeds_preserve_trajectories_restart_selection_and_predictions(self):
        for conditional in (False, True):
            with self.subTest(conditional=conditional):
                inputs = synthetic_batch()
                if conditional:
                    inputs = replace(inputs, local_features={**inputs.local_features, "inverse_folding":
                        inputs.local_features["inverse_folding"] * 0.83},
                        feature_signature={**inputs.feature_signature,
                            "inverse_folding.probability_recipe": json.dumps(CONDITIONAL_RECIPE)})
                train, heldout = inputs.select(range(6)), inputs.select(range(6, 8))
                config = TrainingConfig(max_steps=40, min_steps=10, patience=10, tolerance=0,
                                        seeds=(3, 9), lengthscale_multipliers=(0.5, 2.0))
                with patch.object(ProteinKernel, "prepare_training_geometry", return_value=None):
                    reference = fit_gp(train, small_config().model, config)
                cached = fit_gp(train, small_config().model, config)
                self.assert_fits_equal(cached, reference, heldout)
                self.assertTrue(all(row["status"] == "complete" for row in cached.restart_summaries))
                self.assertEqual(len(cached.history), 2 * (config.max_steps + 1))
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "checkpoint.pt"
                    cached.save(path)
                    state = torch.load(path, weights_only=True)
                    self.assertEqual(state["format_version"], 1)
                    self.assertEqual(state["parameters"].keys(), reference.model.state_dict().keys())
                    restored = FittedGP.load(path)
                    self.assert_fits_equal(restored, reference, heldout)

    def test_identical_seeds_preserve_early_stopping(self):
        inputs = synthetic_batch()
        train, heldout = inputs.select(range(6)), inputs.select(range(6, 8))
        config = TrainingConfig(max_steps=100, min_steps=30, patience=10, tolerance=1e6,
                                seeds=(12,), lengthscale_multipliers=(1.0,))
        with patch.object(ProteinKernel, "prepare_training_geometry", return_value=None):
            reference = fit_gp(train, small_config().model, config)
        cached = fit_gp(train, small_config().model, config)
        self.assert_fits_equal(cached, reference, heldout)
        self.assertEqual(cached.restart_summaries[0]["status"], "early_stopped")
        self.assertEqual(cached.history[-1]["step"], config.min_steps)


if __name__ == "__main__":
    unittest.main()
