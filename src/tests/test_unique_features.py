"""Unique ProteinMPNN pairs preserve occurrence-weighted GP computations."""

from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from config import ModelConfig, TrainingConfig
from gp.model import ExactProteinGP
from gp.training import FittedGP, Prediction, fit_gp
from kernels import ProteinKernel, initial_lengthscale
from tests.test_kernels import batch, model_config
from tests.test_pipeline import small_config, synthetic_batch
from tests.test_triangle import assert_close, asymmetric_sensitivity, conditional_batch


DTYPE = torch.float64


def singles(conditional=False):
    inputs = (conditional_batch() if conditional else synthetic_batch()).select([0, 1, 2, 3, 5, 6])
    values = inputs.local_features["inverse_folding"]
    # Vector equality deliberately differs from position equality: position 1
    # has two vectors, and vector 0 occurs at positions 1, 2, and 4.
    repeated = values[torch.tensor([0, 1, 0, 2, 1, 0])].clone()
    return replace(inputs, local_features={**inputs.local_features, "inverse_folding": repeated},
                   mutation_positions=torch.tensor([1, 1, 2, 2, 3, 4]))


class UniqueFeatureTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def kernels(self, inputs, config=None):
        torch.manual_seed(59)
        reference = ProteinKernel(config or model_config(), inputs)
        actual = deepcopy(reference)
        before = deepcopy(actual.state_dict())
        parameter_ids = {name: id(parameter) for name, parameter in actual.named_parameters()}
        reference.prepare_training_geometry(inputs, use_unique_features=False)
        actual.prepare_training_geometry(inputs)
        self.assertEqual(parameter_ids, {name: id(parameter) for name, parameter in actual.named_parameters()})
        self.assertEqual(before.keys(), actual.state_dict().keys())
        for name, value in before.items():
            torch.testing.assert_close(value, actual.state_dict()[name], rtol=0, atol=0)
        return actual, reference

    @staticmethod
    def mapping(kernel):
        return kernel._training_triangle.branches["inverse_folding"].occurrence_to_unique_pairs

    def gradients_equal(self, actual, expected):
        actual_parameters, expected_parameters = dict(actual.named_parameters()), dict(expected.named_parameters())
        self.assertEqual(actual_parameters.keys(), expected_parameters.keys())
        for name, parameter in actual_parameters.items():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertIsNotNone(expected_parameters[name].grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                assert_close(parameter.grad, expected_parameters[name].grad)

    def kernels_and_gradients_equal(self, actual, reference, *inputs):
        actual.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        observed, expected = actual(*inputs), reference(*inputs)
        assert_close(observed, expected)
        asymmetric_sensitivity(observed).backward()
        asymmetric_sensitivity(expected).backward()
        self.gradients_equal(actual, reference)

    def test_single_and_multi_mutants_evaluate_unique_vectors_not_positions(self):
        for inputs in (singles(), singles(True), synthetic_batch(), conditional_batch()):
            with self.subTest(variants=inputs.num_variants,
                              conditional="inverse_folding.probability_recipe" in inputs.feature_signature):
                actual, reference = self.kernels(inputs)
                unique_count = torch.unique(inputs.local_features["inverse_folding"], dim=0).shape[0]
                self.assertIsNone(self.mapping(reference))
                self.assertIsNotNone(self.mapping(actual))
                observed_rows = []
                hook = actual.branches["inverse_folding"].nkn.register_forward_pre_hook(
                    lambda module, arguments: observed_rows.append(arguments[0].shape[0]))
                try:
                    self.kernels_and_gradients_equal(actual, reference, inputs)
                finally:
                    hook.remove()
                self.assertEqual(observed_rows, [unique_count * (unique_count + 1) // 2])
                for name, prepared in actual._training_triangle.branches.items():
                    if name != "inverse_folding":
                        self.assertIsNone(prepared.occurrence_to_unique_pairs)
                for parameter, expected_parameter in zip(actual.parameters(), reference.parameters()):
                    with torch.no_grad():
                        parameter.add_(0.017)
                        expected_parameter.add_(0.017)
                self.kernels_and_gradients_equal(actual, reference, inputs, inputs)

    def test_no_duplicates_and_all_identical_features_preserve_gradients(self):
        for mode in ("unique", "identical"):
            with self.subTest(mode=mode):
                inputs = singles()
                generator = torch.Generator().manual_seed(37)
                values = torch.softmax(torch.randn(inputs.num_variants, 20, generator=generator, dtype=DTYPE), dim=1)
                if mode == "identical":
                    values = values[:1].expand(inputs.num_variants, -1).clone()
                inputs = replace(inputs, local_features={**inputs.local_features, "inverse_folding": values})
                actual, reference = self.kernels(inputs)
                self.kernels_and_gradients_equal(actual, reference, inputs)
                if mode == "unique":
                    self.assertIsNone(self.mapping(actual))
                else:
                    mapping = self.mapping(actual)
                    self.assertIsNotNone(mapping)
                    self.assertTrue(bool((mapping == 0).all()))

    def test_initialization_keeps_occurrence_weighted_lengthscale(self):
        inputs = singles(True)
        values = torch.tensor([.01, .01, .01, .02, .04, .049], dtype=DTYPE)[:, None].expand(-1, 20).clone()
        inputs = replace(inputs, local_features={**inputs.local_features, "inverse_folding": values})
        occurrence_scale = initial_lengthscale(values, hellinger=True, probability_policy="sequence_conditional")
        unique_scale = initial_lengthscale(torch.unique(values, dim=0), hellinger=True,
                                          probability_policy="sequence_conditional")
        self.assertFalse(torch.isclose(occurrence_scale, unique_scale, rtol=1e-5, atol=1e-10))
        actual, reference = self.kernels(inputs)
        for primitive in actual.branches["inverse_folding"].primitives:
            if primitive.raw_lengthscale is not None:
                assert_close(primitive.lengthscale, occurrence_scale)
        self.kernels_and_gradients_equal(actual, reference, inputs)

    def test_finite_differences_check_repeated_pair_gradient_multiplicity(self):
        inputs = singles(True)
        actual, reference = self.kernels(inputs)
        self.kernels_and_gradients_equal(actual, reference, inputs)
        parameters = dict(actual.named_parameters())
        names = [name for name in parameters if name.startswith("branches.inverse_folding")
                 and "raw_lengthscale" in name][:1]
        names += [name for name in parameters if name.startswith("branches.inverse_folding.nkn")
                  and name.endswith(("raw_weight", "raw_bias"))][:2]
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
            assert_close(analytical, (plus - minus) / (2 * step), rtol=2e-5, atol=2e-7)

    def test_likelihood_and_all_gp_parameter_gradients_match(self):
        for inputs in (singles(True), conditional_batch()):
            torch.manual_seed(18)
            reference = ExactProteinGP(inputs, small_config().model, TrainingConfig())
            actual = deepcopy(reference)
            reference.kernel.prepare_training_geometry(inputs, use_unique_features=False)
            actual.kernel.prepare_training_geometry(inputs)
            observed, observed_jitter = actual.objective(inputs, inputs.targets)
            expected, expected_jitter = reference.objective(inputs, inputs.targets)
            assert_close(observed, expected)
            self.assertEqual(observed_jitter, expected_jitter)
            observed.backward()
            expected.backward()
            self.gradients_equal(actual, reference)

    def test_changed_probabilities_and_mutation_ownership_bypass_unique_mapping(self):
        for change in ("probability", "owners", "positions"):
            with self.subTest(change=change):
                inputs = singles()
                actual, reference = self.kernels(inputs)
                if change == "probability":
                    inputs.local_features["inverse_folding"][0].copy_(inputs.local_features["inverse_folding"][3])
                elif change == "owners":
                    inputs.owners[1] = 0
                else:
                    inputs.mutation_positions[0] += 1
                with patch.object(actual, "_forward_triangle", wraps=actual._forward_triangle) as triangle:
                    self.kernels_and_gradients_equal(actual, reference, inputs)
                    triangle.assert_not_called()

    def test_rectangular_predictions_preserve_prepared_unique_mapping(self):
        inputs = singles()
        actual, reference = self.kernels(inputs)
        original_mapping = self.mapping(actual)
        heldout = inputs.select([1, 4])
        with patch.object(actual, "_forward_triangle", wraps=actual._forward_triangle) as triangle:
            self.kernels_and_gradients_equal(actual, reference, inputs, heldout)
            self.kernels_and_gradients_equal(actual, reference, heldout, inputs)
            triangle.assert_not_called()
            self.assertIs(self.mapping(actual), original_mapping)
            self.kernels_and_gradients_equal(actual, reference, inputs)
            triangle.assert_called_once()

    def test_clear_and_conversion_release_mapping_and_reprepare_restores_it(self):
        inputs = singles()
        actual, reference = self.kernels(inputs)
        for invalidate in (actual.clear_geometry_cache, lambda: actual.to(dtype=DTYPE),
                           lambda: actual.to(device="cpu")):
            self.assertIsNotNone(self.mapping(actual))
            invalidate()
            self.assertIsNone(actual._training_triangle)
            self.kernels_and_gradients_equal(actual, reference, inputs)
            actual.prepare_training_geometry(inputs)
            self.assertIsNotNone(self.mapping(actual))
            self.kernels_and_gradients_equal(actual, reference, inputs)
        actual.prepare_training_geometry(inputs, use_unique_features=False)
        self.assertIsNone(self.mapping(actual))
        self.kernels_and_gradients_equal(actual, reference, inputs)
        actual.prepare_training_geometry(inputs)
        self.assertIsNotNone(self.mapping(actual))

    def test_differentiable_probability_inputs_preserve_feature_gradients(self):
        for before in (False, True):
            with self.subTest(requires_grad_before_preparation=before):
                inputs, expected_inputs = singles(), singles()
                torch.manual_seed(43)
                reference = ProteinKernel(model_config(), inputs)
                actual = deepcopy(reference)
                inputs.local_features["inverse_folding"].requires_grad_(before)
                actual.prepare_training_geometry(inputs)
                inputs.local_features["inverse_folding"].requires_grad_()
                expected_inputs.local_features["inverse_folding"].requires_grad_()
                with patch.object(actual, "_forward_triangle", wraps=actual._forward_triangle) as triangle:
                    actual_values = actual(inputs)
                    triangle.assert_not_called()
                expected_values = reference(expected_inputs)
                assert_close(actual_values, expected_values)
                asymmetric_sensitivity(actual_values).backward()
                asymmetric_sensitivity(expected_values).backward()
                actual_gradient = inputs.local_features["inverse_folding"].grad
                self.assertIsNotNone(actual_gradient)
                self.assertTrue(bool(torch.isfinite(actual_gradient).all()))
                assert_close(actual_gradient, expected_inputs.local_features["inverse_folding"].grad)
                self.gradients_equal(actual, reference)

    def test_absent_inverse_folding_and_empty_batches_need_no_mapping(self):
        inputs = singles()
        actual, reference = self.kernels(inputs, ModelConfig(enabled_branches=("sequence_global",)))
        self.kernels_and_gradients_equal(actual, reference, inputs)
        self.assertNotIn("inverse_folding", actual._training_triangle.branches)
        for global_features in ([[1., 0.], [0., 1.]], []):
            empty = batch(global_features, [], [])
            empty.global_features = empty.global_features.reshape(-1, 2)
            actual, reference = self.kernels(empty)
            self.assertIsNone(self.mapping(actual))
            assert_close(actual(empty), reference(empty))

    def test_inference_preparation_keeps_later_backward_valid(self):
        inputs = singles()
        actual, reference = self.kernels(inputs)
        with torch.inference_mode():
            actual.prepare_training_geometry(inputs)
        self.assertIsNone(actual._training_triangle)
        self.kernels_and_gradients_equal(actual, reference, inputs)

    def test_default_fitting_deduplicates_and_preserves_histories_predictions_and_reload(self):
        original_prepare = ProteinKernel.prepare_training_geometry

        def prepare_occurrences(kernel, train):
            original_prepare(kernel, train, use_unique_features=False)

        for inputs in (synthetic_batch(), conditional_batch()):
            with self.subTest(conditional="inverse_folding.probability_recipe" in inputs.feature_signature):
                train, heldout = inputs.select(range(6)), inputs.select(range(6, 8))
                config = TrainingConfig(max_steps=20, min_steps=10, patience=5, tolerance=0,
                                        seeds=(3, 9), lengthscale_multipliers=(0.5, 2.0))
                with patch.object(ProteinKernel, "prepare_training_geometry", prepare_occurrences):
                    reference = fit_gp(train, small_config().model, config)
                actual = fit_gp(train, small_config().model, config)
                self.assertIsNotNone(self.mapping(actual.model.kernel))
                self.assertIsNone(self.mapping(reference.model.kernel))
                self.assertEqual(actual.selected_seed, reference.selected_seed)
                self.assertEqual(len(actual.history), 2 * (config.max_steps + 1))
                self.assertEqual(len(actual.history), len(reference.history))
                for left, right in zip(actual.history, reference.history):
                    self.assertEqual(left.keys(), right.keys())
                    for name in left:
                        assert_close(left[name], right[name], rtol=1e-9, atol=1e-10)
                for left, right in zip(actual.restart_summaries, reference.restart_summaries):
                    self.assertEqual(left["status"], right["status"])
                    self.assertEqual(left["best_step"], right["best_step"])
                    assert_close(left["best_nll_per_variant"], right["best_nll_per_variant"], rtol=1e-9, atol=1e-10)
                self.assertEqual(actual.model.state_dict().keys(), reference.model.state_dict().keys())
                for name, parameter in actual.model.state_dict().items():
                    assert_close(parameter, reference.model.state_dict()[name], rtol=1e-9, atol=1e-10)
                expected = reference.predict(heldout)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "unique_features.pt"
                    actual.save(path)
                    restored = FittedGP.load(path)
                    self.assertIsNone(restored.model.kernel._training_triangle)
                    for fit in (actual, restored):
                        observed = fit.predict(heldout)
                        for field in fields(Prediction):
                            assert_close(getattr(observed, field.name), getattr(expected, field.name),
                                         rtol=1e-9, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
