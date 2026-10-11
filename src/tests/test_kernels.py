"""Numerical and aggregation checks independent of any pretrained model."""

import math
import json
import unittest
from types import SimpleNamespace

import torch

from kernels import (
    NeuralKernelNetwork,
    NonnegativeLinear,
    PrimitiveKernel,
    PrimitiveNKN,
    ProteinKernel,
    aggregate_mutations,
    initial_lengthscale,
    inverse_softplus,
)


DTYPE = torch.float64
CONDITIONAL_RECIPE = {
    "model_family": "proteinmpnn", "conditioning_mode": "sequence_conditional",
    "inference_method": "conditional_probs", "normalization": "native_21_amino_acid_tokens",
    "noncanonical_tokens": "excluded_without_renormalization",
    "order_aggregation": "geometric_mean_of_probabilities", "num_orders": 10, "seed": 37,
}


def batch(global_features, vectors, owners, positions=None):
    count = len(owners)
    probabilities = torch.zeros((count, 20), dtype=DTYPE)
    for row in range(count):
        probabilities[row, row % 2] = 0.8
        probabilities[row, 2] = 0.2
    return SimpleNamespace(
        num_variants=len(global_features),
        global_features=torch.tensor(global_features, dtype=DTYPE),
        local_features={
            "inverse_folding": probabilities,
            "structure": torch.tensor(vectors, dtype=DTYPE).reshape(count, 2),
            "sequence_site": torch.tensor(vectors, dtype=DTYPE).reshape(count, 2) + 0.3,
        },
        owners=torch.tensor(owners, dtype=torch.long),
        mutation_positions=torch.tensor(positions if positions is not None else list(range(count)), dtype=torch.long),
    )


def model_config(widths=(4, 2, 1)):
    return SimpleNamespace(
        branches={
            name: SimpleNamespace(copies=copies, widths=widths)
            for name, copies in {
                "inverse_folding": {"hellinger_exp": 1, "hellinger_rbf": 1, "bhattacharyya": 1},
                "structure": {"cosine": 1, "rbf": 1, "matern32": 1},
                "sequence_site": {"cosine": 1, "rbf": 1, "matern32": 1},
                "sequence_global": {"cosine": 1, "rbf": 1, "matern32": 1},
            }.items()
        },
        final_widths=widths,
    )


class PrimitiveTests(unittest.TestCase):
    def test_closed_forms_and_signed_cosine(self):
        vectors = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=DTYPE)
        self.assertAlmostEqual(PrimitiveKernel("cosine")(vectors)[0, 1].item(), -1)
        self.assertAlmostEqual(PrimitiveKernel("rbf", 2)(vectors)[0, 1].item(), math.exp(-0.5))
        self.assertAlmostEqual(
            PrimitiveKernel("matern32", 2)(vectors)[0, 1].item(),
            (1 + math.sqrt(3)) * math.exp(-math.sqrt(3)),
        )
        probabilities = torch.zeros((2, 20), dtype=DTYPE)
        probabilities[0, 0] = probabilities[1, 1] = 1
        self.assertAlmostEqual(PrimitiveKernel("hellinger_exp", 2)(probabilities)[0, 1].item(), math.exp(-0.5))
        self.assertAlmostEqual(PrimitiveKernel("hellinger_rbf", 2)(probabilities)[0, 1].item(), math.exp(-0.125))
        torch.testing.assert_close(PrimitiveKernel("bhattacharyya")(probabilities), torch.eye(2, dtype=DTYPE))
        for name in ("cosine", "rbf", "matern32"):
            torch.testing.assert_close(PrimitiveKernel(name)(vectors).diag(), torch.ones(2, dtype=DTYPE))

    def test_psd_and_finite_lengthscale_gradients_at_duplicate_inputs(self):
        torch.manual_seed(12)
        vectors = torch.randn(5, 3, dtype=DTYPE)
        vectors = torch.cat([vectors, vectors[:1]])
        probabilities = torch.softmax(torch.randn(5, 20, dtype=DTYPE), dim=1)
        probabilities = torch.cat([probabilities, probabilities[:1]])
        for name in ("cosine", "rbf", "matern32", "hellinger_exp", "hellinger_rbf", "bhattacharyya"):
            inputs = probabilities if name.startswith("hellinger") or name == "bhattacharyya" else vectors
            kernel = PrimitiveKernel(name, 1.3)
            covariance = kernel(inputs)
            torch.testing.assert_close(covariance, covariance.T)
            self.assertGreaterEqual(torch.linalg.eigvalsh(covariance).min().item(), -1e-10)
            if kernel.raw_lengthscale is not None:
                covariance.sum().backward()
                self.assertTrue(torch.isfinite(kernel.raw_lengthscale.grad).item())
                parameter = kernel.raw_lengthscale.detach().clone()
                step = 1e-5
                with torch.no_grad():
                    kernel.raw_lengthscale.copy_(parameter + step)
                    plus = kernel(inputs).sum()
                    kernel.raw_lengthscale.copy_(parameter - step)
                    minus = kernel(inputs).sum()
                    kernel.raw_lengthscale.copy_(parameter)
                torch.testing.assert_close(kernel.raw_lengthscale.grad, (plus - minus) / (2 * step), rtol=1e-5, atol=1e-7)

    def test_median_retains_occurrences_and_averages_middle_values(self):
        # Repeated occurrences yield median 3; deduplication would instead yield 5.
        repeated = torch.tensor([[0.0], [0.0], [1.0], [3.0], [10.0]], dtype=DTYPE)
        self.assertEqual(initial_lengthscale(repeated).item(), 3)
        self.assertEqual(initial_lengthscale(torch.tensor([[0.0], [1.0], [3.0], [10.0]], dtype=DTYPE)).item(), 5)
        self.assertEqual(initial_lengthscale(torch.ones(3, 2, dtype=DTYPE)).item(), 1)

    def test_rejects_zero_cosine_and_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            PrimitiveKernel("cosine")(torch.zeros(1, 2, dtype=DTYPE))
        with self.assertRaises(ValueError):
            PrimitiveKernel("bhattacharyya")(torch.ones(1, 20, dtype=DTYPE))

    def test_conditional_closed_forms_preserve_mass_without_renormalization(self):
        weights = torch.tensor([[0.01] * 20, [0.04] * 20], dtype=DTYPE)
        untouched = weights.clone()
        distance = math.sqrt(0.1)
        policy = {"probability_policy": "sequence_conditional"}
        self.assertAlmostEqual(initial_lengthscale(weights, hellinger=True, **policy).item(), distance)
        self.assertAlmostEqual(PrimitiveKernel("hellinger_exp", 2, **policy)(weights)[0, 1].item(), math.exp(-distance / 2))
        self.assertAlmostEqual(PrimitiveKernel("hellinger_rbf", 2, **policy)(weights)[0, 1].item(), math.exp(-0.1 / 8))
        torch.testing.assert_close(PrimitiveKernel("bhattacharyya", **policy)(weights),
                                   torch.tensor([[0.2, 0.4], [0.4, 0.8]], dtype=DTYPE))
        torch.testing.assert_close(weights, untouched, rtol=0, atol=0)
        # Identical normalized rows would have zero Hellinger distance; these
        # distinct raw masses intentionally retain a nonzero distance.
        with self.assertRaisesRegex(ValueError, "sum to one"):
            initial_lengthscale(weights, hellinger=True)
        for name in ("hellinger_exp", "hellinger_rbf", "bhattacharyya"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "sum to one"):
                PrimitiveKernel(name)(weights)

    def test_conditional_primitives_are_psd_with_finite_gradients_at_duplicates(self):
        generator = torch.Generator().manual_seed(36)
        weights = torch.softmax(torch.randn(4, 20, dtype=DTYPE, generator=generator), dim=-1)
        weights *= torch.tensor([0.2, 0.5, 0.8, 0.6], dtype=DTYPE)[:, None]
        weights = torch.cat([weights, weights[:1]])
        for name in ("hellinger_exp", "hellinger_rbf", "bhattacharyya"):
            with self.subTest(name=name):
                inputs = weights.clone().requires_grad_()
                kernel = PrimitiveKernel(name, 1.3, probability_policy="sequence_conditional")
                covariance = kernel(inputs)
                torch.testing.assert_close(covariance, covariance.T)
                self.assertGreaterEqual(torch.linalg.eigvalsh(covariance).min().item(), -1e-10)
                covariance.square().sum().backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())
                if kernel.raw_lengthscale is not None:
                    self.assertTrue(torch.isfinite(kernel.raw_lengthscale.grad))

    def test_conditional_policy_rejects_invalid_weights_and_unknown_policy(self):
        invalid_weights = [torch.full((2, 19), 0.01, dtype=DTYPE),
                           torch.full((2, 20), 0.1, dtype=DTYPE)]
        for invalid_value in (0.0, -0.1, float("nan"), float("inf")):
            weights = torch.full((2, 20), 0.01, dtype=DTYPE)
            weights[0, 0] = invalid_value
            invalid_weights.append(weights)
        kernel = PrimitiveKernel("bhattacharyya", probability_policy="sequence_conditional")
        for weights in invalid_weights:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                kernel(weights)
            with self.assertRaises(ValueError):
                initial_lengthscale(weights, hellinger=True, probability_policy="sequence_conditional")
        with self.assertRaisesRegex(ValueError, "probability_policy"):
            PrimitiveKernel("bhattacharyya", probability_policy="anything")


class NKNTests(unittest.TestCase):
    def test_affine_bias_and_adjacent_products(self):
        network = NeuralKernelNetwork(2, (4, 2, 1))
        first, _, final = network.layers
        with torch.no_grad():
            first.raw_weight.copy_(inverse_softplus(torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], dtype=DTYPE)))
            first.raw_bias.copy_(inverse_softplus(torch.full((4,), 0.1, dtype=DTYPE)))
            final.raw_weight.copy_(inverse_softplus(torch.tensor([[0.3, 0.7]], dtype=DTYPE)))
            final.raw_bias.copy_(inverse_softplus(torch.tensor([0.2], dtype=DTYPE)))
        inputs = torch.tensor([[0.2, -0.1], [1., 1.]], dtype=DTYPE)
        hidden = inputs @ first.weight.T + 0.1
        expected = 0.3 * hidden[:, 0] * hidden[:, 1] + 0.7 * hidden[:, 2] * hidden[:, 3] + 0.2
        torch.testing.assert_close(network(inputs), expected)
        self.assertEqual(network(inputs).dtype, DTYPE)
        torch.testing.assert_close(NonnegativeLinear(3, 2).bias, torch.full((2,), 0.01, dtype=DTYPE))

    def test_nkn_preserves_psd_with_negative_entries(self):
        torch.manual_seed(3)
        features = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [1., 1.]], dtype=DTYPE)
        kernel = PrimitiveNKN({"cosine": 1, "rbf": 2}, (4, 2, 1))
        covariance = kernel(features)
        self.assertGreaterEqual(torch.linalg.eigvalsh(covariance).min().item(), -1e-10)
        covariance.square().sum().backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in kernel.parameters()))

    def test_nkn_preserves_conditional_covariance_and_finite_parameter_gradients(self):
        weights = torch.tensor([[0.01] * 20, [0.04] * 20, [0.01] * 20], dtype=DTYPE)
        copies = {"hellinger_exp": 1, "hellinger_rbf": 1, "bhattacharyya": 1}
        kernel = PrimitiveNKN(copies, (4, 2, 1), probability_policy="sequence_conditional")
        self.assertTrue(all(primitive.probability_policy == "sequence_conditional" for primitive in kernel.primitives))
        covariance = kernel(weights)
        self.assertGreaterEqual(torch.linalg.eigvalsh(covariance).min().item(), -1e-10)
        covariance.square().sum().backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in kernel.parameters()))

    def test_invalid_layer_connectivity(self):
        for widths in ((3, 1, 1), (4, 3, 1), (4, 2), (4, 2, 2)):
            with self.assertRaises(ValueError):
                NeuralKernelNetwork(3, widths)

    def test_seed_and_configuration_key_order_preserve_initialization(self):
        torch.manual_seed(18)
        first = PrimitiveNKN({"cosine": 1, "rbf": 2, "matern32": 1}, (4, 2, 1))
        torch.manual_seed(18)
        repeated = PrimitiveNKN({"rbf": 2, "matern32": 1, "cosine": 1}, (4, 2, 1))
        self.assertEqual([primitive.name for primitive in first.primitives], [primitive.name for primitive in repeated.primitives])
        for name, values in first.state_dict().items():
            torch.testing.assert_close(values, repeated.state_dict()[name])
        torch.manual_seed(19)
        different = PrimitiveNKN({"cosine": 1, "rbf": 2, "matern32": 1}, (4, 2, 1))
        self.assertFalse(torch.equal(first.nkn.layers[0].raw_weight, different.nkn.layers[0].raw_weight))


class ProteinKernelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.features = batch([[1., 0.], [0., 1.], [1., 1.]], [[1., 0.], [0., 1.], [1., 1.]], [0, 1, 1])
        self.kernel = ProteinKernel(model_config(), self.features)

    def test_explicit_mutation_pair_sum_and_empty_variant(self):
        parts = []
        for name in ("inverse_folding", "structure", "sequence_site"):
            local = self.kernel.branches[name](self.features.local_features[name])
            manual = local.new_zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    manual[i, j] = local[self.features.owners == i][:, self.features.owners == j].sum()
            self.assertEqual(manual[2].abs().sum().item(), 0)
            parts.append(manual)
        parts.append(self.kernel.branches["sequence_global"](self.features.global_features))
        expected = self.kernel.final_nkn(torch.stack(parts, dim=-1))
        covariance = self.kernel(self.features)
        torch.testing.assert_close(covariance, expected)
        self.assertGreaterEqual(torch.linalg.eigvalsh(covariance).min().item(), -1e-10)
        covariance.square().sum().backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in self.kernel.parameters()))

    def test_local_nkn_cannot_be_moved_after_mutation_sum(self):
        head = self.kernel.branches["structure"]
        primitives = torch.stack([primitive(self.features.local_features["structure"]) for primitive in head.primitives], dim=-1)
        correct = aggregate_mutations(head.nkn(primitives), self.features.owners, self.features.owners, 3, 3)
        wrong = head.nkn(torch.stack([
            aggregate_mutations(primitives[..., index], self.features.owners, self.features.owners, 3, 3)
            for index in range(primitives.shape[-1])
        ], dim=-1))
        self.assertFalse(torch.allclose(correct, wrong))

    def test_occurrence_and_variant_permutation_and_rectangular_covariance(self):
        permutation = torch.tensor([2, 0, 1])
        inverse = permutation.argsort()
        occurrence_order = torch.tensor([2, 0, 1])
        shuffled = SimpleNamespace(
            num_variants=3,
            global_features=self.features.global_features[permutation],
            local_features={name: values[occurrence_order] for name, values in self.features.local_features.items()},
            owners=inverse[self.features.owners[occurrence_order]],
            mutation_positions=self.features.mutation_positions[occurrence_order],
        )
        covariance = self.kernel(self.features)
        torch.testing.assert_close(self.kernel(shuffled), covariance[permutation][:, permutation])
        torch.testing.assert_close(self.kernel(self.features, shuffled), covariance[:, permutation])
        empty_local = batch([[1., 1.]], [], [])
        cross = self.kernel(self.features, empty_local)
        self.assertEqual(cross.shape, (3, 1))
        torch.testing.assert_close(cross, self.kernel(empty_local, self.features).T)

    def test_training_lengthscale_and_restart_multiplier(self):
        restarted = ProteinKernel(model_config(), self.features, lengthscale_multiplier=0.5)
        for name in self.kernel.branches:
            for first, second in zip(self.kernel.branches[name].primitives, restarted.branches[name].primitives):
                if first.lengthscale is not None:
                    torch.testing.assert_close(second.lengthscale, first.lengthscale / 2)

    def test_conditional_recipe_routes_policy_and_rejects_unmarked_or_changed_inputs(self):
        weights = torch.tensor([[0.01] * 20, [0.04] * 20, [0.02] * 20], dtype=DTYPE)
        self.features.local_features["inverse_folding"] = weights
        with self.assertRaisesRegex(ValueError, "sum to one"):
            ProteinKernel(model_config(), self.features)
        self.features.feature_signature = {
            "inverse_folding.probability_recipe": json.dumps(CONDITIONAL_RECIPE),
        }
        kernel = ProteinKernel(model_config(), self.features)
        self.assertEqual(kernel.inverse_folding_policy, "sequence_conditional")
        for name, branch in kernel.branches.items():
            expected = "sequence_conditional" if name == "inverse_folding" else "normalized"
            self.assertEqual({primitive.probability_policy for primitive in branch.primitives}, {expected})
        covariance = kernel(self.features)
        self.assertGreaterEqual(torch.linalg.eigvalsh(covariance).min().item(), -1e-10)
        covariance.square().sum().backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in kernel.parameters()))
        for signature in ({}, {"inverse_folding.probability_recipe": json.dumps({**CONDITIONAL_RECIPE, "seed": 38})}):
            changed = SimpleNamespace(**{**vars(self.features), "feature_signature": signature})
            with self.assertRaisesRegex(ValueError, "recipe differs"):
                kernel(self.features, changed)
            with self.assertRaisesRegex(ValueError, "recipe differs"):
                kernel(changed, self.features)
        # A model initialized from legacy normalized distributions likewise
        # cannot silently change its input contract during prediction.
        with self.assertRaisesRegex(ValueError, "recipe differs"):
            self.kernel(self.features)

    def test_conditional_recipe_must_be_complete_before_kernel_initialization(self):
        for encoded in ('{}', '[]', json.dumps({**CONDITIONAL_RECIPE, "num_orders": 0})):
            with self.subTest(encoded=encoded):
                self.features.feature_signature = {"inverse_folding.probability_recipe": encoded}
                with self.assertRaises(ValueError):
                    ProteinKernel(model_config(), self.features)


if __name__ == "__main__":
    unittest.main()
