"""Kernel equations checked independently of the implementation and reference clone."""

from dataclasses import replace
import json
import unittest

import torch

from baselines.kermut.data import validate_batch
from baselines.kermut.kernel import KermutKernel
from data_loading.features import ALPHABET
from fixtures import CONDITIONAL_PROBABILITY_RECIPE, scalar_kernel, small_config, synthetic_batch


class KernelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.batch = synthetic_batch()
        self.config = small_config()

    def test_single_and_double_kernel_matches_scalar_equations(self):
        kernel = KermutKernel(self.batch, self.config.model).double()
        actual = kernel(self.batch)
        expected = scalar_kernel(self.batch, self.batch, self.config.model)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        self.assertTrue(torch.allclose(actual, actual.T, atol=1e-12, rtol=0))
        self.assertGreater(torch.linalg.eigvalsh(actual).min().item(), -1e-10)
        # Distinct replacement amino acids at the same site are not identical inputs.
        local_self = actual[0, 0] - (1 - kernel.alpha)
        global_cross = (1 - kernel.alpha) * torch.exp(-((self.batch.global_features[0] - self.batch.global_features[1]) ** 2).sum() / (2 * kernel.sequence_lengthscale ** 2))
        self.assertLess((actual[0, 1] - global_cross).item(), local_self.item())

    def test_rectangular_cross_kernel_preserves_requested_variant_order(self):
        train = self.batch.select([11, 1, 7, 19])
        test = self.batch.select([4, 10, 0])
        kernel = KermutKernel(train, self.config.model).double()
        torch.testing.assert_close(kernel(test, train), scalar_kernel(test, train, self.config.model), rtol=1e-10, atol=1e-10)
        self.assertEqual(kernel(test, train).shape, (3, 4))

    def test_conditional_subprobabilities_follow_original_unscaled_kernel_formula(self):
        weights = self.batch.local_features["inverse_folding"] * (0.4 + 0.1 * self.batch.mutation_positions[:, None])
        batch = replace(self.batch, local_features={**self.batch.local_features, "inverse_folding": weights},
                        feature_signature={**self.batch.feature_signature,
                            "inverse_folding.probability_recipe": json.dumps(CONDITIONAL_PROBABILITY_RECIPE)})
        kernel = KermutKernel(batch, self.config.model).double()
        torch.testing.assert_close(kernel(batch), scalar_kernel(batch, batch, self.config.model), rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(batch.local_features["inverse_folding"], weights, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            validate_batch(replace(batch, feature_signature=self.batch.feature_signature))

    def test_invalid_zero_probabilities_and_inconsistent_mutations_fail(self):
        probabilities = self.batch.local_features["inverse_folding"].clone()
        # Preserve normalization and agreement between repeated positions. A
        # zero for the actual replacement AA specifically makes log(p) invalid.
        rows = self.batch.mutation_positions == 1
        column = ALPHABET.index("W")
        probabilities[rows, 0] += probabilities[rows, column]
        probabilities[rows, column] = 0
        zero_probability = replace(self.batch, local_features={**self.batch.local_features, "inverse_folding": probabilities})
        with self.assertRaisesRegex(ValueError, "substituted amino acid.*positive|positive.*substituted amino acid"):
            validate_batch(zero_probability)
        with self.assertRaises(ValueError):
            validate_batch(replace(self.batch, sequences=()))
        sequences = (self.batch.reference_sequence, *self.batch.sequences[1:])
        with self.assertRaises(ValueError):
            validate_batch(replace(self.batch, sequences=sequences))
        coordinates = self.batch.local_features["structure"].clone()
        coordinates[1, 0] += 1
        with self.assertRaises(ValueError):
            validate_batch(replace(self.batch, local_features={**self.batch.local_features, "structure": coordinates}))
        with self.assertRaises(ValueError):
            replace(self.batch, global_features=torch.full_like(self.batch.global_features, float("nan")))

    def test_wild_type_has_only_global_covariance_and_can_be_a_prediction_batch(self):
        batch = self.batch.select([0, 1, 10])
        keep = batch.owners != 0
        batch = replace(batch, variant_ids=("WT", *batch.variant_ids[1:]),
                        sequences=(batch.reference_sequence, *batch.sequences[1:]),
                        owners=batch.owners[keep], mutation_positions=batch.mutation_positions[keep],
                        local_features={name: values[keep] for name, values in batch.local_features.items()})
        kernel = KermutKernel(batch, self.config.model)
        torch.testing.assert_close(kernel(batch), scalar_kernel(batch, batch, self.config.model))
        wild_type = batch.select([0])
        self.assertEqual(wild_type.owners.numel(), 0)
        validate_batch(wild_type)
        torch.testing.assert_close(kernel(wild_type)[0, 0], 1 - kernel.alpha)


if __name__ == "__main__":
    unittest.main()
