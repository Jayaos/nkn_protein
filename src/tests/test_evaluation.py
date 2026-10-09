"""Scientific behavior checks for held-out calibration and rank metrics."""

import json
import math
import unittest

import torch

from evaluation import aggregate_folds, evaluate_fold, spearman
from evaluation.metrics import _confidence_bin_indices


def vector(values):
    return torch.tensor(values, dtype=torch.float64)


class EvaluationTests(unittest.TestCase):
    def test_two_variant_reference_and_observation_covariance(self):
        targets = vector([0, 1])
        covariance = torch.eye(2, dtype=torch.float64) * 0.5
        result = evaluate_fold(targets, targets, covariance, [0.5, 0.9], 4)
        confidence = 0.5 * (1 + math.erf(1 / math.sqrt(2)))
        self.assertEqual(result["n_pairs"], 1)
        self.assertAlmostEqual(result["pair_ece"], 1 - confidence)
        self.assertAlmostEqual(result["interval_ece"], 0.3)
        self.assertEqual([row["empirical"] for row in result["coverage"]], [1.0, 1.0])
        occupied = [row for row in result["bin_stats"] if row["count"]]
        self.assertEqual(occupied[0]["bin"], 2)
        self.assertAlmostEqual(occupied[0]["confidence"], confidence)
        self.assertEqual(occupied[0]["accuracy"], 1.0)
        json.dumps(result, allow_nan=False)

        covariance[0, 1] = covariance[1, 0] = 0.25
        correlated = evaluate_fold(targets, targets, covariance, [0.5, 0.9], 4)
        expected_confidence = 0.5 * (1 + math.erf(1))
        self.assertAlmostEqual(correlated["pair_ece"], 1 - expected_confidence)
        self.assertLess(correlated["pair_ece"], result["pair_ece"])
        self.assertEqual(correlated["interval_ece"], result["interval_ece"])

    def test_bins_include_interior_edge_on_right_and_one_in_final_bin(self):
        confidences = vector([0.5, 0.624, 0.625, 0.75, 0.875, 1.0])
        self.assertEqual(_confidence_bin_indices(confidences, 4).tolist(), [0, 0, 1, 2, 3, 3])
        result = evaluate_fold(
            vector([0, 100]), vector([0, 100]), torch.eye(2, dtype=torch.float64), [0.5], 4
        )
        self.assertEqual(result["bin_stats"][-1]["count"], 1)
        self.assertEqual(result["bin_stats"][-1]["confidence"], 1.0)
        self.assertEqual(result["pair_ece"], 0.0)

    def test_exact_ties_are_excluded_in_targets_and_predictions(self):
        # Only (0, 2) survives: (0, 1) has a target tie; (1, 2) a mean tie.
        result = evaluate_fold(
            vector([0, 0, 2]), vector([0, 1, 1]), torch.eye(3, dtype=torch.float64), [0.5], 3, 1
        )
        self.assertEqual(result["n_pairs"], 1)
        self.assertEqual(sum(row["count"] for row in result["bin_stats"]), 1)
        no_pairs = evaluate_fold(
            vector([0, 1]), vector([0, 0]), torch.eye(2, dtype=torch.float64), [0.5], 3
        )
        self.assertIsNone(no_pairs["pair_ece"])
        self.assertEqual(no_pairs["n_pairs"], 0)
        self.assertTrue(all(row["confidence"] is None for row in no_pairs["bin_stats"]))
        singleton = evaluate_fold(vector([1]), vector([1]), vector([1]).reshape(1, 1), [0.5], 1)
        self.assertIsNone(singleton["pair_ece"])
        self.assertEqual(singleton["interval_ece"], 0.5)

    def test_fold_aggregation_averages_errors_instead_of_pooling(self):
        first = evaluate_fold(
            vector([0, 1]), vector([0, 1]), torch.eye(2, dtype=torch.float64) * 0.5, [0.5], 1
        )
        second = evaluate_fold(
            vector([9, 8, 7]), vector([0, 1, 2]), torch.eye(3, dtype=torch.float64) * 0.5, [0.5], 1
        )
        aggregate = aggregate_folds([first, second])
        self.assertEqual(aggregate["interval_ece"], 0.5)
        # Pooled coverage is 2/5 and gives ECE=0.1, not the fold mean 0.5.
        self.assertNotAlmostEqual(aggregate["interval_ece"], abs(2 / 5 - 0.5))
        expected_pair_ece = (first["pair_ece"] + second["pair_ece"]) / 2
        self.assertAlmostEqual(aggregate["pair_ece"], expected_pair_ece)
        pooled_confidence = (
            first["bin_stats"][0]["confidence"]
            + 3 * second["bin_stats"][0]["confidence"]
        ) / 4
        self.assertNotAlmostEqual(aggregate["pair_ece"], abs(0.25 - pooled_confidence))
        self.assertEqual(aggregate["fold_n_pairs"], [1, 3])
        self.assertEqual(aggregate["n_valid_pair_folds"], 2)
        json.dumps(aggregate, allow_nan=False)

        undefined = evaluate_fold(
            vector([0]), vector([0]), vector([1]).reshape(1, 1), [0.5], 1
        )
        mixed = aggregate_folds([first, undefined])
        self.assertEqual(mixed["pair_ece"], first["pair_ece"])
        self.assertEqual(mixed["n_folds"], 2)
        self.assertEqual(mixed["n_valid_pair_folds"], 1)
        empty_pairs = aggregate_folds([undefined])
        self.assertIsNone(empty_pairs["pair_ece"])
        self.assertEqual(empty_pairs["n_valid_pair_folds"], 0)

    def test_blocking_and_permutation_preserve_metrics(self):
        generator = torch.Generator().manual_seed(15)
        factor = torch.randn(7, 4, generator=generator, dtype=torch.float64)
        covariance = factor @ factor.T + torch.eye(7, dtype=torch.float64) * 0.1
        targets = vector([1, 1, 2, -2, 5, 2, 6])
        means = vector([0, 1, 2, 1, 4, 5, -1])
        baseline = evaluate_fold(targets, means, covariance, [0.2, 0.5, 0.9], 5, 7)
        for block_size in (1, 2, 3, 512):
            result = evaluate_fold(targets, means, covariance, [0.2, 0.5, 0.9], 5, block_size)
            self.assertEqual(result["n_pairs"], baseline["n_pairs"])
            self.assertAlmostEqual(result["pair_ece"], baseline["pair_ece"])
            self.assertEqual(result["interval_ece"], baseline["interval_ece"])
        order = torch.tensor([6, 0, 3, 1, 5, 2, 4])
        permuted = evaluate_fold(
            targets[order], means[order], covariance[order][:, order], [0.2, 0.5, 0.9], 5, 2
        )
        self.assertAlmostEqual(permuted["pair_ece"], baseline["pair_ece"])
        self.assertEqual(permuted["coverage"], baseline["coverage"])
        self.assertEqual(
            [row["count"] for row in permuted["bin_stats"]],
            [row["count"] for row in baseline["bin_stats"]],
        )

    def test_invalid_covariance_and_configuration_are_rejected(self):
        for covariance in (
            vector([1, 0.5, 0, 1]).reshape(2, 2),
            vector([0, 0, 0, 1]).reshape(2, 2),
            vector([1, 1, 1, 1]).reshape(2, 2),
            vector([1, 2, 2, 1]).reshape(2, 2),
            vector([1, 0, 0, math.nan]).reshape(2, 2),
        ):
            with self.subTest(covariance=covariance), self.assertRaises(ValueError):
                evaluate_fold(vector([0, 1]), vector([0, 1]), covariance, [0.5], 2)
        for levels in ([], [0], [1], [math.nan]):
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                evaluate_fold(vector([0, 1]), vector([0, 1]), torch.eye(2), levels, 2)
        for bins in (0, -1, True, 1.5):
            with self.subTest(bins=bins), self.assertRaises(ValueError):
                evaluate_fold(vector([0, 1]), vector([0, 1]), torch.eye(2), [0.5], bins)
        with self.assertRaises(ValueError):
            evaluate_fold(vector([0, 1]), vector([0, math.inf]), torch.eye(2), [0.5], 2)
        with self.assertRaises(ValueError):
            evaluate_fold(vector([]), vector([]), torch.empty(0, 0), [0.5], 2)
        with self.assertRaises(ValueError):
            aggregate_folds([])

    def test_spearman_uses_average_ranks_and_handles_constant_vectors(self):
        self.assertAlmostEqual(spearman(vector([1, 2, 2, 4]), vector([4, 1, 1, 2])), -1 / 3)
        self.assertAlmostEqual(spearman(vector([1, 2, 2, 4]), vector([2, 4, 4, 8])), 1)
        self.assertAlmostEqual(spearman(vector([1, 2, 3]), vector([3, 2, 1])), -1)
        self.assertIsNone(spearman(vector([1, 1]), vector([0, 1])))
        self.assertIsNone(spearman(vector([1]), vector([2])))
        self.assertIsNone(spearman(vector([]), vector([])))
        with self.assertRaises(ValueError):
            spearman(vector([0, math.nan]), vector([0, 1]))


if __name__ == "__main__":
    unittest.main()
