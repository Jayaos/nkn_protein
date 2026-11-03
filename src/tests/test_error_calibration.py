"""Independent references for Kermut-style error calibration and uncertainty CV."""

import json
import math
import statistics
import unittest

import torch

from config import EvaluationConfig, ExperimentConfig
from evaluation import aggregate_folds, error_calibration


def vector(values):
    return torch.tensor(values, dtype=torch.float64)


class ErrorCalibrationTests(unittest.TestCase):
    def test_bin_rmse_rmv_and_sample_cv_match_hand_calculation(self):
        result = error_calibration(vector([0, 0, 6, 8]), vector([0] * 4), vector([1, 4, 9, 16]), 2)
        first, second = result["error_bin_stats"]
        self.assertEqual([first["count"], second["count"]], [2, 2])
        self.assertAlmostEqual(first["rmv"], math.sqrt(2.5))
        self.assertEqual(first["rmse"], 0)
        self.assertAlmostEqual(second["rmv"], math.sqrt(12.5))
        self.assertAlmostEqual(second["rmse"], math.sqrt(50))
        self.assertAlmostEqual(result["ence"], 1)
        self.assertAlmostEqual(result["uncertainty_cv"], statistics.stdev([1, 2, 3, 4]) / 2.5)
        # The denominator is outside the square root (Appendix J.2), unlike
        # the released Kermut code's sqrt(sample_variance / mean_sigma).
        self.assertNotAlmostEqual(result["uncertainty_cv"], math.sqrt(statistics.variance([1, 2, 3, 4]) / 2.5))
        json.dumps(result, allow_nan=False)

    def test_perfect_error_calibration_and_scale_invariance(self):
        targets, means, variance = vector([1, 2, 3, 4, 5, 6]), vector([0] * 6), vector([1, 4, 9, 16, 25, 36])
        result = error_calibration(targets, means, variance, 3)
        self.assertEqual(result["ence"], 0)
        scaled = error_calibration(7 * targets + 20, 7 * means + 20, 49 * variance, 3)
        self.assertAlmostEqual(result["ence"], scaled["ence"])
        self.assertAlmostEqual(result["uncertainty_cv"], scaled["uncertainty_cv"])
        for before, after in zip(result["error_bin_stats"], scaled["error_bin_stats"]):
            self.assertAlmostEqual(7 * before["rmv"], after["rmv"])
            self.assertAlmostEqual(7 * before["rmse"], after["rmse"])

    def test_quantile_ties_drop_edges_and_are_permutation_invariant(self):
        variance = vector([1, 1, 1, 4, 4, 9, 16, 16])
        targets, means = vector([1, 2, 3, 4, 5, 6, 7, 8]), vector([0] * 8)
        result = error_calibration(targets, means, variance, 4)
        # Linear quartiles [1, 1, 4, 10.75, 16], right-closed after deduplication.
        self.assertEqual([row["count"] for row in result["error_bin_stats"]], [5, 1, 2])
        self.assertEqual(result["n_error_bins"], 3)
        self.assertEqual(result["error_n_bins"], 4)
        order = torch.tensor([7, 3, 0, 5, 1, 4, 6, 2])
        permuted = error_calibration(targets[order], means[order], variance[order], 4)
        self.assertEqual(result["error_bin_stats"], permuted["error_bin_stats"])
        self.assertAlmostEqual(result["uncertainty_cv"], permuted["uncertainty_cv"])
        # ENCE gives bins equal weight even when tied values make sizes unequal.
        self.assertAlmostEqual(result["ence"], sum(row["relative_error"] for row in result["error_bin_stats"]) / 3)

    def test_constant_variance_singleton_and_empty_groups(self):
        constant = error_calibration(vector([1, 2, 3]), vector([0, 0, 0]), vector([4, 4, 4]))
        self.assertEqual(constant["uncertainty_cv"], 0)
        self.assertEqual(constant["n_error_bins"], 1)
        self.assertAlmostEqual(constant["ence"], abs(2 - math.sqrt(14 / 3)) / 2)
        singleton = error_calibration(vector([3]), vector([1]), vector([4]))
        self.assertEqual(singleton["ence"], 0)
        self.assertIsNone(singleton["uncertainty_cv"])
        small = error_calibration(vector([1, 2]), vector([0, 0]), vector([1, 4]), 5)
        self.assertEqual(small["n_error_bins"], 2)
        self.assertEqual(small["ence"], 0)
        empty = error_calibration(vector([]), vector([]), vector([]))
        self.assertIsNone(empty["ence"])
        self.assertIsNone(empty["uncertainty_cv"])
        self.assertEqual(empty["error_bin_stats"], [])
        json.dumps([constant, singleton, small, empty], allow_nan=False)

    def test_constant_fractional_variance_has_exact_zero_cv_and_calculated_ence(self):
        for variance in (0.01, 0.3, 3.0):
            for count in (3, 7):
                with self.subTest(variance=variance, count=count):
                    result = error_calibration(
                        vector([i * math.sqrt(variance) for i in range(count)]),
                        vector([0] * count), vector([variance] * count),
                    )
                    expected = abs(1 - math.sqrt(sum(i * i for i in range(count)) / count))
                    self.assertAlmostEqual(result["ence"], expected)
                    self.assertEqual(result["uncertainty_cv"], 0.0)
                    self.assertEqual(result["n_error_bins"], 1)
                    self.assertEqual(result["error_n_bins"], 5)
                    self.assertEqual(result["error_bin_stats"][0]["count"], count)
                    json.dumps(result, allow_nan=False)

    def test_nearly_constant_variance_retains_bins_and_nonzero_cv(self):
        result = error_calibration(vector([0, 1, 2]), vector([0, 0, 0]),
                                   vector([1, 1 + 1e-12, 1 + 2e-12]))
        self.assertGreater(result["n_error_bins"], 1)
        self.assertGreater(result["uncertainty_cv"], 0)

    def test_invalid_inputs_and_bin_configuration_are_rejected(self):
        for variance in ([0, 1], [-1, 1], [math.inf, 1], [math.nan, 1], [1]):
            with self.subTest(variance=variance), self.assertRaises(ValueError):
                error_calibration(vector([0, 1]), vector([1, 2]), vector(variance))
        for bins in (0, -1, 1.5, True):
            with self.subTest(bins=bins), self.assertRaises(ValueError):
                error_calibration(vector([0]), vector([1]), vector([1]), bins)
            with self.assertRaises(ValueError):
                EvaluationConfig(error_n_bins=bins)
        self.assertEqual(ExperimentConfig.from_dict({}).evaluation.error_n_bins, 5)
        config = ExperimentConfig.from_dict({"evaluation": {"error_n_bins": 7}})
        self.assertEqual(config.to_dict()["evaluation"]["error_n_bins"], 7)

    def test_aggregate_uses_equal_fold_means_and_sample_standard_deviations(self):
        base = {"interval_ece": 0.1, "pair_ece": None, "n_pairs": 0}
        folds = [{**base, "n_variants": 2, "ence": .1, "uncertainty_cv": 0},
                 {**base, "n_variants": 100, "ence": .9, "uncertainty_cv": .4},
                 {**base, "n_variants": 1, "ence": .5, "uncertainty_cv": None}]
        result = aggregate_folds(folds)
        self.assertAlmostEqual(result["ence"], .5)
        self.assertAlmostEqual(result["std_ence"], statistics.stdev([.1, .9, .5]))
        self.assertAlmostEqual(result["uncertainty_cv"], .2)
        self.assertAlmostEqual(result["std_uncertainty_cv"], statistics.stdev([0, .4]))
        self.assertEqual(result["n_valid_ence_folds"], 3)
        self.assertEqual(result["n_valid_uncertainty_cv_folds"], 2)
        single = aggregate_folds([folds[2]])
        self.assertIsNone(single["std_ence"])
        self.assertIsNone(single["uncertainty_cv"])
        legacy = aggregate_folds([{**base, "n_variants": 2}])
        self.assertIsNone(legacy["ence"])
        self.assertEqual(legacy["n_valid_ence_folds"], 0)
        for invalid in (math.nan, math.inf, -.1):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                aggregate_folds([{**base, "n_variants": 2, "ence": invalid}])


if __name__ == "__main__":
    unittest.main()
