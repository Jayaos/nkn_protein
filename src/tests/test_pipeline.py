"""Small end-to-end experiments, including checkpoint prediction recovery."""

import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from config import BranchConfig, EvaluationConfig, ExperimentConfig, ModelConfig, TrainingConfig
from data_loading import FeatureBatch
from evaluation import spearman
from experiments import run_cv, run_extrapolation, run_fit, run_predict
from gp.linalg import NumericalError
from gp.training import FittedGP


def synthetic_batch():
    generator = torch.Generator().manual_seed(40)
    owners = torch.tensor([0, 1, 2, 3, 4, 4, 5, 6, 7, 7])
    positions = torch.tensor([1, 2, 3, 4, 1, 2, 2, 3, 4, 5])
    structure = torch.randn(5, 3, dtype=torch.float64, generator=generator)
    probabilities = torch.softmax(torch.randn(5, 20, dtype=torch.float64, generator=generator), dim=-1)
    return FeatureBatch(
        assay_id="synthetic_pipeline",
        variant_ids=tuple(f"variant_{index}" for index in range(8)),
        targets=torch.tensor([-1, 0.2, 1, 1.7, 2.2, -0.5, 0.9, 2.9], dtype=torch.float64),
        scores=torch.randn(8, 2, dtype=torch.float64, generator=generator),
        global_features=torch.randn(8, 4, dtype=torch.float64, generator=generator),
        local_features={
            "inverse_folding": probabilities[positions - 1],
            "structure": structure[positions - 1],
            "sequence_site": torch.randn(10, 4, dtype=torch.float64, generator=generator),
        },
        owners=owners,
        mutation_positions=positions,
        reference_sequence="ACDEF",
        score_names=("synthetic_first", "synthetic_second"),
        feature_signature={"source": "pipeline_test"},
    )


def small_config():
    defaults = ModelConfig()
    return ExperimentConfig(
        model=ModelConfig(
            branches={name: BranchConfig(copies=branch.copies, widths=(2, 1, 1)) for name, branch in defaults.branches.items()},
            final_widths=(2, 1, 1),
        ),
        training=TrainingConfig(max_steps=2, min_steps=1, patience=1, tolerance=0, seeds=(7,), lengthscale_multipliers=(1.0,)),
        evaluation=EvaluationConfig(n_folds=3, coverage_levels=(0.5, 0.9), n_bins=3, pair_block_size=2),
        num_threads=1,
    )


def read_predictions(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.batch = synthetic_batch()
        self.config = small_config()
        self.folds = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

    def test_cv_artifacts_and_reloaded_out_of_fold_predictions(self):
        output = self.root / "cv"
        result = run_cv(self.batch, self.folds, self.config, output, split_source="development")
        self.assertEqual(result["n_folds"], 3)
        self.assertEqual(result["n_variants"], 8)
        self.assertEqual(result["split_source"], "development")
        self.assertAlmostEqual(
            result["interval_ece"], sum(row["interval_ece"] for row in result["fold_metrics"]) / 3
        )
        self.assertAlmostEqual(result["pair_ece"], sum(row["pair_ece"] for row in result["fold_metrics"]) / 3)
        rows = read_predictions(output / "predictions.csv")
        self.assertEqual([row["variant_id"] for row in rows], list(self.batch.variant_ids))
        self.assertEqual([int(row["fold"]) for row in rows], self.folds.tolist())
        means = torch.tensor([float(row["mean"]) for row in rows], dtype=torch.float64)
        self.assertAlmostEqual(result["spearman_oof"], spearman(self.batch.targets, means))
        self.assertEqual(result["variance_summary"]["double_mutants"]["n_variants"], 2)
        self.assertEqual(json.loads((output / "metrics.json").read_text()), result)

        for fold in range(3):
            fold_output = output / f"fold_{fold}"
            self.assertTrue((fold_output / "resolved_config.json").is_file())
            self.assertTrue((fold_output / "history.json").is_file())
            self.assertTrue((fold_output / "restart_summaries.json").is_file())
            fitted = FittedGP.load(fold_output / "checkpoint.pt")
            test = self.batch.select(torch.nonzero(self.folds == fold).flatten())
            self.assertTrue(set(fitted.train.variant_ids).isdisjoint(test.variant_ids))
            prediction = fitted.predict(replace(test, targets=None))
            fold_rows = read_predictions(fold_output / "predictions.csv")
            expected_means = torch.tensor([float(row["mean"]) for row in fold_rows], dtype=torch.float64)
            expected_variances = torch.tensor([float(row["observation_variance"]) for row in fold_rows], dtype=torch.float64)
            torch.testing.assert_close(prediction.mean, expected_means, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(prediction.observation_covariance.diagonal(), expected_variances, rtol=1e-12, atol=1e-12)

    def test_fit_all_then_label_free_predict_and_no_output_overwrite(self):
        fit_output = self.root / "fit"
        fitted = run_fit(self.batch, self.config, fit_output)
        self.assertEqual(fitted["n_train"], 8)
        self.assertEqual(fitted["n_test"], 0)
        self.assertNotIn("metrics", fitted)
        prediction_output = self.root / "prediction"
        summary = run_predict(replace(self.batch, targets=None), fit_output / "checkpoint.pt", self.config, prediction_output)
        self.assertEqual(summary["n_variants"], 8)
        rows = read_predictions(prediction_output / "predictions.csv")
        self.assertTrue(all(row["target"] == "" for row in rows))
        self.assertFalse((prediction_output / "metrics.json").exists())
        with self.assertRaisesRegex(ValueError, "must be empty"):
            run_fit(self.batch, self.config, fit_output)

    def test_single_held_out_fold_and_positional_protocol_guard(self):
        result = run_fit(self.batch, self.config, self.root / "held_out", folds=self.folds, test_fold=2)
        self.assertEqual(result["n_train"], 6)
        self.assertEqual(result["n_test"], 2)
        self.assertEqual(result["metrics"]["test_fold"], 2)
        with self.assertRaisesRegex(ValueError, "single substitutions only"):
            run_cv(self.batch, self.folds, self.config, self.root / "invalid_position", scheme="modulo")
        self.assertFalse((self.root / "invalid_position").exists())

    def test_unsuccessful_fit_retains_failure_evidence(self):
        error = NumericalError("all synthetic restarts failed")
        error.history = [{"seed": 7, "step": 0, "nll_per_variant": 5.0}]
        error.restart_summaries = [{"seed": 7, "status": "failed", "failure": "synthetic failure"}]
        output = self.root / "failure"
        with patch("experiments.runner.fit_gp", side_effect=error):
            with self.assertRaisesRegex(NumericalError, "all synthetic"):
                run_fit(self.batch, self.config, output)
        self.assertEqual(json.loads((output / "history.json").read_text()), error.history)
        self.assertEqual(json.loads((output / "restart_summaries.json").read_text()), error.restart_summaries)
        self.assertEqual(json.loads((output / "failure.json").read_text())["phase"], "training")
        self.assertFalse((output / "checkpoint.pt").exists())

    def test_single_to_double_is_one_fixed_split_and_excludes_higher_mutants(self):
        # Turn the first single into a triple while preserving the other domains.
        batch = replace(
            self.batch,
            owners=torch.cat([self.batch.owners, torch.tensor([0, 0])]),
            mutation_positions=torch.cat([self.batch.mutation_positions, torch.tensor([2, 3])]),
            local_features={name: torch.cat([values, values[1:3]]) for name, values in self.batch.local_features.items()},
        )
        output = self.root / "single_to_double"
        result = run_extrapolation(batch, self.config, output)
        self.assertEqual(result["n_train"], 5)
        self.assertEqual(result["n_test"], 2)
        self.assertEqual(result["n_excluded"], 1)
        self.assertEqual(result["metrics"]["aggregation"], "single_fixed_split")
        self.assertNotIn("n_folds", result["metrics"])
        fitted = FittedGP.load(output / "checkpoint.pt")
        self.assertTrue(bool((torch.bincount(fitted.train.owners, minlength=5) == 1).all()))
        predictions = read_predictions(output / "predictions.csv")
        self.assertEqual({row["variant_id"] for row in predictions}, {"variant_4", "variant_7"})
        self.assertTrue(all(row["n_mutations"] == "2" for row in predictions))
        self.assertNotIn("variant_0", fitted.train.variant_ids)
        with self.assertRaisesRegex(ValueError, "at least two singles and one double"):
            run_extrapolation(batch.select([1, 2, 3]), self.config, self.root / "no_doubles")


if __name__ == "__main__":
    unittest.main()
