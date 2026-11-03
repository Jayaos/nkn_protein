"""All six ProteinGym settings through the baseline CLI and shared evaluator."""

import contextlib
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines.kermut.cli import main
from baselines.kermut.runner import run_extrapolation
from baselines.kermut.training import FittedKermut
from config import EvaluationConfig
from fixtures import make_cache, read_csv, write_csv
from reporting import aggregate_results


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.features, self.config, self.batch, self.index = make_cache(self.root)
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config.to_dict()), encoding="utf-8")
        self.folds = self.root / "published_folds.csv"
        self.assignments = [i // 2 if i < 10 else (i - 10) // 2 for i in range(20)]
        write_csv(self.folds, ["mutant", "fold_random_5", "fold_modulo_5", "fold_contiguous_5", "fold_rand_multiples"],
                  [[variant, fold, fold, fold, fold] for variant, fold in zip(self.batch.variant_ids, self.assignments)])
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)

    def invoke(self, command, output, options=(), use_config=True):
        stdout, stderr = io.StringIO(), io.StringIO()
        arguments = [command, "--features", str(self.features), "--output", str(output), "--quiet", *map(str, options)]
        if use_config:
            arguments.extend(["--config", str(self.config_path)])
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                status = main(arguments)
            except SystemExit as error:
                status = error.code
        return status, stdout.getvalue(), stderr.getvalue()

    def test_constant_variance_calibration_is_saved_and_exported(self):
        variance = .01
        predict = FittedKermut.predict

        def constant_variance(fitted, test):
            prediction = predict(fitted, test)
            covariance = torch.eye(test.num_variants, dtype=prediction.mean.dtype,
                                   device=prediction.mean.device) * variance
            return replace(prediction, observation_covariance=covariance,
                           observation_covariance_standardized=covariance / fitted.transform.target_scale.square())

        results = self.root / "results"
        output = results / self.batch.assay_id / "kermut_constant_1" / "cv" / "singles_random"
        with patch.object(FittedKermut, "predict", constant_variance):
            status, _, errors = self.invoke("cv", output, ["--setting", "singles_random", "--folds", self.folds])
        self.assertEqual(status, 0, errors)
        aggregate = json.loads((output / "metrics.json").read_text())
        expected_fold_ences = []
        for fold in range(5):
            directory = output / f"fold_{fold}"
            rows = read_csv(directory / "predictions.csv")
            self.assertEqual({float(row["observation_variance"]) for row in rows}, {variance})
            rmse = math.sqrt(sum((float(row["target"]) - float(row["mean"])) ** 2
                                 for row in rows) / len(rows))
            expected_ence = abs(math.sqrt(variance) - rmse) / math.sqrt(variance)
            expected_fold_ences.append(expected_ence)
            metrics = json.loads((directory / "metrics.json").read_text())
            self.assertEqual(metrics["error_n_bins"], 5)
            self.assertEqual(metrics["n_error_bins"], 1)
            self.assertEqual(metrics["uncertainty_cv"], 0)
            self.assertTrue(math.isfinite(metrics["ence"]))
            self.assertAlmostEqual(metrics["ence"], expected_ence)
            self.assertEqual(len(metrics["error_bin_stats"]), 1)
            self.assertEqual(metrics["error_bin_stats"][0]["count"], len(rows))
            self.assertEqual(aggregate["fold_metrics"][fold], metrics)
        self.assertEqual(aggregate["uncertainty_cv"], 0)
        self.assertAlmostEqual(aggregate["ence"], sum(expected_fold_ences) / 5)
        report = results / "aggregate"
        aggregate_results(results, report)
        exported = read_csv(report / "folds.csv")
        self.assertEqual(len(exported), 5)
        for row in exported:
            self.assertEqual((row["error_n_bins"], row["n_error_bins"]), ("5", "1"))
            self.assertEqual(float(row["uncertainty_cv"]), 0)
            self.assertAlmostEqual(float(row["ence"]), expected_fold_ences[int(row["fold"])])

    def test_all_six_settings_use_published_partitions_and_expected_evaluation_domains(self):
        self.config = replace(self.config, evaluation=replace(
            self.config.evaluation, coverage_levels=EvaluationConfig().coverage_levels))
        self.config_path.write_text(json.dumps(self.config.to_dict()), encoding="utf-8")
        settings = ("singles_random", "singles_modulo", "singles_contiguous", "mixed_random_singles", "mixed_random_doubles", "single_to_double")
        for setting in settings:
            with self.subTest(setting=setting):
                output = self.root / setting
                command = "fit" if setting == "single_to_double" else "cv"
                options = ["--setting", setting, "--assay-index", self.index, "--full-assay"]
                if command == "cv":
                    options += ["--folds", self.folds]
                status, _, errors = self.invoke(command, output, options)
                self.assertEqual(status, 0, errors)
                metrics = json.loads((output / "metrics.json").read_text())
                rows = read_csv(output / "predictions.csv")
                domains = metrics.get("setting_metrics", {setting: metrics})
                for domain, domain_metrics in domains.items():
                    self.assertGreaterEqual(domain_metrics["ence"], 0)
                    self.assertGreaterEqual(domain_metrics["uncertainty_cv"], 0)
                    selected = rows
                    if domain.startswith("mixed_"):
                        depth = "1" if domain.endswith("singles") else "2"
                        selected = [row for row in rows if row["n_mutations"] == depth]
                    prefix = "mse" if command == "fit" else "mse_oof"
                    for suffix in ("", "_standardized"):
                        expected_mse = sum(
                            (float(row[f"target{suffix}"]) - float(row[f"mean{suffix}"])) ** 2
                            for row in selected
                        ) / len(selected)
                        self.assertAlmostEqual(domain_metrics[f"{prefix}{suffix}"], expected_mse)
                if setting == "single_to_double":
                    self.assertEqual([row["nominal"] for row in metrics["coverage"]],
                                     list(self.config.evaluation.coverage_levels))
                    self.assertEqual(metrics["coverage"][-1]["empirical"], 1)
                    summary = json.loads((output / "run_summary.json").read_text())
                    self.assertEqual(summary["n_train"], 8)
                    self.assertEqual(summary["n_unused_singles"], 2)
                    self.assertNotIn("n_folds", metrics)
                    self.assertEqual({row["n_mutations"] for row in rows}, {"2"})
                    expected = np.random.RandomState(2024).choice(10, size=8, replace=False)
                    fitted = FittedKermut.load(output / "checkpoint.pt")
                    self.assertEqual(fitted.train.variant_ids, tuple(self.batch.variant_ids[i] for i in expected))
                else:
                    self.assertEqual(metrics["n_folds"], 5)
                    self.assertEqual(metrics["split_source"], "provided")
                    self.assertEqual(metrics["n_variants"], 10)
                    self.assertIn("avg_spearman", metrics)
                    self.assertIn("std_spearman", metrics)
                    self.assertIn("spearman_oof", metrics)
                    self.assertIn("interval_ece", metrics)
                    self.assertIn("pair_ece", metrics)
                    expected_count = 10 if setting.startswith("singles_") else 20
                    self.assertEqual(len(rows), expected_count)
                    self.assertEqual([int(row["fold"]) for row in rows], self.assignments[:expected_count])
                    for fold in range(5):
                        fitted = FittedKermut.load(output / f"fold_{fold}/checkpoint.pt")
                        fold_metrics = json.loads((output / f"fold_{fold}/metrics.json").read_text())
                        self.assertEqual([row["nominal"] for row in fold_metrics["coverage"]],
                                         list(self.config.evaluation.coverage_levels))
                        self.assertEqual(fold_metrics["coverage"][-1]["empirical"], 1)
                        for bin_stats in fold_metrics["error_bin_stats"]:
                            self.assertAlmostEqual(bin_stats["rmv_standardized"],
                                                   bin_stats["rmv"] / fitted.transform.target_scale.item())
                        expected_ids = {self.batch.variant_ids[i] for i in range(expected_count) if self.assignments[i] != fold}
                        self.assertEqual(set(fitted.train.variant_ids), expected_ids)
                        summary = json.loads((output / f"fold_{fold}/fit_summary.json").read_text())
                        self.assertEqual(summary["model"], "kermut")
                        self.assertEqual(summary["target_standardization"], "training_population_std")
                        self.assertEqual(summary["target_standardization_ddof"], 0)
                        self.assertEqual(summary["checkpoint_selection"], "final_iterate")
                        self.assertEqual(summary["optimizer_updates"], 2)
                    if setting.startswith("mixed_"):
                        self.assertEqual(set(metrics["setting_metrics"]), {"mixed_random_singles", "mixed_random_doubles"})
                manifest = json.loads((output / "input_manifest.json").read_text())
                self.assertEqual(manifest["setting"], setting)

    def test_saved_fold_config_is_used_for_predict_and_labels_are_optional(self):
        output = self.root / "cv"
        status, _, errors = self.invoke("cv", output, ["--setting", "singles_random", "--folds", self.folds])
        self.assertEqual(status, 0, errors)
        checkpoint = output / "fold_0/checkpoint.pt"
        # Make the input cache label-free; prediction still loads its adjacent
        # baseline config, including the selected scorer and structure mapping.
        variants = read_csv(self.features / "variants.csv")
        for row in variants:
            row.pop("DMS_score")
        write_csv(self.features / "variants.csv", list(variants[0]), [list(row.values()) for row in variants])
        prediction_output = self.root / "predict"
        status, _, errors = self.invoke("predict", prediction_output, ["--checkpoint", checkpoint], use_config=False)
        self.assertEqual(status, 0, errors)
        predictions = read_csv(prediction_output / "predictions.csv")
        self.assertEqual(len(predictions), 20)
        self.assertTrue(all(row["target"] == "" for row in predictions))
        self.assertFalse((prediction_output / "metrics.json").exists())
        saved = json.loads((prediction_output / "resolved_config.json").read_text())
        self.assertEqual(saved["score_sources"], list(self.config.score_sources))
        self.assertEqual(saved["training"]["n_steps"], 2)

    def test_extrapolation_all_singles_and_seeded_subset_match_nkn_protocol(self):
        from baselines.kermut.data import load_feature_bundle

        batch = load_feature_bundle(self.features, self.config)
        for ratio, seed, n_train in ((1.0, 42, 10), (0.5, 17, 5)):
            with self.subTest(ratio=ratio):
                output = self.root / f"sample_{ratio}"
                run_extrapolation(batch, self.config, output, sample_ratio=ratio, split_seed=seed)
                fitted = FittedKermut.load(output / "checkpoint.pt")
                indices = list(range(10)) if ratio == 1 else np.random.RandomState(seed).choice(10, size=n_train, replace=False)
                self.assertEqual(fitted.train.variant_ids, tuple(batch.variant_ids[i] for i in indices))
                self.assertEqual({row["variant_id"] for row in read_csv(output / "predictions.csv")}, set(batch.variant_ids[10:]))
                with self.assertRaisesRegex(ValueError, "must be empty"):
                    run_extrapolation(batch, self.config, output, sample_ratio=ratio, split_seed=seed)


if __name__ == "__main__":
    unittest.main()
