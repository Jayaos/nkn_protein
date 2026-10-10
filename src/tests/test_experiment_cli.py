"""Domain protocol routing and safeguards around the public experiment CLI."""

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.cli import main
from gp.training import FittedGP
from test_data import write_csv
from test_pipeline import read_predictions, small_config, synthetic_batch


class ExperimentSettingCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.features = self.root / "features"
        self.features.mkdir()
        (self.features / "assay.json").write_text(json.dumps({"synthetic": True}), encoding="utf-8")
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(small_config().to_dict()), encoding="utf-8")
        self.batch = synthetic_batch()
        self.folds = self.root / "official.csv"
        # Shared positions receive the same fold in both positional schemes.
        random = [0, 1, 2, 0, 1, 2, 0, 1]
        modulo = [0, 1, 2, 0, 1, 1, 2, 1]
        contiguous = [0, 0, 1, 2, 1, 0, 1, 1]
        write_csv(
            self.folds,
            ["mutant", "fold_random_5", "fold_modulo_5", "fold_contiguous_5", "fold_rand_multiples"],
            [[variant, random[i], modulo[i], contiguous[i], random[i]] for i, variant in enumerate(self.batch.variant_ids)],
        )
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)

    def invoke(self, command, output, options=()):
        stdout, stderr = io.StringIO(), io.StringIO()
        arguments = [
            command, "--features", str(self.features), "--config", str(self.config),
            "--output", str(output), "--quiet", *map(str, options),
        ]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                status = main(arguments)
            except SystemExit as error:
                status = error.code
        return status, stdout.getvalue(), stderr.getvalue()

    def test_all_six_public_settings_complete_with_expected_domains(self):
        settings = (
            "singles_random", "singles_modulo", "singles_contiguous",
            "mixed_random_singles", "mixed_random_doubles", "single_to_double",
        )
        for setting in settings:
            with self.subTest(setting=setting), patch("experiments.cli.load_feature_bundle", return_value=self.batch):
                output = self.root / setting
                command = "fit" if setting == "single_to_double" else "cv"
                options = ["--setting", setting]
                if command == "cv":
                    options += ["--folds", self.folds]
                status, _, errors = self.invoke(command, output, options)
                self.assertEqual(status, 0, errors)
                metrics = json.loads((output / "metrics.json").read_text())
                rows = read_predictions(output / "predictions.csv")
                if setting.startswith("singles_"):
                    self.assertEqual(metrics["n_variants"], 6)
                    self.assertEqual({row["n_mutations"] for row in rows}, {"1"})
                elif setting.startswith("mixed_"):
                    self.assertEqual(metrics["n_variants"], 6 if setting.endswith("singles") else 2)
                    self.assertEqual({row["n_mutations"] for row in rows}, {"1", "2"})
                    self.assertEqual(set(metrics["setting_metrics"]), {"mixed_random_singles", "mixed_random_doubles"})
                else:
                    summary = json.loads((output / "run_summary.json").read_text())
                    self.assertEqual(summary["n_train"], 5)
                    self.assertEqual(summary["n_unused_singles"], 1)
                    self.assertEqual(summary["sample_ratio"], 0.8)
                    self.assertEqual(summary["split_seed"], 2024)
                    self.assertEqual({row["n_mutations"] for row in rows}, {"2"})
                    self.assertNotIn("n_folds", metrics)
                manifest = json.loads((output / "input_manifest.json").read_text())
                self.assertEqual(manifest["setting"], setting)
                self.assertNotIn("development_split_seed", manifest)
                if command == "cv":
                    self.assertEqual(manifest["fold_source"], str(self.folds.resolve()))
                    self.assertEqual(metrics["split_source"], "provided")

    def test_single_to_double_sampling_options_reach_training_and_saved_metadata(self):
        output = self.root / "sampled_extrapolation"
        with patch("experiments.cli.load_feature_bundle", return_value=self.batch):
            status, stdout, errors = self.invoke("fit", output, [
                "--setting", "single_to_double", "--sample-ratio", 0.5, "--split-seed", 17,
            ])
        self.assertEqual(status, 0, errors)
        summary = json.loads((output / "run_summary.json").read_text())
        metrics = json.loads((output / "metrics.json").read_text())
        manifest = json.loads((output / "input_manifest.json").read_text())
        printed = json.loads(stdout)
        self.assertEqual(summary["n_train"], 3)
        self.assertEqual(summary["n_unused_singles"], 3)
        self.assertEqual(summary["train_variant_ids"], ["variant_2", "variant_3", "variant_5"])
        self.assertEqual(summary["test_variant_ids"], ["variant_4", "variant_7"])
        for saved in (summary, metrics, printed):
            self.assertEqual(saved["sample_ratio"], 0.5)
            self.assertEqual(saved["training_single_fraction"], 0.5)
            self.assertEqual(saved["split_seed"], 17)
        self.assertEqual(manifest["extrapolation_sample_ratio"], 0.5)
        self.assertEqual(manifest["extrapolation_split_seed"], 17)
        fitted = FittedGP.load(output / "checkpoint.pt")
        self.assertEqual(list(fitted.train.variant_ids), summary["train_variant_ids"])
        partitions = {row["variant_id"]: row["partition"] for row in read_predictions(output / "partition_assignments.csv")}
        self.assertEqual({key for key, value in partitions.items() if value == "train_single"}, set(fitted.train.variant_ids))
        self.assertEqual({key for key, value in partitions.items() if value == "unused_single"}, {"variant_0", "variant_1", "variant_6"})
        self.assertEqual({row["variant_id"] for row in read_predictions(output / "predictions.csv")}, {"variant_4", "variant_7"})

    def test_too_small_training_sample_fails_without_creating_outputs(self):
        output = self.root / "too_few_training_singles"
        with patch("experiments.cli.load_feature_bundle", return_value=self.batch):
            status, _, errors = self.invoke("fit", output, [
                "--setting", "single_to_double", "--sample-ratio", 0.1,
            ])
        self.assertEqual(status, 2, errors)
        self.assertIn("at least two", errors)
        self.assertFalse(output.exists())

    def test_full_cache_coverage_checked_before_mixed_depth_filter(self):
        batch = replace(
            self.batch,
            owners=torch.cat([self.batch.owners, torch.tensor([0, 0])]),
            mutation_positions=torch.cat([self.batch.mutation_positions, torch.tensor([2, 3])]),
            local_features={name: torch.cat([values, values[1:3]]) for name, values in self.batch.local_features.items()},
        )
        output = self.root / "depth_filter"
        with patch("experiments.cli.load_feature_bundle", return_value=batch), patch("experiments.cli.validate_assay_coverage") as coverage:
            status, _, errors = self.invoke("cv", output, [
                "--setting", "mixed_random_doubles", "--folds", self.folds,
                "--assay-index", self.root / "index.csv", "--full-assay",
            ])
        self.assertEqual(status, 0, errors)
        self.assertIs(coverage.call_args.args[0], batch)
        self.assertFalse(coverage.call_args.kwargs["singles_only"])
        for fold in range(small_config().evaluation.n_folds):
            fitted = FittedGP.load(output / f"fold_{fold}/checkpoint.pt")
            self.assertNotIn("variant_0", fitted.train.variant_ids)
        self.assertNotIn("variant_0", {row["variant_id"] for row in read_predictions(output / "predictions.csv")})
        manifest = json.loads((output / "input_manifest.json").read_text())
        self.assertEqual(manifest["excluded_variant_ids"], ["variant_0"])
        self.assertEqual(manifest["n_variants"], 7)

    def test_invalid_modes_fail_before_loading_features_or_creating_outputs(self):
        invalid = [
            ("fit", []),
            ("fit", ["--setting", "singles_random"]),
            ("fit", ["--setting", "single_to_double", "--folds", self.folds]),
            ("fit", ["--setting", "single_to_double", "--development-splits"]),
            ("fit", ["--setting", "single_to_double", "--test-fold", 0]),
            ("fit", ["--setting", "single_to_double", "--split-seed", -1]),
            ("fit", ["--setting", "single_to_double", "--split-seed", 2**32]),
            ("cv", ["--setting", "single_to_double", "--folds", self.folds]),
            ("cv", ["--setting", "mixed_random_singles"]),
            ("cv", ["--setting", "singles_random"]),
            ("cv", ["--setting", "singles_random", "--development-splits"]),
            ("cv", ["--setting", "singles_random", "--folds", self.folds, "--development-splits"]),
            ("cv", ["--setting", "singles_random", "--folds", self.folds, "--split-seed", 2024]),
            ("cv", ["--setting", "singles_random", "--folds", self.folds, "--sample-ratio", 0.5]),
            ("cv", ["--setting", "singles_random", "--folds", self.folds, "--test-fold", 0]),
            ("fit", ["--setting", "mixed_random_singles", "--folds", self.folds, "--singles-only"]),
        ]
        invalid.extend(
            ("fit", ["--setting", "single_to_double", "--sample-ratio", ratio])
            for ratio in ("nan", "inf", "-inf", "0", "-0.1", "1.01", "invalid")
        )
        invalid.extend(
            ("fit", ["--setting", setting])
            for setting in (
                "singles_modulo", "singles_contiguous", "mixed_random_singles", "mixed_random_doubles",
            )
        )
        for index, (command, options) in enumerate(invalid):
            with self.subTest(command=command, options=options), patch("experiments.cli.load_feature_bundle") as loader:
                output = self.root / f"invalid_{index}"
                status, _, errors = self.invoke(command, output, options)
                self.assertEqual(status, 2, errors)
                loader.assert_not_called()
                self.assertFalse(output.exists())

    def test_single_only_assay_rejects_mixed_and_extrapolation_settings(self):
        counts = torch.bincount(self.batch.owners, minlength=self.batch.num_variants)
        singles = self.batch.select(counts == 1)
        for setting in ("mixed_random_singles", "mixed_random_doubles", "single_to_double"):
            with self.subTest(setting=setting), patch("experiments.cli.load_feature_bundle", return_value=singles):
                output = self.root / f"missing_doubles_{setting}"
                options = ["--setting", setting]
                if setting != "single_to_double":
                    options += ["--folds", self.folds]
                command = "fit" if setting == "single_to_double" else "cv"
                status, _, errors = self.invoke(command, output, options)
                self.assertEqual(status, 2, errors)
                self.assertIn("double", errors.lower())
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
