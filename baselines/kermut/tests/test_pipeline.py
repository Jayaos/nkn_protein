"""All six ProteinGym settings through the baseline CLI and shared evaluator."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from baselines.kermut.cli import main
from baselines.kermut.runner import run_extrapolation
from baselines.kermut.training import FittedKermut
from fixtures import make_cache, read_csv, write_csv


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

    def test_all_six_settings_use_published_partitions_and_expected_evaluation_domains(self):
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
                if setting == "single_to_double":
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
                        expected_ids = {self.batch.variant_ids[i] for i in range(expected_count) if self.assignments[i] != fold}
                        self.assertEqual(set(fitted.train.variant_ids), expected_ids)
                        summary = json.loads((output / f"fold_{fold}/fit_summary.json").read_text())
                        self.assertEqual(summary["model"], "kermut")
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
