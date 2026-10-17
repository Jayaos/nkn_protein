"""Synthetic-only device checks and an end-to-end numerical benchmark smoke test."""

import contextlib
import csv
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from config import BranchConfig, EvaluationConfig, ExperimentConfig, ModelConfig, TrainingConfig
from experiments.synthetic_gpu import main, make_folds, make_synthetic_batch, validate_device_parity
from gp.training import FittedGP


def compact_config():
    defaults = ModelConfig()
    return ExperimentConfig(
        model=ModelConfig(
            enabled_branches=("inverse_folding", "sequence_site", "sequence_global", "structure_distance"),
            branches={name: BranchConfig(copies=value.copies, widths=(1,))
                      for name, value in defaults.branches.items()},
            final_widths=(2, 1, 1),
        ),
        training=TrainingConfig(max_steps=2, min_steps=0, patience=2, tolerance=0,
                                noise_initial=1.0, seeds=(7,), lengthscale_multipliers=(1.0,)),
        evaluation=EvaluationConfig(n_folds=5, coverage_levels=(0.5, 0.9), n_bins=2, pair_block_size=8),
        num_threads=1,
    )


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class SyntheticBatchTests(unittest.TestCase):
    def test_dataset_labels_are_safe_identifiers_and_do_not_select_input_assets(self):
        baseline = make_synthetic_batch(40, 4, 19)
        for label in ("control", "benchmark-2.v1", "7_repeat", "x" * 128):
            with self.subTest(label=label):
                batch = make_synthetic_batch(40, 4, 19, dataset_label=label)
                self.assertEqual(batch.assay_id, f"synthetic_{label}")
                self.assertEqual(batch.variant_ids, baseline.variant_ids)
                torch.testing.assert_close(batch.scores, baseline.scores, rtol=0, atol=0)
                torch.testing.assert_close(batch.targets, baseline.targets, rtol=0, atol=0)
        for label in ("", "../outside", "path/name", "path\\name", "two words", "_leading", "A\n", "x" * 129,
                      3, [], {}):
            with self.subTest(invalid_label=label):
                with patch("experiments.synthetic_gpu.torch.Generator") as generator:
                    with self.assertRaises((ValueError, TypeError)):
                        make_synthetic_batch(40, 4, 19, dataset_label=label)
                    generator.assert_not_called()

    def test_reproducible_features_obey_model_and_position_invariants(self):
        batch = make_synthetic_batch(n_variants=40, feature_dim=6, seed=123)
        repeated = make_synthetic_batch(n_variants=40, feature_dim=6, seed=123)
        different = make_synthetic_batch(n_variants=40, feature_dim=6, seed=124)
        self.assertIn("synthetic", batch.assay_id.lower())
        self.assertEqual(batch.num_variants, 40)
        self.assertEqual(batch.scores.shape, (40, 2))
        self.assertEqual(batch.global_features.shape, (40, 6))
        self.assertEqual(batch.variant_ids, repeated.variant_ids)
        torch.testing.assert_close(torch.bincount(batch.owners, minlength=40), torch.ones(40, dtype=torch.long))
        self.assertFalse(torch.equal(batch.scores, different.scores))
        self.assertGreater(batch.targets.std().item(), 0)
        for name, actual, expected in (
            ("scores", batch.scores, repeated.scores),
            ("targets", batch.targets, repeated.targets),
            ("global", batch.global_features, repeated.global_features),
            *((name, value, repeated.local_features[name]) for name, value in batch.local_features.items()),
        ):
            with self.subTest(feature=name):
                self.assertEqual(actual.device.type, "cpu")
                self.assertEqual(actual.dtype, torch.float64)
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        probabilities = batch.local_features["inverse_folding"]
        self.assertEqual(probabilities.shape, (40, 20))
        self.assertTrue((probabilities > 0).all())
        self.assertTrue((probabilities.sum(dim=1) <= 1 + 1e-6).all())
        coordinates = batch.local_features["structure_distance"]
        self.assertEqual(coordinates.shape, (40, 3))
        for position in batch.mutation_positions.unique():
            rows = coordinates[batch.mutation_positions == position]
            torch.testing.assert_close(rows, rows[:1].expand_as(rows), rtol=0, atol=0)

    def test_fold_assignments_are_balanced_reproducible_and_position_aware(self):
        batch = make_synthetic_batch(n_variants=40, feature_dim=4, seed=10)
        for scheme in ("random", "modulo", "contiguous"):
            with self.subTest(scheme=scheme):
                folds = make_folds(batch, scheme, n_folds=5, seed=42)
                self.assertEqual(folds.shape, (40,))
                self.assertEqual(folds.dtype, torch.long)
                self.assertEqual(folds.device.type, "cpu")
                torch.testing.assert_close(folds, make_folds(batch, scheme, n_folds=5, seed=42))
                counts = torch.bincount(folds, minlength=5)
                self.assertEqual(counts.numel(), 5)
                self.assertTrue((counts > 0).all())
                largest_position_group = torch.unique(batch.mutation_positions, return_counts=True)[1].max().item()
                self.assertLessEqual((counts.max() - counts.min()).item(), largest_position_group)
                if scheme != "random":
                    for position in batch.mutation_positions.unique():
                        owners = batch.owners[batch.mutation_positions == position]
                        self.assertEqual(folds[owners].unique().numel(), 1)
        self.assertFalse(torch.equal(make_folds(batch, "random", 5, 42), make_folds(batch, "random", 5, 43)))


class SyntheticBenchmarkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = compact_config()
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config.to_dict()), encoding="utf-8")
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())

    def invoke(self, output, selection=None, device="cpu", dataset_label=None):
        arguments = ([] if selection is None else [selection]) + [
            "--config", str(self.config_path), "--output", str(output), "--device", device,
            "--variants", "40", "--feature-dim", "4", "--seed", "19",
            "--num-threads", "1", "--max-steps", "1", "--min-steps", "0", "--log-every", "1",
        ]
        if dataset_label is not None:
            arguments.extend(("--dataset-label", dataset_label))
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                status = main(arguments)
            except SystemExit as error:
                status = error.code
        return status, stdout.getvalue() + stderr.getvalue()

    def test_cpu_singles_cv_artifacts_and_checkpoint_predictions(self):
        output = self.root / "benchmark"
        status, logs = self.invoke(output)
        self.assertEqual(status, 0, logs)
        for filename in ("resolved_config.json", "synthetic_manifest.json", "benchmark.json"):
            self.assertIsInstance(json.loads((output / filename).read_text(encoding="utf-8")), dict)
        resolved = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
        self.assertEqual(resolved["device"], "cpu")
        self.assertEqual(resolved["training"]["max_steps"], 1)
        benchmark = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
        self.assertEqual(benchmark["device"], "cpu")
        self.assertEqual(set(benchmark["settings"]), {
            "singles_random", "singles_modulo", "singles_contiguous",
        })
        for result in benchmark["settings"].values():
            self.assertTrue(math.isfinite(result["cv_seconds"]) and result["cv_seconds"] > 0)
        batch = make_synthetic_batch(n_variants=40, feature_dim=4, seed=19)
        for scheme in ("random", "modulo", "contiguous"):
            with self.subTest(scheme=scheme):
                directory = output / "cv" / f"singles_{scheme}"
                metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(metrics["n_folds"], 5)
                self.assertEqual(metrics["n_variants"], 40)
                predictions = read_csv(directory / "predictions.csv")
                self.assertEqual({row["variant_id"] for row in predictions}, set(batch.variant_ids))
                self.assertTrue(all(math.isfinite(float(row["mean"])) for row in predictions))
                for fold in range(5):
                    fold_directory = directory / f"fold_{fold}"
                    self.assertTrue((fold_directory / "checkpoint.pt").is_file())
                    history = json.loads((fold_directory / "history.json").read_text(encoding="utf-8"))
                    self.assertEqual([row["step"] for row in history], [0, 1])
                fitted = FittedGP.load(directory / "fold_0" / "checkpoint.pt")
                rows = read_csv(directory / "fold_0" / "predictions.csv")
                ids = [batch.variant_ids.index(row["variant_id"]) for row in rows]
                heldout = replace(batch.select(ids), targets=None)
                self.assertTrue(set(heldout.variant_ids).isdisjoint(fitted.train.variant_ids))
                restored = fitted.predict(heldout)
                expected = torch.tensor([float(row["mean"]) for row in rows], dtype=torch.float64)
                torch.testing.assert_close(restored.mean, expected, rtol=1e-9, atol=1e-10)
                expected_variance = torch.tensor(
                    [float(row["observation_variance"]) for row in rows], dtype=torch.float64,
                )
                torch.testing.assert_close(restored.observation_covariance.diagonal(), expected_variance,
                                           rtol=1e-9, atol=1e-10)

    def test_existing_output_is_preserved(self):
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("original", encoding="utf-8")
        status, _ = self.invoke(output, "singles_random")
        self.assertNotEqual(status, 0)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "original")
        self.assertEqual(list(output.iterdir()), [sentinel])

    def test_dataset_label_survives_cv_artifacts_and_checkpoint_reload(self):
        label = "custom-benchmark_2.v1"
        output = self.root / label
        status, logs = self.invoke(output, "singles_random", dataset_label=label)
        self.assertEqual(status, 0, logs)
        for filename in ("synthetic_manifest.json", "benchmark.json"):
            metadata = json.loads((output / filename).read_text())
            self.assertEqual(metadata["assay_id"], f"synthetic_{label}")
            self.assertEqual(metadata["dataset_label"], label)
            self.assertEqual(metadata["seed"], 19)
        fitted = FittedGP.load(output / "cv/singles_random/fold_0/checkpoint.pt")
        self.assertEqual(fitted.train.assay_id, f"synthetic_{label}")
        batch = make_synthetic_batch(40, 4, 19, dataset_label=label)
        rows = read_csv(output / "cv/singles_random/fold_0/predictions.csv")
        ids = [batch.variant_ids.index(row["variant_id"]) for row in rows]
        prediction = fitted.predict(replace(batch.select(ids), targets=None))
        expected = torch.tensor([float(row["mean"]) for row in rows], dtype=torch.float64)
        torch.testing.assert_close(prediction.mean, expected, rtol=1e-9, atol=1e-10)

    def test_invalid_cli_labels_fail_without_generating_data_or_creating_outputs(self):
        for label in ("", "../outside", "two words", "x" * 129):
            with self.subTest(label=label):
                output = self.root / "invalid_label"
                with patch("experiments.synthetic_gpu.torch.Generator") as generator:
                    status, logs = self.invoke(output, "singles_random", dataset_label=label)
                self.assertNotEqual(status, 0, logs)
                self.assertFalse(output.exists())
                generator.assert_not_called()

    def test_missing_cuda_fails_before_outputs_are_created(self):
        output = self.root / "no_cuda"
        with patch("experiments.synthetic_gpu.torch.cuda.is_available", return_value=False):
            status, logs = self.invoke(output, "singles_random", device="cuda")
        self.assertNotEqual(status, 0)
        self.assertIn("cuda", logs.lower())
        self.assertFalse(output.exists())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_matches_cpu_kernel_gradients_and_posterior(self):
        result = validate_device_parity(self.config, device="cuda")
        self.assertEqual(result["status"], "passed")
        errors = result["max_abs_error"]
        self.assertEqual(set(errors), {
            "kernel", "objective", "gradients", "posterior_mean", "posterior_latent", "posterior_observation",
        })
        self.assertTrue(all(math.isfinite(error) and error >= 0 for error in errors.values()))


if __name__ == "__main__":
    unittest.main()
