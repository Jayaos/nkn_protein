"""Small end-to-end experiments, including checkpoint prediction recovery."""

import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import weakref
from unittest.mock import patch

import torch

from config import BranchConfig, EvaluationConfig, ExperimentConfig, ModelConfig, TrainingConfig
from data_loading import FeatureBatch
from evaluation import evaluate_fold, spearman
from experiments import run_cv, run_extrapolation, run_predict
from gp.linalg import NumericalError
from gp.training import FittedGP, fit_gp


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

    def test_cpu_cv_does_not_collect_or_touch_cuda_cache(self):
        events = []
        with patch("experiments.runner.gc.collect") as collect, \
                patch("experiments.runner.torch.cuda.empty_cache") as empty_cache:
            run_cv(self.batch, self.folds, self.config, self.root / "cpu_cache", progress=events.append)
        collect.assert_not_called()
        empty_cache.assert_not_called()
        self.assertEqual([event["fold"] for event in events if event.get("event") == "fold_returned"], [0, 1, 2])
        self.assertFalse(any(event.get("event") == "fold_cleanup_complete" for event in events))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_cv_releases_fold_objects_before_next_fit_with_or_without_logging(self):
        config = replace(self.config, device="cuda")
        release_cache = torch.cuda.empty_cache
        for log_progress in (True, False):
            with self.subTest(log_progress=log_progress):
                fitted_refs, releases, events = [], [], []
                device_index = torch.cuda.current_device()

                def fit(*args, **kwargs):
                    # Every preceding fold must have been cleaned before a new fit.
                    self.assertEqual(len(releases), len(fitted_refs))
                    # Ensure releasable cached blocks even for this tiny model.
                    scratch = torch.empty(8 * 1024**2, dtype=torch.uint8, device=config.device)
                    del scratch
                    fitted = fit_gp(*args, **kwargs)
                    fitted_refs.append(weakref.ref(fitted))
                    # Exercise collection of cycles as well as ordinary local release.
                    fitted._test_cycle = fitted
                    return fitted

                def empty_cache():
                    self.assertTrue(all(reference() is None for reference in fitted_refs))
                    self.assertEqual(torch.cuda.current_device(), device_index)
                    before = torch.cuda.memory_reserved()
                    release_cache()
                    releases.append((before, torch.cuda.memory_reserved()))
                    events.append({"event": "cache_released", "fold": len(releases) - 1})

                with patch("experiments.runner.torch.cuda.empty_cache", side_effect=empty_cache):
                    result = run_cv(
                        self.batch, self.folds, config, self.root / f"cuda_cache_{log_progress}",
                        fit_function=fit, progress=events.append if log_progress else None,
                    )
                self.assertEqual(len(releases), result["n_folds"])
                # Live inputs/library workspaces can pin a segment in a tiny fold.
                self.assertTrue(all(after <= before for before, after in releases), releases)
                self.assertTrue(any(after < before for before, after in releases), releases)
                self.assertTrue(all(reference() is None for reference in fitted_refs))
                if log_progress:
                    boundaries = [(event["event"], event["fold"]) for event in events
                                  if event.get("event") in ("fold_start", "fold_returned",
                                                            "cache_released", "fold_cleanup_complete")]
                    self.assertEqual(boundaries, [(event, fold) for fold in range(3) for event in
                                                  ("fold_start", "fold_returned", "cache_released",
                                                   "fold_cleanup_complete")])

    def test_cv_artifacts_and_reloaded_out_of_fold_predictions(self):
        output = self.root / "cv"
        result = run_cv(self.batch, self.folds, self.config, output, split_source="provided")
        self.assertEqual(result["n_folds"], 3)
        self.assertEqual(result["fold_ids"], [0, 1, 2])
        self.assertEqual(result["n_variants"], 8)
        self.assertEqual(result["split_source"], "provided")
        self.assertAlmostEqual(
            result["interval_ece"], sum(row["interval_ece"] for row in result["fold_metrics"]) / 3
        )
        self.assertAlmostEqual(result["pair_ece"], sum(row["pair_ece"] for row in result["fold_metrics"]) / 3)
        rows = read_predictions(output / "predictions.csv")
        self.assertEqual([row["variant_id"] for row in rows], list(self.batch.variant_ids))
        self.assertEqual([int(row["fold"]) for row in rows], self.folds.tolist())
        means = torch.tensor([float(row["mean"]) for row in rows], dtype=torch.float64)
        self.assertAlmostEqual(result["spearman_oof"], spearman(self.batch.targets, means))
        fold_spearmans = torch.tensor([row["spearman"] for row in result["fold_metrics"]], dtype=torch.float64)
        self.assertAlmostEqual(result["avg_spearman"], fold_spearmans.mean().item())
        self.assertAlmostEqual(result["std_spearman"], fold_spearmans.std(unbiased=False).item())
        self.assertEqual(result["n_valid_spearman_folds"], 3)
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

    def test_cv_runs_only_observed_folds_and_preserves_assignments(self):
        config = replace(self.config, evaluation=replace(self.config.evaluation, n_folds=5))
        for fold_ids in ([0, 1, 2, 3], [0, 2, 4]):
            with self.subTest(fold_ids=fold_ids):
                folds = torch.tensor([fold_ids[index % len(fold_ids)] for index in range(self.batch.num_variants)])
                output = self.root / f"observed_{len(fold_ids)}"
                events = []
                result = run_cv(self.batch, folds, config, output, progress=events.append)
                self.assertEqual(result["n_folds"], len(fold_ids))
                self.assertEqual(result["fold_ids"], fold_ids)
                self.assertEqual([fold["fold"] for fold in result["fold_metrics"]], fold_ids)
                self.assertEqual(result["n_variants"], self.batch.num_variants)
                self.assertEqual(
                    {event["fold"] for event in events if event.get("event") == "fold_start"},
                    set(fold_ids),
                )
                self.assertEqual({path.name for path in output.glob("fold_*") if path.is_dir()}, {f"fold_{fold}" for fold in fold_ids})
                for filename in ("fold_assignments.csv", "predictions.csv"):
                    rows = read_predictions(output / filename)
                    self.assertEqual([row["variant_id"] for row in rows], list(self.batch.variant_ids))
                    self.assertEqual([int(row["fold"]) for row in rows], folds.tolist())
                for fold in fold_ids:
                    fitted = FittedGP.load(output / f"fold_{fold}" / "checkpoint.pt")
                    expected_train = {variant for index, variant in enumerate(self.batch.variant_ids) if folds[index] != fold}
                    self.assertEqual(set(fitted.train.variant_ids), expected_train)
                    torch.testing.assert_close(fitted.transform.target_mean, self.batch.targets[folds != fold].mean())
                self.assertEqual(json.loads((output / "metrics.json").read_text()), result)

    def test_invalid_observed_folds_fail_before_training_and_artifacts(self):
        config = replace(self.config, evaluation=replace(self.config.evaluation, n_folds=5))
        assignments = ([0] * 8, [-1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 5, 0, 1, 0, 1, 0])
        for index, folds in enumerate(assignments):
            output = self.root / f"invalid_folds_{index}"
            with self.subTest(folds=folds), patch("experiments.runner.fit_gp") as trainer:
                with self.assertRaises(ValueError):
                    run_cv(self.batch, torch.tensor(folds), config, output)
                trainer.assert_not_called()
                self.assertFalse(output.exists())

    def test_validation_stopping_still_requires_five_populated_folds(self):
        batch = self.batch.select([0, 1, 2, 3, 5, 6])
        config = replace(
            self.config,
            training=replace(self.config.training, stopping_criterion="validation_spearman"),
            evaluation=replace(self.config.evaluation, n_folds=5),
        )
        output = self.root / "incomplete_validation"
        with patch("experiments.runner.fit_gp") as trainer:
            with self.assertRaisesRegex(ValueError, "requires five populated folds"):
                run_cv(batch, torch.tensor([0, 1, 2, 3, 0, 1]), config, output)
            trainer.assert_not_called()
        self.assertFalse(output.exists())

    def test_cv_checkpoint_then_label_free_predict_and_no_output_overwrite(self):
        cv_output = self.root / "cv"
        run_cv(self.batch, self.folds, self.config, cv_output)
        prediction_output = self.root / "prediction"
        summary = run_predict(
            replace(self.batch, targets=None), cv_output / "fold_2/checkpoint.pt", self.config, prediction_output
        )
        self.assertEqual(summary["n_variants"], 8)
        rows = read_predictions(prediction_output / "predictions.csv")
        self.assertTrue(all(row["target"] == "" for row in rows))
        self.assertFalse((prediction_output / "metrics.json").exists())
        with self.assertRaisesRegex(ValueError, "must be empty"):
            run_cv(self.batch, self.folds, self.config, cv_output)

    def test_single_fold_fit_is_removed_from_experiment_api(self):
        import experiments
        from experiments import runner

        self.assertFalse(hasattr(experiments, "run_fit"))
        self.assertFalse(hasattr(runner, "run_fit"))

    def test_positional_protocol_guard(self):
        with self.assertRaisesRegex(ValueError, "single substitutions only"):
            run_cv(self.batch, self.folds, self.config, self.root / "invalid_position", scheme="modulo")
        self.assertFalse((self.root / "invalid_position").exists())

    def test_missing_folds_do_not_relax_positional_leakage_guard(self):
        batch = self.batch.select([0, 1, 2, 3, 5, 6])
        config = replace(self.config, evaluation=replace(self.config.evaluation, n_folds=5))
        # The two variants at position 2 belong to different observed folds.
        folds = torch.tensor([0, 2, 4, 0, 0, 4])
        for scheme in ("modulo", "contiguous"):
            output = self.root / f"position_leakage_{scheme}"
            with self.subTest(scheme=scheme), patch("experiments.runner.fit_gp") as trainer:
                with self.assertRaisesRegex(ValueError, "mutation position occurs in multiple"):
                    run_cv(batch, folds, config, output, scheme=scheme)
                trainer.assert_not_called()
                self.assertFalse(output.exists())

    def test_unsuccessful_fit_retains_failure_evidence(self):
        error = NumericalError("all synthetic restarts failed")
        error.history = [{"seed": 7, "step": 0, "nll_per_variant": 5.0}]
        error.restart_summaries = [{"seed": 7, "status": "failed", "failure": "synthetic failure"}]
        output = self.root / "failure"
        with patch("experiments.runner.fit_gp", side_effect=error):
            with self.assertRaisesRegex(NumericalError, "all synthetic"):
                run_extrapolation(self.batch, self.config, output)
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
        result = run_extrapolation(batch, self.config, output, split_seed=17)
        self.assertEqual(result["n_train"], 4)
        self.assertEqual(result["n_test"], 2)
        self.assertEqual(result["n_excluded"], 1)
        self.assertEqual(result["n_available_singles"], 5)
        self.assertEqual(result["n_unused_singles"], 1)
        self.assertEqual(result["sample_ratio"], 0.8)
        self.assertEqual(result["training_single_fraction"], 0.8)
        self.assertEqual(result["split_seed"], 17)
        self.assertEqual(result["metrics"]["sample_ratio"], 0.8)
        self.assertEqual(result["metrics"]["aggregation"], "single_fixed_split")
        self.assertNotIn("n_folds", result["metrics"])
        fitted = FittedGP.load(output / "checkpoint.pt")
        self.assertTrue(bool((torch.bincount(fitted.train.owners, minlength=4) == 1).all()))
        predictions = read_predictions(output / "predictions.csv")
        self.assertEqual({row["variant_id"] for row in predictions}, {"variant_4", "variant_7"})
        self.assertTrue(all(row["n_mutations"] == "2" for row in predictions))
        self.assertNotIn("variant_0", fitted.train.variant_ids)
        # pandas.DataFrame.sample(frac=.8, random_state=17) selects offsets
        # [4, 0, 2, 3] from these five singles using the legacy NumPy RNG.
        self.assertEqual(set(fitted.train.variant_ids), {"variant_6", "variant_1", "variant_3", "variant_5"})
        partitions = {row["variant_id"]: row["partition"] for row in read_predictions(output / "partition_assignments.csv")}
        self.assertEqual(partitions["variant_0"], "excluded")
        self.assertEqual(partitions["variant_2"], "unused_single")
        self.assertEqual({key for key, value in partitions.items() if value == "train_single"}, set(fitted.train.variant_ids))
        with self.assertRaises(ValueError):
            run_extrapolation(batch.select([1, 2, 3]), self.config, self.root / "no_doubles")

    def test_single_to_double_full_sample_keeps_all_singles_in_input_order(self):
        expected_singles = ["variant_0", "variant_1", "variant_2", "variant_3", "variant_5", "variant_6"]
        for seed in (0, 2024):
            with self.subTest(seed=seed):
                output = self.root / f"all_singles_{seed}"
                result = run_extrapolation(self.batch, self.config, output, sample_ratio=1.0, split_seed=seed)
                self.assertEqual(result["train_variant_ids"], expected_singles)
                self.assertEqual(result["n_train"], 6)
                self.assertEqual(result["n_unused_singles"], 0)
                self.assertEqual(result["unused_single_variant_ids"], [])
                self.assertEqual(result["sample_ratio"], 1.0)
                self.assertEqual(result["training_single_fraction"], 1.0)
                self.assertEqual(result["test_variant_ids"], ["variant_4", "variant_7"])
                fitted = FittedGP.load(output / "checkpoint.pt")
                self.assertEqual(list(fitted.train.variant_ids), expected_singles)
                partitions = read_predictions(output / "partition_assignments.csv")
                self.assertNotIn("unused_single", {row["partition"] for row in partitions})

    def test_single_to_double_fractional_sample_rounding_and_seed_reproducibility(self):
        # Five singles make 50% a half-integer; pandas uses round(2.5) == 2.
        batch = self.batch.select([0, 1, 2, 3, 4, 5, 7])
        for attempt in range(2):
            output = self.root / f"half_of_five_{attempt}"
            result = run_extrapolation(batch, self.config, output, sample_ratio=0.5, split_seed=17)
            self.assertEqual(result["n_train"], 2)
            self.assertEqual(result["n_unused_singles"], 3)
            self.assertEqual(result["train_variant_ids"], ["variant_5", "variant_0"])
            self.assertEqual(result["test_variant_ids"], ["variant_4", "variant_7"])

    def test_single_to_double_invalid_or_too_small_samples_fail_before_artifacts(self):
        invalid_ratios = (0, -0.1, 1.01, float("nan"), float("inf"), -float("inf"), True, "0.8", None, 0.1)
        for index, ratio in enumerate(invalid_ratios):
            with self.subTest(sample_ratio=ratio), patch("experiments.runner.fit_gp") as trainer:
                output = self.root / f"invalid_sample_{index}"
                with self.assertRaises(ValueError):
                    run_extrapolation(self.batch, self.config, output, sample_ratio=ratio)
                trainer.assert_not_called()
                self.assertFalse(output.exists())

    def test_mixed_cv_reuses_fits_and_evaluates_covariance_submatrices(self):
        output = self.root / "mixed"
        # Both doubles are in fold 1; other double-test folds are empty.
        from gp.training import fit_gp
        with patch("experiments.runner.fit_gp", wraps=fit_gp) as trainer:
            result = run_cv(
                self.batch, self.folds, self.config, output,
                scheme="random_multiples", setting="mixed_random_doubles",
            )
        self.assertEqual(trainer.call_count, 3)
        self.assertEqual(result["n_predicted_variants"], 8)
        self.assertEqual(result["n_variants"], 2)
        self.assertEqual(result["fold_n_variants"], [0, 2, 0])
        self.assertEqual(result["n_valid_spearman_folds"], 1)
        self.assertEqual(result["std_spearman"], 0.0)
        self.assertIsNone(result["fold_metrics"][0]["interval_ece"])
        self.assertIsNone(result["fold_metrics"][2]["spearman"])
        rows = read_predictions(output / "predictions.csv")
        self.assertEqual(len(rows), 8)
        singles = result["setting_metrics"]["mixed_random_singles"]
        doubles = result["setting_metrics"]["mixed_random_doubles"]
        self.assertEqual(singles["n_variants"], 6)
        self.assertEqual(doubles["n_variants"], 2)
        self.assertEqual(result["pair_ece"], doubles["pair_ece"])

        for fold in range(3):
            fitted = FittedGP.load(output / f"fold_{fold}" / "checkpoint.pt")
            test_indices = torch.nonzero(self.folds == fold).flatten()
            test = self.batch.select(test_indices)
            self.assertEqual(set(fitted.train.variant_ids), set(self.batch.variant_ids) - set(test.variant_ids))
            torch.testing.assert_close(fitted.transform.target_mean, self.batch.targets[self.folds != fold].mean())
            prediction = fitted.predict(test)
            counts = torch.bincount(test.owners, minlength=test.num_variants)
            for setting, depth in (("mixed_random_singles", 1), ("mixed_random_doubles", 2)):
                indices = torch.nonzero(counts == depth).flatten()
                observed = result["setting_metrics"][setting]["fold_metrics"][fold]
                self.assertEqual(observed["n_variants"], len(indices))
                if not len(indices):
                    self.assertIsNone(observed["interval_ece"])
                    continue
                expected = evaluate_fold(
                    test.targets[indices], prediction.mean[indices],
                    prediction.observation_covariance[indices][:, indices],
                    self.config.evaluation.coverage_levels, self.config.evaluation.n_bins,
                    self.config.evaluation.pair_block_size,
                )
                self.assertEqual(observed["n_pairs"], expected["n_pairs"])
                self.assertAlmostEqual(observed["interval_ece"], expected["interval_ece"])
                if expected["pair_ece"] is None:
                    self.assertIsNone(observed["pair_ece"])
                else:
                    self.assertAlmostEqual(observed["pair_ece"], expected["pair_ece"])
            fold_rows = [row for row in rows if int(row["fold"]) == fold]
            actual_targets = torch.tensor([float(row["target_standardized"]) for row in fold_rows], dtype=torch.float64)
            torch.testing.assert_close(actual_targets, fitted.transform.standardize_targets(test.targets))

        for setting, depth in (("mixed_random_singles", 1), ("mixed_random_doubles", 2)):
            selected = [row for row in rows if int(row["n_mutations"]) == depth]
            target = torch.tensor([float(row["target_standardized"]) for row in selected], dtype=torch.float64)
            mean = torch.tensor([float(row["mean_standardized"]) for row in selected], dtype=torch.float64)
            self.assertAlmostEqual(result["setting_metrics"][setting]["spearman_oof_standardized"], spearman(target, mean))
        self.assertEqual(json.loads((output / "metrics.json").read_text()), result)


if __name__ == "__main__":
    unittest.main()
