"""Calibration plots preserve fold metrics, missing data, and result layout."""

from copy import deepcopy
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest

from plotting.calibration import discover_runs, load_run, plot_run, summarize_setting


REPOSITORY = Path(__file__).resolve().parents[2]
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


def fold_metrics(index):
    """Unequal pair counts make pooled and equal-fold summaries distinguishable."""
    if index == 0:
        empirical, counts, confidence, accuracy = (0.2, 0.6), (100, 1), (0.6, 0.8), (0.9, 0.0)
        n_variants = 101
    else:
        empirical, counts, confidence, accuracy = (0.8, 1.0), (1, 9), (0.7, 0.9), (0.1, 1.0)
        n_variants = 10
    coverage = [{"nominal": nominal, "empirical": value}
                for nominal, value in zip((0.5, 0.9), empirical)]
    bins = [{"bin": number, "lower": 0.5 + number / 4, "upper": 0.75 + number / 4,
             "count": count, "confidence": predicted, "accuracy": observed}
            for number, (count, predicted, observed) in enumerate(zip(counts, confidence, accuracy))]
    return {
        "fold": index, "n_variants": n_variants, "n_pairs": sum(counts),
        "interval_ece": sum(abs(row["nominal"] - row["empirical"]) for row in coverage) / len(coverage),
        "pair_ece": sum(count * abs(predicted - observed)
                        for count, predicted, observed in zip(counts, confidence, accuracy)) / sum(counts),
        "coverage": coverage, "bin_stats": bins,
    }


def with_error_calibration(fold):
    result = deepcopy(fold)
    offset = fold["fold"]
    result.update(ence=.1 + .2 * offset, uncertainty_cv=.2 + .2 * offset, error_n_bins=2,
                  n_error_bins=2, error_bin_stats=[
                      {"bin": index, "count": count, "rmv": 10 * (index + 1 + offset),
                       "rmse": 10 * (index + .5 + offset), "rmv_standardized": index + 1 + offset,
                       "rmse_standardized": index + .5 + offset}
                      for index, count in enumerate((fold["n_variants"] - 1, 1))])
    return result


def legacy_fold(index):
    return {**fold_metrics(index), "n_variants": 4, "n_pairs": 0, "pair_ece": None, "bin_stats": []}


def legacy_predictions(index):
    targets = [0, 0, 6, 8] if index == 0 else [1, 2, 3, 4]
    return [{"variant_id": f"f{index}_v{i}", "fold": index, "n_mutations": 1,
             "target": target, "mean": 0, "observation_variance": (i + 1) ** 2,
             "observation_variance_standardized": (i + 1) ** 2 / 4, "latent_variance": .01}
            for i, target in enumerate(targets)]


class CalibrationSummaryTests(unittest.TestCase):
    def test_error_curves_use_standardized_units_and_equal_fold_statistics(self):
        folds = [with_error_calibration(fold_metrics(index)) for index in (0, 1)]
        summary = summarize_setting(folds)
        self.assertAlmostEqual(summary["ence"]["mean"], .2)
        self.assertAlmostEqual(summary["uncertainty_cv"]["mean"], .3)
        self.assertAlmostEqual(summary["ence"]["std"], statistics.stdev([.1, .3]))
        self.assertEqual(summary["error_calibration_units"], "training_standardized")
        self.assertEqual(summary["error_bin_stats"][0]["rmv"], 1.5)
        self.assertEqual(summary["error_bin_stats"][0]["rmse"], 1)
        self.assertAlmostEqual(summary["error_bin_stats"][0]["rmv_std"], statistics.stdev([1, 2]))
        self.assertEqual(summary["n_error_curve_folds"], 2)
        self.assertTrue(summary["error_curve_comparable"])
        # Use saved fold ENCEs, not an ENCE recomputed from the mean curve.
        self.assertNotAlmostEqual(summary["ence"]["mean"], ((1.5 - 1) / 1.5 + (2.5 - 2) / 2.5) / 2)

    def test_collapsed_quantiles_and_legacy_folds_do_not_invent_mean_curves(self):
        fold = with_error_calibration(fold_metrics(0))
        fold["error_n_bins"] = 5
        summary = summarize_setting([fold, fold_metrics(1)])
        self.assertFalse(summary["error_curve_comparable"])
        self.assertEqual(summary["error_bin_stats"], [])
        self.assertEqual(summary["n_error_curve_folds"], 1)
        self.assertEqual(summary["ence"], {"mean": .1, "std": None, "n_folds": 1})
        legacy = summarize_setting([fold_metrics(0)])
        self.assertIsNone(legacy["ence"]["mean"])
        self.assertEqual(legacy["n_error_curve_folds"], 0)

    def test_error_curve_units_counts_and_bin_indices_are_validated(self):
        for invalid in ("units", "count", "bin", "variance"):
            first, second = [with_error_calibration(fold_metrics(index)) for index in (0, 1)]
            if invalid == "units":
                for row in second["error_bin_stats"]:
                    row.pop("rmv_standardized")
                    row.pop("rmse_standardized")
            elif invalid == "count":
                first["error_bin_stats"][0]["count"] = 0
            elif invalid == "bin":
                first["error_bin_stats"][0]["bin"] = 2
            else:
                first["error_bin_stats"][0]["rmv_standardized"] = 0
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                summarize_setting([first, second])

    def test_fold_errors_and_reliability_curves_are_averaged_equally(self):
        folds = [fold_metrics(0), fold_metrics(1)]
        summary = summarize_setting(folds)
        self.assertEqual(summary["n_folds"], 2)
        self.assertEqual(summary["n_variants"], 111)
        self.assertEqual(summary["n_pairs"], 111)
        for metric in ("interval_ece", "pair_ece"):
            expected = [fold[metric] for fold in folds]
            self.assertAlmostEqual(summary[metric]["mean"], statistics.mean(expected))
            self.assertAlmostEqual(summary[metric]["std"], statistics.stdev(expected))
            self.assertEqual(summary[metric]["n_folds"], 2)
        self.assertAlmostEqual(summary["coverage"][0]["empirical"], 0.5)
        self.assertAlmostEqual(summary["coverage"][0]["std"], statistics.stdev((0.2, 0.8)))
        self.assertEqual(summary["coverage"][0]["n_folds"], 2)
        self.assertAlmostEqual(summary["bin_stats"][0]["confidence"], 0.65)
        self.assertAlmostEqual(summary["bin_stats"][0]["accuracy"], 0.5)
        self.assertEqual(summary["bin_stats"][0]["count"], 101)
        self.assertEqual(summary["bin_stats"][0]["n_folds"], 2)
        self.assertAlmostEqual(summary["bin_stats"][0]["confidence_std"], statistics.stdev((0.6, 0.7)))
        self.assertAlmostEqual(summary["bin_stats"][0]["accuracy_std"], statistics.stdev((0.9, 0.1)))
        interval_error_of_mean_curve = statistics.mean(
            abs(row["nominal"] - row["empirical"]) for row in summary["coverage"])
        self.assertNotAlmostEqual(summary["interval_ece"]["mean"], interval_error_of_mean_curve)
        pair_error_of_mean_curve = sum(row["count"] * abs(row["confidence"] - row["accuracy"])
                                       for row in summary["bin_stats"]) / summary["n_pairs"]
        pooled_fold_error = sum(fold["n_pairs"] * fold["pair_ece"] for fold in folds) / summary["n_pairs"]
        self.assertNotAlmostEqual(summary["pair_ece"]["mean"], pair_error_of_mean_curve)
        self.assertNotAlmostEqual(summary["pair_ece"]["mean"], pooled_fold_error)

    def test_empty_bins_and_undefined_pair_ece_are_never_replaced_with_zero(self):
        first, second = fold_metrics(0), fold_metrics(1)
        first["bin_stats"][1].update(count=0, confidence=None, accuracy=None)
        first.update(n_pairs=100, pair_ece=0.3)
        for row in second["bin_stats"]:
            row.update(count=0, confidence=None, accuracy=None)
        second.update(n_pairs=0, pair_ece=None)
        summary = summarize_setting([first, second])
        self.assertEqual(summary["pair_ece"], {"mean": 0.3, "std": None, "n_folds": 1})
        self.assertEqual(summary["coverage"][0]["n_folds"], 2)
        nonempty, empty = summary["bin_stats"]
        self.assertEqual(nonempty["n_folds"], 1)
        self.assertEqual(nonempty["confidence"], 0.6)
        self.assertEqual(nonempty["accuracy"], 0.9)
        self.assertIsNone(nonempty["confidence_std"])
        self.assertIsNone(nonempty["accuracy_std"])
        self.assertEqual(empty["count"], 0)
        self.assertEqual(empty["n_folds"], 0)
        for key in ("confidence", "accuracy", "confidence_std", "accuracy_std"):
            self.assertIsNone(empty[key])

    def test_no_evaluations_produce_explicit_undefined_summaries(self):
        summary = summarize_setting([])
        self.assertEqual(summary["n_folds"], 0)
        self.assertEqual(summary["coverage"], [])
        self.assertEqual(summary["bin_stats"], [])
        for metric in ("interval_ece", "pair_ece"):
            self.assertEqual(summary[metric], {"mean": None, "std": None, "n_folds": 0})

    def test_empty_domain_fold_does_not_contribute_zero_to_calibration(self):
        populated = [fold_metrics(0), fold_metrics(1)]
        empty = {"fold": 2, "n_variants": 0, "n_pairs": 0,
                 "interval_ece": None, "pair_ece": None, "coverage": [], "bin_stats": []}
        expected = summarize_setting(populated)
        summary = summarize_setting([*populated, empty])
        self.assertEqual(summary["n_folds"], 3)
        for name in ("interval_ece", "pair_ece", "coverage", "bin_stats", "n_variants", "n_pairs"):
            self.assertEqual(summary[name], expected[name])

    def test_incompatible_coverage_grids_or_bin_edges_are_rejected(self):
        for change in ("coverage", "bins"):
            with self.subTest(change=change):
                first, second = fold_metrics(0), fold_metrics(1)
                if change == "coverage":
                    second["coverage"][0]["nominal"] = 0.4
                else:
                    second["bin_stats"][0]["upper"] = 0.74
                with self.assertRaises(ValueError):
                    summarize_setting([first, second])

    def test_unsorted_coverage_levels_are_aligned_and_sorted(self):
        first, second = fold_metrics(0), fold_metrics(1)
        expected = summarize_setting([first, second])
        first["coverage"].reverse()
        summary = summarize_setting([first, second])
        self.assertEqual(summary["coverage"], expected["coverage"])
        self.assertEqual([row["nominal"] for row in summary["coverage"]], [0.5, 0.9])
        self.assertEqual(summary["interval_ece"], expected["interval_ece"])

    def test_coverage_endpoints_are_preserved_in_plot_summaries(self):
        fold = fold_metrics(0)
        fold["coverage"] = [{"nominal": q / 10, "empirical": q / 10} for q in range(11)]
        fold["interval_ece"] = 0
        summary = summarize_setting([fold])
        self.assertEqual([row["nominal"] for row in summary["coverage"]], [q / 10 for q in range(11)])
        self.assertEqual(summary["coverage"][0]["empirical"], 0)
        self.assertEqual(summary["coverage"][-1]["empirical"], 1)
        self.assertEqual(summary["interval_ece"]["mean"], 0)

    def test_duplicate_coverage_levels_are_rejected_even_if_unsorted(self):
        first = fold_metrics(0)
        first["coverage"] = [first["coverage"][1], first["coverage"][0], deepcopy(first["coverage"][1])]
        with self.assertRaises(ValueError):
            summarize_setting([first])


class CalibrationResultTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.results = self.root / "results"

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def write_predictions(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def make_run(self, assay="assay_a", run_name="nkn_test", settings=("singles_random",)):
        run = self.results / assay / run_name
        self.write(run / "config.json", {"evaluation": {"n_folds": 2}})
        self.write(run / "execution_plan.json", {"settings": {name: {"fits": 2} for name in settings}})
        return run

    def write_complete_setting(self, run, name="singles_random"):
        folds = [fold_metrics(0), fold_metrics(1)]
        aggregate = {
            "setting": name, "n_folds": 2, "n_valid_interval_folds": 2, "n_valid_pair_folds": 2,
            "interval_ece": statistics.mean(fold["interval_ece"] for fold in folds),
            "pair_ece": statistics.mean(fold["pair_ece"] for fold in folds), "fold_metrics": folds,
        }
        self.write(run / "cv" / name / "metrics.json", aggregate)
        return folds

    @staticmethod
    def setting(run_data, name):
        return next(setting for setting in run_data["settings"] if setting["name"] == name)

    def invoke_cli(self, *arguments):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            path for path in (str(REPOSITORY / "src"), environment.get("PYTHONPATH")) if path)
        environment["MPLBACKEND"] = "Agg"
        environment["MPLCONFIGDIR"] = str(self.root / "matplotlib_cache")
        return subprocess.run(
            [sys.executable, "-m", "plotting.calibration", *map(str, arguments)],
            cwd=self.root, env=environment, capture_output=True, text=True, timeout=60,
        )

    def test_discovery_filters_assays_and_ignores_nonrun_directories(self):
        first = self.make_run()
        second = self.make_run("assay_b", "nkn_second")
        self.write_complete_setting(first)
        self.write_complete_setting(second)
        (self.results / "assay_a" / "unrelated").mkdir()
        self.assertEqual(set(discover_runs(self.results)), {first, second})
        self.assertEqual(discover_runs(self.results, assays=["assay_b"]), [second])
        baseline = self.make_run("assay_b", "kermut_first")
        self.write_complete_setting(baseline)
        self.assertEqual(set(discover_runs(self.results, assays=["assay_b"])), {second, baseline})
        self.assertEqual(set(discover_runs(self.results, model_family="nkn")), {first, second})
        self.assertEqual(discover_runs(self.results, model_family="kermut"), [baseline])
        self.assertEqual(discover_runs(self.results, assays=["assay_a"], model_family="kermut"), [])

    def test_embedded_aggregate_folds_are_authoritative_without_double_counting(self):
        run = self.make_run()
        expected = self.write_complete_setting(run)
        for fold in expected:
            conflicting = deepcopy(fold)
            conflicting["interval_ece"] = 0.99
            self.write(run / "cv/singles_random" / f"fold_{fold['fold']}" / "metrics.json", conflicting)
        loaded = self.setting(load_run(run), "singles_random")
        self.assertEqual(loaded["status"], "complete")
        self.assertEqual(loaded["expected_folds"], 2)
        self.assertEqual(loaded["folds"], expected)
        self.assertEqual(loaded["summary"]["n_folds"], 2)
        self.assertAlmostEqual(loaded["summary"]["interval_ece"]["mean"], 0.25)
        self.assertEqual(len(loaded["source_files"]), 1)

    @unittest.skipUnless(HAS_TORCH, "PyTorch is required to recover legacy ENCE")
    def test_legacy_ence_uses_separate_folds_observation_variance_and_training_scale(self):
        run = self.make_run()
        self.write(run / "config.json", {"evaluation": {"n_folds": 2, "error_n_bins": 2}})
        path = run / "cv/singles_random"
        self.write(path / "metrics.json", {"n_folds": 2, "fold_metrics": [legacy_fold(i) for i in (0, 1)]})
        # A pooled CSV must be split by fold, regardless of row order.
        self.write_predictions(path / "predictions.csv", list(reversed(legacy_predictions(0) + legacy_predictions(1))))
        originals = {p: p.read_bytes() for p in path.iterdir()}
        setting = self.setting(load_run(run), "singles_random")
        self.assertEqual([fold["ence"] for fold in setting["folds"]], [1, 0])
        self.assertAlmostEqual(setting["summary"]["ence"]["mean"], .5)
        first = setting["folds"][0]
        self.assertAlmostEqual(first["uncertainty_cv"], statistics.stdev([1, 2, 3, 4]) / 2.5)
        self.assertAlmostEqual(first["error_bin_stats"][0]["rmv"], math.sqrt(2.5))
        self.assertAlmostEqual(first["error_bin_stats"][0]["rmv_standardized"], math.sqrt(2.5) / 2)
        self.assertEqual(first["error_bin_stats"][1]["rmse_standardized"], math.sqrt(50) / 2)
        self.assertTrue(all(record["status"] == "recovered" for record in setting["error_calibration_recovery"]))
        self.assertIn(str(path / "predictions.csv"), setting["source_files"])
        self.assertEqual(originals, {p: p.read_bytes() for p in originals})

    @unittest.skipUnless(HAS_TORCH, "PyTorch is required to recover legacy ENCE")
    def test_legacy_ence_filters_mixed_domains_and_supports_single_fit(self):
        run = self.make_run(settings=("mixed_random_singles", "single_to_double"))
        names = ("mixed_random_singles", "mixed_random_doubles")
        children = {name: {"fold_metrics": [legacy_fold(0)]} for name in names}
        path = run / "cv/mixed_random_singles"
        self.write(path / "metrics.json", {"setting_metrics": children})
        doubles = [{**row, "fold": 0, "n_mutations": 2} for row in legacy_predictions(1)]
        self.write_predictions(path / "fold_0/predictions.csv", legacy_predictions(0) + doubles)
        fit = run / "fit/single_to_double"
        self.write(fit / "metrics.json", legacy_fold(0))
        self.write_predictions(fit / "predictions.csv", [{**row, "fold": ""} for row in doubles])
        loaded = load_run(run)
        self.assertAlmostEqual(self.setting(loaded, names[0])["summary"]["ence"]["mean"], 1)
        self.assertAlmostEqual(self.setting(loaded, names[1])["summary"]["ence"]["mean"], 0)
        self.assertAlmostEqual(self.setting(loaded, "single_to_double")["summary"]["ence"]["mean"], 0)

    @unittest.skipUnless(HAS_TORCH, "PyTorch is required to recover legacy ENCE")
    def test_invalid_legacy_predictions_leave_other_metrics_available_with_reason(self):
        run = self.make_run()
        path = run / "cv/singles_random"
        self.write(path / "metrics.json", {"fold_metrics": [legacy_fold(0)]})
        for invalid in ("count", "duplicates", "variance", "scale", "columns", "fold"):
            rows = legacy_predictions(0)
            if invalid == "count": rows.pop()
            elif invalid == "duplicates": rows[1]["variant_id"] = rows[0]["variant_id"]
            elif invalid == "variance": rows[0]["observation_variance"] = -1
            elif invalid == "scale": rows[0]["observation_variance_standardized"] *= 2
            elif invalid == "columns":
                for row in rows: row.pop("target")
            else: rows[0]["fold"] = 1
            self.write_predictions(path / "fold_0/predictions.csv", rows)
            with self.subTest(invalid=invalid):
                setting = self.setting(load_run(run), "singles_random")
                self.assertEqual(setting["summary"]["n_error_curve_folds"], 0)
                self.assertIsNotNone(setting["summary"]["interval_ece"]["mean"])
                self.assertTrue(setting["error_calibration_recovery"][0]["reason"])

    @unittest.skipUnless(HAS_TORCH, "PyTorch is required to recover legacy ENCE")
    def test_saved_error_bins_take_precedence_over_prediction_recovery(self):
        run = self.make_run()
        expected = with_error_calibration(fold_metrics(0))
        path = run / "cv/singles_random"
        self.write(path / "metrics.json", {"fold_metrics": [expected]})
        self.write_predictions(path / "predictions.csv", [{"unusable": "data"}])
        setting = self.setting(load_run(run), "singles_random")
        self.assertEqual(setting["folds"], [expected])
        self.assertEqual(setting["error_calibration_recovery"], [])

    def test_fold_files_supply_ids_and_partial_failed_settings_stay_explicit(self):
        run = self.make_run(settings=("singles_random", "singles_modulo", "singles_contiguous"))
        available = fold_metrics(0)
        available.pop("fold")
        self.write(run / "cv/singles_random/fold_0/metrics.json", available)
        self.write(run / "cv/singles_random/fold_1/failure.json", {"phase": "training", "error": "test failure"})
        self.write(run / "cv/singles_modulo/fold_0/failure.json", {"phase": "training", "error": "test failure"})
        loaded = load_run(run)
        partial = self.setting(loaded, "singles_random")
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["expected_folds"], 2)
        self.assertEqual(partial["folds"][0]["fold"], 0)
        self.assertEqual(partial["summary"]["n_folds"], 1)
        self.assertTrue(partial["reason"])
        for name in ("singles_modulo", "singles_contiguous"):
            unavailable = self.setting(loaded, name)
            self.assertEqual(unavailable["status"], "unavailable")
            self.assertTrue(unavailable["reason"])
            self.assertEqual(unavailable["folds"], [])
            self.assertIsNone(unavailable["summary"]["interval_ece"]["mean"])
            self.assertIsNone(unavailable["summary"]["pair_ece"]["mean"])

    def test_populated_published_ids_determine_completeness(self):
        run = self.make_run()
        self.write(run / "config.json", {"evaluation": {"n_folds": 5}})
        ids = [0, 1, 3, 4]
        folds = [fold_metrics(index % 2) | {"fold": fold} for index, fold in enumerate(ids)]
        directory = run / "cv/singles_random"
        self.write(directory / "metrics.json", {"n_folds": 4, "fold_ids": ids, "fold_metrics": folds})
        (directory / "fold_assignments.csv").write_text(
            "variant_id,fold\na,0\nb,1\nc,3\nd,4\n", encoding="utf-8")
        setting = self.setting(load_run(run), "singles_random")
        self.assertEqual((setting["status"], setting["expected_folds"]), ("complete", 4))
        self.assertEqual([fold["fold"] for fold in setting["folds"]], ids)
        self.write(directory / "metrics.json", {"n_folds": 4, "fold_ids": ids, "fold_metrics": folds[:-1]})
        setting = self.setting(load_run(run), "singles_random")
        self.assertEqual((setting["status"], setting["expected_folds"]), ("partial", 4))

    def test_partial_fold_files_use_published_ids_before_aggregate_exists(self):
        run = self.make_run()
        self.write(run / "config.json", {"evaluation": {"n_folds": 5}})
        directory = run / "cv/singles_random"
        self.write(directory / "fold_3/metrics.json", fold_metrics(0) | {"fold": 3})
        (directory / "fold_assignments.csv").write_text(
            "variant_id,fold\na,1\nb,3\n", encoding="utf-8")
        setting = self.setting(load_run(run), "singles_random")
        self.assertEqual((setting["status"], setting["expected_folds"]), ("partial", 2))
        self.assertEqual(setting["folds"][0]["fold"], 3)

    def test_conflicting_metric_ids_cannot_hide_missing_published_fold(self):
        run = self.make_run()
        self.write(run / "config.json", {"evaluation": {"n_folds": 5}})
        self.write_complete_setting(run)
        directory = run / "cv/singles_random"
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        metrics["fold_ids"] = [0, 1]
        self.write(directory / "metrics.json", metrics)
        (directory / "fold_assignments.csv").write_text(
            "variant_id,fold\na,0\nb,1\nc,3\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "disagree"):
            load_run(run)

    def test_unstarted_setting_uses_manifest_populated_ids(self):
        run = self.make_run()
        self.write(run / "config.json", {"evaluation": {"n_folds": 5}})
        self.write(run / "fold_manifest.json", {"fold_ids": {"singles_random": [0, 1, 2, 3]}})
        setting = self.setting(load_run(run), "singles_random")
        self.assertEqual((setting["status"], setting["expected_folds"]), ("unavailable", 4))

    def test_shared_mixed_aggregate_replaces_planned_missing_domain_placeholder(self):
        names = ("mixed_random_singles", "mixed_random_doubles")
        run = self.make_run(settings=names)
        singles = [fold_metrics(0), fold_metrics(1)]
        doubles = deepcopy(singles)
        for fold in doubles:
            fold["coverage"][0]["empirical"] += 0.1
            fold["interval_ece"] = sum(abs(row["nominal"] - row["empirical"])
                                       for row in fold["coverage"]) / len(fold["coverage"])
        children = {name: {"setting": name, "n_folds": 2, "fold_metrics": folds}
                    for name, folds in zip(names, (singles, doubles))}
        aggregate = {**children[names[0]], "setting_metrics": children}
        self.write(run / "cv" / names[0] / "metrics.json", aggregate)
        loaded = load_run(run)
        self.assertEqual({setting["name"] for setting in loaded["settings"]}, set(names))
        self.assertEqual(len(loaded["settings"]), 2)
        for name, expected in zip(names, (singles, doubles)):
            setting = self.setting(loaded, name)
            self.assertEqual(setting["status"], "complete")
            self.assertEqual(setting["folds"], expected)
            self.assertEqual(setting["summary"]["n_folds"], 2)

    def test_fold_only_nested_mixed_metrics_expand_into_two_partial_domains(self):
        names = ("mixed_random_singles", "mixed_random_doubles")
        run = self.make_run(settings=names)
        singles, doubles = fold_metrics(0), fold_metrics(1)
        singles.pop("fold")
        doubles.pop("fold")
        children = {name: {**fold, "setting": name} for name, fold in zip(names, (singles, doubles))}
        fold_record = {**children[names[0]], "setting_metrics": children}
        self.write(run / "cv" / names[0] / "fold_0/metrics.json", fold_record)
        loaded = load_run(run)
        self.assertEqual({setting["name"] for setting in loaded["settings"]}, set(names))
        self.assertEqual(len(loaded["settings"]), 2)
        for name, expected in zip(names, (singles, doubles)):
            setting = self.setting(loaded, name)
            self.assertEqual(setting["status"], "partial")
            self.assertEqual(setting["expected_folds"], 2)
            self.assertEqual(len(setting["folds"]), 1)
            self.assertEqual(setting["folds"][0]["fold"], 0)
            self.assertEqual(setting["summary"]["n_folds"], 1)
            self.assertAlmostEqual(setting["summary"]["interval_ece"]["mean"], expected["interval_ece"])

    @unittest.skipUnless(HAS_MATPLOTLIB and HAS_TORCH, "Matplotlib and PyTorch are required for legacy plot rendering")
    def test_error_plot_saves_png_pdf_and_preserves_source_metrics(self):
        run = self.make_run(run_name="kermut_calibration", settings=("singles_random", "singles_contiguous"))
        folds = [legacy_fold(index) for index in (0, 1)]
        self.write(run / "cv/singles_random/metrics.json", {"n_folds": 2, "fold_metrics": folds})
        for index in (0, 1):
            self.write_predictions(run / f"cv/singles_random/fold_{index}/predictions.csv", legacy_predictions(index))
        result = plot_run(run, formats=("png", "pdf"), dpi=60)
        expected_files = {"interval_ece.png", "interval_ece.pdf", "pair_ece.png", "pair_ece.pdf", "ence.png", "ence.pdf",
                          "calibration_summary.json"}
        self.assertEqual({path.name for path in (run / "plots").iterdir()}, expected_files)
        self.assertEqual({Path(path).name for path in result["files"]}, expected_files)
        self.assertEqual((run / "plots/ence.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual((run / "plots/ence.pdf").read_bytes()[:4], b"%PDF")
        self.assertEqual(result["skipped_plots"], {})
        self.assertEqual(self.setting(result, "singles_random")["summary"]["ence"]["n_folds"], 2)
        saved = json.loads((run / "cv/singles_random/metrics.json").read_text())
        self.assertEqual(saved["fold_metrics"], folds)

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is required for plot rendering")
    def test_plot_run_defaults_to_run_plots_and_writes_json_summary(self):
        run = self.make_run(settings=("singles_random", "singles_contiguous"))
        self.write_complete_setting(run)
        result = plot_run(run, formats=("png",), dpi=60)
        self.assertEqual(Path(result["output_dir"]), run / "plots")
        for name in ("interval_ece.png", "pair_ece.png"):
            self.assertEqual((run / "plots" / name).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        saved = json.loads((run / "plots/calibration_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(self.setting(saved, "singles_contiguous")["status"], "unavailable")
        self.assertEqual(self.setting(saved, "singles_random")["summary"]["n_folds"], 2)
        self.assertIn("No saved error_bin_stats", saved["skipped_plots"]["ence"])
        self.assertFalse((run / "plots/ence.png").exists())

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is required for plot rendering")
    def test_cli_png_pdf_output_root_groups_by_results_directory_and_assay(self):
        self.results /= "kermut_experiment"
        run = self.make_run(run_name="kermut_calibration", settings=("singles_random", "singles_contiguous"))
        folds = [with_error_calibration(fold) for fold in self.write_complete_setting(run)]
        self.write(run / "cv/singles_random/metrics.json", {"n_folds": 2, "fold_metrics": folds})
        other = self.make_run("assay_b", "nkn_second")
        self.write_complete_setting(other)
        output = self.root / "exported"
        completed = self.invoke_cli("--results-dir", self.results, "--assay", "assay_a",
                                    "--output-dir", output, "--formats", "png", "pdf", "--dpi", "60")
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        destination = output / "kermut_experiment" / "assay_a"
        for plot in ("interval_ece", "pair_ece", "ence"):
            self.assertEqual((destination / f"{plot}.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual((destination / f"{plot}.pdf").read_bytes()[:4], b"%PDF")
        self.assertTrue((destination / "calibration_summary.json").is_file())
        self.assertFalse((output / "kermut_experiment" / "assay_b").exists())
        self.assertFalse((destination / run.name).exists())
        self.assertFalse((run / "plots").exists())

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is required for plot rendering")
    def test_cli_plots_both_results_groups_once_with_shared_assay_filter(self):
        runs = []
        for group, model in (("hierarchical_nkn", "nkn_first"), ("kermut_experiment", "kermut_second")):
            self.results = self.root / "results" / group
            run = self.make_run(run_name=model)
            folds = [with_error_calibration(fold) for fold in self.write_complete_setting(run)]
            self.write(run / "cv/singles_random/metrics.json", {"n_folds": 2, "fold_metrics": folds})
            self.write_complete_setting(self.make_run("assay_b", model))
            runs.append(run)
        output = self.root / "exported"
        completed = self.invoke_cli(
            "--results-dir", runs[0].parent.parent, f"--results-dir={runs[1].parent.parent}",
            "--results-dir", runs[0].parent.parent, "--assay", "assay_a",
            "--output-dir", output, "--formats", "png", "--dpi", "60")
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(completed.stdout.count("Saved calibration plots:"), 2)
        for run in runs:
            destination = output / run.parent.parent.name / "assay_a"
            for plot in ("interval_ece", "pair_ece", "ence"):
                self.assertEqual((destination / f"{plot}.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            summary = json.loads((destination / "calibration_summary.json").read_text())
            self.assertEqual(Path(summary["run_dir"]), run)
            self.assertEqual(summary["model"], run.name)
            self.assertFalse((destination.parent / "assay_b").exists())

    def test_cli_checks_every_input_root_before_writing_plots(self):
        run = self.make_run()
        self.write_complete_setting(run)
        empty = self.root / "empty_results"
        empty.mkdir()
        output = self.root / "exported"
        completed = self.invoke_cli("--results-dir", self.results, "--results-dir", empty,
                                    "--output-dir", output)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(f"No matching NKN/Kermut runs found in {empty}", completed.stderr)
        self.assertFalse(output.exists())

    def test_cli_rejects_wrong_model_roots_and_explicit_run_dirs(self):
        for family, other in (("nkn", "kermut"), ("kermut", "nkn")):
            self.results = self.root / f"{other}_experiment"
            run = self.make_run(run_name=f"{other}_first")
            self.write_complete_setting(run)
            output = self.root / "exported"
            for option, path in (("--results-dir", self.results), ("--run-dir", run)):
                with self.subTest(family=family, option=option):
                    completed = self.invoke_cli(option, path, "--model-family", family, "--output-dir", output)
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn("No matching" if option == "--results-dir" else f"requires a {family}_* run", completed.stderr)
                    self.assertFalse(output.exists())

    def test_cli_rejects_groups_that_would_share_output_folder(self):
        roots = [self.root / location / "same_name" for location in ("first", "second")]
        for root in roots:
            self.results = root
            self.write_complete_setting(self.make_run())
        output = self.root / "exported"
        completed = self.invoke_cli("--results-dir", roots[0], "--results-dir", roots[1],
                                    "--output-dir", output)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("share a folder name", completed.stderr)
        self.assertFalse(output.exists())

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is required for plot rendering")
    def test_multiple_runs_in_one_assay_keep_separate_files_and_summaries(self):
        self.results /= "comparison"
        runs = [self.make_run(run_name=name) for name in ("nkn_first", "kermut_second")]
        for run in runs:
            folds = [with_error_calibration(fold) for fold in self.write_complete_setting(run)]
            self.write(run / "cv/singles_random/metrics.json", {"n_folds": 2, "fold_metrics": folds})
        output = self.root / "exported"
        completed = self.invoke_cli("--results-dir", self.results, "--model-family", "nkn",
                                    "--output-dir", output, "--formats", "png", "--dpi", "60")
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(completed.stdout.count("Saved calibration plots:"), 1)
        destination = output / "comparison" / "assay_a"
        first = json.loads((destination / "nkn_first__calibration_summary.json").read_text())
        self.assertEqual(len(list(destination.iterdir())), 4)
        original = {path: Path(path).read_bytes() for path in first["files"]}
        completed = self.invoke_cli("--run-dir", runs[1], "--model-family", "kermut",
                                    "--output-dir", output, "--formats", "png", "--dpi", "60")
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(completed.stdout.count("Saved calibration plots:"), 1)
        second = json.loads((destination / "kermut_second__calibration_summary.json").read_text())
        for run, result in zip(runs, (first, second)):
            self.assertEqual(Path(result["output_dir"]), destination)
            expected = {f"{run.name}__{name}" for name in
                        ("interval_ece.png", "pair_ece.png", "ence.png", "calibration_summary.json")}
            self.assertEqual({Path(path).name for path in result["files"]}, expected)
            summary = json.loads((destination / f"{run.name}__calibration_summary.json").read_text())
            self.assertEqual(Path(summary["run_dir"]), run)
            self.assertEqual({path.name for path in run.iterdir()}, {"config.json", "execution_plan.json", "cv"})
        self.assertEqual(len(list(destination.iterdir())), 8)
        self.assertEqual(original, {path: Path(path).read_bytes() for path in first["files"]})


if __name__ == "__main__":
    unittest.main()
