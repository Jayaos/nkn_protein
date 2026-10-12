"""Calibration plots preserve fold metrics, missing data, and result layout."""

from copy import deepcopy
import importlib.util
import json
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


class CalibrationSummaryTests(unittest.TestCase):
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

    def test_discovery_filters_assays_and_ignores_nonrun_directories(self):
        first = self.make_run()
        second = self.make_run("assay_b", "nkn_second")
        self.write_complete_setting(first)
        self.write_complete_setting(second)
        (self.results / "assay_a" / "unrelated").mkdir()
        self.assertEqual(set(discover_runs(self.results)), {first, second})
        self.assertEqual(discover_runs(self.results, assays=["assay_b"]), [second])

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

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is required for plot rendering")
    def test_cli_png_pdf_output_root_preserves_assay_run_hierarchy(self):
        run = self.make_run(settings=("singles_random", "singles_contiguous"))
        self.write_complete_setting(run)
        other = self.make_run("assay_b", "nkn_second")
        self.write_complete_setting(other)
        output = self.root / "exported"
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            path for path in (str(REPOSITORY / "src"), environment.get("PYTHONPATH")) if path)
        environment["MPLBACKEND"] = "Agg"
        environment["MPLCONFIGDIR"] = str(self.root / "matplotlib_cache")
        completed = subprocess.run(
            [sys.executable, "-m", "plotting.calibration", "--results-dir", str(self.results),
             "--assay", "assay_a", "--output-dir", str(output), "--formats", "png", "pdf", "--dpi", "60"],
            cwd=self.root, env=environment, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        destination = output / run.parent.name / run.name
        for plot in ("interval_ece", "pair_ece"):
            self.assertEqual((destination / f"{plot}.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual((destination / f"{plot}.pdf").read_bytes()[:4], b"%PDF")
        self.assertTrue((destination / "calibration_summary.json").is_file())
        self.assertFalse((output / "assay_b").exists())
        self.assertFalse((run / "plots").exists())


if __name__ == "__main__":
    unittest.main()
