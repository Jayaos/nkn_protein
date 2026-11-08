"""Check assay weighting, matched cohorts, and Figure 2 export edge cases."""

from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from plotting.predictive_variance import (
    SETTINGS, build_comparison, load_setting, main, select_runs, write_comparison,
)


HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def variance_metrics(count, mean):
    return {"n_variants": count, "variance_summary": {
        "single_mutants": {"n_variants": count,
                           "mean_observation_variance_standardized": mean,
                           "mean_latent_variance_standardized": mean / 2}}}


class PredictiveVarianceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def make_run(self, model, assay="assay_a", suffix="test", factor=1):
        run = self.root / "results" / assay / f"{model}_{suffix}"
        self.write(run / "config.json", {"evaluation": {"n_folds": 3}})
        for setting in SETTINGS:
            # The populated folds need not be consecutive; fold 1 is empty.
            folds = [{"fold": 0, **variance_metrics(1, .2 * factor)},
                     {"fold": 2, **variance_metrics(3, .6 * factor)}]
            metrics = {**variance_metrics(4, .5 * factor), "n_folds": 2,
                       "fold_ids": [0, 2], "fold_metrics": folds,
                       "setting": setting, "assay_id": assay}
            self.write(run / "cv" / setting / "metrics.json", metrics)
            (run / "cv" / setting / "fold_assignments.csv").write_text(
                "variant_id,fold\nA1V,0\nA2V,2\nA3V,2\nA4V,2\n", encoding="utf-8")
        return run

    def pair(self):
        return ({"assay_a": self.make_run("kermut")},
                {"assay_a": self.make_run("nkn", factor=2)})

    def mutate_metrics(self, run, setting, change):
        path = run / "cv" / setting / "metrics.json"
        value = json.loads(path.read_text())
        change(value)
        self.write(path, value)

    def test_saved_mean_is_one_observation_per_assay_and_setting(self):
        runs = self.pair()
        # Unrelated domains are deliberately unreadable and must not be read.
        self.write(runs[0]["assay_a"] / "cv/mixed_random_singles/metrics.json", {"invalid": True})
        summary = build_comparison(*runs)
        self.assertEqual(summary["n_assays"], 1)
        self.assertEqual(summary["issues"], [])
        self.assertEqual(len(summary["rows"]), 6)
        for row in summary["rows"]:
            self.assertIn(row["setting"], SETTINGS)
            self.assertEqual(row["n_variants"], 4)
            self.assertEqual(row["n_folds"], 2)
            self.assertEqual(row["mean_variance"], .5 if row["model"] == "kermut" else 1)
            self.assertEqual(row["pairing_verification"], "variant_fold_assignments")
        latent = build_comparison(*runs, kind="latent")
        self.assertEqual([r["mean_variance"] for r in latent["rows"]], [.25, .5] * 3)

    def test_missing_aggregate_summary_weights_fold_means_by_variant_count(self):
        run = self.make_run("kermut")
        self.mutate_metrics(run, SETTINGS[0], lambda data: data.pop("variance_summary"))
        row = load_setting(run, SETTINGS[0])
        self.assertAlmostEqual(row["mean_variance"], (.2 + 3 * .6) / 4)
        self.assertNotAlmostEqual(row["mean_variance"], (.2 + .6) / 2)
        self.assertEqual(row["aggregation_source"], "variant_weighted_fold_summaries")

    def test_all_six_distributions_share_a_complete_cohort(self):
        kermut, nkn = self.pair()
        kermut["only_kermut"] = self.make_run("kermut", "only_kermut")
        for model, runs in (("kermut", kermut), ("nkn", nkn)):
            runs["partial"] = self.make_run(model, "partial")
        self.mutate_metrics(nkn["partial"], SETTINGS[2], lambda data: data["fold_metrics"].pop())
        summary = build_comparison(kermut, nkn)
        self.assertEqual(summary["assays"], ["assay_a"])
        self.assertEqual(len(summary["rows"]), 6)
        self.assertEqual({issue["assay_id"] for issue in summary["issues"]}, {"only_kermut", "partial"})
        self.assertTrue(all(issue["status"] == "excluded" for issue in summary["issues"]))

    def test_different_variants_or_fold_assignments_are_excluded(self):
        for assignments in ("B1V,0\nA2V,2\nA3V,2\nA4V,2\n",
                            "A1V,2\nA2V,0\nA3V,2\nA4V,2\n"):
            with self.subTest(assignments=assignments):
                kermut, nkn = self.pair()
                path = nkn["assay_a"] / "cv" / SETTINGS[0] / "fold_assignments.csv"
                path.write_text("variant_id,fold\n" + assignments, encoding="utf-8")
                summary = build_comparison(kermut, nkn)
                self.assertEqual(summary["n_assays"], 0)
                self.assertIn("different variant/fold assignments", summary["issues"][0]["reason"])

    def test_missing_assignments_are_reported_as_count_only_pairing(self):
        kermut, nkn = self.pair()
        (nkn["assay_a"] / "cv" / SETTINGS[0] / "fold_assignments.csv").unlink()
        summary = build_comparison(kermut, nkn)
        self.assertEqual(summary["n_assays"], 1)
        self.assertEqual(summary["issues"][0]["status"], "warning")
        self.assertEqual(summary["rows"][0]["pairing_verification"], "fold_counts_only")

    def test_invalid_variances_counts_and_failed_settings_are_not_plotted(self):
        run = self.make_run("kermut")
        path = run / "cv" / SETTINGS[0] / "metrics.json"
        original = json.loads(path.read_text())
        for invalid in (None, float("nan"), float("inf"), -1, 0, True):
            with self.subTest(value=invalid):
                data = deepcopy(original)
                data["variance_summary"]["single_mutants"]["mean_observation_variance_standardized"] = invalid
                self.write(path, data)
                with self.assertRaises(ValueError):
                    load_setting(run, SETTINGS[0])
        data = deepcopy(original)
        data["variance_summary"]["single_mutants"]["n_variants"] = 3
        self.write(path, data)
        with self.assertRaisesRegex(ValueError, "only single mutants"):
            load_setting(run, SETTINGS[0])
        self.write(path, original)
        self.write(path.parent / "failure.json", {"error": "interrupted"})
        with self.assertRaisesRegex(ValueError, "failure.json"):
            load_setting(run, SETTINGS[0])

    def test_manifest_and_assignment_fold_mismatch_is_rejected(self):
        run = self.make_run("kermut")
        self.write(run / "fold_manifest.json", {"fold_ids": {SETTINGS[0]: [0, 1]}})
        with self.assertRaisesRegex(ValueError, "disagree"):
            load_setting(run, SETTINGS[0])

    def test_ambiguous_runs_require_explicit_selection(self):
        first = self.make_run("kermut", suffix="one")
        self.make_run("kermut", suffix="two")
        with self.assertRaisesRegex(ValueError, "--kermut-run-dir"):
            select_runs(self.root / "results", "kermut")
        self.assertEqual(select_runs(self.root / "missing", "kermut", [first, first]), {"assay_a": first})
        with self.assertRaisesRegex(ValueError, "Expected a nkn_"):
            select_runs(self.root / "results", "nkn", [first])

    def test_no_shared_complete_assays_exports_exclusion_reasons_and_fails(self):
        kermut, nkn = self.pair()
        self.write(nkn["assay_a"] / "cv" / SETTINGS[1] / "failure.json", {"error": "failed"})
        summary = build_comparison(kermut, nkn)
        output = self.root / "plots"
        with self.assertRaisesRegex(ValueError, "No matched complete assays"):
            write_comparison(summary, output)
        saved = json.loads((output / "predictive_variance_summary.json").read_text())
        self.assertEqual(saved["n_assays"], 0)
        self.assertEqual(saved["issues"][0]["status"], "excluded")
        self.assertFalse((output / "predictive_variance.png").exists())

    @unittest.skipUnless(HAS_MATPLOTLIB, "Matplotlib is required for rendering")
    def test_exports_singleton_constant_and_variable_distributions(self):
        kermut, nkn = self.pair()
        for cohort in ("singleton", "constant", "variable"):
            if cohort != "singleton":
                factor = 1 if cohort == "constant" else 3
                for model, runs in (("kermut", kermut), ("nkn", nkn)):
                    runs[cohort] = self.make_run(model, cohort, factor=factor)
            summary = build_comparison(kermut, nkn)
            output = self.root / cohort
            files = write_comparison(summary, output, formats=("png", "pdf", "svg"), dpi=70)
            self.assertEqual(len(files), 5)
            for path in files:
                self.assertGreater(Path(path).stat().st_size, 100)
            with (output / "predictive_variance_per_assay.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6 * len(kermut))
            self.assertEqual(json.loads(rows[0]["source_files"]), summary["rows"][0]["source_files"])

    @unittest.skipUnless(HAS_MATPLOTLIB, "Matplotlib is required for rendering")
    def test_cli_with_separate_model_run_paths(self):
        kermut, nkn = self.pair()
        output = self.root / "cli"
        status = main(["--kermut-run-dir", str(kermut["assay_a"]), "--nkn-run-dir", str(nkn["assay_a"]),
                       "--output-dir", str(output), "--formats", "png", "--variance-kind", "latent"])
        self.assertEqual(status, 0)
        self.assertTrue((output / "predictive_variance.png").is_file())
        summary = json.loads((output / "predictive_variance_summary.json").read_text())
        self.assertEqual(summary["variance_kind"], "latent")


if __name__ == "__main__":
    unittest.main()
