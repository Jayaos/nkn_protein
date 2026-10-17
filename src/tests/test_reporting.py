"""Saved-result aggregation preserves runs and never fills in partial CV scores."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from reporting import aggregate_results


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def run_dir(self, model, job):
        run = self.root / "assay" / f"{model}_singles_{job}"
        self.write(run / "config.json", {"evaluation": {"n_folds": 2},
                                         "training": {"stopping_criterion": "training_nll"}})
        return run

    def metrics(self, setting="singles_random"):
        return {"assay_id": "assay", "setting": setting, "n_folds": 2,
                "avg_spearman": 0.3, "spearman_oof": 0.7, "pair_ece": None,
                "fold_metrics": [{"fold": 0, "spearman": 0.2}, {"fold": 1, "spearman": 0.4}]}

    def test_models_repeated_runs_and_csv_roundtrip(self):
        for model, job in [("nkn", 1), ("nkn", 2), ("kermut", 3)]:
            run = self.run_dir(model, job)
            self.write(run / "cv/singles_random/metrics.json", self.metrics())
        output = self.root / "aggregate"
        tables = aggregate_results(self.root, output)
        self.assertEqual(len(tables["summary"]), 3)
        self.assertEqual(len(tables["folds"]), 6)
        self.assertEqual(tables["issues"], [])
        for row in tables["summary"]:
            self.assertEqual(row["status"], "complete")
            self.assertEqual((row["avg_spearman"], row["spearman_oof"]), (0.3, 0.7))
            self.assertIsNone(row["pair_ece"])
        with (output / "summary.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["pair_ece"], "")
        self.assertEqual(aggregate_results(self.root), tables)

    def test_partial_unstarted_and_corrupt_settings(self):
        run = self.run_dir("nkn", 1)
        self.write(run / "fold_manifest.json", {"requested_settings": [
            "singles_random", "singles_modulo", "singles_contiguous"]})
        self.write(run / "cv/singles_random/fold_0/metrics.json", {"fold": 0, "spearman": 0.9})
        broken = run / "cv/singles_contiguous/metrics.json"
        broken.parent.mkdir(parents=True)
        broken.write_text('{"avg_spearman":')
        tables = aggregate_results(self.root)
        rows = {row["setting"]: row for row in tables["summary"]}
        self.assertEqual(rows["singles_random"]["completed_folds"], 1)
        self.assertNotIn("avg_spearman", rows["singles_random"])
        self.assertEqual(rows["singles_modulo"]["status"], "incomplete")
        self.assertEqual(rows["singles_contiguous"]["status"], "invalid")
        self.assertEqual(len(tables["issues"]), 1)
        self.assertEqual(len(tables["folds"]), 1)

    def test_mixed_depths_and_extrapolation(self):
        run = self.run_dir("kermut", 1)
        mixed = self.metrics("mixed_random_singles")
        mixed["setting_metrics"] = {name: self.metrics(name) for name in
                                    ["mixed_random_singles", "mixed_random_doubles"]}
        self.write(run / "cv/mixed_random_singles/metrics.json", mixed)
        self.write(run / "fit/single_to_double/metrics.json", {
            "assay_id": "assay", "setting": "single_to_double", "spearman": 0.6})
        tables = aggregate_results(self.root)
        self.assertEqual(len(tables["summary"]), 3)
        self.assertEqual(len(tables["folds"]), 4)
        self.assertTrue(all(r["status"] == "complete" for r in tables["summary"]))

    def test_invalid_identity_and_missing_fold(self):
        run = self.run_dir("nkn", 1)
        metrics = self.metrics()
        metrics["assay_id"] = "different_assay"
        self.write(run / "cv/singles_random/metrics.json", metrics)
        metrics = self.metrics("singles_modulo")
        metrics["fold_metrics"] = metrics["fold_metrics"][:1]
        self.write(run / "cv/singles_modulo/metrics.json", metrics)
        tables = aggregate_results(self.root)
        self.assertEqual({r["status"] for r in tables["summary"]}, {"invalid", "incomplete"})
        self.assertEqual(len(tables["issues"]), 1)

    def test_published_subset_is_complete_without_renumbering(self):
        run = self.run_dir("nkn", 1)
        self.write(run / "config.json", {"evaluation": {"n_folds": 5}})
        directory = run / "cv/singles_random"
        ids = [0, 1, 3, 4]
        metrics = {**self.metrics(), "n_folds": 4, "fold_ids": ids,
                   "fold_metrics": [{"fold": fold, "spearman": 0.2} for fold in ids]}
        self.write(directory / "metrics.json", metrics)
        (directory / "fold_assignments.csv").write_text(
            "variant_id,fold\na,0\nb,1\nc,3\nd,4\ne,4\n", encoding="utf-8")
        tables = aggregate_results(self.root)
        row = tables["summary"][0]
        self.assertEqual((row["status"], row["expected_folds"], row["completed_folds"]),
                         ("complete", 4, 4))
        self.assertEqual([fold["fold"] for fold in tables["folds"]], ids)
        self.assertEqual(tables["issues"], [])

    def test_missing_published_fold_remains_incomplete(self):
        run = self.run_dir("nkn", 1)
        self.write(run / "config.json", {"evaluation": {"n_folds": 5}})
        directory = run / "cv/singles_random"
        self.write(directory / "metrics.json", {**self.metrics(), "n_folds": 4,
                   "fold_ids": [0, 1, 3, 4]})
        (directory / "fold_assignments.csv").write_text(
            "variant_id,fold\na,0\nb,1\nc,3\nd,4\n", encoding="utf-8")
        row = aggregate_results(self.root)["summary"][0]
        self.assertEqual((row["status"], row["expected_folds"], row["completed_folds"]),
                         ("incomplete", 4, 2))

    def test_metric_fold_ids_cannot_hide_missing_published_folds(self):
        run = self.run_dir("nkn", 1)
        self.write(run / "config.json", {"evaluation": {"n_folds": 5}})
        directory = run / "cv/singles_random"
        self.write(directory / "metrics.json", {**self.metrics(), "fold_ids": [0, 1]})
        (directory / "fold_assignments.csv").write_text(
            "variant_id,fold\na,0\nb,1\nc,3\nd,4\n", encoding="utf-8")
        tables = aggregate_results(self.root)
        self.assertEqual(tables["summary"][0]["status"], "invalid")
        self.assertIn("disagree", tables["issues"][0]["error"])

    def test_manifest_and_metric_fold_ids_support_runs_without_assignments(self):
        for job, manifest in enumerate((
            {"sources": {"singles": {"settings": {"singles_random": {"fold_ids": [0, 3]}}}}},
            {"fold_ids": {"singles_random": [0, 3]}},
            {},
        )):
            with self.subTest(manifest=manifest):
                run = self.run_dir("nkn", job)
                self.write(run / "config.json", {"evaluation": {"n_folds": 5}})
                self.write(run / "fold_manifest.json", manifest)
                metrics = {**self.metrics(), "fold_ids": [0, 3],
                           "fold_metrics": [{"fold": 0}, {"fold": 3}]}
                self.write(run / "cv/singles_random/metrics.json", metrics)
        rows = aggregate_results(self.root)["summary"]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["status"] == "complete" and row["expected_folds"] == 2 for row in rows))

    def test_unstarted_setting_uses_manifest_populated_fold_count(self):
        run = self.run_dir("nkn", 1)
        self.write(run / "config.json", {"evaluation": {"n_folds": 5}})
        self.write(run / "fold_manifest.json", {"requested_settings": ["singles_random"],
                   "fold_ids": {"singles_random": [0, 1, 2, 3]}})
        row = aggregate_results(self.root)["summary"][0]
        self.assertEqual((row["status"], row["expected_folds"], row["completed_folds"]),
                         ("incomplete", 4, 0))

    def test_missing_and_empty_results(self):
        self.assertEqual(aggregate_results(self.root), {"summary": [], "folds": [], "issues": []})
        with self.assertRaises(FileNotFoundError):
            aggregate_results(self.root / "missing")


if __name__ == "__main__":
    unittest.main()
