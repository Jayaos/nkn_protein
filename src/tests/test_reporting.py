"""Aggregation selects the latest run with metrics for every requested setting."""

import csv
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reporting import aggregate_results
from reporting.__main__ import main


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
        self.assertEqual(len(tables["summary"]), 2)
        self.assertEqual(len(tables["folds"]), 4)
        self.assertEqual({row["run_id"] for row in tables["summary"]},
                         {"nkn_singles_2", "kermut_singles_3"})
        self.assertEqual({row["run_id"] for row in tables["folds"]},
                         {"nkn_singles_2", "kermut_singles_3"})
        self.assertEqual(tables["issues"], [])
        for row in tables["summary"]:
            self.assertEqual(row["status"], "complete")
            self.assertEqual((row["avg_spearman"], row["spearman_oof"]), (0.3, 0.7))
            self.assertIsNone(row["pair_ece"])
        with (output / "summary.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["pair_ece"], "")
        self.assertEqual(aggregate_results(self.root), tables)

    def test_all_runs_preserves_configurations_and_reruns_in_csvs(self):
        runs = [self.run_dir(model, job) for model, job in (
            ("nkn_esm2_proteinmpnn", 14521254),
            ("nkn_esm2_proteinmpnn", 14521255),
            ("nkn_prosst_proteinmpnn", 14522762),
            ("kermut_esm2_proteinmpnn", 14522763),
        )]
        for run in runs:
            self.write(run / "cv/singles_random/metrics.json", self.metrics())
        output = self.root / "aggregate"
        tables = aggregate_results(self.root, output, all_runs=True)
        self.assertEqual(len(tables["summary"]), 4)
        self.assertEqual(len(tables["folds"]), 8)
        self.assertEqual(tables["issues"], [])
        for table in ("summary", "folds"):
            with (output / f"{table}.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["run_id"] for row in rows}, {run.name for run in runs})
            self.assertEqual({row["model"] for row in rows}, {"nkn", "kermut"})
        self.assertEqual({row["run_id"] for row in aggregate_results(self.root)["summary"]},
                         {runs[2].name, runs[3].name})

    def test_all_runs_cli_still_excludes_unstarted_settings(self):
        runs = [self.run_dir(model, job) for model, job in (
            ("nkn_esm2_proteinmpnn", 14521255),
            ("nkn_prosst_proteinmpnn", 14522762),
            ("nkn_prosst_proteinmpnn", 14522763),
        )]
        for run in runs:
            self.write(run / "cv/singles_random/metrics.json", self.metrics())
        self.write(runs[2] / "fold_manifest.json", {
            "requested_settings": ["singles_random", "singles_modulo"]})
        output = self.root / "aggregate"
        with patch("sys.argv", ["reporting", "--results", str(self.root),
                               "--output", str(output), "--all-runs"]):
            with redirect_stdout(io.StringIO()):
                main()
        with (output / "summary.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({row["run_id"] for row in rows}, {run.name for run in runs[:2]})
        with (output / "issues.csv").open(newline="", encoding="utf-8") as handle:
            issues = list(csv.DictReader(handle))
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["path"], str(runs[2] / "cv/singles_modulo/metrics.json"))

    def test_partial_unstarted_and_corrupt_settings(self):
        run = self.run_dir("nkn", 1)
        self.write(run / "fold_manifest.json", {"requested_settings": [
            "singles_random", "singles_modulo", "singles_contiguous"]})
        self.write(run / "cv/singles_random/fold_0/metrics.json", {"fold": 0, "spearman": 0.9})
        broken = run / "cv/singles_contiguous/metrics.json"
        broken.parent.mkdir(parents=True)
        broken.write_text('{"avg_spearman":')
        tables = aggregate_results(self.root)
        self.assertEqual(tables["summary"], [])
        self.assertEqual(tables["folds"], [])
        self.assertEqual(len(tables["issues"]), 3)
        self.assertEqual({Path(issue["path"]).parent.name for issue in tables["issues"]},
                         {"singles_random", "singles_modulo", "singles_contiguous"})

    def test_latest_gpu_job_ignores_task_index_and_directory_mtime(self):
        older = self.run_dir("nkn_esm2_proteinmpnn", "gpu_14144978_81")
        newer = self.run_dir("nkn_esm2_proteinmpnn", "gpu_14151727_56")
        for run in (older, newer):
            self.write(run / "cv/singles_random/metrics.json", self.metrics())
        os.utime(older, (2000, 2000))
        os.utime(newer, (1000, 1000))
        tables = aggregate_results(self.root)
        self.assertEqual([row["run_id"] for row in tables["summary"]], [newer.name])
        self.assertEqual({row["run_id"] for row in tables["folds"]}, {newer.name})
        self.assertTrue((older / "cv/singles_random/metrics.json").is_file())

    def test_job_ids_sort_numerically(self):
        for job in (9, 10):
            run = self.run_dir("nkn", job)
            self.write(run / "cv/singles_random/metrics.json", self.metrics())
        self.assertEqual(aggregate_results(self.root)["summary"][0]["run_id"], "nkn_singles_10")

    def test_newer_run_missing_existing_setting_falls_back_as_a_whole(self):
        older = self.run_dir("nkn", 1)
        newer = self.run_dir("nkn", 2)
        for setting in ("singles_random", "singles_modulo"):
            self.write(older / "cv" / setting / "metrics.json", self.metrics(setting))
        self.write(newer / "cv/singles_random/metrics.json", {**self.metrics(), "avg_spearman": 0.9})
        (newer / "cv/singles_modulo").mkdir()
        tables = aggregate_results(self.root)
        self.assertEqual(len(tables["summary"]), 2)
        self.assertEqual({row["run_id"] for row in tables["summary"]}, {older.name})
        self.assertEqual({row["run_id"] for row in tables["folds"]}, {older.name})
        self.assertTrue(all(row["avg_spearman"] == 0.3 for row in tables["summary"]))
        self.assertIn("missing aggregate metrics", tables["issues"][0]["error"])

    def test_unstarted_settings_from_all_metadata_formats_exclude_newer_run(self):
        older = self.run_dir("nkn", 1)
        self.write(older / "cv/singles_random/metrics.json", self.metrics())
        metadata = [
            ("fold_manifest.json", {"requested_settings": ["singles_random", "singles_modulo"]}),
            ("fold_manifest.json", {"fold_ids": {"singles_random": [0, 1], "singles_modulo": [0, 1]}}),
            ("fold_manifest.json", {"sources": {"singles": {"settings": {
                "singles_random": {"fold_ids": [0, 1]}, "singles_modulo": {"fold_ids": [0, 1]}}}}}),
            ("execution_plan.json", {"settings": {"singles_random": {}, "singles_modulo": {}}}),
        ]
        for job, (filename, value) in enumerate(metadata, start=2):
            with self.subTest(metadata=value):
                newer = self.run_dir("nkn", job)
                self.write(newer / filename, value)
                self.write(newer / "cv/singles_random/metrics.json", self.metrics())
                tables = aggregate_results(self.root)
                self.assertEqual([row["run_id"] for row in tables["summary"]], [older.name])
                self.assertTrue(any(issue["path"] == str(newer / "cv/singles_modulo/metrics.json")
                                    for issue in tables["issues"]))

    def test_corrupt_empty_and_non_object_metrics_fall_back(self):
        older = self.run_dir("nkn", 1)
        self.write(older / "cv/singles_random/metrics.json", self.metrics())
        for job, contents in enumerate(('{"avg_spearman":', '[]', '{}', ''), start=2):
            with self.subTest(contents=contents):
                newer = self.run_dir("nkn", job)
                path = newer / "cv/singles_random/metrics.json"
                path.parent.mkdir(parents=True)
                path.write_text(contents, encoding="utf-8")
                tables = aggregate_results(self.root)
                self.assertEqual([row["run_id"] for row in tables["summary"]], [older.name])
                self.assertTrue(any(issue["path"] == str(path) for issue in tables["issues"]))

    def test_explicit_single_setting_run_is_complete(self):
        older = self.run_dir("nkn", 1)
        newer = self.run_dir("nkn", 2)
        for setting in ("singles_random", "singles_modulo", "singles_contiguous"):
            self.write(older / "cv" / setting / "metrics.json", self.metrics(setting))
        self.write(newer / "fold_manifest.json", {"requested_settings": ["singles_random"]})
        self.write(newer / "cv/singles_random/metrics.json", self.metrics())
        tables = aggregate_results(self.root)
        self.assertEqual([(row["run_id"], row["setting"]) for row in tables["summary"]],
                         [(newer.name, "singles_random")])
        self.assertEqual(tables["issues"], [])

    def test_each_assay_is_selected_independently(self):
        for assay in ("assay", "other_assay"):
            for job in (1, 2):
                run = self.root / assay / f"nkn_singles_{job}"
                self.write(run / "cv/singles_random/metrics.json", {**self.metrics(), "assay_id": assay})
        rows = aggregate_results(self.root)["summary"]
        self.assertEqual([(row["assay_id"], row["run_id"]) for row in rows],
                         [("assay", "nkn_singles_2"), ("other_assay", "nkn_singles_2")])

    def test_manual_gpu_runs_use_mtime_not_task_index(self):
        older = self.run_dir("nkn", "gpu_manual_20261017T120000Z_99")
        newer = self.run_dir("nkn", "gpu_manual_20261018T120000Z_1")
        for run in (older, newer):
            self.write(run / "cv/singles_random/metrics.json", self.metrics())
        os.utime(older, (1000, 1000))
        os.utime(newer, (2000, 2000))
        self.assertEqual(aggregate_results(self.root)["summary"][0]["run_id"], newer.name)

    def test_unreadable_or_invalid_setting_metadata_excludes_run(self):
        older = self.run_dir("nkn", 1)
        self.write(older / "cv/singles_random/metrics.json", self.metrics())
        values = [None, {"requested_settings": "singles_random"}, {"fold_ids": []},
                  {"sources": []}, {"sources": {"singles": None}}]
        for job, value in enumerate(values, start=2):
            with self.subTest(value=value):
                newer = self.run_dir("nkn", job)
                self.write(newer / "fold_manifest.json", value)
                self.write(newer / "cv/singles_random/metrics.json", self.metrics())
                tables = aggregate_results(self.root)
                self.assertEqual([row["run_id"] for row in tables["summary"]], [older.name])
                self.assertTrue(any(str(newer) in issue["path"] for issue in tables["issues"]))

    def test_run_without_settings_is_reported(self):
        run = self.run_dir("nkn", 1)
        tables = aggregate_results(self.root)
        self.assertEqual(tables["summary"], [])
        self.assertEqual(tables["folds"], [])
        self.assertEqual(tables["issues"], [{"path": str(run),
                         "error": "No setting directories or requested settings found"}])

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
                self.assertEqual(len(rows), 1)
                self.assertEqual((rows[0]["run_id"], rows[0]["status"], rows[0]["expected_folds"]),
                                 (run.name, "complete", 2))

    def test_unstarted_setting_is_excluded_even_with_manifest_fold_ids(self):
        run = self.run_dir("nkn", 1)
        self.write(run / "config.json", {"evaluation": {"n_folds": 5}})
        self.write(run / "fold_manifest.json", {"requested_settings": ["singles_random"],
                   "fold_ids": {"singles_random": [0, 1, 2, 3]}})
        tables = aggregate_results(self.root)
        self.assertEqual(tables["summary"], [])
        self.assertEqual(tables["folds"], [])
        self.assertEqual(tables["issues"][0]["path"], str(run / "cv/singles_random/metrics.json"))

    def test_missing_and_empty_results(self):
        self.assertEqual(aggregate_results(self.root), {"summary": [], "folds": [], "issues": []})
        with self.assertRaises(FileNotFoundError):
            aggregate_results(self.root / "missing")


if __name__ == "__main__":
    unittest.main()
