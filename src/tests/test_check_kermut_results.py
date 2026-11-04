"""Check saved Kermut completion without importing models or reading tensors."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sbatch_run/run_kermut/check_kermut_results.py"
SPEC = importlib.util.spec_from_file_location("check_kermut_results", SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)

SETTINGS = ("singles_random", "singles_modulo", "singles_contiguous")
PREFIX = "kermut_prosst_esmif"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def snapshot(root):
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


class KermutResultCompletionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.results = self.workspace / "results with spaces"
        self.results.mkdir()
        self.catalog = self.workspace / "catalog.csv"
        self.set_catalog("assay")

    def set_catalog(self, *assays):
        write_csv(self.catalog, ["DMS_id", "DMS_number_single_mutants"], [
            {"DMS_id": assay, "DMS_number_single_mutants": 4} for assay in assays
        ])

    def make_run(self, assay="assay", job=100, *, prefix=PREFIX, settings=SETTINGS):
        run = self.results / assay / f"{prefix}_singles_{job}"
        config = {"evaluation": {"n_folds": 5}, "training": {"n_steps": 150},
                  "model": {"use_structure_distance": True}, "device": "cpu"}
        write_json(run / "config.json", config)
        write_json(run / "execution_plan.json", {
            "settings": {setting: {"fits": 5, "fold_workers": 5} for setting in settings},
        })
        write_json(run / "fold_manifest.json", {
            "assay_id": assay, "requested_settings": list(settings),
            "sources": {"singles": {"settings": {
                setting: {"num_variants": 4, "test_counts": [0, 2, 0, 0, 2]}
                for setting in settings
            }}},
        })
        write_json(run / "validation/validation.json", {
            "valid": True, "assay_id": assay, "n_variants": 4,
        })
        manifest = {"assay_id": assay, "n_variants": 4, "synthetic": False,
                    "coverage": "full", "singles_only": True,
                    "assay_metadata": {"assay_id": assay, "synthetic": False}}
        write_json(run / "validation/input_manifest.json", dict(manifest, setting=None))
        write_json(run / "validation/resolved_config.json", config)
        rows = [{
            "assay_id": assay, "variant_id": f"v{index}", "fold": fold,
            "target": float(index), "target_standardized": float(index), "n_mutations": 1,
            "mean": float(index) + 0.1, "mean_standardized": float(index) + 0.1,
            "latent_variance": 0.5, "observation_variance": 1.0,
            "latent_variance_standardized": 0.5, "observation_variance_standardized": 1.0,
        } for index, fold in enumerate((1, 1, 4, 4))]
        for setting in settings:
            directory = run / "cv" / setting
            write_json(directory / "resolved_config.json", config)
            write_json(directory / "input_manifest.json", dict(manifest, setting=setting))
            write_csv(directory / "fold_assignments.csv", ["variant_id", "fold"], [
                {key: row[key] for key in ("variant_id", "fold")} for row in rows
            ])
            write_csv(directory / "predictions.csv", list(rows[0]), rows)
            fold_metrics = []
            for fold in (1, 4):
                folder = directory / f"fold_{fold}"
                selected = [row for row in rows if row["fold"] == fold]
                metrics = {"fold": fold, "setting": setting, "n_variants": 2,
                           "n_pairs": 1, "spearman": 1.0, "pair_ece": None}
                fold_metrics.append(metrics)
                write_json(folder / "metrics.json", metrics)
                write_json(folder / "resolved_config.json", config)
                write_json(folder / "history.json", [{"step": 150, "nll_per_variant": 1.0}])
                write_json(folder / "restart_summaries.json", [{"status": "complete", "seed": 2024}])
                write_json(folder / "fit_summary.json", {
                    "model": "kermut", "n_train": 2, "optimizer_updates": 150,
                    "checkpoint_selection": "final_iterate",
                })
                # Non-pickle bytes prove the checker does not deserialize checkpoints.
                (folder / "checkpoint.pt").write_bytes(b"nonempty checkpoint placeholder")
                write_csv(folder / "predictions.csv", list(rows[0]), selected)
            write_json(directory / "metrics.json", {
                "assay_id": assay, "setting": setting, "scheme": setting.removeprefix("singles_"),
                "split_source": "provided", "n_variants": 4,
                "n_folds": 2, "fold_ids": [1, 4], "fold_n_variants": [2, 2],
                "fold_metrics": fold_metrics, "avg_spearman": 1.0, "pair_ece": None,
            })
        return run

    def check(self, **kwargs):
        return checker.check_results(self.results, self.catalog, **kwargs)

    def assert_incomplete(self):
        report = self.check()
        self.assertEqual(report["summary"], {"expected": 1, "complete": 0, "missing": 0, "incomplete": 1})
        self.assertEqual(report["assays"][0]["status"], "incomplete")
        self.assertEqual(report["rerun_indices"], [0])
        self.assertTrue(report["assays"][0]["runs"][0]["issues"])

    def test_complete_noncontiguous_populated_folds_are_read_only(self):
        run = self.make_run()
        before = snapshot(self.results)
        report = self.check()
        self.assertEqual(report["summary"], {"expected": 1, "complete": 1, "missing": 0, "incomplete": 0})
        self.assertEqual(report["assays"][0]["complete_runs"], [str(run.resolve())])
        self.assertEqual(report["rerun_indices"], [])
        self.assertEqual(snapshot(self.results), before)

    def test_missing_assay_and_wrong_model_prefix_are_reported_in_catalog_order(self):
        self.set_catalog("assay", "missing", "wrong_model")
        self.make_run()
        self.make_run("wrong_model", prefix="kermut_esm2_proteinmpnn")
        report = self.check()
        self.assertEqual([row["assay_id"] for row in report["assays"]], ["assay", "missing", "wrong_model"])
        self.assertEqual([row["status"] for row in report["assays"]], ["complete", "missing", "missing"])
        self.assertEqual(report["rerun_indices"], [1, 2])

    def test_complete_rerun_counts_despite_failed_older_or_newer_attempts(self):
        successful = self.make_run(job=200)
        for job in (100, 300):
            failed = self.make_run(job=job)
            (failed / "cv/singles_random/input_manifest.json").unlink()
        report = self.check()
        self.assertEqual(report["summary"]["complete"], 1)
        self.assertEqual(report["assays"][0]["complete_runs"], [str(successful.resolve())])
        self.assertEqual(len(report["assays"][0]["runs"]), 3)

    def test_settings_from_separate_partial_runs_do_not_complete_an_assay(self):
        self.make_run(job=100, settings=SETTINGS[:2])
        self.make_run(job=200, settings=SETTINGS[2:])
        self.assert_incomplete()

    def test_missing_final_manifest_and_missing_populated_fold_artifact_fail(self):
        run = self.make_run()
        for relative in ("cv/singles_random/input_manifest.json",
                         "cv/singles_contiguous/fold_4/checkpoint.pt"):
            with self.subTest(path=relative):
                path = run / relative
                original = path.read_bytes()
                path.unlink()
                self.assert_incomplete()
                path.write_bytes(original)

    def test_explicit_failure_or_cancellation_marker_rejects_saved_results(self):
        run = self.make_run()
        for relative in ("cv/singles_random/failure.json", "cv/singles_modulo/fold_1/cancelled.json"):
            with self.subTest(path=relative):
                path = run / relative
                write_json(path, {"error": "interrupted"})
                self.assert_incomplete()
                path.unlink()

    def test_truncated_or_empty_final_json_is_incomplete(self):
        run = self.make_run()
        for relative, content in (("cv/singles_random/metrics.json", b'{"assay_id":'),
                                  ("cv/singles_modulo/fold_1/history.json", b"[]")):
            with self.subTest(path=relative):
                path = run / relative
                original = path.read_bytes()
                path.write_bytes(content)
                self.assert_incomplete()
                path.write_bytes(original)

    def test_truncated_prediction_table_does_not_count_as_complete(self):
        run = self.make_run()
        path = run / "cv/singles_random/predictions.csv"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        self.assert_incomplete()

    def test_csv_parser_errors_do_not_prevent_checking_other_assays(self):
        self.set_catalog("assay", "other")
        run = self.make_run()
        self.make_run("other")
        path = run / "cv/singles_random/predictions.csv"
        original = path.read_text()
        cases = (
            original.splitlines()[0] + '\n"unterminated quoted field\n',
            original.replace("v0,", "x" * (csv.field_size_limit() + 1) + ",", 1),
        )
        for contents in cases:
            with self.subTest(length=len(contents)):
                path.write_text(contents, encoding="utf-8")
                report = self.check()
                self.assertEqual(report["summary"], {
                    "expected": 2, "complete": 1, "missing": 0, "incomplete": 1,
                })
                self.assertTrue(report["assays"][0]["runs"][0]["issues"])
                self.assertEqual(report["assays"][1]["status"], "complete")

    def test_duplicate_variants_wrong_folds_and_nonfinite_predictions_are_rejected(self):
        run = self.make_run()
        path = run / "cv/singles_random/predictions.csv"
        original = path.read_bytes()
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for field, value in (("variant_id", rows[0]["variant_id"]), ("fold", "4"), ("mean", "nan")):
            with self.subTest(field=field):
                changed = [dict(row) for row in rows]
                changed[1][field] = value
                write_csv(path, list(rows[0]), changed)
                self.assert_incomplete()
                path.write_bytes(original)

    def test_wrong_assay_or_setting_identity_is_incomplete(self):
        run = self.make_run()
        for relative, field, value in (("cv/singles_random/input_manifest.json", "assay_id", "different"),
                                       ("cv/singles_random/metrics.json", "setting", "singles_modulo")):
            with self.subTest(path=relative, field=field):
                path = run / relative
                original = path.read_bytes()
                data = json.loads(original)
                data[field] = value
                write_json(path, data)
                self.assert_incomplete()
                path.write_bytes(original)

    def test_manifest_fold_counts_must_agree_with_saved_assignments(self):
        run = self.make_run()
        path = run / "fold_manifest.json"
        data = json.loads(path.read_text())
        data["sources"]["singles"]["settings"]["singles_random"]["test_counts"] = [1, 1, 0, 0, 2]
        write_json(path, data)
        self.assert_incomplete()

    def test_settings_must_cover_the_same_variants(self):
        run = self.make_run()
        directory = run / "cv/singles_modulo"
        for relative in ("predictions.csv", "fold_assignments.csv", "fold_1/predictions.csv"):
            path = directory / relative
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                if row["variant_id"] == "v0":
                    row["variant_id"] = "different_variant"
            write_csv(path, list(rows[0]), rows)
        self.assert_incomplete()

    def test_cli_json_report_and_exit_status(self):
        self.make_run()
        report_path = self.workspace / "reports with spaces" / "completion.json"
        arguments = [str(self.results), "--assay-index", str(self.catalog), "--report", str(report_path)]
        before = snapshot(self.results)
        output = io.StringIO()
        with redirect_stdout(output):
            status = checker.main(arguments)
        self.assertEqual(status, 0)
        report = json.loads(report_path.read_text())
        self.assertEqual(report["summary"]["complete"], 1)
        self.assertEqual(snapshot(self.results), before)
        original_report = report_path.read_bytes()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(checker.main(arguments), 2)
        self.assertEqual(report_path.read_bytes(), original_report)
        self.set_catalog("assay", "missing")
        report_path = report_path.with_name("incomplete.json")
        arguments[-1] = str(report_path)
        with redirect_stdout(io.StringIO()):
            status = checker.main(arguments)
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(report_path.read_text())["rerun_indices"], [1])

    def test_invalid_results_root_or_catalog_returns_configuration_error(self):
        for arguments in ([str(self.workspace / "missing"), "--assay-index", str(self.catalog)],
                          [str(self.results), "--assay-index", str(self.workspace / "absent.csv")]):
            with self.subTest(arguments=arguments), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(checker.main(arguments), 2)

    def test_explicit_settings_and_prefix_check_the_requested_results(self):
        run = self.make_run(prefix="kermut_custom", settings=("singles_random",))
        report = self.check(run_prefix="kermut_custom", settings=("singles_random",))
        self.assertEqual(report["summary"]["complete"], 1)
        self.assertEqual(report["assays"][0]["complete_runs"], [str(run.resolve())])


if __name__ == "__main__":
    unittest.main()
