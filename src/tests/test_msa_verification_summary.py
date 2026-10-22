"""Catalog completeness and run selection for the MSA verification summary."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("msa_summary", ROOT / "sbatch_run/run_prepare_rep/summarize_msa_verification.py")
summary_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary_tool)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.reference = self.root / "catalog.csv"
        self.reference.write_text("DMS_id\nA\nB\nC\n")
        self.digest = hashlib.sha256(self.reference.read_bytes()).hexdigest()

    def report(self, assay="A", suffix="one", status="weights_reproduced", **comparison):
        path = self.root / f"{assay}_{suffix}"
        path.mkdir()
        comp = dict(recomputed_count=2, supplied_count=2, finite_positive_weights_at_most_one=True,
                    counts_match=True, values_match_in_order=True, exact_match=True,
                    max_absolute_difference=0.0, mismatching_indices_zero_based=[])
        comp.update(comparison)
        report = dict(assay_id=assay, status=status, created_at="2026-10-22T00:00:00+00:00",
                      inputs={"reference": {"sha256": self.digest}}, comparisons={"specialized": comp})
        (path / "verification_report.json").write_text(json.dumps(report))
        return path

    def summarize(self, job_id=None):
        return summary_tool.summarize(self.reference, self.root, self.root, job_id)

    def test_missing_assays_are_included_in_denominator(self):
        self.report()
        rows, summary = self.summarize()
        self.assertEqual(len(rows), 3)
        self.assertEqual(summary["counts"], {"exact_match": 1, "no_report": 2})
        self.assertFalse(summary["all_assays_matched"])

    def test_newer_failed_run_does_not_use_older_pass(self):
        self.report()
        path = self.report(suffix="two", status="failed")
        report_path = path / "verification_report.json"
        report = json.loads(report_path.read_text())
        report["created_at"] = "2026-10-23T00:00:00+00:00"
        report["error"] = "weight computation failed"
        report_path.write_text(json.dumps(report))
        rows, summary = self.summarize()
        self.assertEqual(rows[0]["status"], "failed")
        self.assertEqual(rows[0]["note"], "weight computation failed")
        self.assertEqual(summary["assays_with_multiple_run_directories"], 1)

    def test_array_scope_does_not_reuse_old_pass_when_task_failed_early(self):
        self.report()
        (self.root / "verify_msa_all-123_0.out").write_text("FileNotFoundError: reference CSV\n")
        rows, summary = self.summarize("123")
        self.assertEqual(rows[0]["status"], "no_report")
        self.assertEqual(summary["matched_assays"], 0)

    def test_array_scope_selects_logged_report(self):
        path = self.report()
        self.report(suffix="other", status="failed")
        (self.root / "verify_msa_all-123_0.out").write_text(f"Verification output: {path}\n")
        rows, _ = self.summarize("123")
        self.assertEqual(rows[0]["status"], "exact_match")
        self.assertEqual(rows[0]["selected_run"], str(path))

    def test_count_and_value_mismatches_are_separate(self):
        self.report(status="weight_mismatch", counts_match=False, values_match_in_order=False, exact_match=False)
        self.report(assay="B", status="weight_mismatch", values_match_in_order=False, exact_match=False)
        self.report(assay="C", exact_match=False)
        _, summary = self.summarize()
        self.assertEqual(summary["counts"], {"count_mismatch": 1, "value_mismatch": 1, "match_within_tolerance": 1})

    def test_changed_catalog_does_not_count_stale_report_as_pass(self):
        self.report()
        self.reference.write_text("DMS_id\nA\nC\nB\n")
        rows, _ = self.summarize()
        self.assertEqual(rows[0]["status"], "reference_mismatch")

    def test_partial_and_invalid_reports_are_not_passes(self):
        (self.root / "A_partial").mkdir()
        path = self.root / "B_invalid"
        path.mkdir()
        (path / "verification_report.json").write_text("{broken")
        rows, summary = self.summarize()
        self.assertEqual(rows[0]["status"], "no_report")
        self.assertEqual(rows[1]["status"], "invalid_report")
        self.assertEqual(summary["matched_assays"], 0)

    def test_completed_catalog_passes(self):
        for assay in ("A", "B", "C"):
            self.report(assay=assay)
        _, summary = self.summarize()
        self.assertTrue(summary["all_assays_matched"])


if __name__ == "__main__":
    unittest.main()
