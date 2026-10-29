"""Across-assay Spearman summaries retain exact run IDs and equal assay weights."""

from contextlib import redirect_stdout
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reporting.summarize import main, print_summary, summarize_csv


class ReportingSummaryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "summary.csv"

    def write(self, rows):
        fields = ("assay_id", "run_id", "stage", "setting", "status",
                  "avg_spearman", "std_spearman", "n_variants")
        with self.path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def row(self, assay, value, run="nkn_esm2_proteinmpnn_singles_14521255",
            setting="singles_random", **changes):
        return {"assay_id": assay, "run_id": run, "stage": "cv", "setting": setting,
                "status": "complete", "avg_spearman": value, "std_spearman": 0.99,
                "n_variants": 1, **changes}

    def test_equal_assay_weights_and_exact_run_ids(self):
        rows = []
        for setting, values in (("random", (0.2, 0.6)), ("modulo", (0.1, 0.7)),
                                ("contiguous", (-0.4, 0.2))):
            rows.extend([self.row("a", values[0], setting=f"singles_{setting}"),
                         self.row("b", values[1], setting=f"singles_{setting}", n_variants=999)])
        other_run = "nkn_prosst_proteinmpnn_singles_14522762"
        rows.append(self.row("a", 0.9, run=other_run))
        self.write(rows)
        summary = summarize_csv(self.path)
        self.assertEqual(len(summary), 6)
        for actual, setting, mean, std in zip(summary[:3],
                ("random", "modulo", "contiguous"), (0.4, 0.4, -0.1), (0.2, 0.3, 0.3)):
            self.assertEqual(actual["setting"], setting)
            self.assertEqual(actual["n_assays"], 2)
            self.assertAlmostEqual(actual["mean"], mean)
            self.assertAlmostEqual(actual["std"], std)
        self.assertEqual(summary[3], {"run_id": other_run, "setting": "random",
                         "n_assays": 1, "mean": 0.9, "std": 0.0, "skipped": 0})
        self.assertIsNone(summary[4]["mean"])
        self.assertIsNone(summary[4]["std"])

    def test_skipped_scores_and_unrelated_settings(self):
        rows = [self.row(str(i), value) for i, value in enumerate(("", "nan", "inf", "bad"))]
        rows.extend([self.row("incomplete", 0.7, status="incomplete"),
                     self.row("invalid", 0.8, status="invalid"),
                     self.row("valid", -0.2, setting="singles_modulo"),
                     self.row("mixed", 1, setting="mixed_random_singles"),
                     self.row("fit", 1, stage="fit")])
        self.write(rows)
        summary = summarize_csv(self.path)
        self.assertEqual((summary[0]["n_assays"], summary[0]["skipped"]), (0, 6))
        self.assertIsNone(summary[0]["mean"])
        self.assertEqual((summary[1]["n_assays"], summary[1]["mean"]), (1, -0.2))
        output = io.StringIO()
        with redirect_stdout(output):
            print_summary(summary)
        self.assertIn("N/A", output.getvalue())

    def test_cli_prints_both_runs_without_writing_files(self):
        self.write([self.row("a", 0.2, run="nkn_singles_10"),
                    self.row("b", 0.6, run="nkn_singles_10"),
                    self.row("a", 0.9, run="nkn_singles_11")])
        original = self.path.read_bytes()
        output = io.StringIO()
        with patch("sys.argv", ["summarize", str(self.path)]), redirect_stdout(output):
            main()
        self.assertIn("run_id: nkn_singles_10", output.getvalue())
        self.assertIn("run_id: nkn_singles_11", output.getvalue())
        self.assertIn("0.400000", output.getvalue())
        self.assertIn("0.200000", output.getvalue())
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_duplicate_assay_does_not_silently_change_weights(self):
        self.write([self.row("a", 0.2), self.row("a", 0.6)])
        with self.assertRaisesRegex(ValueError, "Duplicate assay/setting"):
            summarize_csv(self.path)

    def test_missing_columns_and_empty_summary(self):
        self.path.write_text("run_id,setting\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            summarize_csv(self.path)
        self.write([])
        self.assertEqual(summarize_csv(self.path), [])


if __name__ == "__main__":
    unittest.main()
