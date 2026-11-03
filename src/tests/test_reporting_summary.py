"""Across-assay metric summaries retain equal weights and independent coverage."""

from contextlib import redirect_stdout
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reporting.summarize import DEFAULT_METRICS, METRIC_LABELS, main, print_summary, summarize_csv


class ReportingSummaryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "summary.csv"

    def write(self, rows):
        fields = ("assay_id", "run_id", "stage", "setting", "status",
                  "avg_spearman", "std_spearman", "n_variants")
        fields = list(dict.fromkeys((*fields, *(key for row in rows for key in row))))
        with self.path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def row(self, assay, value, run="nkn_esm2_proteinmpnn_singles_14521255",
            setting="singles_random", **changes):
        return {"assay_id": assay, "run_id": run, "stage": "cv", "setting": setting,
                "status": "complete", "avg_spearman": value, "std_spearman": 0.99,
                "n_variants": 1, **changes}

    def test_equal_assay_weights_and_run_prefixes(self):
        rows = []
        for setting, values in (("random", (0.2, 0.6)), ("modulo", (0.1, 0.7)),
                                ("contiguous", (-0.4, 0.2))):
            rows.extend([self.row("a", values[0], setting=f"singles_{setting}"),
                         self.row("b", values[1], run="nkn_esm2_proteinmpnn_singles_14521140",
                                  setting=f"singles_{setting}", n_variants=999)])
        other_run = "nkn_prosst_proteinmpnn_singles_14522762"
        rows.append(self.row("a", 0.9, run=other_run))
        self.write(rows)
        summary = summarize_csv(self.path)
        self.assertEqual(len(summary), 8)
        for actual, setting, mean, std in zip(summary[:3],
                ("random", "modulo", "contiguous"), (0.4, 0.4, -0.1), (0.2, 0.3, 0.3)):
            self.assertEqual(actual["setting"], setting)
            self.assertEqual(actual["n_assays"], 2)
            self.assertAlmostEqual(actual["mean"], mean)
            self.assertAlmostEqual(actual["std"], std)
        self.assertEqual(summary[0]["run_id"], "nkn_esm2_proteinmpnn_singles")
        self.assertEqual((summary[3]["setting"], summary[3]["n_assays"], summary[3]["skipped"]),
                         ("average", 2, 0))
        self.assertAlmostEqual(summary[3]["mean"], 7 / 30)
        self.assertAlmostEqual(summary[3]["std"], 4 / 15)
        self.assertEqual(summary[4], {"run_id": "nkn_prosst_proteinmpnn_singles", "setting": "random",
                         "n_assays": 1, "mean": 0.9, "std": 0.0, "skipped": 0})
        self.assertIsNone(summary[5]["mean"])
        self.assertIsNone(summary[5]["std"])
        self.assertEqual(summary[7], {"run_id": "nkn_prosst_proteinmpnn_singles", "setting": "average",
                         "n_assays": 0, "mean": None, "std": None, "skipped": 1})

    def test_average_uses_matching_assays_and_std_of_per_assay_averages(self):
        rows = []
        for assay, scores in (("a", (0.2, 0.8, 0.5)), ("b", (0.8, 0.2, 0.5)),
                              ("invalid", (0.4, 0.5, "nan"))):
            for setting, score in zip(("random", "modulo", "contiguous"), scores):
                rows.append(self.row(assay, score, setting=f"singles_{setting}"))
        rows.extend([self.row("partial", 0.9),
                     self.row("a", 0.0, run="nkn_esm2_proteinmpnn_singles_14521254")])
        self.write(rows)
        summary = summarize_csv(self.path)
        self.assertEqual(summary[-1], {
            "run_id": "nkn_esm2_proteinmpnn_singles", "setting": "average",
            "n_assays": 2, "mean": 0.5, "std": 0.0, "skipped": 2,
        })
        self.assertGreater(summary[0]["std"], 0)
        self.assertGreater(summary[1]["std"], 0)

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
        self.assertEqual((summary[3]["n_assays"], summary[3]["skipped"]), (0, 7))
        self.assertIsNone(summary[3]["mean"])
        self.assertIsNone(summary[3]["std"])
        output = io.StringIO()
        with redirect_stdout(output):
            print_summary(summary)
        self.assertIn("N/A", output.getvalue())

    def test_cli_prints_both_runs_without_writing_files(self):
        self.write([self.row("a", 0.2, run="nkn_singles_10"),
                    self.row("b", 0.6, run="nkn_singles_11"),
                    self.row("a", 0.9, run="kermut_singles_12")])
        original = self.path.read_bytes()
        output = io.StringIO()
        with patch("sys.argv", ["summarize", str(self.path), "--metrics", "avg_spearman"]), redirect_stdout(output):
            main()
        self.assertIn("run prefix: nkn_singles\n", output.getvalue())
        self.assertIn("run prefix: kermut_singles\n", output.getvalue())
        self.assertNotIn("nkn_singles_10", output.getvalue())
        self.assertNotIn("nkn_singles_11", output.getvalue())
        self.assertIn("0.400000", output.getvalue())
        self.assertIn("0.200000", output.getvalue())
        self.assertEqual(sum(line.startswith("average ") for line in output.getvalue().splitlines()), 2)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_new_metrics_weight_assays_equally_and_average_within_assays(self):
        values = {
            "mse_oof_standardized": ((1, 4, 7), (3, 8, 1)),
            "interval_ece": ((.1, .2, .3), (.3, .4, .5)),
            "ence": ((.4, .6, .8), (.8, 1, 1.2)),
            "uncertainty_cv": ((0, .1, .2), (.2, .3, .4)),
        }
        rows = [
            self.row(assay, "nan" if assay == "a" else .5,
                     setting=f"singles_{setting}", n_variants=1 if assay == "a" else 999,
                     mse_oof=9999, pair_ece=.99, std_ence=99, std_uncertainty_cv=99,
                     **{metric: scores[index][split] for metric, scores in values.items()})
            for index, assay in enumerate(("a", "b"))
            for split, setting in enumerate(("random", "modulo", "contiguous"))
        ]
        self.write(rows)
        for metric, random_mean, random_std, average_mean, average_std in (
            ("mse_oof_standardized", 2, 1, 4, 0),
            ("interval_ece", .2, .1, .3, .1),
            ("ence", .6, .2, .8, .2),
            ("uncertainty_cv", .1, .1, .2, .1),
        ):
            with self.subTest(metric=metric):
                summary = summarize_csv(self.path, metric)
                self.assertTrue(all(row["n_assays"] == 2 for row in summary))
                self.assertAlmostEqual(summary[0]["mean"], random_mean)
                self.assertAlmostEqual(summary[0]["std"], random_std)
                self.assertEqual(summary[-1]["setting"], "average")
                self.assertAlmostEqual(summary[-1]["mean"], average_mean)
                self.assertAlmostEqual(summary[-1]["std"], average_std)
        self.assertEqual(summarize_csv(self.path)[0]["n_assays"], 1)

    def test_each_metric_has_its_own_assay_intersection(self):
        rows = []
        for index, setting in enumerate(("random", "modulo", "contiguous")):
            rows.extend([
                self.row("a", .2, setting=f"singles_{setting}",
                         mse_oof_standardized=index + 1,
                         ence=None if setting == "contiguous" else .1),
                self.row("b", .6, setting=f"singles_{setting}",
                         mse_oof_standardized=None if setting == "random" else 10,
                         ence=.6),
            ])
        self.write(rows)
        for metric, counts, mean in (("mse_oof_standardized", (1, 2, 2, 1), 2),
                                    ("ence", (2, 2, 1, 1), .6)):
            with self.subTest(metric=metric):
                summary = summarize_csv(self.path, metric)
                self.assertEqual(tuple(row["n_assays"] for row in summary), counts)
                self.assertAlmostEqual(summary[-1]["mean"], mean)
                self.assertEqual(summary[-1]["std"], 0)
                self.assertEqual(summary[-1]["skipped"], 1)

    def test_new_metrics_skip_invalid_values_and_keep_zero(self):
        for metric in DEFAULT_METRICS[1:]:
            with self.subTest(metric=metric):
                rows = [self.row(str(i), .5, **{metric: value})
                        for i, value in enumerate((None, "", "nan", "inf", "bad"))]
                rows.extend([
                    self.row("incomplete", .5, status="incomplete", **{metric: .8}),
                    self.row("invalid", .5, status="invalid", **{metric: .9}),
                    self.row("valid", .5, **{metric: 0}),
                    self.row("fit", .5, stage="fit", **{metric: 99}),
                    self.row("mixed", .5, setting="mixed_random_singles", **{metric: 99}),
                ])
                self.write(rows)
                summary = summarize_csv(self.path, metric)
                self.assertEqual((summary[0]["n_assays"], summary[0]["skipped"]), (1, 7))
                self.assertEqual(summary[0]["mean"], 0)
                self.assertEqual(summary[0]["std"], 0)

    def test_latest_valid_jobs_are_selected_independently_for_each_metric(self):
        rows = [
            self.row("a", .9, run="nkn_singles_9", mse_oof_standardized=4, ence=.8),
            self.row("a", .2, run="nkn_singles_10", mse_oof_standardized=2, ence=None),
            self.row("a", .5, run="nkn_singles_11", status="incomplete",
                     mse_oof_standardized=10, ence=10),
            self.row("a", "nan", run="nkn_singles_12", mse_oof_standardized="nan", ence=.4),
        ]
        for order in (rows, list(reversed(rows))):
            self.write(order)
            for metric, mean in (("avg_spearman", .2), ("mse_oof_standardized", 2), ("ence", .4)):
                with self.subTest(metric=metric, order=order):
                    summary = summarize_csv(self.path, metric)
                    self.assertEqual(summary[0]["mean"], mean)
                    self.assertEqual((summary[0]["n_assays"], summary[0]["skipped"]), (1, 3))

    def test_legacy_columns_do_not_substitute_raw_mse_or_spearman(self):
        self.write([self.row("a", .5, mse_oof=9)])
        for metric in DEFAULT_METRICS[1:]:
            with self.subTest(metric=metric):
                summary = summarize_csv(self.path, metric)
                self.assertIsNone(summary[0]["mean"])
                self.assertIsNone(summary[0]["std"])
                self.assertEqual((summary[0]["n_assays"], summary[0]["skipped"]), (0, 1))
        self.assertEqual(summarize_csv(self.path, "mse_oof")[0]["mean"], 9)

    def test_calibration_only_csv_does_not_require_spearman(self):
        self.path.write_text(
            "assay_id,run_id,setting,ence\na,nkn_singles_10,singles_random,0.2\n",
            encoding="utf-8",
        )
        self.assertEqual(summarize_csv(self.path, "ence")[0]["mean"], .2)
        self.assertIsNone(summarize_csv(self.path)[0]["mean"])

    def test_cli_defaults_to_all_five_metrics_without_writing_files(self):
        self.write([self.row("a", .2, mse_oof_standardized=2, interval_ece=.1,
                             ence=.3, uncertainty_cv=.4)])
        original = self.path.read_bytes()
        output = io.StringIO()
        with patch("sys.argv", ["summarize", str(self.path)]), redirect_stdout(output):
            main()
        for metric in DEFAULT_METRICS:
            self.assertIn(f"{METRIC_LABELS[metric]}: mean", output.getvalue())
            self.assertIn(f"across assays of {metric}.", output.getvalue())
        self.assertIn("2.000000", output.getvalue())
        self.assertEqual(sum(line.startswith("average ") for line in output.getvalue().splitlines()), 5)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_cli_metric_selection_and_legacy_na_tables(self):
        self.write([self.row("a", .2)])
        output = io.StringIO()
        with patch("sys.argv", ["summarize", str(self.path), "--metrics", "ence", "ence", "uncertainty_cv"]), redirect_stdout(output):
            main()
        self.assertIn("ENCE: mean", output.getvalue())
        self.assertIn("Uncertainty CV: mean", output.getvalue())
        self.assertNotIn("Spearman: mean", output.getvalue())
        self.assertIn("N/A", output.getvalue())
        self.assertEqual(sum(line.startswith("average ") for line in output.getvalue().splitlines()), 2)

    def test_gpu_job_and_task_suffixes_group_by_prefix(self):
        self.write([self.row("a", 0.2, run="nkn_esm2_proteinmpnn_singles_gpu_14521140_99"),
                    self.row("b", 0.6, run="nkn_esm2_proteinmpnn_singles_gpu_14521255_1"),
                    self.row("a", 0.9, run="nkn_prosst_proteinmpnn_singles_gpu_14522762_0")])
        summary = summarize_csv(self.path)
        self.assertEqual(len(summary), 8)
        self.assertEqual(summary[0]["run_id"], "nkn_esm2_proteinmpnn_singles_gpu")
        self.assertEqual(summary[0]["n_assays"], 2)
        self.assertAlmostEqual(summary[0]["mean"], 0.4)
        self.assertAlmostEqual(summary[0]["std"], 0.2)
        self.assertEqual(summary[4]["run_id"], "nkn_prosst_proteinmpnn_singles_gpu")

    def test_latest_valid_job_per_assay_is_independent_of_csv_order(self):
        rows = [self.row("a", 0.9, run="nkn_singles_gpu_9_99"),
                self.row("a", 0.2, run="nkn_singles_gpu_10_1"),
                self.row("a", 0.8, run="nkn_singles_gpu_11_2", status="incomplete"),
                self.row("a", "nan", run="nkn_singles_gpu_12_3"),
                self.row("b", 0.6, run="nkn_singles_gpu_13_4")]
        for order in (rows, list(reversed(rows))):
            with self.subTest(order=order):
                self.write(order)
                summary = summarize_csv(self.path)
                self.assertEqual(len(summary), 4)
                self.assertEqual(summary[0]["n_assays"], 2)
                self.assertEqual(summary[0]["skipped"], 3)
                self.assertAlmostEqual(summary[0]["mean"], 0.4)
                self.assertAlmostEqual(summary[0]["std"], 0.2)

    def test_names_without_job_ids_and_manual_runs_keep_their_names(self):
        names = ("nkn_esm2_proteinmpnn_singles", "nkn_custom_v2",
                 "nkn_singles_gpu_manual_20261017T120000Z_99")
        self.write([self.row("a", 0.2, run=name) for name in names])
        self.assertEqual({row["run_id"] for row in summarize_csv(self.path)}, set(names))

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
