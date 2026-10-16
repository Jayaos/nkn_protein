"""Preparation audits select four exact caches, enforce coverage, and never infer."""

import contextlib
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from feature_generation import readiness as r


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        index = self.root / "index.csv"
        index.write_text("DMS_id,DMS_filename,target_seq\ntiny,tiny.csv,ACD\n" +
                         f"{r.BRCA2},brca.csv,ACD\n")
        for filename in ("tiny.csv", "brca.csv"):
            (self.root / filename).write_text("mutant,mutated_sequence,DMS_score\nA1V,VCD,1\n")
        self.esm = replace(r.ExtractionConfig.load(r.CONFIG_DIR / "esm2_650m_centered.json"), assay_index=index)
        self.features = replace(r.InverseFoldingConfig.load(r.CONFIG_DIR / "proteinmpnn_v48_020_kermut.json"), assay_index=index)
        self.scores = replace(r.InverseFoldingConfig.load(r.CONFIG_DIR / "proteinmpnn_v48_020.json"), assay_index=index)

    def run_audit(self, failure=None, shared_failure=False):
        calls = []

        def output(config, assay, inputs, kind, structure):
            label = (assay, getattr(config, "model_family", "esm2"), kind)
            calls.append(label)
            if label == failure:
                raise ValueError("fixture corrupt cache")

        log = io.StringIO()
        summary = io.StringIO()
        with patch.object(r, "esm2_checkpoint", side_effect=ValueError("missing model") if shared_failure else None), \
                patch.object(r, "proteinmpnn_checkpoint"), patch.object(r, "load_inverse_structure"), \
                patch.object(r, "validate_output", side_effect=output), contextlib.redirect_stdout(log):
            status = r.audit(self.esm, self.features, self.scores, summary_stream=summary)
        self.summary_rows = list(csv.DictReader(io.StringIO(summary.getvalue())))
        return status, log.getvalue(), calls

    def test_complete_outputs_skip_only_brca2_score(self):
        status, log, calls = self.run_audit()
        self.assertEqual(status, 0, log)
        self.assertEqual(len(calls), 7)
        self.assertNotIn((r.BRCA2, "proteinmpnn", "scores"), calls)
        self.assertIn((r.BRCA2, "proteinmpnn", "features"), calls)
        self.assertIn("Assay outputs ready: 2 / 2", log)
        self.assertIn("SKIP   ProteinMPNN scores", log)
        self.assertEqual([row["assay_id"] for row in self.summary_rows], ["tiny", r.BRCA2])
        self.assertEqual([row["array_index"] for row in self.summary_rows], ["0", "1"])
        self.assertEqual([row["status"] for row in self.summary_rows], ["READY", "READY"])
        self.assertEqual(self.summary_rows[1]["proteinmpnn_scores"], "SKIP")
        self.assertIn("BRCA2", self.summary_rows[1]["notes"])

    def test_failed_cache_does_not_hide_other_outputs_or_assays(self):
        status, log, calls = self.run_audit(("tiny", "esm2", "features"))
        self.assertEqual(status, 1)
        self.assertEqual(len(calls), 7)
        self.assertIn("fixture corrupt cache", log)
        self.assertIn("Assay outputs ready: 1 / 2", log)
        self.assertEqual(self.summary_rows[0]["esm2_embeddings"], "FAILED")
        self.assertEqual(self.summary_rows[0]["status"], "FAILED")
        self.assertIn("fixture corrupt cache", self.summary_rows[0]["errors"])
        self.assertEqual(self.summary_rows[1]["status"], "READY")

    def test_domain_assays_require_features_but_skip_full_reference_scores(self):
        for assay in r.PROTEINMPNN_SCORE_SKIP_REASONS:
            if assay == r.BRCA2:
                continue
            with self.subTest(assay=assay):
                self.esm.assay_index.write_text(
                    f"DMS_id,DMS_filename,target_seq\n{assay},tiny.csv,ACD\n")
                status, log, calls = self.run_audit()
                self.assertEqual(status, 0, log)
                self.assertIn((assay, "proteinmpnn", "features"), calls)
                self.assertNotIn((assay, "proteinmpnn", "scores"), calls)
                self.assertEqual(self.summary_rows[0]["proteinmpnn_scores"], "SKIP")
                self.assertEqual(self.summary_rows[0]["status"], "READY")

    def test_missing_model_is_reported_even_when_outputs_pass(self):
        status, log, calls = self.run_audit(shared_failure=True)
        self.assertEqual(status, 1)
        self.assertEqual(len(calls), 7)
        self.assertIn("shared asset failures: 1", log)
        for row in self.summary_rows:
            self.assertEqual(row["esm2_model"], "FAILED")
            self.assertEqual(row["outputs_status"], "READY")
            self.assertEqual(row["status"], "FAILED")
            self.assertIn("missing model", row["errors"])

    def test_existing_summary_is_not_overwritten(self):
        summary = self.root / "previous.csv"
        summary.write_text("previous audit\n")
        with patch.object(r, "audit") as audit, contextlib.redirect_stdout(io.StringIO()):
            status = r.main(["--summary-csv", str(summary)])
        self.assertEqual(status, 1)
        audit.assert_not_called()
        self.assertEqual(summary.read_text(), "previous audit\n")

    def test_selected_score_cache_and_current_coverage_are_checked(self):
        leaf = self.root / "score"
        leaf.mkdir()
        saved = {key: getattr(self.esm, key) for key in ("model_id", "revision", "inference_dtype", "scoring_window")}
        (leaf / "config.json").write_text(json.dumps(saved))
        report = {"path": str(leaf), "input_signature": "current"}
        inputs = SimpleNamespace(input_signature="current")
        with patch.object(r, "validate_score_cache", return_value=report) as validate:
            r.validate_output(self.esm, "tiny", inputs, "scores")
            validate.assert_called_once_with(self.esm.output_root / "tiny", "esm2", r.default_cache_id(self.esm))
            with self.assertRaisesRegex(ValueError, "coverage"):
                r.validate_output(self.esm, "tiny", SimpleNamespace(input_signature="changed"), "scores")
            saved["revision"] = "wrong"
            (leaf / "config.json").write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "revision"):
                r.validate_output(self.esm, "tiny", inputs, "scores")

    def test_storage_validation_failure_is_not_ignored(self):
        with patch.object(r, "validate_sequence_cache", side_effect=ValueError("checksum mismatch")):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                r.validate_output(self.esm, "tiny", None, "features")

    def test_structure_change_invalidates_proteinmpnn_output(self):
        leaf = self.root / "inverse"
        leaf.mkdir()
        saved = {key: getattr(self.features, key) for key in
                 ("model_id", "revision", "inference_dtype", "conditioning_mode", "seed", "num_orders")}
        saved["dtype"] = self.features.output_dtype
        (leaf / "config.json").write_text(json.dumps(saved))
        report = {"path": str(leaf), "input_signature": "current", "structure_input_signature": "old"}
        with patch.object(r, "validate_inverse_cache", return_value=report):
            with self.assertRaisesRegex(ValueError, "different registered structure"):
                r.validate_output(self.features, "tiny", SimpleNamespace(input_signature="current"),
                                  "features", SimpleNamespace(input_signature="new"))


if __name__ == "__main__":
    unittest.main()
