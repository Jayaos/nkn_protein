"""Checks for MSA identity tracking and weight-comparison failure cases."""

import importlib.util
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "msa_preprocessing_verification", ROOT / "sbatch_run/run_prepare_rep/verify_msa_preprocessing.py",
)
verification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verification)


class ReferenceResolutionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.nested = self.root / "data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv"
        self.nested.parent.mkdir(parents=True)
        self.nested.write_text("nested reference")
        self.legacy = self.root / "data/DMS_substitutions.csv"
        patcher = patch.object(verification, "PROJECT_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_missing_legacy_path_uses_existing_pipeline_reference(self):
        self.assertEqual(verification.resolve_reference("data/DMS_substitutions.csv"), self.nested.resolve())

    def test_existing_configured_file_takes_precedence(self):
        self.legacy.write_text("different legacy reference")
        self.assertEqual(verification.resolve_reference(self.legacy), self.legacy.resolve())

    def test_missing_explicit_path_does_not_fall_back(self):
        with self.assertRaisesRegex(FileNotFoundError, "Sync the reference CSV"):
            verification.resolve_reference(self.nested, Path("wrong.csv"))

    def test_explicit_override_wins(self):
        self.legacy.write_text("explicit reference")
        self.assertEqual(verification.resolve_reference(self.nested, self.legacy), self.legacy.resolve())

    def test_conflicting_fallbacks_require_explicit_selection(self):
        self.legacy.write_text("different legacy reference")
        with self.assertRaisesRegex(ValueError, "Multiple different reference"):
            verification.resolve_reference("missing.csv")

    def test_missing_reference_reports_checked_locations(self):
        self.nested.unlink()
        with self.assertRaises(FileNotFoundError) as error:
            verification.resolve_reference("missing.csv")
        self.assertIn(str(self.nested), str(error.exception))
        self.assertIn("--reference-csv PATH", str(error.exception))


class WeightComparisonTests(unittest.TestCase):
    def compare(self, calculated, supplied):
        return verification.compare_weights(calculated, supplied, rtol=1e-10, atol=1e-12)

    def test_count_mismatch_is_not_truncated_into_agreement(self):
        result = self.compare([1.0, 0.5, 0.25], [1.0, 0.5])
        self.assertFalse(result["counts_match"])
        self.assertFalse(result["values_match_in_order"])
        self.assertIsNone(result["max_absolute_difference"])

    def test_equal_length_reordering_fails(self):
        result = self.compare([1.0, 0.5, 0.25], [1.0, 0.25, 0.5])
        self.assertTrue(result["counts_match"])
        self.assertFalse(result["values_match_in_order"])
        self.assertEqual(result["mismatching_indices_zero_based"], [1, 2])

    def test_invalid_weights_cannot_pass(self):
        for value in (0.0, -1.0, float("nan"), float("inf"), 1.1):
            with self.subTest(value=value):
                result = self.compare([value], [value])
                self.assertFalse(result["values_match_in_order"])
                json.dumps(result, allow_nan=False)

    def test_roundoff_can_pass_without_claiming_exact_match(self):
        result = self.compare([1.0, 0.5 + 1e-13], [1.0, 0.5])
        self.assertTrue(result["values_match_in_order"])
        self.assertFalse(result["exact_match"])

    def test_empty_arrays_do_not_pass(self):
        self.assertFalse(self.compare([], [])["values_match_in_order"])


class RowMappingTests(unittest.TestCase):
    def setUp(self):
        # A realigned query-gap column is removed during ProteinGym cleaning.
        self.raw = [(">query/1-3", "aC.D"), (">seq.a", "AC-E"), (">seq.b", "AC-F")]
        self.normalized = [(verification.normalize(name), verification.normalize(seq))
                           for name, seq in self.raw]

    def test_lowercase_positions_are_preserved(self):
        self.assertEqual(self.normalized[0], (">QUERY/1-3", "AC-D"))

    def test_tracks_removed_and_reordered_hhfilter_rows(self):
        filtered = [self.normalized[0], self.normalized[2], self.normalized[1]]
        processed = [(">QUERY/1-3", "ACD"), (">SEQ-B", "ACF")]
        mapping, columns = verification.retained_rows(self.raw, self.normalized, filtered, processed)
        self.assertEqual(columns, [0, 1, 3])
        self.assertEqual(mapping[1]["raw_row_one_based"], 3)
        self.assertEqual(mapping[1]["hhfilter_row_one_based"], 2)
        self.assertEqual(mapping[1]["weight_index_zero_based"], 1)
        self.assertEqual(mapping[1]["original_header"], ">seq.b")

    def test_matching_header_with_changed_sequence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Retained sequence changed"):
            verification.retained_rows(self.raw, self.normalized, self.normalized,
                                       [(">QUERY/1-3", "ACD"), (">SEQ-A", "ACF")])

    def test_query_cannot_be_removed(self):
        with self.assertRaisesRegex(ValueError, "query must remain"):
            verification.retained_rows(self.raw, self.normalized, self.normalized,
                                       [(">SEQ-A", "ACE")])

    def test_postprocessing_cannot_silently_reorder_rows(self):
        with self.assertRaisesRegex(ValueError, "row order"):
            verification.retained_rows(self.raw, self.normalized, self.normalized,
                                       [(">QUERY/1-3", "ACD"), (">SEQ-B", "ACF"), (">SEQ-A", "ACE")])

    def test_changed_hhfilter_header_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown or changed"):
            verification.retained_rows(self.raw, self.normalized,
                                       [self.normalized[0], (">UNKNOWN", "AC-E")],
                                       [(">QUERY/1-3", "ACD")])


class AlignmentReaderTests(unittest.TestCase):
    def read(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.a2m"
            path.write_text(text, encoding="ascii")
            return verification.read_alignment(path)

    def test_multiline_sequence_preserves_full_header(self):
        self.assertEqual(self.read(">query description\naC\nD\n>other\nACD\n"),
                         [(">query description", "aCD"), (">other", "ACD")])

    def test_duplicate_headers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.read(">query\nACD\n>query\nACE\n")

    def test_unaligned_rows_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "different lengths"):
            self.read(">query\nACD\n>other\nAC\n")


@unittest.skipIf(np is None, "NumPy required for report/artifact integration checks")
class VerificationReportTests(unittest.TestCase):
    """Exercise real artifact I/O with controlled HHfilter/ProteinGym stand-ins.

These tests check orchestration, not the upstream filtering/weighting algorithm.
"""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "msa").mkdir()
        (self.root / "weights/specialized").mkdir(parents=True)
        verification.write_alignment(self.root / "msa/input.a2m",
                                     [(">query", "aCD"), (">a", "ACE"), (">b", "ACE")])
        with (self.root / "reference.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["DMS_id", "MSA_filename", "weight_file_name",
                                                       "MSA_start", "MSA_end", "MSA_theta", "target_seq"])
            writer.writeheader()
            writer.writerow(dict(DMS_id="test", MSA_filename="input.a2m", weight_file_name="weights.npy",
                                 MSA_start=1, MSA_end=3, MSA_theta=0.2, target_seq="ACD"))
        np.save(self.root / "weights/weights.npy", [1.0, 0.5, 0.5])
        np.save(self.root / "weights/specialized/weights.npy", [1.0, 1.0])
        config = json.loads((ROOT / "configs/setup/msa_preprocessing_brca1.json").read_text())
        config.update(assay_id="test", reference_csv="reference.csv", msa_folder="msa",
                      weights_folder="weights", specialized_weights_subfolder="specialized",
                      output_folder="output")
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(config))
        self.runtime_config = self.root / "runtime.json"
        self.runtime_config.write_text(json.dumps({"proteingym_commit": "test", "hhfilter_version": "3.3.0"}))
        self.source = self.root / "source"
        (self.source / "proteingym/utils").mkdir(parents=True)
        for name in ("msa_utils.py", "weights.py"):
            (self.source / "proteingym/utils" / name).write_text("# controlled test stand-in\n")

    def processor(self, **kwargs):
        weight_path = Path(kwargs["weights_location"])
        self.assertFalse(weight_path.exists(), "Recomputation must never load a cached weight file")
        rows = verification.read_alignment(kwargs["MSA_location"])
        values = np.array([1.0, 0.5, 0.5] if len(rows) == 3 else [1.0, 1.0])
        np.save(weight_path, values)
        return SimpleNamespace(seq_name_to_sequence=dict(rows), weights=values)

    def hhfilter(self, command, **kwargs):
        self.assertEqual(command[1:7], ["-cov", "75", "-id", "90", "-qid", "0"])
        rows = verification.read_alignment(command[command.index("-i") + 1])
        verification.write_alignment(command[command.index("-o") + 1], [rows[0], rows[2]])

    def run_pipeline(self, processor=None):
        with patch.object(verification, "PROJECT_ROOT", self.root), \
             patch.object(verification, "RUNTIME_CONFIG", self.runtime_config), \
             patch.object(verification, "check_runtime", return_value=(processor or self.processor,
                          sys.executable, self.source, {})), \
             patch.object(verification.subprocess, "run", side_effect=self.hhfilter):
            return verification.run_verification(self.config_path, 1)

    def test_report_mapping_and_repeated_run_preserve_original_inputs(self):
        input_paths = [self.root / "msa/input.a2m", self.root / "weights/weights.npy",
                       self.root / "weights/specialized/weights.npy"]
        original = [path.read_bytes() for path in input_paths]
        for _ in range(2):
            report = self.run_pipeline()
            self.assertEqual(report["status"], "weights_reproduced")
            self.assertFalse(report["historical_row_identity_verified"])
        outputs = list((self.root / "output").iterdir())
        self.assertEqual(len(outputs), 2)
        self.assertEqual(original, [path.read_bytes() for path in input_paths])
        with (outputs[0] / "specialized_rows.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[1]["original_header"], ">b")
        self.assertEqual(rows[1]["raw_row_one_based"], "3")
        saved = json.loads((outputs[0] / "verification_report.json").read_text())
        self.assertEqual(saved["status"], "weights_reproduced")
        self.assertIn("columns.csv", saved["artifacts"])

    def test_count_mismatch_keeps_diagnostics_without_assigning_weights(self):
        np.save(self.root / "weights/specialized/weights.npy", [1.0, 0.5, 0.5])
        report = self.run_pipeline()
        self.assertEqual(report["status"], "weight_mismatch")
        output = next((self.root / "output").iterdir())
        with (output / "specialized_rows.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(all(row["supplied_weight_at_index"] == "" for row in rows))
        self.assertTrue((output / "specialized_recomputed_weights.npy").is_file())

    def test_processing_error_leaves_failed_report(self):
        def fail(**kwargs):
            raise ValueError("controlled processing failure")
        with self.assertRaisesRegex(ValueError, "controlled processing failure"):
            self.run_pipeline(processor=fail)
        output = next((self.root / "output").iterdir())
        saved = json.loads((output / "verification_report.json").read_text())
        self.assertEqual(saved["status"], "failed")
        self.assertIn("controlled processing failure", saved["error"])


if __name__ == "__main__":
    unittest.main()
