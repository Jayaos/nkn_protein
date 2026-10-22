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

    def run_pipeline(self, processor=None, **options):
        with patch.object(sys, "path", [str(ROOT / "sbatch_run/run_prepare_rep"), *sys.path]), \
             patch.object(verification, "PROJECT_ROOT", self.root), \
             patch.object(verification, "RUNTIME_CONFIG", self.runtime_config), \
             patch.object(verification, "check_runtime", return_value=(processor or self.processor,
                          sys.executable, self.source, {})), \
             patch.object(verification.subprocess, "run", side_effect=self.hhfilter):
            return verification.run_verification(self.config_path, 1, **options)

    def test_generation_does_not_require_downloaded_weight_files_or_config(self):
        (self.root / "weights/weights.npy").unlink()
        (self.root / "weights/specialized/weights.npy").unlink()
        config = json.loads(self.config_path.read_text())
        for key in ("weights_folder", "specialized_weights_subfolder", "comparison"):
            del config[key]
        self.config_path.write_text(json.dumps(config))
        report = self.run_pipeline(generate_only=True)
        self.assertEqual(report["status"], "generated")
        self.assertTrue(report["generated_pair_validated"])
        self.assertEqual(set(report["inputs"]), {"reference", "msa"})
        self.assertEqual(report["comparisons"], {})
        output = next((self.root / "output").iterdir())
        for name in ("alignment.a2m", "weights.npy", "rows.csv", "columns.csv", "manifest.json"):
            self.assertTrue((output / name).is_file(), name)
        module = sys.modules["generated_msa"]
        rows, weights = module.validate_pair(output)
        self.assertEqual([name for name, _ in rows], [">QUERY", ">B"])
        self.assertEqual(weights.tolist(), [1.0, 1.0])

    def test_generated_pair_rejects_modified_weights(self):
        self.run_pipeline(generate_only=True)
        output = next((self.root / "output").iterdir())
        np.save(output / "weights.npy", [1.0, 0.5])
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            sys.modules["generated_msa"].validate_pair(output)

    def test_generated_pair_rejects_reordered_rows_even_with_updated_file_hash(self):
        self.run_pipeline(generate_only=True)
        output = next((self.root / "output").iterdir())
        path = output / "alignment.a2m"
        verification.write_alignment(path, list(reversed(verification.read_alignment(path))))
        manifest = json.loads((output / "manifest.json").read_text())
        manifest["artifacts"]["alignment.a2m"] = verification.sha256(path)
        with self.assertRaisesRegex(ValueError, "correspondence mismatch"):
            sys.modules["generated_msa"].validate_pair(output, manifest)

    def test_invalid_generated_weights_never_publish_manifest(self):
        def invalid(**kwargs):
            result = self.processor(**kwargs)
            result.weights[0] = float("nan")
            np.save(kwargs["weights_location"], result.weights)
            return result
        with self.assertRaisesRegex(ValueError, "Generated weights are invalid"):
            self.run_pipeline(processor=invalid, generate_only=True)
        output = next((self.root / "output").iterdir())
        self.assertFalse((output / "manifest.json").exists())
        self.assertEqual(json.loads((output / "generation_report.json").read_text())["status"], "failed")

    def test_generation_batch_summary_validates_pairs_and_rejects_corruption(self):
        self.run_pipeline(generate_only=True)
        (self.root / "output").rename(self.root / "assays")
        (self.root / "DMS_substitutions.csv").write_bytes((self.root / "reference.csv").read_bytes())
        module = sys.modules["generated_msa"]
        summary = module.summarize_batch(self.root)
        self.assertTrue(summary["all_assays_generated"])
        output = next((self.root / "assays").iterdir())
        (output / "weights.npy").write_bytes(b"corrupted")
        summary = module.summarize_batch(self.root)
        self.assertEqual(summary["counts"], {"invalid": 1})
        self.assertFalse(summary["all_assays_generated"])

    def test_generation_batch_summary_includes_missing_assays(self):
        reference = self.root / "reference.csv"
        with reference.open("a") as handle:
            handle.write("missing,input.a2m,weights.npy,1,3,0.2,ACD\n")
        self.run_pipeline(generate_only=True)
        (self.root / "output").rename(self.root / "assays")
        (self.root / "DMS_substitutions.csv").write_bytes(reference.read_bytes())
        summary = sys.modules["generated_msa"].summarize_batch(self.root)
        self.assertEqual(summary["expected_assays"], 2)
        self.assertEqual(summary["counts"], {"generated": 1, "no_report": 1})
        self.assertEqual(summary["assays_needing_attention"], 1)
        self.assertFalse(summary["all_assays_generated"])

    def test_generation_batch_summary_records_malformed_failure_report(self):
        self.run_pipeline(generate_only=True)
        (self.root / "output").rename(self.root / "assays")
        (self.root / "DMS_substitutions.csv").write_bytes((self.root / "reference.csv").read_bytes())
        output = next((self.root / "assays").iterdir())
        (output / "manifest.json").unlink()
        (output / "generation_report.json").write_text("[]")
        summary = sys.modules["generated_msa"].summarize_batch(self.root)
        self.assertEqual(summary["counts"], {"invalid": 1})
        self.assertFalse(summary["all_assays_generated"])

    def test_catalog_task_runs_only_specialized_comparison(self):
        report = self.run_pipeline(assay_index=0, specialized_only=True)
        self.assertEqual(report["assay_id"], "test")
        self.assertEqual(set(report["comparisons"]), {"specialized"})
        self.assertEqual(report["status"], "weights_reproduced")
        self.assertEqual(report["assay_index_zero_based"], 0)
        output = next((self.root / "output").iterdir())
        self.assertFalse((output / "general_recomputed_weights.npy").exists())

    def test_catalog_task_rejects_out_of_range_index(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            self.run_pipeline(assay_index=1, specialized_only=True)

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
