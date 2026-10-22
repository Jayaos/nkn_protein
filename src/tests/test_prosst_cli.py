"""ProSST CLI routing, override, selection, and error behavior without downloads."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from feature_generation.cli import main
from feature_generation.prosst_config import ProSSTConfig


class ProSSTCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "prosst.json"
        self.config.write_text(json.dumps({"model_family": "prosst"}), encoding="utf-8")

    def invoke(self, arguments):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            status = main([str(value) for value in arguments])
        return status, output.getvalue(), errors.getvalue()

    def test_prosst_scoring_progress_uses_masked_positions(self):
        from feature_generation.cli import _progress

        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            _progress({"assay_id": "tiny", "status": "scoring", "model_family": "prosst",
                       "num_variants": 4, "num_masked_positions": 2})
        self.assertIn("ProSST masked marginals at 2 WT positions", errors.getvalue())

    def test_score_routes_prosst_config_and_structure_token_override(self):
        with patch("feature_generation.score_runner.score", return_value=[{"status": "completed"}]) as scoring:
            status, _, errors = self.invoke([
                "score", "--config", self.config, "--assay", "tiny", "--singles-only", "--resume",
                "--structure-token-dir", self.root / "score-tokens", "--max-tokens-per-batch", "512",
            ])
        self.assertEqual(status, 0, errors)
        config, assays = scoring.call_args.args
        self.assertIsInstance(config, ProSSTConfig)
        self.assertEqual(assays, ["tiny"])
        self.assertEqual(config.structure_token_dir, (self.root / "score-tokens").resolve())
        self.assertTrue(scoring.call_args.kwargs["singles_only"])
        self.assertTrue(scoring.call_args.kwargs["resume"])

    def test_download_selects_prosst_pinned_configuration_and_downloader(self):
        with patch("feature_generation.prosst_checkpoints.download", return_value=self.root / "checkpoint") as prosst, patch(
            "feature_generation.checkpoints.download"
        ) as esm2, patch("feature_generation.inverse_checkpoints.download") as inverse:
            status, output, errors = self.invoke(["download", "--config", self.config])
        self.assertEqual(status, 0, errors)
        esm2.assert_not_called()
        inverse.assert_not_called()
        config = prosst.call_args.args[0]
        self.assertIsInstance(config, ProSSTConfig)
        self.assertEqual(config.model_id, "AI4Protein/ProSST-2048")
        report = json.loads(output)
        self.assertEqual(report["revision"], config.revision)
        self.assertRegex(report["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(report["path"], str(self.root / "checkpoint"))

    def test_extraction_passes_structure_paths_batching_and_subset_flags(self):
        with patch("feature_generation.prosst_runner.extract", return_value=[{"assay_id": "tiny", "status": "completed"}]) as prosst, patch(
            "feature_generation.runner.extract"
        ) as esm2, patch("feature_generation.inverse_runner.extract") as inverse:
            status, output, errors = self.invoke([
                "extract", "--config", self.config, "--assay", "tiny", "--assay", "other",
                "--assay-index", self.root / "index.csv", "--output-root", self.root / "features",
                "--structure-manifest-dir", self.root / "manifests", "--structure-token-dir", self.root / "tokens",
                "--max-tokens-per-batch", "700", "--device", "cpu", "--limit", "4",
                "--singles-only", "--resume", "--continue-on-error", "--quiet",
            ])
        self.assertEqual(status, 0, errors)
        self.assertEqual(json.loads(output)["results"][0]["status"], "completed")
        esm2.assert_not_called()
        inverse.assert_not_called()
        config, assays = prosst.call_args.args
        self.assertIsInstance(config, ProSSTConfig)
        self.assertEqual(assays, ["tiny", "other"])
        self.assertEqual(config.assay_index, (self.root / "index.csv").resolve())
        self.assertEqual(config.output_root, (self.root / "features").resolve())
        self.assertEqual(config.structure_manifest_dir, (self.root / "manifests").resolve())
        self.assertEqual(config.structure_token_dir, (self.root / "tokens").resolve())
        self.assertEqual(config.max_tokens_per_batch, 700)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(prosst.call_args.kwargs, {
            "singles_only": True, "limit": 4, "resume": True,
            "continue_on_error": True, "progress": None,
        })

    def test_all_assays_uses_overridden_catalog_and_progress_callback(self):
        records = [SimpleNamespace(assay_id="first"), SimpleNamespace(assay_id="second")]
        with patch("data_loading.AssayCatalog", return_value=records) as catalog, patch(
            "feature_generation.prosst_runner.extract", return_value=[]
        ) as prosst:
            status, _, errors = self.invoke([
                "extract", "--config", self.config, "--all-assays", "--assay-index", self.root / "custom.csv",
            ])
        self.assertEqual(status, 0, errors)
        catalog.assert_called_once_with((self.root / "custom.csv").resolve())
        self.assertEqual(prosst.call_args.args[1], ["first", "second"])
        self.assertTrue(callable(prosst.call_args.kwargs["progress"]))

    def test_explicit_sequence_family_is_forwarded_instead_of_default_selection(self):
        # Storage resolves the registered default only when family is absent.
        # Explicit ProSST must reach storage even when ESM-2 is registered first.
        for family in (None, "esm2", "prosst"):
            with self.subTest(family=family):
                arguments = ["validate", "--features", self.root]
                if family:
                    arguments.extend(["--model-family", family, "--cache-id", "requested_recipe"])
                report = {"model_family": family or "esm2", "cache_id": "requested_recipe"}
                with patch("feature_generation.storage.validate_sequence_cache", return_value=report) as sequence, patch(
                    "feature_generation.inverse_storage.validate_inverse_cache"
                ) as inverse:
                    status, output, errors = self.invoke(arguments)
                self.assertEqual(status, 0, errors)
                inverse.assert_not_called()
                self.assertEqual(json.loads(output), report)
                if family:
                    sequence.assert_called_once_with(self.root, "requested_recipe", model_family=family)
                else:
                    sequence.assert_called_once_with(self.root, None)

    def test_conflicting_validation_types_and_model_specific_flags_fail_early(self):
        with patch("feature_generation.storage.validate_sequence_cache") as sequence, patch(
            "feature_generation.inverse_storage.validate_inverse_cache"
        ) as inverse:
            status, output, errors = self.invoke([
                "validate", "--features", self.root, "--model-family", "prosst", "--feature-type", "inverse_folding",
            ])
        self.assertEqual(status, 1)
        self.assertEqual(output, "")
        self.assertIn("different feature types", errors)
        sequence.assert_not_called()
        inverse.assert_not_called()
        for family in ("esm2", "proteinmpnn", "esm_if1"):
            with self.subTest(family=family):
                values = {"model_id": "facebook/esm2_t6_8M_UR50D"} if family == "esm2" else {"model_family": family}
                config = self.root / f"{family}.json"
                config.write_text(json.dumps(values), encoding="utf-8")
                with patch("feature_generation.prosst_runner.extract") as prosst, patch(
                    "feature_generation.runner.extract"
                ) as esm2, patch("feature_generation.inverse_runner.extract") as inverse:
                    status, output, errors = self.invoke([
                        "extract", "--config", config, "--assay", "tiny", "--structure-token-dir", self.root / "tokens",
                    ])
                self.assertEqual(status, 1)
                self.assertEqual(output, "")
                self.assertIn("does not apply", errors)
                prosst.assert_not_called()
                esm2.assert_not_called()
                inverse.assert_not_called()

    def test_failed_assay_download_or_validation_returns_nonzero(self):
        results = [
            {"assay_id": "tiny", "status": "completed"},
            {"assay_id": "other", "status": "failed", "error": "missing structure"},
        ]
        with patch("feature_generation.prosst_runner.extract", return_value=results):
            status, output, _ = self.invoke([
                "extract", "--config", self.config, "--assay", "tiny", "--assay", "other", "--continue-on-error",
            ])
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(output)["results"], results)
        with patch("feature_generation.prosst_checkpoints.download", side_effect=ValueError("checkpoint checksum mismatch")):
            status, output, errors = self.invoke(["download", "--config", self.config])
        self.assertEqual(status, 1)
        self.assertEqual(output, "")
        self.assertIn("checkpoint checksum mismatch", errors)
        with patch("feature_generation.storage.validate_sequence_cache", side_effect=ValueError("structure token mismatch")):
            status, output, errors = self.invoke(["validate", "--features", self.root, "--model-family", "prosst"])
        self.assertEqual(status, 1)
        self.assertEqual(output, "")
        self.assertIn("structure token mismatch", errors)

    def test_invalid_prosst_batch_override_and_conflicting_config_never_run(self):
        with patch("feature_generation.prosst_runner.extract") as prosst:
            status, output, errors = self.invoke([
                "extract", "--config", self.config, "--assay", "tiny", "--max-tokens-per-batch", "0",
            ])
        self.assertEqual(status, 1)
        self.assertEqual(output, "")
        self.assertIn("positive integer", errors)
        prosst.assert_not_called()
        self.config.write_text(json.dumps({"model_family": "prosst", "feature_type": "inverse_folding"}), encoding="utf-8")
        with patch("feature_generation.prosst_checkpoints.download") as download:
            status, output, errors = self.invoke(["download", "--config", self.config])
        self.assertEqual(status, 1)
        self.assertEqual(output, "")
        self.assertIn("feature_type=sequence", errors)
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
