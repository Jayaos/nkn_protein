"""CLI routing and failure behavior for ESM-2 and inverse-folding workflows."""

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
from feature_generation.config import ExtractionConfig
from feature_generation.inverse_config import InverseFoldingConfig


class InverseCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.configs = {}
        for family, values in (
            ("esm2", {"model_id": "facebook/esm2_t6_8M_UR50D"}),
            ("proteinmpnn", {"model_family": "proteinmpnn"}),
            ("esm_if1", {"model_family": "esm_if1"}),
        ):
            path = self.root / f"{family}.json"
            path.write_text(json.dumps(values), encoding="utf-8")
            self.configs[family] = path

    def invoke(self, arguments):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            status = main([str(value) for value in arguments])
        return status, output.getvalue(), errors.getvalue()

    def test_download_dispatch_preserves_esm2_and_supports_both_inverse_models(self):
        for family in self.configs:
            with self.subTest(family=family):
                with patch("feature_generation.checkpoints.download", return_value=self.root / "esm2") as sequence, patch(
                    "feature_generation.inverse_checkpoints.download", return_value=self.root / family
                ) as inverse:
                    status, output, errors = self.invoke(["download", "--config", self.configs[family]])
                self.assertEqual(status, 0, errors)
                report = json.loads(output)
                self.assertRegex(report["revision"], r"^[0-9a-f]{40}$")
                if family == "esm2":
                    inverse.assert_not_called()
                    config = sequence.call_args.args[0]
                    self.assertIsInstance(config, ExtractionConfig)
                else:
                    sequence.assert_not_called()
                    config = inverse.call_args.args[0]
                    self.assertIsInstance(config, InverseFoldingConfig)
                    self.assertEqual(config.model_family, family)
                self.assertEqual(report["model_id"], config.model_id)

    def test_inverse_extraction_routes_overrides_and_flags_for_both_families(self):
        for family in ("proteinmpnn", "esm_if1"):
            with self.subTest(family=family):
                with patch("feature_generation.runner.extract") as sequence, patch(
                    "feature_generation.inverse_runner.extract",
                    return_value=[{"assay_id": "tiny", "status": "completed"}],
                ) as inverse:
                    status, output, errors = self.invoke([
                        "extract", "--config", self.configs[family],
                        "--assay", "tiny", "--assay", "other",
                        "--assay-index", self.root / "index.csv",
                        "--output-root", self.root / "outputs",
                        "--structure-manifest-dir", self.root / "manifests",
                        "--device", "cpu", "--limit", "4", "--singles-only",
                        "--resume", "--continue-on-error", "--quiet",
                    ])
                self.assertEqual(status, 0, errors)
                sequence.assert_not_called()
                config, ids = inverse.call_args.args
                self.assertEqual(ids, ["tiny", "other"])
                self.assertEqual(config.model_family, family)
                self.assertEqual(config.assay_index, (self.root / "index.csv").resolve())
                self.assertEqual(config.output_root, (self.root / "outputs").resolve())
                self.assertEqual(config.structure_manifest_dir, (self.root / "manifests").resolve())
                self.assertEqual(config.device, "cpu")
                self.assertEqual(inverse.call_args.kwargs, {
                    "singles_only": True, "limit": 4, "resume": True,
                    "continue_on_error": True, "progress": None,
                })
                self.assertEqual(json.loads(output)["results"][0]["status"], "completed")

    def test_esm2_extraction_keeps_sequence_batching_and_catalog_selection(self):
        records = [SimpleNamespace(assay_id="first"), SimpleNamespace(assay_id="second")]
        with patch("data_loading.AssayCatalog", return_value=records) as catalog, patch(
            "feature_generation.runner.extract", return_value=[]
        ) as sequence, patch("feature_generation.inverse_runner.extract") as inverse:
            status, _, errors = self.invoke([
                "extract", "--config", self.configs["esm2"], "--all-assays",
                "--assay-index", self.root / "index.csv", "--max-tokens-per-batch", "800",
            ])
        self.assertEqual(status, 0, errors)
        inverse.assert_not_called()
        config, ids = sequence.call_args.args
        self.assertIsInstance(config, ExtractionConfig)
        self.assertEqual(ids, ["first", "second"])
        self.assertEqual(config.max_tokens_per_batch, 800)
        catalog.assert_called_once_with((self.root / "index.csv").resolve())
        self.assertTrue(callable(sequence.call_args.kwargs["progress"]))

    def test_model_inappropriate_extraction_flags_are_rejected_before_inference(self):
        for family, extra in (
            ("proteinmpnn", ["--max-tokens-per-batch", "800"]),
            ("esm_if1", ["--max-tokens-per-batch", "800"]),
            ("esm2", ["--structure-manifest-dir", str(self.root / "manifests")]),
        ):
            with self.subTest(family=family):
                with patch("feature_generation.runner.extract") as sequence, patch(
                    "feature_generation.inverse_runner.extract"
                ) as inverse:
                    status, output, errors = self.invoke([
                        "extract", "--config", self.configs[family], "--assay", "tiny", *extra,
                    ])
                self.assertEqual(status, 1)
                self.assertEqual(output, "")
                self.assertIn("does not apply", errors)
                sequence.assert_not_called()
                inverse.assert_not_called()

    def test_validation_dispatch_and_recipe_selection(self):
        cases = (
            ([], None, None),
            (["--model-family", "esm2", "--cache-id", "sequence_recipe"], None, "sequence_recipe"),
            (["--feature-type", "inverse_folding"], None, None),
            (["--model-family", "proteinmpnn", "--cache-id", "mpnn_recipe"], "proteinmpnn", "mpnn_recipe"),
            (["--model-family", "esm_if1"], "esm_if1", None),
        )
        for extra, family, cache_id in cases:
            with self.subTest(extra=extra):
                with patch("feature_generation.storage.validate_sequence_cache", return_value={"kind": "sequence"}) as sequence, patch(
                    "feature_generation.inverse_storage.validate_inverse_cache", return_value={"kind": "inverse"}
                ) as inverse:
                    status, output, errors = self.invoke(["validate", "--features", self.root, *extra])
                self.assertEqual(status, 0, errors)
                is_inverse = family is not None or "inverse_folding" in extra
                if is_inverse:
                    sequence.assert_not_called()
                    inverse.assert_called_once_with(self.root, model_family=family, cache_id=cache_id)
                    self.assertEqual(json.loads(output)["kind"], "inverse")
                else:
                    inverse.assert_not_called()
                    if "esm2" in extra:
                        sequence.assert_called_once_with(self.root, cache_id, model_family="esm2")
                    else:
                        sequence.assert_called_once_with(self.root, cache_id)
                    self.assertEqual(json.loads(output)["kind"], "sequence")

    def test_conflicting_explicit_validation_flags_are_rejected(self):
        for feature_type, family in (
            ("sequence", "proteinmpnn"), ("sequence", "esm_if1"), ("inverse_folding", "esm2"),
        ):
            with self.subTest(feature_type=feature_type, family=family):
                with patch("feature_generation.storage.validate_sequence_cache") as sequence, patch(
                    "feature_generation.inverse_storage.validate_inverse_cache"
                ) as inverse:
                    status, output, errors = self.invoke([
                        "validate", "--features", self.root,
                        "--feature-type", feature_type, "--model-family", family,
                    ])
                self.assertEqual(status, 1)
                self.assertEqual(output, "")
                self.assertIn("error:", errors)
                sequence.assert_not_called()
                inverse.assert_not_called()

    def test_prepare_structure_passes_local_mapping_options(self):
        manifest = self.root / "manifests" / "tiny.json"
        with patch("feature_generation.structure_inputs.prepare_structure", return_value=manifest) as prepare:
            status, output, errors = self.invoke([
                "prepare-structure", "--assay", "tiny", "--assay-index", self.root / "index.csv",
                "--structure", self.root / "tiny.cif", "--chain", "B", "--model-index", "2",
                "--manifest-dir", self.root / "manifests", "--source-url", "https://example.org/tiny.cif",
            ])
        self.assertEqual(status, 0, errors)
        prepare.assert_called_once_with(
            self.root / "index.csv", "tiny", self.root / "tiny.cif",
            output_dir=self.root / "manifests", source_url="https://example.org/tiny.cif",
            chain_id="B", model_index=2,
        )
        self.assertEqual(json.loads(output), {"assay_id": "tiny", "manifest": str(manifest)})

    def test_download_structure_selects_pinned_archive_or_explicit_url(self):
        for url in (None, "https://example.org/tiny.pdb"):
            with self.subTest(url=url):
                with patch("feature_generation.structure_inputs.download_proteingym_structure", return_value=self.root / "tiny.json") as archive, patch(
                    "feature_generation.structure_inputs.download_structure", return_value=self.root / "tiny.json"
                ) as direct:
                    arguments = [
                        "download-structure", "--assay", "tiny", "--assay-index", self.root / "index.csv",
                        "--chain", "A", "--raw-dir", self.root / "raw", "--manifest-dir", self.root / "manifests",
                    ]
                    if url:
                        arguments.extend(["--url", url])
                    status, _, errors = self.invoke(arguments)
                self.assertEqual(status, 0, errors)
                expected = dict(raw_dir=self.root / "raw", manifest_dir=self.root / "manifests", chain_id="A", model_index=1)
                if url:
                    archive.assert_not_called()
                    direct.assert_called_once_with(self.root / "index.csv", "tiny", url, **expected)
                else:
                    direct.assert_not_called()
                    archive.assert_called_once_with(self.root / "index.csv", "tiny", **expected)

    def test_failed_assay_and_download_error_return_nonzero(self):
        with patch("feature_generation.inverse_runner.extract", return_value=[
            {"assay_id": "tiny", "status": "failed", "error": "missing structure"},
        ]):
            status, output, _ = self.invoke([
                "extract", "--config", self.configs["proteinmpnn"], "--assay", "tiny", "--continue-on-error",
            ])
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(output)["results"][0]["error"], "missing structure")
        with patch("feature_generation.inverse_checkpoints.download", side_effect=ValueError("checkpoint checksum failed")):
            status, output, errors = self.invoke(["download", "--config", self.configs["esm_if1"]])
        self.assertEqual(status, 1)
        self.assertEqual(output, "")
        self.assertIn("checkpoint checksum failed", errors)


if __name__ == "__main__":
    unittest.main()
