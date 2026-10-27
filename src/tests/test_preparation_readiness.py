"""Preparation audits select all ten caches, enforce coverage, and never infer."""

import contextlib
import csv
from dataclasses import replace
import io
import json
import os
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
        self.if1 = replace(r.InverseFoldingConfig.load(r.CONFIG_DIR / "esm_if1_142m.json"), assay_index=index)
        self.prosst = replace(r.ProSSTConfig.load(r.CONFIG_DIR / "prosst_2048_singles.json"), assay_index=index,
                              structure_token_dir=self.root / "tokens")
        self.msa = replace(r.MSAConfig.load(r.CONFIG_DIR / "msa_transformer.json"), assay_index=index)

    def run_audit(self, failure=None, shared_failure=False, missing_model=None, missing_msa=False, legacy=False):
        calls = []

        def output(config, assay, inputs, kind, structure, **kwargs):
            label = (assay, getattr(config, "model_family", "esm2"), kind)
            calls.append(label)
            if label == failure:
                raise ValueError("fixture corrupt cache")
            if legacy and label[1] == "proteinmpnn":
                return {"legacy_cache_id": "old-cache", "path": f"{assay}/{kind}/old-cache"}

        def inverse_model(config):
            if missing_model == "if1" and config.model_family == "esm_if1":
                raise ValueError("missing ESM-IF1 model")

        log = io.StringIO()
        summary = io.StringIO()
        with patch.object(r, "esm2_checkpoint", side_effect=ValueError("missing model") if shared_failure else None), \
                patch.object(r, "inverse_checkpoint", side_effect=inverse_model), patch.object(r, "load_inverse_structure"), \
                patch.object(r, "load_structure"), \
                patch.object(r, "prosst_checkpoint", side_effect=ValueError("missing ProSST model") if missing_model == "prosst" else None), \
                patch.object(r, "msa_checkpoint", side_effect=ValueError("missing MSA model") if missing_model == "msa" else None), \
                patch.object(r, "msa_inputs", side_effect=ValueError("missing MSA pair") if missing_msa else None), \
                patch.object(r, "validate_output", side_effect=output), contextlib.redirect_stdout(log):
            status = r.audit(self.esm, self.features, self.scores, self.if1, self.prosst, self.msa, summary_stream=summary)
        self.summary_rows = list(csv.DictReader(io.StringIO(summary.getvalue())))
        return status, log.getvalue(), calls

    def test_complete_outputs_skip_only_brca2_score(self):
        status, log, calls = self.run_audit()
        self.assertEqual(status, 0, log)
        self.assertEqual(len(calls), 19)
        self.assertNotIn((r.BRCA2, "proteinmpnn", "scores"), calls)
        self.assertIn((r.BRCA2, "proteinmpnn", "features"), calls)
        self.assertIn("Assay outputs ready: 2 / 2", log)
        self.assertIn("SKIP   ProteinMPNN scores", log)
        self.assertEqual([row["assay_id"] for row in self.summary_rows], ["tiny", r.BRCA2])
        self.assertEqual([row["array_index"] for row in self.summary_rows], ["0", "1"])
        self.assertEqual([row["status"] for row in self.summary_rows], ["READY", "READY"])
        self.assertEqual(self.summary_rows[1]["proteinmpnn_scores"], "SKIP")
        self.assertIn("BRCA2", self.summary_rows[1]["notes"])
        for row in self.summary_rows:
            for family, output in (("esm_if1", "features"), ("prosst", "embeddings"), ("msa_transformer", "embeddings")):
                self.assertEqual(row[f"{family}_{output}"], "READY")
                self.assertEqual(row[f"{family}_scores"], "READY")
                self.assertEqual(row[f"{family}_model"], "READY")

    def test_failed_cache_does_not_hide_other_outputs_or_assays(self):
        status, log, calls = self.run_audit(("tiny", "esm2", "features"))
        self.assertEqual(status, 1)
        self.assertEqual(len(calls), 19)
        self.assertIn("fixture corrupt cache", log)
        self.assertIn("Assay outputs ready: 1 / 2", log)
        self.assertEqual(self.summary_rows[0]["esm2_embeddings"], "FAILED")
        self.assertEqual(self.summary_rows[0]["status"], "FAILED")
        self.assertIn("fixture corrupt cache", self.summary_rows[0]["errors"])
        self.assertEqual(self.summary_rows[1]["status"], "READY")

    def test_legacy_selection_is_reported_without_losing_skip_notes(self):
        status, log, _ = self.run_audit(legacy=True)
        self.assertEqual(status, 0, log)
        self.assertIn("ProteinMPNN features: validated legacy cache tiny/features/old-cache", log)
        self.assertIn("ProteinMPNN scores: validated legacy cache tiny/scores/old-cache", self.summary_rows[0]["notes"])
        self.assertIn("validated legacy cache", self.summary_rows[1]["notes"])
        self.assertIn("intentionally omitted", self.summary_rows[1]["notes"])

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
        self.assertEqual(len(calls), 19)
        self.assertIn("shared asset failures: 1", log)
        for row in self.summary_rows:
            self.assertEqual(row["esm2_model"], "FAILED")
            self.assertEqual(row["outputs_status"], "READY")
            self.assertEqual(row["status"], "FAILED")
            self.assertIn("missing model", row["errors"])

    def test_new_model_failures_do_not_hide_other_outputs(self):
        for family, feature in (("esm_if1", "features"), ("prosst", "embeddings"), ("msa_transformer", "embeddings")):
            for kind, column in (("features", feature), ("scores", "scores")):
                with self.subTest(family=family, kind=kind):
                    status, log, calls = self.run_audit(("tiny", family, kind))
                    self.assertEqual(status, 1, log)
                    self.assertEqual(len(calls), 19)
                    self.assertEqual(self.summary_rows[0][f"{family}_{column}"], "FAILED")
                    self.assertEqual(self.summary_rows[0]["outputs_status"], "FAILED")
                    self.assertEqual(self.summary_rows[1]["status"], "READY")

    def test_missing_new_shared_assets_preserve_output_readiness(self):
        for model, column in (("if1", "esm_if1_model"), ("prosst", "prosst_model"), ("msa", "msa_transformer_model")):
            with self.subTest(model=model):
                status, log, calls = self.run_audit(missing_model=model)
                self.assertEqual(status, 1, log)
                self.assertEqual(len(calls), 19)
                for row in self.summary_rows:
                    self.assertEqual(row[column], "FAILED")
                    self.assertEqual(row["outputs_status"], "READY")
                    self.assertEqual(row["status"], "FAILED")

    def test_missing_msa_inputs_are_failures_not_skips(self):
        status, log, calls = self.run_audit(missing_msa=True)
        self.assertEqual(status, 1, log)
        self.assertEqual(len(calls), 19)
        for row in self.summary_rows:
            self.assertEqual(row["msa_inputs"], "FAILED")
            self.assertEqual(row["outputs_status"], "FAILED")

    def test_cli_defaults_match_preparation_jobs_and_config_overrides(self):
        for override in (False, True):
            with self.subTest(override=override):
                summary = self.root / f"summary-{override}.csv"
                arguments = ["--summary-csv", str(summary)]
                env = {"ESM2_CONFIG": str(r.CONFIG_DIR / "esm2_650m_centered.json"),
                       "PROSST_CONFIG": str(r.CONFIG_DIR / "prosst_2048_singles.json"),
                       "MSA_TRANSFORMER_CONFIG": str(r.CONFIG_DIR / "msa_transformer.json")}
                if override:
                    arguments += ["--prosst-config", str(r.CONFIG_DIR / "prosst_2048.json"),
                                  "--msa-transformer-config", str(r.CONFIG_DIR / "msa_transformer.json")]
                    env["PROSST_CONFIG"] = env["MSA_TRANSFORMER_CONFIG"] = "missing.json"
                with patch.dict(os.environ, env), patch.object(r, "audit", return_value=0) as audit, \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(r.main(arguments), 0)
                _, _, _, if1, prosst, msa = audit.call_args.args
                self.assertEqual(if1.output_root, r.PROJECT_ROOT / "data/features/pretrained/singles")
                self.assertEqual(prosst.structure_source, "proteingym" if override else "protssn_proteingym")
                self.assertEqual(msa.cache_id, self.msa.cache_id)

    def test_mismatched_new_catalog_fails_before_any_checks(self):
        with patch.object(r, "esm2_checkpoint") as checkpoint, self.assertRaisesRegex(ValueError, "same assay catalog"):
            r.audit(self.esm, self.features, self.scores, self.if1, self.prosst,
                    replace(self.msa, assay_index=self.root / "different.csv"))
        checkpoint.assert_not_called()

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

    def test_if1_features_and_scores_use_distinct_caches_and_current_structure(self):
        leaf = self.root / "if1"
        leaf.mkdir()
        saved = {key: getattr(self.if1, key) for key in
                 ("model_id", "revision", "inference_dtype", "conditioning_mode", "seed")}
        saved["dtype"] = self.if1.output_dtype
        (leaf / "config.json").write_text(json.dumps(saved))
        inputs = SimpleNamespace(input_signature="current")
        report = {"path": str(leaf), "input_signature": "current", "structure_input_signature": "backbone"}
        for kind, validator, cache_id in (("features", "validate_inverse_cache", self.if1.cache_id),
                                          ("scores", "validate_score_cache", r.default_cache_id(self.if1))):
            with self.subTest(kind=kind), patch.object(r, validator, return_value=report) as validate:
                r.validate_output(self.if1, "tiny", inputs, kind, SimpleNamespace(input_signature="backbone"))
                validate.assert_called_once_with(self.if1.output_root / "tiny", "esm_if1", cache_id)
                with self.assertRaisesRegex(ValueError, "different registered structure"):
                    r.validate_output(self.if1, "tiny", inputs, kind, SimpleNamespace(input_signature="changed"))

    def test_prosst_output_checks_structure_and_referenced_tokens(self):
        leaf = self.root / "prosst"
        leaf.mkdir()
        saved = {key: getattr(self.prosst, key) for key in
                 ("model_id", "revision", "inference_dtype", "layer", "pooling", "truncation", "max_length",
                  "site_embeddings", "global_window", "scoring_window")}
        saved.update(dtype="float32", structure_input_signature="backbone",
                     structure_token_input_signature="a" * 64, structure_tokens={"recipe": {"batch_size": 32}})
        (leaf / "config.json").write_text(json.dumps(saved))
        inputs = SimpleNamespace(input_signature="current")
        structure = SimpleNamespace(input_signature="backbone", reference_sequence="ACD")
        report = {"path": str(leaf), "input_signature": "current"}
        tokens = self.prosst.structure_token_dir / "prosst2048-fixture" / ("a" * 64)
        tokens.mkdir(parents=True)
        for kind, validator in (("features", "validate_sequence_cache"), ("scores", "validate_score_cache")):
            with self.subTest(kind=kind), patch.object(r, validator, return_value=report) as validate, \
                    patch.object(r, "validate_structure_token_cache", return_value=SimpleNamespace(metadata=saved["structure_tokens"])) as validate_tokens:
                r.validate_output(self.prosst, "tiny", inputs, kind, structure)
                self.assertEqual(validate.call_args.args[0], self.prosst.output_root / "tiny")
                validate_tokens.assert_called_once_with(tokens, {
                    "input_signature": "a" * 64, "structure_input_signature": "backbone", "reference_sequence": "ACD"})
                with self.assertRaisesRegex(ValueError, "different registered structure"):
                    r.validate_output(self.prosst, "tiny", inputs, kind, SimpleNamespace(input_signature="changed"))
                validate_tokens.side_effect = ValueError("token checksum mismatch")
                with self.assertRaisesRegex(ValueError, "token checksum"):
                    r.validate_output(self.prosst, "tiny", inputs, kind, structure)
        tokens.rmdir()
        with self.assertRaisesRegex(ValueError, "Expected one ProSST structure-token cache"):
            r.validate_prosst_tokens(self.prosst, saved, structure)

    def test_msa_checkpoint_verification_is_offline_and_detects_corruption(self):
        checkpoint = self.root / "runtime/model.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"synthetic checkpoint")
        spec = {"model_id": "fixture", "filename": checkpoint.name, "url": "https://example.invalid/model",
                "size_bytes": checkpoint.stat().st_size}
        import hashlib
        manifest = dict(spec, sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        (checkpoint.parent / "checkpoint_manifest.json").write_text(json.dumps(manifest))
        runtime = self.root / "runtime.json"
        runtime.write_text(json.dumps({"checkpoint": spec}))
        config = replace(self.msa, runtime_dir=checkpoint.parent, runtime_config=runtime)
        helper = r.preparation_helper("setup_msa_transformer")
        with patch.object(helper, "validate_runtime", side_effect=AssertionError("must not infer")), \
                patch.object(helper, "download_checkpoint", side_effect=AssertionError("must not download")):
            self.assertEqual(r.msa_checkpoint(config), manifest)
            checkpoint.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum or size"):
                r.msa_checkpoint(config)

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
