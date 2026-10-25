"""Offline inverse-folding cache checks with deterministic model outputs."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from feature_generation.inputs import sha256_file, stable_json
from feature_generation.inverse_runner import extract
from feature_generation.inverse_storage import validate_inverse_cache


def write_csv(path, header, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def structure_metadata():
    mapping = [
        {
            "reference_position": i + 1, "chain_id": "A", "residue_number": i + 10,
            "insertion_code": "", "residue_name": name, "amino_acid": aa,
            "backbone_atoms": ["N", "CA", "C", "O"],
        }
        for i, (aa, name) in enumerate(zip("ACD", ("ALA", "CYS", "ASP")))
    ]
    result = {
        "schema_version": 1,
        "reference_sequence": "ACD",
        "structure_sha256": "a" * 64,
        "chain_id": "A",
        "model_index": 1,
        "residue_mapping": mapping,
        "assay_id": "tiny",
        "structure_path": "../raw/tiny.pdb",
        "source_url": "https://example.org/tiny.pdb",
        "source_index_sha256": "b" * 64,
        "catalog_structure": {"pdb_file": "tiny.pdb", "pdb_range": "1-3"},
        "parser": "biopython",
        "parser_version": "test",
        "atom_order": ["N", "CA", "C", "O"],
        "mapping_policy": "exact_full_chain_sequence",
        "altloc_policy": "reject_backbone_alternates",
    }
    refresh_structure_signature(result)
    return result


def refresh_structure_signature(metadata):
    identity = {key: metadata[key] for key in (
        "schema_version", "reference_sequence", "structure_sha256",
        "chain_id", "model_index", "residue_mapping",
    )}
    metadata["input_signature"] = hashlib.sha256(stable_json(identity).encode()).hexdigest()


class FakeAdapter:
    def __init__(self, config):
        self.config = config
        self.load_count = 0
        self.predict_count = 0

    def metadata(self):
        return {
            "model_id": self.config.model_id,
            "revision": self.config.revision,
            "conditioning_mode": self.config.conditioning_mode,
            "checkpoint_sha256": "c" * 64,
        }

    def load(self):
        self.load_count += 1

    def predict(self, structure):
        self.predict_count += 1
        if torch.is_grad_enabled():
            raise AssertionError("Inference must disable gradients")
        logits = torch.arange(60, dtype=torch.float32).reshape(3, 20) / 20
        log_probs = logits.log_softmax(-1)
        return {"probabilities": log_probs.exp(), "log_probs": log_probs}


class FakeConditionalAdapter(FakeAdapter):
    def metadata(self):
        return dict(
            super().metadata(), normalization="native_21_amino_acid_tokens",
            noncanonical_tokens="excluded_without_renormalization",
            order_aggregation="geometric_mean_of_probabilities",
            inference_method="conditional_probs", num_orders=self.config.num_orders,
        )

    def predict(self, structure):
        prediction = super().predict(structure)
        # Model geometric means with missing X mass; storage must keep it.
        prediction["log_probs"] -= 0.25
        prediction["probabilities"] = prediction["log_probs"].exp()
        return prediction


class InversePipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        raw = self.root / "raw"
        raw.mkdir()
        self.index = raw / "DMS_substitutions.csv"
        write_csv(self.index, ["DMS_id", "DMS_filename", "target_seq", "seq_len"], [["tiny", "tiny.csv", "ACD", 3]])
        write_csv(raw / "tiny.csv", ["variant_id", "mutant", "mutated_sequence", "DMS_score"], [
            ["wt", "WT", "ACD", 0],
            ["first", "A1V", "VCD", 1],
            ["second", "C2D", "ADD", 2],
            ["double", "A1V:C2D", "VDD", 3],
        ])
        self.config = SimpleNamespace(
            model_family="proteinmpnn", model_id="v_48_020",
            revision="d" * 40, cache_id="tiny_test",
            output_root=self.root / "features", assay_index=self.index,
            structure_manifest_dir=self.root / "manifests",
            model_dir=self.root / "models", max_length=10, seed=0,
            conditioning_mode="backbone_only", device="cpu",
            inference_dtype="float32", output_dtype="float32",
        )
        self.metadata = structure_metadata()
        self.adapters = []
        self.structure_loader = patch(
            "feature_generation.inverse_runner.load_structure",
            side_effect=self.load_structure,
        )
        self.mock_structure = self.structure_loader.start()
        self.addCleanup(self.structure_loader.stop)

    def load_structure(self, manifest_path, reference_sequence, require_oxygen=True):
        metadata = copy.deepcopy(self.metadata)
        return SimpleNamespace(
            metadata=metadata,
            reference_sequence=reference_sequence,
            positions=torch.arange(1, 4),
            coords=torch.zeros(3, 4, 3),
            structure_sha256=metadata["structure_sha256"],
            input_signature=metadata["input_signature"],
        )

    def factory(self, config):
        adapter_type = FakeConditionalAdapter if config.conditioning_mode == "sequence_conditional" else FakeAdapter
        adapter = adapter_type(config)
        self.adapters.append(adapter)
        return adapter

    @property
    def assay_root(self):
        return self.config.output_root / "tiny"

    @property
    def leaf(self):
        return self.assay_root / "inverse_folding" / self.config.model_family / self.config.cache_id

    def run_extraction(self, **kwargs):
        return extract(self.config, ["tiny"], adapter_factory=self.factory, **kwargs)

    def rehash(self, name):
        path = self.leaf / "completion.json"
        completion = json.loads(path.read_text(encoding="utf-8"))
        completion["files_sha256"][name] = sha256_file(self.leaf / name)
        path.write_text(json.dumps(completion), encoding="utf-8")

    def test_reference_distribution_roundtrip_and_resume_skip_inference(self):
        result = self.run_extraction()
        self.assertEqual(result[0]["shape"], [3, 20])
        self.assertEqual(self.adapters[0].predict_count, 1)
        self.assertEqual(self.adapters[0].load_count, 1)
        payload = torch.load(self.leaf / "features.pt", weights_only=True)
        self.assertEqual(payload["alphabet"], "ACDEFGHIKLMNPQRSTVWY")
        self.assertNotIn("variant_ids", payload)
        torch.testing.assert_close(payload["positions"], torch.arange(1, 4))
        torch.testing.assert_close(payload["probabilities"].sum(-1), torch.ones(3))
        report = validate_inverse_cache(self.assay_root)
        self.assertEqual(report["num_variants"], 4)
        self.assertEqual(report["conditioning_mode"], "backbone_only")
        result = self.run_extraction(resume=True)
        self.assertEqual(result[0]["status"], "skipped")
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 1)
        self.assertEqual(sum(adapter.predict_count for adapter in self.adapters), 1)
        self.assertTrue(self.mock_structure.call_args.kwargs["require_oxygen"])

    def test_models_coexist_and_preserve_existing_defaults(self):
        self.run_extraction()
        assay_path = self.assay_root / "assay.json"
        metadata = json.loads(assay_path.read_text(encoding="utf-8"))
        sequence_descriptor = {"model_family": "esm2", "checkpoint": "user_selected", "path": "sequence/esm2/user_selected"}
        inverse_descriptor = metadata["features"]["inverse_folding"].copy()
        metadata["features"]["sequence"] = sequence_descriptor
        assay_path.write_text(json.dumps(metadata), encoding="utf-8")
        self.config.model_family = "esm_if1"
        self.config.model_id = "esm_if1_gvp4_t16_142M_UR50"
        self.config.conditioning_mode = "reference_prefix"
        self.run_extraction()
        metadata = json.loads(assay_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["features"]["sequence"], sequence_descriptor)
        self.assertEqual(metadata["features"]["inverse_folding"], inverse_descriptor)
        self.assertEqual(validate_inverse_cache(self.assay_root)["model_family"], "proteinmpnn")
        self.assertEqual(validate_inverse_cache(self.assay_root, "esm_if1")["conditioning_mode"], "reference_prefix")
        self.assertFalse(self.mock_structure.call_args.kwargs["require_oxygen"])

    def use_conditional_recipe(self):
        self.config.conditioning_mode = "sequence_conditional"
        self.config.cache_id = "conditional-orders10-seed37"
        self.config.num_orders = 10
        self.config.seed = 37

    def test_esm_if1_cleanup_respects_resume_and_removes_only_its_other_recipes(self):
        self.run_extraction()
        proteinmpnn_leaf = self.leaf
        self.config.model_family = "esm_if1"
        self.config.model_id = "esm_if1_gvp4_t16_142M_UR50"
        self.config.conditioning_mode = "reference_prefix"
        self.run_extraction()
        self.config.cache_id = "second"
        self.run_extraction()
        before = {path.name: path.read_bytes() for path in self.leaf.iterdir()}
        result = self.run_extraction(resume=True, cleanup_and_save=True)[0]
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(list(self.leaf.parent.iterdir()), [self.leaf])
        self.assertEqual(self.adapters[-1].predict_count, 0)
        self.assertEqual({path.name: path.read_bytes() for path in self.leaf.iterdir()}, before)
        self.assertTrue(proteinmpnn_leaf.is_dir())
        self.assertEqual(validate_inverse_cache(self.assay_root)["model_family"], "proteinmpnn")
        self.assertEqual(validate_inverse_cache(self.assay_root, "esm_if1")["cache_id"], "second")
        (self.leaf / "features.pt").write_bytes(b"corrupted")
        result = self.run_extraction(resume=True, cleanup_and_save=True)[0]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.adapters[-1].predict_count, 1)
        validate_inverse_cache(self.assay_root, "esm_if1")

    def test_conditional_cache_preserves_subprobabilities_and_backbone_cache(self):
        self.run_extraction()
        backbone_leaf = self.leaf
        backbone_bytes = (backbone_leaf / "features.pt").read_bytes()
        variants_bytes = (self.assay_root / "variants.csv").read_bytes()
        self.use_conditional_recipe()
        self.run_extraction()
        payload = torch.load(self.leaf / "features.pt", weights_only=True)
        expected_logs = (torch.arange(60, dtype=torch.float32).reshape(3, 20) / 20).log_softmax(-1) - 0.25
        torch.testing.assert_close(payload["probabilities"], expected_logs.exp(), atol=0, rtol=0)
        torch.testing.assert_close(payload["log_probs"], expected_logs, atol=0, rtol=0)
        self.assertTrue((payload["probabilities"].sum(-1) < 1).all())
        self.assertEqual((backbone_leaf / "features.pt").read_bytes(), backbone_bytes)
        self.assertEqual((self.assay_root / "variants.csv").read_bytes(), variants_bytes)
        self.assertEqual(validate_inverse_cache(self.assay_root)["conditioning_mode"], "backbone_only")
        report = validate_inverse_cache(self.assay_root, "proteinmpnn", self.config.cache_id)
        self.assertEqual(report["conditioning_mode"], "sequence_conditional")
        self.assertEqual(self.run_extraction(resume=True)[0]["status"], "skipped")
        self.config.num_orders = 11
        with self.assertRaisesRegex(ValueError, "structure differs"):
            self.run_extraction(resume=True)

    def test_conditional_cache_requires_declared_recipe(self):
        self.use_conditional_recipe()
        original = FakeConditionalAdapter.metadata
        for field, value in (("normalization", "canonical_amino_acids_only"),
                             ("order_aggregation", "arithmetic_mean"),
                             ("num_orders", 0)):
            with self.subTest(field=field):
                def malformed(adapter):
                    return dict(original(adapter), **{field: value})
                with patch.object(FakeConditionalAdapter, "metadata", malformed), self.assertRaises(ValueError):
                    self.run_extraction()
                self.assertFalse(self.leaf.exists())

    def test_conditional_cache_rejects_excess_mass_and_inconsistent_logs(self):
        self.use_conditional_recipe()
        original = FakeConditionalAdapter.predict
        for kind in ("mass", "logs"):
            with self.subTest(kind=kind):
                def invalid(adapter, structure):
                    output = original(adapter, structure)
                    if kind == "mass":
                        output["probabilities"] *= 2
                        output["log_probs"] = output["probabilities"].log()
                    else:
                        output["log_probs"] += 0.1
                    return output
                with patch.object(FakeConditionalAdapter, "predict", invalid), self.assertRaises(ValueError):
                    self.run_extraction()
                self.assertFalse(self.leaf.exists())

    def test_structure_content_mapping_and_source_changes_block_resume(self):
        self.run_extraction()
        original = copy.deepcopy(self.metadata)
        for change in ("structure_sha256", "residue_mapping", "source_url"):
            with self.subTest(change=change):
                self.metadata = copy.deepcopy(original)
                if change == "residue_mapping":
                    self.metadata["residue_mapping"][0]["residue_number"] = 9
                elif change == "structure_sha256":
                    self.metadata[change] = "f" * 64
                else:
                    self.metadata[change] = "https://example.org/other.pdb"
                refresh_structure_signature(self.metadata)
                with self.assertRaisesRegex(ValueError, "structure differs"):
                    self.run_extraction(resume=True)
        self.assertEqual(sum(adapter.predict_count for adapter in self.adapters), 1)

    def test_checkpoint_seed_and_conditioning_changes_block_resume(self):
        self.run_extraction()
        self.config.revision = "e" * 40
        with self.assertRaisesRegex(ValueError, "structure differs"):
            self.run_extraction(resume=True)
        self.config.revision = "d" * 40
        self.config.seed = 10
        with self.assertRaisesRegex(ValueError, "structure differs"):
            self.run_extraction(resume=True)
        self.config.conditioning_mode = "silently_different"
        with self.assertRaisesRegex(ValueError, "conditioning"):
            self.run_extraction(resume=True)
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 1)

    def test_variant_table_is_immutable_for_other_modalities(self):
        self.run_extraction()
        before = (self.assay_root / "variants.csv").read_bytes()
        with self.assertRaisesRegex(ValueError, "coverage differ"):
            self.run_extraction(resume=True, limit=2)
        self.assertEqual((self.assay_root / "variants.csv").read_bytes(), before)

    def test_invalid_distributions_never_publish_a_cache(self):
        original = FakeAdapter.predict

        def invalid(adapter, structure):
            output = original(adapter, structure)
            output["probabilities"][0, 0] = float("nan")
            return output

        with patch.object(FakeAdapter, "predict", invalid):
            with self.assertRaisesRegex(ValueError, "finite CPU float32"):
                self.run_extraction()
        self.assertFalse(self.leaf.exists())
        self.run_extraction(resume=True)
        validate_inverse_cache(self.assay_root)

    def test_unnormalized_probabilities_and_wrong_logs_are_rejected(self):
        original = FakeAdapter.predict
        for kind in ("sum", "logs", "negative", "dtype"):
            with self.subTest(kind=kind):
                def invalid(adapter, structure):
                    output = original(adapter, structure)
                    if kind == "sum":
                        output["probabilities"] *= 2
                    elif kind == "logs":
                        output["log_probs"] += 1
                    elif kind == "negative":
                        output["probabilities"][0, 0] = -0.1
                    else:
                        output["probabilities"] = output["probabilities"].double()
                    return output
                with patch.object(FakeAdapter, "predict", invalid), self.assertRaises(ValueError):
                    self.run_extraction()
                self.assertFalse(self.leaf.exists())

    def test_storage_interruption_is_not_a_completed_cache_and_can_retry(self):
        with patch("feature_generation.inverse_storage.torch.save", side_effect=OSError("disk failed")):
            with self.assertRaisesRegex(OSError, "disk failed"):
                self.run_extraction()
        self.assertFalse(self.leaf.exists())
        self.assertTrue(list(self.leaf.parent.glob(".tiny_test.pending-*")))
        result = self.run_extraction(resume=True)
        self.assertEqual(result[0]["status"], "completed")
        validate_inverse_cache(self.assay_root)

    def test_missing_completion_or_modified_payload_fails_validation(self):
        self.run_extraction()
        path = self.leaf / "features.pt"
        payload = torch.load(path, weights_only=True)
        payload["alphabet"] = payload["alphabet"][::-1]
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_inverse_cache(self.assay_root)
        self.rehash("features.pt")
        with self.assertRaisesRegex(ValueError, "payload metadata"):
            validate_inverse_cache(self.assay_root)
        (self.leaf / "completion.json").unlink()
        self.assertEqual(self.run_extraction(resume=True)[0]["status"], "completed")
        validate_inverse_cache(self.assay_root)

    def test_structure_manifest_integrity_is_checked_without_original_structure(self):
        self.run_extraction()
        path = self.leaf / "config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["structure"]["residue_mapping"][0]["residue_number"] = 900
        path.write_text(json.dumps(config), encoding="utf-8")
        self.rehash("config.json")
        with self.assertRaises(ValueError):
            validate_inverse_cache(self.assay_root)

    def test_length_guard_and_missing_structure_fail_before_model_load(self):
        self.config.max_length = 2
        with self.assertRaisesRegex(ValueError, "max_length"):
            self.run_extraction()
        self.mock_structure.assert_not_called()
        self.config.max_length = 10
        self.mock_structure.side_effect = FileNotFoundError("no structure manifest")
        with self.assertRaisesRegex(FileNotFoundError, "no structure manifest"):
            self.run_extraction()
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 0)

    def test_continue_on_error_reports_failure_and_processes_other_assays(self):
        result = extract(self.config, ["missing", "tiny"], continue_on_error=True, adapter_factory=self.factory)
        self.assertEqual([item["status"] for item in result], ["failed", "completed"])
        validate_inverse_cache(self.assay_root)

    def test_real_structure_registration_and_changed_source_block_resume(self):
        from feature_generation.structure_inputs import load_structure, prepare_structure

        lines = []
        serial = 1
        for position, name in enumerate(("ALA", "CYS", "ASP"), 1):
            for atom_index, atom in enumerate(("N", "CA", "C", "O")):
                x = position * 4.0 + atom_index * 0.3
                lines.append(
                    f"ATOM  {serial:5d} {atom:^4s} {name} A{position:4d}    "
                    f"{x:8.3f}{1.2:8.3f}{2.4:8.3f}{1.0:6.2f}{20.0:6.2f}          {atom[0]:>2s}  "
                )
                serial += 1
        pdb = self.root / "tiny.pdb"
        pdb.write_text("\n".join(lines) + "\nTER\n", encoding="utf-8")
        prepare_structure(self.index, "tiny", pdb, output_dir=self.config.structure_manifest_dir)
        with patch("feature_generation.inverse_runner.load_structure", load_structure):
            self.run_extraction()
            self.assertEqual(self.run_extraction(resume=True)[0]["status"], "skipped")
            with pdb.open("a", encoding="utf-8") as handle:
                handle.write("REMARK changed source file\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                self.run_extraction(resume=True)
        # Standalone validation verifies the cached snapshot, without requiring
        # the original source file to still be present at its old path.
        validate_inverse_cache(self.assay_root)
        self.assertEqual(sum(adapter.predict_count for adapter in self.adapters), 1)

    def test_existing_cache_never_overwritten_without_resume(self):
        self.run_extraction()
        before = (self.leaf / "features.pt").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_extraction()
        self.assertEqual((self.leaf / "features.pt").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
