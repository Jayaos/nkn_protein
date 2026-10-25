"""Offline integration checks for ProSST structure-conditioned sequence caches."""

from __future__ import annotations

import copy
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from feature_generation.inputs import prepare_assay, sha256_file, stable_json
from feature_generation.prosst_config import ProSSTConfig
from feature_generation.prosst_runner import extract
from feature_generation.storage import make_leaf_config, validate_sequence_cache, write_sequence_cache


def write_csv(path, header, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def refresh_signature(metadata):
    keys = ("schema_version", "reference_sequence", "structure_sha256", "chain_id", "model_index", "residue_mapping")
    metadata["input_signature"] = hashlib.sha256(stable_json({key: metadata[key] for key in keys}).encode()).hexdigest()


def structure_metadata():
    metadata = {
        "schema_version": 1, "reference_sequence": "ACD", "structure_sha256": "a" * 64,
        "chain_id": "A", "model_index": 1,
        "residue_mapping": [{
            "reference_position": i + 1, "chain_id": "A", "residue_number": i + 10,
            "insertion_code": "", "residue_name": name, "amino_acid": aa,
            "backbone_atoms": ["N", "CA", "C", "O"],
        } for i, (aa, name) in enumerate(zip("ACD", ("ALA", "CYS", "ASP")))],
        "atom_order": ["N", "CA", "C", "O"], "mapping_policy": "exact_full_chain_sequence",
        "altloc_policy": "reject_backbone_alternates",
    }
    refresh_signature(metadata)
    return metadata


def residue_values(sequence, tokens):
    return torch.tensor([[ord(aa), position + 1, int(tokens[position])] for position, aa in enumerate(sequence)], dtype=torch.float32)


class FakeAdapter:
    hidden_size = 3

    def __init__(self, config):
        self.config = config
        self.load_count = 0
        self.bindings = []

    def metadata(self):
        return {
            "model_id": self.config.model_id, "revision": self.config.revision,
            "model_family": "prosst", "hidden_size": self.hidden_size,
            "conditioning_mode": "sequence_and_fixed_reference_structure", "layer": self.config.layer,
        }

    def load(self):
        self.load_count += 1

    def set_structure_tokens(self, tokens):
        self.tokens = tokens.clone()
        self.bindings.append(tokens.clone())

    def iter_embeddings(self, sequences):
        for start in range(len(sequences), 0, -2):
            indices = list(range(max(0, start - 2), start))[::-1]
            yield indices, torch.stack([residue_values(sequences[index], self.tokens) for index in indices])


class ProSSTPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        raw = self.root / "raw"
        raw.mkdir()
        self.index = raw / "DMS_substitutions.csv"
        write_csv(self.index, ["DMS_id", "DMS_filename", "target_seq", "seq_len"], [
            ["tiny", "tiny.csv", "ACD", 3], ["other", "other.csv", "ACD", 3],
        ])
        self.sequences = ("ACD", "VCD", "ADD", "VDD")
        write_csv(raw / "tiny.csv", ["variant_id", "mutant", "mutated_sequence", "DMS_score"], [
            ["wt", "WT", "ACD", 0], ["first", "A1V", "VCD", 1],
            ["second", "C2D", "ADD", 2], ["double", "A1V:C2D", "VDD", 3],
        ])
        (raw / "other.csv").write_bytes((raw / "tiny.csv").read_bytes())
        self.config = ProSSTConfig(
            cache_id="tiny_test", output_root=self.root / "features", assay_index=self.index,
            structure_manifest_dir=self.root / "manifests", structure_token_dir=self.root / "tokens",
            model_dir=self.root / "models", max_length=10, max_tokens_per_batch=10, device="cpu",
        )
        self.metadata = structure_metadata()
        self.tokens = torch.tensor([31, 60, 2047])
        self.adapters = []
        self.quantizations = []
        self.loader = patch("feature_generation.prosst_runner.load_structure", side_effect=self.load_structure)
        self.mock_structure = self.loader.start()
        self.addCleanup(self.loader.stop)

    def load_structure(self, path, reference_sequence, require_oxygen=True):
        metadata = copy.deepcopy(self.metadata)
        if Path(path).stem == "other":
            metadata["structure_sha256"] = "b" * 64
            refresh_signature(metadata)
        return SimpleNamespace(
            metadata=metadata, reference_sequence=reference_sequence,
            positions=torch.arange(1, 4), coords=torch.zeros(3, 4, 3),
            residue_mapping=metadata["residue_mapping"],
            structure_sha256=metadata["structure_sha256"], input_signature=metadata["input_signature"],
        )

    def token_factory(self, config, structure):
        # Uses the quantizer's metadata builder but never imports model runtimes.
        from feature_generation.prosst_structure import DEPENDENCIES, _input_metadata, canonical_pdb_bytes, token_ids_sha256
        tokens = self.tokens.clone()
        if structure.structure_sha256 == "b" * 64:
            tokens = torch.tensor([20, 40, 60])
        runtime = {name: "test" for name in DEPENDENCIES}
        runtime.update(python="test", platform="test", machine="test")
        runtime["scikit-learn"] = "1.2.2"
        with patch("feature_generation.prosst_structure._runtime_versions", return_value=runtime):
            metadata = _input_metadata(config, structure, canonical_pdb_bytes(structure))
        metadata.update(num_residues=len(tokens), tokens_sha256=token_ids_sha256(tokens), token_ids_sha256=token_ids_sha256(tokens))
        self.quantizations.append(structure.input_signature)
        return SimpleNamespace(tokens=tokens, metadata=metadata, input_signature=metadata["input_signature"], path=config.structure_token_dir)

    def factory(self, config):
        adapter = FakeAdapter(config)
        self.adapters.append(adapter)
        return adapter

    @property
    def assay_root(self):
        return self.config.output_root / "tiny"

    @property
    def leaf(self):
        return self.assay_root / "sequence" / "prosst" / self.config.cache_id

    def run_extraction(self, **kwargs):
        return extract(self.config, ["tiny"], adapter_factory=self.factory, token_factory=self.token_factory, **kwargs)

    def rehash(self, name):
        path = self.leaf / "completion.json"
        completion = json.loads(path.read_text(encoding="utf-8"))
        completion["files_sha256"][name] = sha256_file(self.leaf / name)
        path.write_text(json.dumps(completion), encoding="utf-8")

    def test_reordered_batches_preserve_variant_position_and_token_alignment(self):
        result = self.run_extraction()
        self.assertEqual(result[0]["shape"], [4, 3, 3])
        payload = torch.load(self.leaf / "features.pt", weights_only=True)
        self.assertEqual(payload["variant_ids"], ["wt", "first", "second", "double"])
        torch.testing.assert_close(payload["embeddings"], torch.stack([residue_values(sequence, self.tokens) for sequence in self.sequences]))
        torch.testing.assert_close(payload["positions"], torch.arange(1, 4))
        report = validate_sequence_cache(self.assay_root)
        self.assertEqual(report["model_family"], "prosst")
        self.assertEqual(report["num_variants"], 4)
        self.assertTrue(self.mock_structure.call_args.kwargs["require_oxygen"])

    def test_resume_revalidates_inputs_without_loading_or_binding_model(self):
        self.run_extraction()
        self.assertEqual(self.run_extraction(resume=True)[0]["status"], "skipped")
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 1)
        self.assertEqual(sum(len(adapter.bindings) for adapter in self.adapters), 1)
        self.assertEqual(len(self.quantizations), 2)

    def test_cleanup_recomputes_over_resume_and_removes_old_recipes(self):
        self.run_extraction()
        self.config = replace(self.config, cache_id="second")
        self.run_extraction()
        self.tokens = torch.tensor([1, 2, 3])
        result = self.run_extraction(resume=True, cleanup_and_save=True)[0]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(list(self.leaf.parent.iterdir()), [self.leaf])
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 3)
        payload = torch.load(self.leaf / "features.pt", weights_only=True)
        torch.testing.assert_close(payload["embeddings"], torch.stack([
            residue_values(sequence, self.tokens) for sequence in self.sequences
        ]))
        self.assertEqual(validate_sequence_cache(self.assay_root)["cache_id"], "second")

    def test_cleanup_inference_failure_preserves_previous_cache(self):
        self.run_extraction()
        before = {path.name: path.read_bytes() for path in self.leaf.iterdir()}
        with patch.object(FakeAdapter, "iter_embeddings", side_effect=RuntimeError("inference failed")):
            result = self.run_extraction(cleanup_and_save=True, continue_on_error=True)[0]
        self.assertEqual(result["status"], "failed")
        self.assertEqual({path.name: path.read_bytes() for path in self.leaf.iterdir()}, before)

    def test_changed_structure_tokens_layer_or_coverage_cannot_resume(self):
        self.run_extraction()
        original_metadata = copy.deepcopy(self.metadata)
        original_tokens = self.tokens.clone()
        original_config = self.config
        self.metadata["structure_sha256"] = "c" * 64
        refresh_signature(self.metadata)
        with self.assertRaises(ValueError):
            self.run_extraction(resume=True)
        self.metadata = original_metadata
        self.tokens = torch.tensor([1, 2, 3])
        with self.assertRaises(ValueError):
            self.run_extraction(resume=True)
        self.tokens = original_tokens
        self.config = replace(self.config, layer=11)
        with self.assertRaises(ValueError):
            self.run_extraction(resume=True)
        self.config = original_config
        before = (self.assay_root / "variants.csv").read_bytes()
        with self.assertRaises(ValueError):
            self.run_extraction(resume=True, limit=2)
        self.assertEqual((self.assay_root / "variants.csv").read_bytes(), before)
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 1)

    def test_two_sequence_families_coexist_and_keep_first_default(self):
        inputs = prepare_assay(self.index, "tiny")
        esm_config = make_leaf_config("esm_test", {"hidden_size": 1})
        write_sequence_cache(self.assay_root, "esm_test", inputs, esm_config, torch.ones(4, 3, 1), {})
        self.run_extraction()
        self.assertEqual(validate_sequence_cache(self.assay_root)["model_family"], "esm2")
        self.assertEqual(validate_sequence_cache(self.assay_root, model_family="prosst")["embedding_dim"], 3)
        self.assertEqual(validate_sequence_cache(self.assay_root, "tiny_test")["model_family"], "prosst")
        old_payload = torch.load(self.assay_root / "sequence/esm2/esm_test/features.pt", weights_only=True)
        self.assertNotIn("structure_input_signature", old_payload)
        self.assertNotIn("model_family", old_payload)

    def test_invalid_default_descriptor_is_rejected_but_explicit_family_is_discoverable(self):
        self.run_extraction()
        path = self.assay_root / "assay.json"
        original = json.loads(path.read_text())
        cases = (
            {},
            {"model_family": "unknown", "checkpoint": self.config.cache_id},
            {"model_family": "prosst"},
        )
        for descriptor in cases:
            with self.subTest(descriptor=descriptor):
                metadata = copy.deepcopy(original)
                metadata["features"]["sequence"] = descriptor
                path.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, "default sequence descriptor"):
                    validate_sequence_cache(self.assay_root)
                self.assertEqual(validate_sequence_cache(self.assay_root, model_family="prosst")["model_family"], "prosst")
        metadata = copy.deepcopy(original)
        del metadata["features"]["sequence"]
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "default sequence descriptor"):
            validate_sequence_cache(self.assay_root)
        self.assertEqual(validate_sequence_cache(self.assay_root, self.config.cache_id, model_family="prosst")["model_family"], "prosst")

    def test_model_reuse_binds_each_assay_and_metadata_is_not_shared(self):
        results = extract(self.config, ["tiny", "other"], adapter_factory=self.factory, token_factory=self.token_factory)
        self.assertEqual([result["status"] for result in results], ["completed", "completed"])
        self.assertEqual(self.adapters[0].load_count, 1)
        self.assertEqual(len(self.adapters[0].bindings), 2)
        first = json.loads((self.leaf / "config.json").read_text())
        second_leaf = self.config.output_root / "other/sequence/prosst" / self.config.cache_id
        second = json.loads((second_leaf / "config.json").read_text())
        self.assertNotEqual(first["structure_input_signature"], second["structure_input_signature"])
        self.assertNotEqual(first["structure_token_input_signature"], second["structure_token_input_signature"])
        torch.testing.assert_close(torch.load(second_leaf / "features.pt", weights_only=True)["embeddings"][0], residue_values("ACD", torch.tensor([20, 40, 60])))
        validate_sequence_cache(self.config.output_root / "other")

    def test_standalone_validation_needs_no_structure_or_model_artifacts(self):
        self.run_extraction()
        with patch("feature_generation.prosst_runner.load_structure", side_effect=AssertionError("must not open structure")):
            self.assertEqual(validate_sequence_cache(self.assay_root)["reference_length"], 3)

    def test_completion_token_identity_corruption_is_rejected(self):
        self.run_extraction()
        path = self.leaf / "completion.json"
        value = json.loads(path.read_text())
        value["structure_token_input_signature"] = "f" * 64
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "completion"):
            validate_sequence_cache(self.assay_root)
        with self.assertRaises(ValueError):
            self.run_extraction(resume=True)

    def test_rehashed_payload_misalignment_is_rejected(self):
        self.run_extraction()
        path = self.leaf / "features.pt"
        value = torch.load(path, weights_only=True)
        value["variant_ids"] = list(reversed(value["variant_ids"]))
        torch.save(value, path)
        self.rehash("features.pt")
        with self.assertRaisesRegex(ValueError, "IDs"):
            validate_sequence_cache(self.assay_root)

    def test_rehashed_family_config_mismatch_is_rejected(self):
        self.run_extraction()
        path = self.leaf / "config.json"
        value = json.loads(path.read_text())
        value["model_family"] = "esm2"
        path.write_text(json.dumps(value))
        self.rehash("config.json")
        with self.assertRaisesRegex(ValueError, "model family"):
            validate_sequence_cache(self.assay_root, model_family="prosst")

    def test_interruption_exposes_no_leaf_and_resume_retries(self):
        def interrupted(adapter, sequences):
            yield [0], residue_values(sequences[0], adapter.tokens).unsqueeze(0)
            raise RuntimeError("simulated interruption")
        with patch.object(FakeAdapter, "iter_embeddings", interrupted):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                self.run_extraction()
        self.assertFalse(self.leaf.exists())
        self.assertEqual(self.run_extraction(resume=True)[0]["status"], "completed")

    def test_missing_completion_is_not_reused(self):
        self.run_extraction()
        (self.leaf / "completion.json").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.run_extraction(resume=True)

    def test_invalid_adapter_outputs_never_create_a_leaf(self):
        invalid_batches = [
            [([0, 0], torch.zeros(2, 3, 3))],
            [([4], torch.zeros(1, 3, 3))],
            [([0], torch.zeros(1, 3, 3, dtype=torch.float64))],
            [([0], torch.full((1, 3, 3), float("nan")))],
            [([0], torch.zeros(1, 4, 3))],
            [([0], torch.zeros(1, 3, 3))],
        ]
        for batches in invalid_batches:
            with self.subTest(batches=batches):
                with patch.object(FakeAdapter, "iter_embeddings", return_value=iter(batches)):
                    with self.assertRaises(ValueError):
                        self.run_extraction()
                self.assertFalse(self.leaf.exists())

    def test_guards_run_before_quantization_and_model_loading(self):
        original = self.config
        for overrides in ({"max_length": 2}, {"max_tokens_per_batch": 4}, {"max_output_gib": 1e-12}):
            with self.subTest(overrides=overrides):
                self.config = replace(original, **overrides)
                with self.assertRaises(ValueError):
                    self.run_extraction()
        self.assertEqual(self.quantizations, [])
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 0)

    def test_continue_on_error_preserves_successful_assays(self):
        results = extract(self.config, ["missing", "tiny"], adapter_factory=self.factory, token_factory=self.token_factory, continue_on_error=True)
        self.assertEqual([result["status"] for result in results], ["failed", "completed"])
        validate_sequence_cache(self.assay_root)

    def test_unknown_family_or_conflicting_model_metadata_is_rejected(self):
        for family in ("../escape", "other"):
            with self.assertRaises(ValueError):
                make_leaf_config("test", {"hidden_size": 1}, model_family=family)
        with self.assertRaises(ValueError):
            make_leaf_config("test", {"hidden_size": 1, "model_family": "prosst"})


if __name__ == "__main__":
    unittest.main()
