"""Offline extraction checks using explicit residue values and small assay tables."""

from __future__ import annotations

import contextlib
import csv
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from feature_generation.config import ExtractionConfig, MODEL_PRESETS
from feature_generation.checkpoints import MANIFEST_NAME, checkpoint_manifest, checkpoint_path, download
from feature_generation.adapters.esm2 import ESM2Adapter
from feature_generation.cli import main
from feature_generation.inputs import prepare_assay
from feature_generation.runner import extract
from feature_generation.storage import validate_sequence_cache


def write_csv(path, header, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def residue_values(sequence):
    """A transparent, order-sensitive expected value independent of tokenization."""
    return torch.tensor([[ord(amino_acid), position + 1] for position, amino_acid in enumerate(sequence)], dtype=torch.float32)


class FakeAdapter:
    """Yield out-of-order batches to exercise the writer's variant alignment."""

    hidden_size = 2

    def __init__(self, config):
        self.config = config
        self.load_count = 0

    def metadata(self):
        return {
            "model_id": self.config.model_id,
            "revision": self.config.revision,
            "layer": self.config.layer,
            "hidden_size": self.hidden_size,
            "inference_dtype": self.config.inference_dtype,
            "output_dtype": "float32",
        }

    def load(self):
        self.load_count += 1

    def iter_embeddings(self, sequences):
        for start in range(len(sequences), 0, -2):
            indices = list(range(max(0, start - 2), start))[::-1]
            yield indices, torch.stack([residue_values(sequences[index]) for index in indices])


class FakeTokenizer:
    """Include BOS, EOS and a non-special padding token to exercise both masks."""

    def __init__(self):
        self.calls = []
        self.corrupt = False

    def __call__(self, sequences, **kwargs):
        self.calls.append(kwargs)
        input_ids = torch.tensor([[1, *map(ord, sequence), 2, 0] for sequence in sequences])
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(0).long(),
            "special_tokens_mask": ((input_ids == 1) | (input_ids == 2)).long(),
        }

    def convert_ids_to_tokens(self, indices):
        tokens = list(map(chr, indices))
        if self.corrupt:
            tokens[0] = "X"
        return tokens


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_ids, attention_mask, output_hidden_states, return_dict):
        self.calls.append({"grad_enabled": torch.is_grad_enabled(), "hidden_states": output_hidden_states})
        # Values encode the token, its original token position, and the layer.
        positions = torch.arange(input_ids.shape[1], dtype=torch.float64)
        values = (input_ids.double() * 100 + positions).unsqueeze(-1).expand(-1, -1, 320)
        states = tuple(values + layer * 10000 for layer in range(7))
        return SimpleNamespace(last_hidden_state=states[-1], hidden_states=states)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.index = self.raw / "DMS_substitutions.csv"
        self.ids = ("wildtype", "single_a", "single_c", "double")
        self.mutants = ("WT", "A1V", "C2D", "A1V:C2D")
        self.sequences = ("ACD", "VCD", "ADD", "VDD")
        write_csv(self.index, ["DMS_id", "DMS_filename", "target_seq", "seq_len"], [["tiny", "tiny.csv", "ACD", 3]])
        write_csv(self.raw / "tiny.csv", ["variant_id", "mutant", "mutated_sequence", "DMS_score"], [[variant_id, mutant, sequence, index / 2] for index, (variant_id, mutant, sequence) in enumerate(zip(self.ids, self.mutants, self.sequences))])
        self.config = ExtractionConfig(
            model_id="facebook/esm2_t6_8M_UR50D",
            revision="a" * 40,
            cache_id="tiny_test",
            model_dir=self.root / "models",
            output_root=self.root / "features",
            assay_index=self.index,
            layer=6,
            device="cpu",
            max_tokens_per_batch=10,
            max_length=10,
            inference_dtype="float32",
            output_dtype="float32",
            max_output_gib=1.0,
        )
        self.adapters = []

    def factory(self, config):
        adapter = FakeAdapter(config)
        self.adapters.append(adapter)
        return adapter

    @property
    def assay_root(self):
        return self.config.output_root / "tiny"

    @property
    def leaf(self):
        return self.assay_root / "sequence" / "esm2" / self.config.cache_id

    def run_extraction(self, **kwargs):
        return extract(self.config, ["tiny"], adapter_factory=self.factory, **kwargs)

    def test_reordered_batches_roundtrip_without_other_feature_types(self):
        result = self.run_extraction()
        self.assertEqual(result[0]["status"], "completed")
        self.assertEqual(self.adapters[0].load_count, 1)
        payload = torch.load(self.leaf / "features.pt", weights_only=True)
        self.assertEqual(tuple(payload["variant_ids"]), self.ids)
        self.assertEqual(payload["reference_sequence"], "ACD")
        self.assertEqual(tuple(payload["sequence_hashes"]), tuple(hashlib.sha256(sequence.encode()).hexdigest() for sequence in self.sequences))
        torch.testing.assert_close(payload["positions"], torch.tensor([1, 2, 3]))
        torch.testing.assert_close(payload["embeddings"], torch.stack([residue_values(sequence) for sequence in self.sequences]))
        self.assertEqual(payload["embeddings"].dtype, torch.float32)
        with (self.assay_root / "variants.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["variant_id"] for row in rows], list(self.ids))
        self.assertEqual([int(row["source_row_index"]) for row in rows], [0, 1, 2, 3])
        self.assertEqual([float(row["DMS_score"]) for row in rows], [0, 0.5, 1, 1.5])
        self.assertFalse((self.assay_root / "structure").exists())
        self.assertFalse((self.assay_root / "inverse_folding").exists())
        validate_sequence_cache(self.assay_root, self.config.cache_id)

    def test_resume_skips_inference_and_changed_coverage_is_rejected(self):
        self.run_extraction()
        result = self.run_extraction(resume=True)
        self.assertEqual(result[0]["status"], "skipped")
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 1)
        before = (self.assay_root / "variants.csv").read_bytes()
        with self.assertRaises(ValueError):
            self.run_extraction(resume=True, limit=2)
        self.assertEqual((self.assay_root / "variants.csv").read_bytes(), before)

    def test_same_cache_name_cannot_resume_with_changed_layer(self):
        self.run_extraction()
        self.config = replace(self.config, layer=5)
        with self.assertRaises(ValueError):
            self.run_extraction(resume=True)
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 1)

    def test_corrupt_cache_is_regenerated_on_resume(self):
        self.run_extraction()
        path = self.leaf / "features.pt"
        payload = torch.load(path, weights_only=True)
        payload["sequence_hashes"][0] = "incorrect"
        torch.save(payload, path)
        with self.assertRaises(ValueError):
            validate_sequence_cache(self.assay_root, self.config.cache_id)
        self.assertEqual(self.run_extraction(resume=True)[0]["status"], "completed")
        validate_sequence_cache(self.assay_root, self.config.cache_id)

    def test_missing_completion_metadata_is_not_a_complete_cache(self):
        self.run_extraction()
        (self.leaf / "completion.json").unlink()
        self.assertEqual(self.run_extraction(resume=True)[0]["status"], "completed")
        validate_sequence_cache(self.assay_root, self.config.cache_id)

    def test_interrupted_inference_can_retry_without_a_partial_cache(self):
        def interrupted(adapter, sequences):
            yield [0], residue_values(sequences[0]).unsqueeze(0)
            raise RuntimeError("simulated interruption")

        with patch.object(FakeAdapter, "iter_embeddings", interrupted):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                self.run_extraction()
        self.assertFalse((self.leaf / "completion.json").exists())
        result = self.run_extraction(resume=True)
        self.assertEqual(result[0]["status"], "completed")
        validate_sequence_cache(self.assay_root)

    def test_storage_failure_never_exposes_a_completed_leaf(self):
        with patch("feature_generation.storage.torch.save", side_effect=OSError("simulated disk failure")):
            with self.assertRaisesRegex(OSError, "simulated disk failure"):
                self.run_extraction()
        self.assertFalse(self.leaf.exists())
        self.run_extraction(resume=True)
        validate_sequence_cache(self.assay_root)

    def test_size_and_length_guards_fail_before_loading_weights(self):
        original = self.config
        for overrides in ({"max_length": 2}, {"max_output_gib": 1e-12}):
            with self.subTest(overrides=overrides):
                self.config = replace(original, **overrides)
                with self.assertRaises(ValueError):
                    self.run_extraction()
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 0)

    def test_singles_filter_keeps_original_source_rows_and_custom_ids(self):
        inputs = prepare_assay(self.index, "tiny", singles_only=True)
        self.assertEqual(inputs.variant_ids, ("single_a", "single_c"))
        self.assertEqual(inputs.sequences, ("VCD", "ADD"))
        rows = list(csv.DictReader(io.StringIO(inputs.variants_csv())))
        self.assertEqual([int(row["source_row_index"]) for row in rows], [1, 2])

    def test_cli_validation_reports_a_sequence_only_cache(self):
        self.run_extraction()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(["validate", "--features", str(self.assay_root)])
        self.assertEqual(status, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["num_variants"], 4)
        self.assertEqual(report["embedding_dim"], 2)


class AdapterTests(unittest.TestCase):
    def adapter(self, **overrides):
        config = ExtractionConfig(model_id="facebook/esm2_t6_8M_UR50D", device="cpu", max_tokens_per_batch=12, max_length=10, **overrides)
        adapter = ESM2Adapter(config)
        adapter._model = FakeModel()
        adapter._tokenizer = FakeTokenizer()
        adapter._device = torch.device("cpu")
        return adapter

    def test_special_tokens_padding_layers_and_frozen_inference(self):
        sequences = ("ACD", "VCD", "ADD")
        for layer in (0, 3, 6):
            with self.subTest(layer=layer):
                adapter = self.adapter(layer=layer)
                batches = list(adapter.iter_embeddings(sequences))
                self.assertEqual([indices for indices, _ in batches], [[0, 1], [2]])
                actual = torch.cat([values for _, values in batches])
                expected = torch.tensor([[[ord(amino_acid) * 100 + position + 1 + layer * 10000] * 320 for position, amino_acid in enumerate(sequence)] for sequence in sequences], dtype=torch.float32)
                torch.testing.assert_close(actual, expected)
                self.assertEqual(actual.shape, (3, 3, 320))
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(actual.device.type, "cpu")
                self.assertFalse(actual.requires_grad)
                self.assertTrue(all(not call["grad_enabled"] for call in adapter._model.calls))
                self.assertTrue(all(call["hidden_states"] == (layer != 6) for call in adapter._model.calls))
                self.assertTrue(all(call["truncation"] is False for call in adapter._tokenizer.calls))

    def test_tokenizer_must_preserve_every_residue(self):
        adapter = self.adapter()
        adapter._tokenizer.corrupt = True
        with self.assertRaisesRegex(ValueError, "Tokenizer residues"):
            list(adapter.iter_embeddings(["ACD"]))
        self.assertEqual(adapter._model.calls, [])

    def test_invalid_inputs_rejected_before_model_loading(self):
        for sequences in (["A" * 11], ["ACD", "AC"], ["ACX"], []):
            with self.subTest(sequences=sequences):
                adapter = self.adapter()
                with patch.object(adapter, "load", side_effect=AssertionError("must not load")):
                    with self.assertRaises(ValueError):
                        list(adapter.iter_embeddings(sequences))


class ConfigurationTests(unittest.TestCase):
    def test_pins_revisions_and_rejects_unsafe_or_silent_configuration(self):
        self.assertRegex(ExtractionConfig().revision, r"^[0-9a-f]{40}$")
        for overrides in ({"revision": "main"}, {"cache_id": "../elsewhere"}, {"max_length": 1023}, {"layer": 34}, {"output_dtype": "float16"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                ExtractionConfig(**overrides)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps({"max_token_per_batch": 8}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown"):
                ExtractionConfig.load(path)

    def test_cli_requires_explicit_assay_selection(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main(["extract", "--config", "unused.json"])
        self.assertEqual(raised.exception.code, 2)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.config = ExtractionConfig(model_id="facebook/esm2_t6_8M_UR50D", model_dir=Path(temporary.name))
        self.path = checkpoint_path(self.config)
        self.path.mkdir(parents=True)
        self.contents = {
            "model.safetensors": b"small stand-in for model weights",
            "config.json": b'{"model_type": "esm"}',
            "tokenizer_config.json": b'{"tokenizer_class": "EsmTokenizer"}',
            "special_tokens_map.json": b'{"cls_token": "<cls>"}',
            "vocab.txt": b"<cls>\n<pad>\n<eos>\n<unk>\nA\nC\nD\n",
        }
        self.weights_hash = hashlib.sha256(self.contents["model.safetensors"]).hexdigest()
        self.preset = MODEL_PRESETS[self.config.model_id]
        # No network dependency, including on installations without the optional Hub package.
        self.hub = Mock(return_value=str(self.path))
        modules = patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=self.hub)})
        modules.start()
        self.addCleanup(modules.stop)

    def stage(self):
        for name, content in self.contents.items():
            (self.path / name).write_bytes(content)

    def test_complete_pinned_staging_is_verified_and_reused_offline(self):
        self.stage()
        with patch.dict(self.preset, weights_sha256=self.weights_hash):
            self.assertEqual(download(self.config), self.path)
            self.assertEqual(download(self.config), self.path)
            manifest = checkpoint_manifest(self.config)
        self.hub.assert_not_called()
        self.assertEqual(manifest["revision"], self.config.revision)
        self.assertEqual(manifest["files_sha256"], {name: hashlib.sha256(content).hexdigest() for name, content in self.contents.items()})

    def test_bad_staged_weights_never_receive_a_completion_manifest(self):
        self.stage()
        with patch.dict(self.preset, weights_sha256="0" * 64):
            with self.assertRaisesRegex(ValueError, "published.*checkpoint SHA256"):
                download(self.config)
        self.hub.assert_not_called()
        self.assertFalse((self.path / MANIFEST_NAME).exists())

    def test_overlapping_downloads_publish_independent_manifests(self):
        self.stage()
        replace_path = Path.replace
        temporary_paths = []

        def publish_after_other_download(temporary, destination):
            temporary_paths.append(temporary)
            if len(temporary_paths) == 1:
                # A second array task finishes while the first is ready to publish.
                self.assertEqual(download(self.config), self.path)
            return replace_path(temporary, destination)

        with patch.dict(self.preset, weights_sha256=self.weights_hash):
            with patch.object(Path, "replace", publish_after_other_download):
                self.assertEqual(download(self.config), self.path)
            checkpoint_manifest(self.config)
        self.hub.assert_not_called()
        self.assertEqual(len(set(temporary_paths)), 2)
        self.assertEqual(list(self.path.glob("*.tmp")), [])

    def test_failed_manifest_publication_cleans_up_temporary_file(self):
        self.stage()
        with patch.dict(self.preset, weights_sha256=self.weights_hash):
            with patch.object(Path, "replace", side_effect=OSError("simulated publication failure")):
                with self.assertRaisesRegex(OSError, "simulated publication failure"):
                    download(self.config)
        self.assertFalse((self.path / MANIFEST_NAME).exists())
        self.assertEqual(list(self.path.glob("*.tmp")), [])

    def test_incomplete_hub_response_is_reported_without_marking_complete(self):
        (self.path / "config.json").write_bytes(self.contents["config.json"])
        with self.assertRaisesRegex(RuntimeError, "incomplete.*network/TLS"):
            download(self.config)
        self.hub.assert_called_once()
        self.assertEqual(self.hub.call_args.kwargs["revision"], self.config.revision)
        self.assertEqual(self.hub.call_args.kwargs["local_dir"], str(self.path))
        self.assertFalse((self.path / MANIFEST_NAME).exists())

    def test_model_and_tokenizer_corruption_each_fail_offline_verification(self):
        self.stage()
        with patch.dict(self.preset, weights_sha256=self.weights_hash):
            download(self.config)
            for name in ("model.safetensors", "vocab.txt"):
                with self.subTest(name=name):
                    (self.path / name).write_bytes(self.contents[name] + b"corruption")
                    with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                        checkpoint_manifest(self.config)
                    (self.path / name).write_bytes(self.contents[name])
        self.hub.assert_not_called()

    def test_additional_tokenizer_artifact_is_not_accepted_without_verification(self):
        self.stage()
        with patch.dict(self.preset, weights_sha256=self.weights_hash):
            download(self.config)
            (self.path / "tokenizer.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly match"):
                checkpoint_manifest(self.config)

    def test_rehashed_local_weights_still_must_match_the_pinned_checkpoint(self):
        self.stage()
        with patch.dict(self.preset, weights_sha256=self.weights_hash):
            download(self.config)
            altered = b"different weights with a matching local manifest"
            (self.path / "model.safetensors").write_bytes(altered)
            path = self.path / MANIFEST_NAME
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["files_sha256"]["model.safetensors"] = hashlib.sha256(altered).hexdigest()
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "published pinned checkpoint"):
                checkpoint_manifest(self.config)


if __name__ == "__main__":
    unittest.main()
