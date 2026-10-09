"""ProSST backend alignment and provenance checks without model downloads."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from feature_generation.adapters.prosst import ProSSTAdapter
from feature_generation.prosst_config import PRESET, ProSSTConfig
from feature_generation import prosst_checkpoints


class FakeTokenizer:
    cls_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    alphabet = "ACDEFGHIKLMNPQRSTVWY"

    def __call__(self, sequences, **kwargs):
        self.kwargs = kwargs
        ids = torch.tensor([[1] + [self.alphabet.index(aa) + 3 for aa in sequence] + [2] for sequence in sequences])
        special = torch.zeros_like(ids)
        special[:, 0] = special[:, -1] = 1
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "special_tokens_mask": special}

    def convert_ids_to_tokens(self, ids):
        return [self.alphabet[index - 3] for index in ids]


class FakeBackbone:
    def __init__(self, invalid=None):
        self.calls = []
        self.invalid = invalid

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        hidden = (kwargs["input_ids"] + kwargs["ss_input_ids"]).float().unsqueeze(-1).expand(-1, -1, 768).clone()
        if self.invalid == "nonfinite":
            hidden[:, 1] = float("nan")
        elif self.invalid == "shape":
            hidden = hidden[:, :-1]
        layers = tuple(hidden + index for index in range(13))
        return SimpleNamespace(last_hidden_state=layers[-1], hidden_states=layers)


def ready_adapter(**kwargs):
    adapter = ProSSTAdapter(ProSSTConfig(device="cpu", **kwargs))
    adapter._device = torch.device("cpu")
    adapter._tokenizer = FakeTokenizer()
    adapter._model = SimpleNamespace(prosst=FakeBackbone())
    adapter.set_structure_tokens(torch.tensor([0, 17, 2047]))
    return adapter


class ProSSTBackendTests(unittest.TestCase):
    def test_config_pinned_architecture_and_operational_limit(self):
        config = ProSSTConfig(max_length=2048)
        self.assertEqual(config.model_id, "AI4Protein/ProSST-2048")
        self.assertEqual(config.layer, 12)
        self.assertEqual(PRESET["hidden_size"], 768)
        self.assertEqual(config.max_length, 2048)
        self.assertIn("fixed-backbone", config.cache_id)
        self.assertEqual(config.model_dir.name, "prosst")

    def test_config_rejects_unsupported_recipes(self):
        for values in ({"model_id": "elsewhere/model"}, {"revision": "main"}, {"feature_type": "structure"},
                       {"model_family": "esm2"}, {"cache_id": "../escape"}, {"cache_id": "NUL.txt"},
                       {"layer": True}, {"layer": 13}, {"inference_dtype": "float16"}, {"device": "mps"},
                       {"quantizer_batch_size": 0}, {"max_length": True}, {"max_output_gib": float("nan")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ProSSTConfig(**values)

    def test_config_roundtrip_and_unknown_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            config = ProSSTConfig()
            path.write_text(json.dumps(config.to_dict()))
            self.assertEqual(ProSSTConfig.load(path), config)
            path.write_text(json.dumps({"conditioning_mode": "backbone_only"}))
            with self.assertRaisesRegex(ValueError, "Unknown"):
                ProSSTConfig.load(path)

    def test_structure_tokens_are_raw_indices_copied_on_bind(self):
        adapter = ProSSTAdapter(ProSSTConfig())
        tokens = torch.tensor([0, 17, 2047])
        adapter.set_structure_tokens(tokens)
        tokens[0] = 100
        self.assertEqual(adapter._structure_tokens.tolist(), [0, 17, 2047])
        for invalid in (torch.tensor([]).long(), torch.tensor([1.0]), torch.tensor([[1]]),
                        torch.tensor([-1]), torch.tensor([2048]), [1, 2]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                adapter.set_structure_tokens(invalid)

    def test_invalid_inputs_rejected_before_loading(self):
        for sequences, kwargs, bind in ((["ACD"], {}, False), ([], {}, True), (["AXD"], {}, True),
                                       (["AC"], {}, True), (["ACD"], {"max_length": 2}, True),
                                       (["ACD"], {"max_tokens_per_batch": 4}, True)):
            adapter = ProSSTAdapter(ProSSTConfig(**kwargs))
            adapter.load = Mock(side_effect=AssertionError("should not load"))
            if bind:
                adapter.set_structure_tokens(torch.tensor([0, 1, 2]))
            with self.subTest(sequences=sequences, kwargs=kwargs), self.assertRaises(ValueError):
                list(adapter.iter_embeddings(sequences))
            adapter.load.assert_not_called()

    def test_unmasked_amino_acids_structure_offset_special_tokens_and_batch_budget(self):
        adapter = ready_adapter(max_tokens_per_batch=10)
        results = list(adapter.iter_embeddings(["ACD", "WCD", "AYD"]))
        self.assertEqual([indices for indices, _ in results], [[0, 1], [2]])
        self.assertEqual(tuple(results[0][1].shape), (2, 3, 768))
        self.assertEqual(results[0][1].dtype, torch.float32)
        self.assertEqual(results[0][1].device.type, "cpu")
        first = adapter._model.prosst.calls[0]
        self.assertEqual(first["ss_input_ids"].tolist(), [[1, 3, 20, 2050, 2]] * 2)
        self.assertEqual(first["input_ids"].tolist()[0], [1, 3, 4, 5, 2])
        self.assertEqual(first["input_ids"].tolist()[1], [1, 21, 4, 5, 2])
        self.assertFalse((first["input_ids"] == 24).any())
        self.assertFalse(adapter._tokenizer.kwargs["truncation"])
        self.assertFalse(results[0][1].requires_grad)
        torch.testing.assert_close(results[0][1][0, :, 0], torch.tensor([18., 36., 2067.]))

    def test_intermediate_layer_is_selected(self):
        adapter = ready_adapter(layer=0)
        values = list(adapter.iter_embeddings(["ACD"]))[0][1]
        self.assertTrue(adapter._model.prosst.calls[0]["output_hidden_states"])
        torch.testing.assert_close(values[0, :, 0], torch.tensor([6., 24., 2055.]))

    def test_output_shape_and_nonfinite_values_rejected(self):
        for invalid in ("shape", "nonfinite"):
            adapter = ready_adapter()
            adapter._model.prosst = FakeBackbone(invalid)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                list(adapter.iter_embeddings(["ACD"]))

    def test_tokenizer_residue_alignment_rejected(self):
        adapter = ready_adapter()
        adapter._tokenizer.convert_ids_to_tokens = Mock(return_value=list("WCD"))
        with self.assertRaisesRegex(ValueError, "complete unmasked"):
            list(adapter.iter_embeddings(["ACD"]))
        self.assertEqual(adapter._model.prosst.calls, [])

    def test_metadata_does_not_load_model_or_bind_structure(self):
        adapter = ProSSTAdapter(ProSSTConfig())
        manifest = {"files_sha256": {"model/test": "a" * 64}, "quantizer_revision": "b" * 40}
        with patch("feature_generation.adapters.prosst.checkpoint_manifest", return_value=manifest), \
             patch("feature_generation.adapters.prosst.version", return_value="test"):
            first = adapter.metadata()
            adapter.set_structure_tokens(torch.tensor([0, 1, 2]))
            self.assertEqual(first, adapter.metadata())
        self.assertIsNone(adapter._model)
        self.assertEqual(first["conditioning_mode"], "sequence_and_fixed_reference_structure")
        self.assertEqual(first["amino_acid_masking"], "none")


class ProSSTCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = ProSSTConfig(model_dir=self.temporary.name)
        self.contents = {"model/model.safetensors": b"safe weights", "model/modeling_prosst.py": b"# pinned source\n"}
        self.inventory = {name: {"url": f"https://example.org/{name}", "sha256": hashlib.sha256(value).hexdigest(), "size_bytes": len(value)} for name, value in self.contents.items()}
        self.inventory_patch = patch.object(prosst_checkpoints, "artifacts", return_value=self.inventory)
        self.quantizer_patch = patch.object(prosst_checkpoints, "quantizer_preset", return_value={"revision": "b" * 40, "upstream_repository": "https://example.org/quantizer"})
        self.inventory_patch.start()
        self.quantizer_patch.start()
        self.addCleanup(self.inventory_patch.stop)
        self.addCleanup(self.quantizer_patch.stop)

    def download(self):
        def write_artifact(url, destination, expected):
            name = destination.relative_to(prosst_checkpoints.checkpoint_path(self.config)).as_posix()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.contents[name])
        with patch.object(prosst_checkpoints, "download_verified", side_effect=write_artifact):
            return prosst_checkpoints.download(self.config)

    def test_download_completion_offline_validation_and_reuse(self):
        path = self.download()
        with patch.object(prosst_checkpoints, "download_verified", side_effect=AssertionError("no network")):
            self.assertEqual(prosst_checkpoints.download(self.config), path)
            manifest = prosst_checkpoints.checkpoint_manifest(self.config)
        self.assertEqual(set(manifest["files_sha256"]), set(self.contents))

    def test_missing_manifest_rejects_uncompleted_checkpoint(self):
        with self.assertRaisesRegex(FileNotFoundError, "run download"):
            prosst_checkpoints.checkpoint_manifest(self.config)

    def test_modified_weights_rejected_even_when_manifest_rewritten(self):
        path = self.download()
        (path / "model/model.safetensors").write_bytes(b"bad weights!")
        with self.assertRaisesRegex(ValueError, "checksum"):
            prosst_checkpoints.checkpoint_manifest(self.config)
        manifest_path = path / prosst_checkpoints.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        manifest["files_sha256"]["model/model.safetensors"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "identity"):
            prosst_checkpoints.checkpoint_manifest(self.config)

    def test_extra_source_modules_rejected(self):
        path = self.download()
        (path / "model/shadow.py").write_text("raise RuntimeError('must never execute')")
        with self.assertRaisesRegex(ValueError, "source inventory"):
            prosst_checkpoints.checkpoint_manifest(self.config)


if __name__ == "__main__":
    unittest.main()
