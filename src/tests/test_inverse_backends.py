"""Semantic checks for native inverse folding adapters; no model download required."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from feature_generation.adapters.inverse_base import InverseFoldingAdapter
from feature_generation.adapters.proteinmpnn import ProteinMPNNAdapter
from feature_generation.adapters.esm_if1 import ESMIF1Adapter, import_pinned_esm
from feature_generation.downloads import download_verified
from feature_generation.inverse_config import ALPHABET, InverseFoldingConfig
from feature_generation import inverse_checkpoints


def structure(sequence="ACD"):
    return SimpleNamespace(reference_sequence=sequence, coords=torch.arange(len(sequence)*12).reshape(len(sequence), 4, 3).float())


class InverseBackendTests(unittest.TestCase):
    def test_presets_separate_native_conditioning_and_model_locations(self):
        mpnn = InverseFoldingConfig()
        esm = InverseFoldingConfig(model_family="esm_if1")
        self.assertEqual(mpnn.conditioning_mode, "backbone_only")
        self.assertEqual(esm.conditioning_mode, "reference_prefix")
        self.assertEqual(mpnn.model_dir.name, "proteinmpnn")
        self.assertEqual(esm.model_dir.name, "esm_if1")
        self.assertNotEqual(mpnn.cache_id, esm.cache_id)

    def test_rejects_unsupported_recipes_and_unsafe_cache_paths(self):
        for kwargs in ({"conditioning_mode": "reference_prefix"}, {"revision": "main"},
                       {"cache_id": "../escape"}, {"cache_id": "CON"}, {"seed": True},
                       {"inference_dtype": "float16"}, {"model_family": "esm_if1", "max_length": 1023},
                       {"feature_type": "sequence"}, {"model_family": []}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                InverseFoldingConfig(**kwargs)

    def test_config_json_rejects_unknown_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps({"temperature": 2.0}))
            with self.assertRaisesRegex(ValueError, "Unknown"):
                InverseFoldingConfig.load(path)

    def test_canonical_vocabulary_mapping_and_renormalization(self):
        values = torch.arange(25).float().repeat(3, 1)
        values[:, 24] = 10000  # Excluded token must not consume canonical probability mass.
        indices = list(reversed(range(20)))
        output = InverseFoldingAdapter._probabilities(values, indices, 3)
        self.assertEqual(output["alphabet"], ALPHABET)
        torch.testing.assert_close(output["probabilities"], values[:, indices].softmax(-1))
        torch.testing.assert_close(output["probabilities"].sum(-1), torch.ones(3))
        torch.testing.assert_close(output["log_probs"].exp(), output["probabilities"])
        self.assertEqual(output["probabilities"].dtype, torch.float32)
        self.assertEqual(output["probabilities"].device.type, "cpu")

    def test_nonfinite_and_underflow_outputs_rejected(self):
        for value in (float("nan"), float("inf"), -10000):
            values = torch.zeros(3, 20)
            values[0, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                InverseFoldingAdapter._probabilities(values, list(range(20)), 3)

    def test_missing_oxygen_only_accepted_by_esm_if(self):
        inputs = structure()
        inputs.coords[0, 3] = float("nan")
        InverseFoldingAdapter(InverseFoldingConfig())._coords(inputs, 3)
        with self.assertRaisesRegex(ValueError, "N, CA, C, O"):
            InverseFoldingAdapter(InverseFoldingConfig())._coords(inputs, 4)

    def test_invalid_inputs_rejected_before_model_load(self):
        adapter = ProteinMPNNAdapter(InverseFoldingConfig(max_length=2))
        adapter.load = Mock(side_effect=AssertionError("must not load"))
        with self.assertRaisesRegex(ValueError, "max_length"):
            adapter.predict(structure())
        adapter.load.assert_not_called()

    def test_mpnn_backbone_only_does_not_receive_reference_amino_acids(self):
        adapter = ProteinMPNNAdapter(InverseFoldingConfig())
        adapter._device = torch.device("cpu")
        adapter._model = SimpleNamespace(unconditional_probs=Mock(return_value=torch.arange(21).float().repeat(1, 3, 1)))
        adapter.load = lambda: None
        first = adapter.predict(structure("ACD"))
        second = adapter.predict(structure("WYG"))
        torch.testing.assert_close(first["probabilities"], second["probabilities"])
        coords, mask, residue_idx, chain = adapter._model.unconditional_probs.call_args.args
        self.assertEqual(tuple(coords.shape), (1, 3, 4, 3))
        torch.testing.assert_close(residue_idx, torch.tensor([[0, 1, 2]]))
        torch.testing.assert_close(chain, torch.ones(1, 3, dtype=torch.long))
        torch.testing.assert_close(mask, torch.ones(1, 3))

    def test_esm_if_prefix_shift_and_canonical_token_mapping(self):
        adapter = ESMIF1Adapter(InverseFoldingConfig(model_family="esm_if1"))
        adapter._device = torch.device("cpu")
        vocabulary = "<bos>", *reversed(ALPHABET), "<unk>"
        adapter._alphabet = SimpleNamespace(get_idx=vocabulary.index, get_tok=lambda index: vocabulary[index])
        def converter(batch, device):
            coords, confidence, seq = batch[0]
            self.assertEqual(coords.shape, (3, 3, 3))
            self.assertIsNone(confidence)
            tokens = torch.tensor([[0] + [vocabulary.index(aa) for aa in seq]])
            return torch.as_tensor(coords)[None], torch.ones(1, 3), [seq], tokens, torch.zeros(1, 3).bool()
        adapter._converter = converter
        def forward(coords, padding, confidence, previous):
            self.assertEqual(previous.tolist(), [[0, vocabulary.index("A"), vocabulary.index("C")]])
            return torch.arange(len(vocabulary)).float().repeat(1, 3, 1).transpose(1, 2), None
        adapter._model = forward
        adapter.load = lambda: None
        output = adapter.predict(structure())
        expected = torch.tensor([vocabulary.index(aa) for aa in ALPHABET]).float().softmax(-1)
        torch.testing.assert_close(output["probabilities"], expected.repeat(3, 1))

    def test_loading_models_preserves_cpu_and_cuda_random_states(self):
        class NativeModel(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(1))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            for family, adapter_class in (("proteinmpnn", ProteinMPNNAdapter), ("esm_if1", ESMIF1Adapter)):
                config = InverseFoldingConfig(model_family=family, device="cpu", seed=12345)
                adapter = adapter_class(config)
                adapter._checkpoint = lambda: path
                checkpoint = {"num_edges": 48, "model_state_dict": {"weight": torch.ones(1)},
                              "args": SimpleNamespace(arch="vt_medium_with_invariant_gvp"), "model": {}}
                name = f"_nkn_proteinmpnn_{config.revision}"
                upstream = ModuleType(name)
                upstream.__file__ = str(path / "source" / "protein_mpnn_utils.py")
                upstream.ProteinMPNN = NativeModel
                fake_esm = SimpleNamespace(
                    pretrained=SimpleNamespace(_load_model_and_alphabet_core_v1=lambda data: (NativeModel(), None, {"weight": torch.ones(1)})),
                    inverse_folding=SimpleNamespace(util=SimpleNamespace(CoordBatchConverter=lambda alphabet: None)),
                )
                cpu_state = torch.get_rng_state().clone()
                cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
                with self.subTest(family=family), patch.dict("sys.modules", {name: upstream}), patch("torch.load", return_value=checkpoint), patch("feature_generation.adapters.esm_if1.import_pinned_esm", return_value=fake_esm):
                    adapter.load()
                    self.assertTrue(torch.equal(cpu_state, torch.get_rng_state()))
                    if cuda_states:
                        self.assertTrue(all(torch.equal(first, second) for first, second in zip(cuda_states, torch.cuda.get_rng_state_all())))
                    self.assertFalse(adapter._model.training)
                    self.assertFalse(adapter._model.weight.requires_grad)

    def test_conflicting_esm_package_rejected_before_import(self):
        conflicting = ModuleType("esm")
        conflicting.__file__ = str(Path(tempfile.gettempdir()) / "unrelated_esm" / "__init__.py")
        with tempfile.TemporaryDirectory() as temporary, patch.dict("sys.modules", {"esm": conflicting}):
            with self.assertRaisesRegex(RuntimeError, "fresh CLI"):
                import_pinned_esm(Path(temporary))

    def test_verified_download_reuses_matching_file_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.pt"
            path.write_bytes(b"verified model")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch("feature_generation.downloads.urlopen", side_effect=AssertionError("offline")):
                self.assertEqual(download_verified("https://example.com/model", path, digest), path)

    def test_verified_download_requires_https_and_expected_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            for url, digest in (("http://example.com/model", "a"*64), ("https://example.com/model", "bad")):
                with self.subTest(url=url, digest=digest), self.assertRaises(ValueError):
                    download_verified(url, Path(temporary) / "artifact", digest)

    def test_checkpoint_verification_rejects_mutated_source_and_extra_module(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = InverseFoldingConfig(model_dir=Path(temporary))
            path = inverse_checkpoints.checkpoint_path(config)
            artifacts = {}
            for name, content in {"source/native.py": b"x=1", "weights.pt": b"weights"}.items():
                item = path / name
                item.parent.mkdir(parents=True, exist_ok=True)
                item.write_bytes(content)
                artifacts[name] = {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content), "url": "https://example.com/" + name}
            preset = {"upstream_repository": "https://example.com/repo", "weights_filename": "weights.pt", "artifacts": artifacts}
            with patch.dict(inverse_checkpoints.PRESETS, {"proteinmpnn": preset}):
                inverse_checkpoints.download(config)
                inverse_checkpoints.checkpoint_manifest(config)
                extra = path / "source" / "injected.py"
                extra.write_text("x=2")
                with self.assertRaisesRegex(ValueError, "inventory"):
                    inverse_checkpoints.checkpoint_manifest(config)
                extra.unlink()
                (path / "source" / "native.py").write_text("x=2")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    inverse_checkpoints.checkpoint_manifest(config)


if __name__ == "__main__":
    unittest.main()
