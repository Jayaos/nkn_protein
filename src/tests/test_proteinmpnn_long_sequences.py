"""Full ProteinMPNN inputs beyond 1022 residues retain structure and sequence coverage."""

import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from feature_generation.adapters.esm_if1 import ESMIF1Adapter
from feature_generation.adapters.inverse_base import InverseFoldingAdapter
from feature_generation.adapters.proteinmpnn import ProteinMPNNAdapter
from feature_generation.inverse_config import InverseFoldingConfig
from feature_generation.inverse_runner import extract
from feature_generation.inverse_storage import validate_inverse_cache
from feature_generation.score_runner import score
from feature_generation.score_storage import validate_score_cache
from feature_generation.structure_inputs import load_structure, prepare_structure
from tests.test_feature_generation import write_csv
from tests.test_structure_inputs import pdb_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NATIVE_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


class LongNativeModel(torch.nn.Module):
    """Cheap native-shaped outputs with position-dependent distributions."""

    def __init__(self):
        super().__init__()
        self.distribution_inputs = []
        self.score_inputs = []

    @staticmethod
    def logits(residue_idx):
        alphabet = torch.arange(21, dtype=torch.float32)
        return -(alphabet - (residue_idx % 20).unsqueeze(-1)).square() / 20

    def unconditional_probs(self, coords, mask, residue_idx, chain):
        self.distribution_inputs.append((coords.clone(), None, mask.clone(), residue_idx.clone()))
        return self.logits(residue_idx)

    def conditional_probs(self, coords, sequence, mask, chain_mask, residue_idx, chain, randn, backbone_only=False):
        assert not backbone_only
        self.distribution_inputs.append((coords.clone(), sequence.clone(), mask.clone(), residue_idx.clone()))
        return self.logits(residue_idx).log_softmax(-1)

    def forward(self, coords, sequence, mask, chain_mask, residue_idx, chain, randn):
        assert not torch.is_grad_enabled()
        self.score_inputs.append((coords.clone(), sequence.clone(), mask.clone(), residue_idx.clone()))
        return self.logits(residue_idx).log_softmax(-1)


class ProteinMPNNLengthConfigurationTests(unittest.TestCase):
    def test_default_and_both_shipped_presets_have_no_fixed_limit(self):
        default = InverseFoldingConfig()
        self.assertIsNone(default.max_length)
        for name in ("proteinmpnn_v48_020.json", "proteinmpnn_v48_020_kermut.json"):
            path = PROJECT_ROOT / "configs" / "extraction" / name
            with self.subTest(preset=name):
                self.assertIsNone(json.loads(path.read_text())["max_length"])
                config = InverseFoldingConfig.load(path)
                self.assertIsNone(config.max_length)
                restored = InverseFoldingConfig(**config.to_dict())
                self.assertIsNone(restored.max_length)
                self.assertEqual(restored.cache_id, config.cache_id)

    def test_positive_proteinmpnn_limits_remain_optional_and_do_not_identify_a_new_model(self):
        unlimited = InverseFoldingConfig()
        for limit in (1, 1022, 1023, 3423, 10000):
            config = InverseFoldingConfig(max_length=limit)
            self.assertEqual(config.max_length, limit)
            self.assertEqual(config.cache_id, unlimited.cache_id)
        for limit in (0, -1, True, False, 1023.0, "1023", []):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                InverseFoldingConfig(max_length=limit)

    def test_esm_if1_default_and_explicit_limits_are_unchanged(self):
        self.assertEqual(InverseFoldingConfig(model_family="esm_if1").max_length, 1022)
        self.assertEqual(InverseFoldingConfig(model_family="esm_if1", max_length=None).max_length, 1022)
        self.assertEqual(InverseFoldingConfig(model_family="esm_if1", max_length=512).max_length, 512)
        for limit in (1023, 3423, 0, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                InverseFoldingConfig(model_family="esm_if1", max_length=limit)


class LongProteinMPNNPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.length = 1023
        self.reference = "A" * (self.length - 1) + "C"
        self.variant_ids = ("wt", "first", "tail", "double")
        self.sequences = (
            self.reference, "V" + self.reference[1:], self.reference[:-1] + "V",
            "V" + self.reference[1:-1] + "V",
        )
        self.index = self.raw / "DMS_substitutions.csv"
        write_csv(self.index, ["DMS_id", "DMS_filename", "target_seq", "seq_len", "pdb_file", "pdb_range"], [
            ["long", "long.csv", self.reference, self.length, "long.pdb", f"1-{self.length}"],
        ])
        write_csv(self.raw / "long.csv", ["variant_id", "mutant", "mutated_sequence", "DMS_score"], [
            [variant_id, mutation, sequence, float(index)]
            for index, (variant_id, mutation, sequence) in enumerate(zip(
                self.variant_ids, ("WT", "A1V", f"C{self.length}V", f"A1V:C{self.length}V"), self.sequences,
            ))
        ])
        self.config = InverseFoldingConfig(
            model_family="proteinmpnn", device="cpu", output_root=self.root / "features",
            assay_index=self.index, structure_manifest_dir=self.root / "manifests",
            model_dir=self.root / "unused_models", num_orders=2,
        )
        self.pdb = self.root / "long.pdb"
        self.pdb.write_text(pdb_text(self.reference), encoding="utf-8")
        self.manifest = prepare_structure(
            self.index, "long", self.pdb, output_dir=self.config.structure_manifest_dir, chain_id="A",
        )
        self.structure = load_structure(self.manifest, self.reference, require_oxygen=True)
        self.adapters = []

    def factory(self, config):
        # Exercise real ProteinMPNN adapter methods, replacing only native
        # weights/computation and checkpoint provenance for this offline test.
        value = ProteinMPNNAdapter(config)
        value._device = torch.device("cpu")
        value._model = LongNativeModel()
        value._manifest = {"files_sha256": {"weights.pt": "a" * 64}}
        value._checkpoint = Mock(return_value=self.root / "unused_checkpoint")
        self.adapters.append(value)
        return value

    def test_full_long_structure_reaches_backbone_and_conditional_extraction_and_cache(self):
        for mode in ("backbone_only", "sequence_conditional"):
            with self.subTest(conditioning_mode=mode):
                config = replace(self.config, conditioning_mode=mode, cache_id=mode)
                result = extract(config, ["long"], adapter_factory=self.factory)[0]
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["shape"], [self.length, 20])
                model = self.adapters[-1]._model
                self.assertEqual(len(model.distribution_inputs), 2 if mode == "sequence_conditional" else 1)
                for coords, tokens, mask, residue_idx in model.distribution_inputs:
                    torch.testing.assert_close(coords[0], self.structure.coords)
                    torch.testing.assert_close(mask, torch.ones(1, self.length))
                    torch.testing.assert_close(residue_idx, torch.arange(self.length)[None])
                    if tokens is not None:
                        self.assertEqual(tokens.shape, (1, self.length))
                        self.assertEqual(int(tokens[0, -1]), NATIVE_ALPHABET.index("C"))
                payload = torch.load(Path(result["path"]) / "features.pt", weights_only=True)
                self.assertEqual(payload["reference_sequence"], self.reference)
                torch.testing.assert_close(payload["positions"], torch.arange(1, self.length + 1))
                self.assertEqual(payload["probabilities"].shape, (self.length, 20))
                self.assertTrue(torch.isfinite(payload["probabilities"][-1]).all())
                validate_inverse_cache(self.config.output_root / "long", "proteinmpnn", config.cache_id)

    def test_full_variant_sequences_and_tail_mutations_reach_scoring_and_saved_scores(self):
        result = score(self.config, ["long"], batch_size=2, num_orders=2, adapter_factory=self.factory)[0]
        self.assertEqual(result["status"], "completed")
        model = self.adapters[-1]._model
        self.assertEqual(len(model.score_inputs), 4)
        for batch_index in (0, 1):
            coords, tokens, mask, residue_idx = model.score_inputs[batch_index * 2]
            self.assertEqual(coords.shape, (2, self.length, 4, 3))
            torch.testing.assert_close(coords[0], self.structure.coords)
            torch.testing.assert_close(mask, torch.ones(2, self.length))
            torch.testing.assert_close(residue_idx, torch.arange(self.length)[None].expand(2, -1))
            decoded = ["".join(NATIVE_ALPHABET[int(index)] for index in row) for row in tokens]
            self.assertEqual(tuple(decoded), self.sequences[batch_index * 2:batch_index * 2 + 2])
        with (Path(result["path"]) / "mean_score.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["variant_id"] for row in rows], list(self.variant_ids))
        actual = torch.tensor([float(row["score"]) for row in rows])
        alphabet = torch.arange(21, dtype=torch.float32)
        log_probs = (-(alphabet - (torch.arange(self.length) % 20)[:, None]).square() / 20).log_softmax(-1)
        expected = torch.stack([
            log_probs[torch.arange(self.length), [NATIVE_ALPHABET.index(aa) for aa in sequence]].mean()
            for sequence in self.sequences
        ])
        torch.testing.assert_close(actual, expected)
        self.assertNotEqual(float(actual[0]), float(actual[2]))
        validate_score_cache(self.config.output_root / "long", "proteinmpnn", Path(result["path"]).name)

    def test_optional_finite_limit_rejects_before_structure_loading_and_allows_larger_limit(self):
        limited = replace(self.config, max_length=1022)
        for operation in (extract, score):
            loader = "feature_generation.inverse_runner.load_structure" if operation is extract else "feature_generation.score_runner.load_structure"
            with self.subTest(operation=operation.__name__), patch(loader) as structure_loader:
                with self.assertRaisesRegex(ValueError, "max_length=1022"):
                    operation(limited, ["long"], adapter_factory=self.factory)
                structure_loader.assert_not_called()
                self.adapters[-1]._checkpoint.assert_not_called()
                self.assertFalse(self.adapters[-1]._model.distribution_inputs)
                self.assertFalse(self.adapters[-1]._model.score_inputs)
        finite = replace(self.config, max_length=self.length)
        result = extract(finite, ["long"], adapter_factory=self.factory)[0]
        self.assertEqual(result["shape"], [self.length, 20])

    def test_actual_coordinate_validation_preserves_long_tail_and_rejects_missing_backbone(self):
        value = InverseFoldingAdapter(self.config)
        coords = value._coords(self.structure, 4)
        self.assertEqual(coords.shape, (self.length, 4, 3))
        torch.testing.assert_close(coords[-1], self.structure.coords[-1])
        limited = InverseFoldingAdapter(replace(self.config, max_length=1022))
        with self.assertRaisesRegex(ValueError, "max_length=1022"):
            limited._coords(self.structure, 4)
        for bad_coords in (self.structure.coords[:-1], self.structure.coords.clone()):
            if len(bad_coords) == self.length:
                bad_coords[-1, 3, 0] = float("nan")
            with self.subTest(shape=bad_coords.shape), self.assertRaises(ValueError):
                value._coords(SimpleNamespace(reference_sequence=self.reference, coords=bad_coords), 4)

    def test_esm_if1_still_rejects_long_reference_before_loading_weights_or_structure(self):
        config = InverseFoldingConfig(model_family="esm_if1", device="cpu")
        value = ESMIF1Adapter(config)
        value.load = Mock(side_effect=AssertionError("long ESM-IF1 reference must fail before load"))
        with self.assertRaisesRegex(ValueError, "max_length=1022"):
            value.predict(self.structure)
        value.load.assert_not_called()
        config = replace(config, assay_index=self.index, output_root=self.config.output_root,
                         structure_manifest_dir=self.config.structure_manifest_dir)
        with patch("feature_generation.inverse_runner.load_structure") as structure_loader:
            with self.assertRaisesRegex(ValueError, "max_length=1022"):
                extract(config, ["long"], adapter_factory=lambda selected: value)
            structure_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
