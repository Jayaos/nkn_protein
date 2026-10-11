"""Configuration selections and structure provenance exercised through actual files."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from baselines.kermut.config import KermutConfig
from baselines.kermut.data import load_feature_bundle
from fixtures import make_cache


class DataConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.features, self.config, self.batch, self.index = make_cache(self.root)

    def test_config_selects_representation_and_single_or_combined_mean_scores(self):
        for names in (self.batch.score_names[:1], self.batch.score_names[1:], tuple(reversed(self.batch.score_names))):
            with self.subTest(scores=names):
                config = replace(self.config, score_sources=names)
                config_path = self.root / "settings.json"
                config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
                loaded_config = KermutConfig.load(config_path)
                actual = load_feature_bundle(self.features, loaded_config)
                self.assertEqual(actual.score_names, names)
                selected = [self.batch.score_names.index(name) for name in names]
                torch.testing.assert_close(actual.scores, self.batch.scores[:, selected])
                torch.testing.assert_close(actual.global_features, self.batch.global_features)
                self.assertEqual(actual.variant_ids, self.batch.variant_ids)
                self.assertEqual(actual.local_features["structure"].shape, (30, 3))
                expected_x = actual.mutation_positions * 4 + 0.3
                torch.testing.assert_close(actual.local_features["structure"][:, 0], expected_x.double(), atol=2e-6, rtol=0)
                self.assertEqual(actual.feature_signature["structure.coordinates"], "reference_CA_angstrom")

    def test_custom_feature_leaf_selection_changes_signature_and_embeddings(self):
        old_leaf = self.features / "sequence/synthetic/v1"
        new_leaf = self.features / "sequence/synthetic/v2"
        new_leaf.mkdir()
        payload = torch.load(old_leaf / "features.pt", weights_only=True)
        payload["embeddings"] = payload["embeddings"] + 2
        torch.save(payload, new_leaf / "features.pt")
        metadata = json.loads((old_leaf / "config.json").read_text())
        metadata["checkpoint"] = "v2"
        (new_leaf / "config.json").write_text(json.dumps(metadata), encoding="utf-8")
        original = load_feature_bundle(self.features, self.config)
        custom = replace(self.config, feature_selection={"sequence": "synthetic/v2", "inverse_folding": "synthetic/v1"})
        alternate = load_feature_bundle(self.features, custom)
        torch.testing.assert_close(alternate.global_features, original.global_features + 2)
        self.assertNotEqual(alternate.feature_signature, original.feature_signature)
        torch.testing.assert_close(alternate.scores, original.scores)

    def test_missing_or_changed_structure_and_wrong_provenance_are_rejected(self):
        absent = replace(self.config, structure_manifest_dir=str(self.root / "absent"))
        with self.assertRaises((FileNotFoundError, ValueError)):
            load_feature_bundle(self.features, absent)
        inverse_path = self.features / "inverse_folding/synthetic/v1/features.pt"
        payload = torch.load(inverse_path, weights_only=True)
        payload["provenance"] = {"structure_input_signature": "wrong_signature"}
        torch.save(payload, inverse_path)
        with self.assertRaisesRegex(ValueError, "provenance"):
            load_feature_bundle(self.features, self.config)
        payload.pop("provenance")
        torch.save(payload, inverse_path)
        pdb = self.root / "reference.pdb"
        pdb.write_text(pdb.read_text().replace("   4.300", "   5.300"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum|SHA|changed"):
            load_feature_bundle(self.features, self.config)

    def test_json_rejects_unknown_and_invalid_hyperparameters(self):
        invalid = [
            {"unsupported": 1}, {"training": {"max_steps": 10}},
            {"training": {"n_steps": 0}}, {"training": {"n_steps": True}},
            {"training": {"learning_rate": float("nan")}},
            {"training": {"dtype": "float16"}}, {"training": {"noise_prior_scale": 0}},
            {"model": {"raw_alpha": float("inf")}},
            {"score_sources": []}, {"score_sources": ["sequence/synthetic/v1"] * 2},
            {"feature_selection": {"sequence": "../outside"}},
        ]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                KermutConfig.from_dict({**self.config.to_dict(), **change})


if __name__ == "__main__":
    unittest.main()
