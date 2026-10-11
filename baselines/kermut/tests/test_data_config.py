"""Configuration selections and structure provenance exercised through actual files."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from baselines.kermut.config import KermutConfig
from baselines.kermut.data import load_feature_bundle
from data_loading import load_feature_bundle as load_normalized_features
from fixtures import CONDITIONAL_PROBABILITY_RECIPE, make_cache


class DataConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.features, self.config, self.batch, self.index = make_cache(self.root)

    def _conditional_cache(self):
        original = self.features / "inverse_folding/synthetic/v1"
        leaf = self.features / "inverse_folding/proteinmpnn/conditional"
        leaf.mkdir(parents=True)
        payload = torch.load(original / "features.pt", weights_only=True)
        payload["probabilities"] *= torch.linspace(0.45, 0.85, 5)[:, None]
        payload["log_probs"] = payload["probabilities"].log()
        torch.save(payload, leaf / "features.pt")
        metadata = json.loads((original / "config.json").read_text())
        metadata.update(CONDITIONAL_PROBABILITY_RECIPE, checkpoint="conditional")
        (leaf / "config.json").write_text(json.dumps(metadata), encoding="utf-8")
        config = replace(self.config, feature_selection={
            **self.config.feature_selection, "inverse_folding": "proteinmpnn/conditional",
        })
        return leaf, payload, metadata, config

    def test_conditional_probabilities_are_preserved_and_require_kermut_policy(self):
        _, payload, _, config = self._conditional_cache()
        actual = load_feature_bundle(self.features, config)
        expected = payload["probabilities"][actual.mutation_positions - 1]
        torch.testing.assert_close(actual.local_features["inverse_folding"], expected, atol=0, rtol=0)
        self.assertTrue((actual.local_features["inverse_folding"].sum(-1) < 0.9).all())
        recipe = json.loads(actual.feature_signature["inverse_folding.probability_recipe"])
        self.assertEqual(recipe, CONDITIONAL_PROBABILITY_RECIPE)
        with self.assertRaisesRegex(ValueError, "explicit Kermut loader policy"):
            load_normalized_features(
                self.features, feature_selection=config.feature_selection,
                score_sources=config.score_sources, enabled_branches=config.model.enabled_branches,
            )

    def test_conditional_recipe_metadata_and_probability_mass_are_validated(self):
        leaf, payload, metadata, config = self._conditional_cache()
        for changes in (
            {"order_aggregation": "arithmetic_mean_of_probabilities"},
            {"normalization": "canonical_amino_acids_only"},
            {"num_orders": 0}, {"num_orders": True}, {"seed": -1}, {"seed": True},
        ):
            with self.subTest(changes=changes):
                (leaf / "config.json").write_text(json.dumps({**metadata, **changes}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_feature_bundle(self.features, config)
        (leaf / "config.json").write_text(json.dumps(metadata), encoding="utf-8")
        probabilities = payload["probabilities"].clone()
        for invalid in ("zero", "negative", "nonfinite", "excess_mass", "wrong_log_probs"):
            with self.subTest(invalid=invalid):
                values = probabilities.clone()
                if invalid == "zero":
                    values[0, 0] = 0
                elif invalid == "negative":
                    values[0, 0] = -0.1
                elif invalid == "nonfinite":
                    values[0, 0] = float("nan")
                elif invalid == "excess_mass":
                    values[0] /= values[0].sum() / 1.1
                payload["probabilities"] = values
                payload["log_probs"] = values.log() + (1 if invalid == "wrong_log_probs" else 0)
                torch.save(payload, leaf / "features.pt")
                with self.assertRaises(ValueError):
                    load_feature_bundle(self.features, config)

    def test_kermut_policy_keeps_existing_normalized_caches_strict(self):
        path = self.features / "inverse_folding/synthetic/v1/features.pt"
        payload = torch.load(path, weights_only=True)
        payload["probabilities"] *= 0.5
        payload["log_probs"] = payload["probabilities"].log()
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            load_feature_bundle(self.features, self.config)

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
