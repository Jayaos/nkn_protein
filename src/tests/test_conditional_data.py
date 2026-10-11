"""Explicit NKN loading of sequence-conditioned ProteinMPNN probabilities."""

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from config import ExperimentConfig
from data_loading import load_feature_bundle
from experiments.cli import _load_batch, main
from tests.test_data import make_bundle


RECIPE = {
    "model_family": "proteinmpnn", "conditioning_mode": "sequence_conditional",
    "inference_method": "conditional_probs", "normalization": "native_21_amino_acid_tokens",
    "noncanonical_tokens": "excluded_without_renormalization",
    "order_aggregation": "geometric_mean_of_probabilities", "num_orders": 10, "seed": 37,
}


class ConditionalDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.features = self.root / "features"
        self.features.mkdir()
        make_bundle(self.features)
        original = self.features / "inverse_folding/synthetic/v1"
        self.leaf = self.features / "inverse_folding/proteinmpnn/conditional"
        self.leaf.mkdir(parents=True)
        self.payload = torch.load(original / "features.pt", weights_only=True)
        self.payload["probabilities"] *= torch.tensor([0.4, 0.7, 0.8])[:, None]
        self.payload["log_probs"] = self.payload["probabilities"].log()
        torch.save(self.payload, self.leaf / "features.pt")
        self.metadata = json.loads((original / "config.json").read_text())
        self.metadata.update(RECIPE, checkpoint="conditional")
        (self.leaf / "config.json").write_text(json.dumps(self.metadata), encoding="utf-8")
        self.config = ExperimentConfig(
            model={"enabled_branches": ["inverse_folding", "sequence_global"]},
            feature_selection={"inverse_folding": "proteinmpnn/conditional"},
            score_sources=["sequence/synthetic/v1"],
            inverse_folding_policy="sequence_conditional",
        )
        self.args = SimpleNamespace(features=self.features, full_assay=False, allow_subset=False,
                                    assay_index=None, singles_only=False)

    def _load(self, policy="sequence_conditional"):
        return load_feature_bundle(
            self.features, feature_selection=self.config.feature_selection,
            score_sources=self.config.score_sources, enabled_branches=self.config.model.enabled_branches,
            inverse_folding_policy=policy,
        )

    def test_config_defaults_validation_and_serialization(self):
        self.assertEqual(ExperimentConfig().inverse_folding_policy, "normalized")
        self.assertEqual(ExperimentConfig.from_dict(self.config.to_dict()), self.config)
        for policy in (None, True, {}, "automatic", "kermut"):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "inverse_folding_policy"):
                ExperimentConfig(inverse_folding_policy=policy)

    def test_opt_in_preserves_exact_values_and_recipe_through_batch_selection(self):
        batch = self._load()
        expected = self.payload["probabilities"].double()[batch.mutation_positions - 1]
        torch.testing.assert_close(batch.local_features["inverse_folding"], expected, rtol=0, atol=0)
        self.assertTrue(torch.all(batch.local_features["inverse_folding"].sum(-1) < 0.9))
        self.assertEqual(json.loads(batch.feature_signature["inverse_folding.probability_recipe"]), RECIPE)
        selected = batch.select([3, 1]).to("cpu")
        self.assertEqual(selected.feature_signature, batch.feature_signature)
        torch.testing.assert_close(selected.local_features["inverse_folding"], expected[[2, 3, 0]], rtol=0, atol=0)
        legacy = self._load("kermut")
        torch.testing.assert_close(legacy.local_features["inverse_folding"], expected, rtol=0, atol=0)
        self.assertEqual(legacy.feature_signature, batch.feature_signature)

    def test_default_policy_still_rejects_raw_conditional_probabilities(self):
        with self.assertRaisesRegex(ValueError, "explicit sequence_conditional or Kermut loader policy"):
            self._load("normalized")
        with self.assertRaisesRegex(ValueError, "inverse_folding_policy"):
            self._load("unvalidated")

    def test_generic_policy_keeps_normalized_input_validation(self):
        batch = load_feature_bundle(self.features, inverse_folding_policy="sequence_conditional")
        torch.testing.assert_close(batch.local_features["inverse_folding"].sum(-1), torch.ones(4).double())
        self.assertNotIn("inverse_folding.probability_recipe", batch.feature_signature)
        path = self.features / "inverse_folding/synthetic/v1/features.pt"
        payload = torch.load(path, weights_only=True)
        payload["probabilities"] *= 0.5
        payload["log_probs"] = payload["probabilities"].log()
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            load_feature_bundle(self.features, inverse_folding_policy="sequence_conditional")

    def test_generic_policy_rejects_incomplete_recipe_and_invalid_raw_values(self):
        for changes in ({"normalization": "canonical_amino_acids_only"}, {"num_orders": 0},
                        {"seed": True}, {"noncanonical_tokens": "excluded_then_renormalized"}):
            (self.leaf / "config.json").write_text(json.dumps({**self.metadata, **changes}), encoding="utf-8")
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self._load()
        (self.leaf / "config.json").write_text(json.dumps(self.metadata), encoding="utf-8")
        for value in (0., -0.1, float("nan"), 2.):
            payload = {**self.payload, "probabilities": self.payload["probabilities"].clone()}
            payload["probabilities"][0, 0] = value
            payload["log_probs"] = payload["probabilities"].log()
            torch.save(payload, self.leaf / "features.pt")
            with self.subTest(value=value), self.assertRaises(ValueError):
                self._load()

    def test_cli_loads_configured_policy_and_records_it_in_resolved_config(self):
        config_path = self.root / "experiment.json"
        config_path.write_text(json.dumps(self.config.to_dict()), encoding="utf-8")
        output = self.root / "validation"
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = main(["validate", "--features", str(self.features), "--config", str(config_path),
                           "--output", str(output)])
        self.assertEqual(status, 0)
        saved = json.loads((output / "resolved_config.json").read_text())
        self.assertEqual(saved["inverse_folding_policy"], "sequence_conditional")
        batch = _load_batch(self.args, self.config)
        torch.testing.assert_close(batch.local_features["inverse_folding"],
                                   self.payload["probabilities"].double()[batch.mutation_positions - 1], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "explicit sequence_conditional"):
            _load_batch(self.args, replace(self.config, inverse_folding_policy="normalized"))

    def test_custom_baseline_loader_keeps_its_existing_config_interface(self):
        batch = self._load()
        baseline_config = SimpleNamespace(custom_configuration=True)
        loader = Mock(return_value=batch)
        self.assertIs(_load_batch(self.args, baseline_config, feature_loader=loader), batch)
        loader.assert_called_once_with(self.features, baseline_config)


if __name__ == "__main__":
    unittest.main()
