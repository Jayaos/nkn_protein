"""The branch rename preserves saved configurations, covariance and predictions."""

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from config import ExperimentConfig, ModelConfig, TrainingConfig
from data_loading import load_feature_bundle
from feature_generation.structure_inputs import prepare_structure
from gp.training import FittedGP, fit_gp
from tests.test_data import make_bundle, write_csv
from tests.test_structure_inputs import pdb_text


def legacy_state(canonical):
    """Represent an actual pre-rename artifact, including all saved identifiers."""
    state = copy.deepcopy(canonical)
    model = state["model_config"]
    model["enabled_branches"] = tuple(
        "physical_distance" if name == "structure_distance" else name
        for name in model["enabled_branches"]
    )
    model["branches"]["physical_distance"] = model["branches"].pop("structure_distance")
    local = state["train"]["local_features"]
    if "structure_distance" in local:
        local["physical_distance"] = local.pop("structure_distance")
    signature = state["train"]["feature_signature"]
    for key in tuple(signature):
        if key.startswith("structure_distance."):
            signature[key.replace("structure_distance.", "physical_distance.", 1)] = signature.pop(key)
    if "enabled_branches" in signature:
        signature["enabled_branches"] = signature["enabled_branches"].replace("structure_distance", "physical_distance")
    state["parameters"] = {
        key.replace("kernel.branches.structure_distance.", "kernel.branches.physical_distance.", 1): value
        for key, value in state["parameters"].items()
    }
    return state


class StructureDistanceConfigurationCompatibilityTests(unittest.TestCase):
    def test_old_config_loads_and_serializes_with_only_canonical_identifiers(self):
        original = {
            "model": {
                "enabled_branches": ["physical_distance", "sequence_global"],
                "branches": {"physical_distance": {"copies": {"exp": 2}, "widths": [4, 2, 1]}},
            },
        }
        untouched = copy.deepcopy(original)
        config = ExperimentConfig.from_dict(original)
        self.assertEqual(config.model.enabled_branches, ("sequence_global", "structure_distance"))
        self.assertEqual(config.model.branches["structure_distance"].copies, {"exp": 2})
        self.assertEqual(config.model.branches["structure_distance"].widths, (4, 2, 1))
        self.assertEqual(original, untouched)
        saved = json.dumps(config.to_dict())
        self.assertNotIn("physical_distance", saved)
        self.assertEqual(ExperimentConfig.from_dict(json.loads(saved)).to_dict(), config.to_dict())

    def test_aliases_cannot_specify_the_same_branch_twice(self):
        for selection in (("physical_distance", "structure_distance"),
                          ("structure_distance", "physical_distance")):
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, "duplicate"):
                ModelConfig(enabled_branches=selection)
        with self.assertRaisesRegex(ValueError, "both physical_distance and structure_distance"):
            ModelConfig(branches={"physical_distance": {}, "structure_distance": {}})


class StructureDistanceCheckpointCompatibilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        self.features = self.root / "features"
        self.features.mkdir()
        make_bundle(self.features)
        index = self.root / "index.csv"
        write_csv(index, ["DMS_id", "DMS_filename", "target_seq"], [["tiny", "tiny.csv", "ACD"]])
        write_csv(self.root / "tiny.csv", ["mutant", "mutated_sequence", "DMS_score"], [["WT", "ACD", 0]])
        pdb = self.root / "tiny.pdb"
        pdb.write_text(pdb_text(), encoding="utf-8")
        self.manifests = self.root / "manifests"
        prepare_structure(index, "tiny", pdb, output_dir=self.manifests)

    def load_batch(self, branches):
        return load_feature_bundle(
            self.features, enabled_branches=branches,
            score_sources=["sequence/synthetic/v1"], structure_manifest_dir=self.manifests,
        )

    def fitted_state(self, branches):
        model = ModelConfig(
            enabled_branches=branches,
            branches={name: {"widths": (4, 2, 1)} for name in branches},
            final_widths=(4, 2, 1),
        )
        training = TrainingConfig(max_steps=2, min_steps=1, patience=1, tolerance=0)
        fitted = fit_gp(self.load_batch(branches).select([1, 2]), model, training)
        path = self.root / "canonical.pt"
        fitted.save(path)
        return fitted, path, torch.load(path, weights_only=True)

    def assert_canonical_state(self, value):
        if isinstance(value, dict):
            for key, nested in value.items():
                self.assertNotIn("physical_distance", str(key))
                self.assert_canonical_state(nested)
        elif isinstance(value, (tuple, list)):
            for nested in value:
                self.assert_canonical_state(nested)
        elif isinstance(value, str):
            self.assertNotIn("physical_distance", value)

    def test_legacy_checkpoint_matches_canonical_predictions_and_resaves_canonically(self):
        for branches in (("structure_distance",), ("inverse_folding", "sequence_global", "structure_distance")):
            with self.subTest(branches=branches):
                fitted, canonical_path, state = self.fitted_state(branches)
                legacy_path = self.root / "legacy.pt"
                torch.save(legacy_state(state), legacy_path)
                original_digest = hashlib.sha256(legacy_path.read_bytes()).hexdigest()
                legacy = FittedGP.load(legacy_path)
                canonical = FittedGP.load(canonical_path)
                # A fresh loader supplies the new field and provenance names.
                test = replace(self.load_batch(branches).select([0, 3]), targets=None)
                expected = fitted.predict(test)
                for restored in (canonical, legacy):
                    actual = restored.predict(test)
                    torch.testing.assert_close(actual.mean, expected.mean, rtol=1e-12, atol=1e-12)
                    torch.testing.assert_close(actual.latent_covariance, expected.latent_covariance, rtol=1e-12, atol=1e-12)
                    torch.testing.assert_close(actual.observation_covariance, expected.observation_covariance, rtol=1e-12, atol=1e-12)
                    torch.testing.assert_close(restored.model.kernel(test), fitted.model.kernel(test), rtol=1e-12, atol=1e-12)
                    self.assertEqual(restored.train.feature_signature, test.feature_signature)
                    self.assertEqual(restored.model_config.enabled_branches, fitted.model_config.enabled_branches)
                resaved_path = self.root / "resaved.pt"
                legacy.save(resaved_path)
                self.assert_canonical_state(torch.load(resaved_path, weights_only=True))
                self.assertEqual(hashlib.sha256(legacy_path.read_bytes()).hexdigest(), original_digest)

    def test_checkpoint_rejects_ambiguous_feature_signature_and_parameter_aliases(self):
        _, _, canonical = self.fitted_state(("structure_distance",))
        for duplicate in ("feature", "signature", "parameter", "selection"):
            with self.subTest(duplicate=duplicate):
                state = legacy_state(canonical)
                if duplicate == "feature":
                    local = state["train"]["local_features"]
                    local["structure_distance"] = local["physical_distance"]
                elif duplicate == "signature":
                    signature = state["train"]["feature_signature"]
                    signature["structure_distance.coordinates"] = signature["physical_distance.coordinates"]
                elif duplicate == "parameter":
                    parameters = state["parameters"]
                    key = next(key for key in parameters if key.startswith("kernel.branches.physical_distance."))
                    parameters[key.replace("physical_distance", "structure_distance", 1)] = parameters[key]
                else:
                    state["train"]["feature_signature"]["enabled_branches"] = json.dumps(
                        ["physical_distance", "structure_distance"]
                    )
                path = self.root / "ambiguous.pt"
                torch.save(state, path)
                with self.assertRaisesRegex(ValueError, "both|duplicate"):
                    FittedGP.load(path)

    def test_unrelated_custom_feature_signature_is_unchanged(self):
        _, path, state = self.fitted_state(("structure_distance",))
        signature = state["train"]["feature_signature"]
        signature["enabled_branches"] = "custom feature signature, not a JSON selection"
        torch.save(state, path)
        restored = FittedGP.load(path)
        self.assertEqual(restored.train.feature_signature, signature)


if __name__ == "__main__":
    unittest.main()
