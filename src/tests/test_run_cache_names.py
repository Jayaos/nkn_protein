"""Training consumers select preparation recipes and retain inputs after migration."""

import contextlib
import io
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import torch

from baselines.kermut.config import KermutConfig
from baselines.kermut.data import load_feature_bundle as load_kermut
from config import ExperimentConfig
from data_loading import load_feature_bundle as load_nkn
from feature_generation.inverse_config import InverseFoldingConfig
from feature_generation.inverse_runner import extract
from feature_generation.prosst_config import ProSSTConfig
from feature_generation.score_runner import default_cache_id, score
from tests import test_migrate_structure_cache_names as migration_tests
from tests.test_inverse_pipeline import FakeConditionalAdapter
from tests.test_mean_score_pipeline import FakeScoreAdapter


ROOT = Path(__file__).resolve().parents[2]


class RunCacheRecipeTests(unittest.TestCase):
    def test_all_launcher_presets_select_current_structure_model_recipes(self):
        extraction = ROOT / "configs/extraction"
        configs = [InverseFoldingConfig.load(extraction / filename) for filename in (
            "proteinmpnn_v48_020.json", "proteinmpnn_v48_020_kermut.json", "esm_if1_142m.json")]
        configs.extend(ProSSTConfig.load(extraction / filename) for filename in (
            "prosst_2048.json", "prosst_2048_singles.json"))
        features = {f"{config.model_family}/{config.cache_id}" for config in configs}
        scores = {f"scores/{config.model_family}/{default_cache_id(config)}" for config in configs}
        models = {config.model_family for config in configs}
        presets = set()
        for folder in ("run_nkn", "run_kermut"):
            for script in (ROOT / "sbatch_run" / folder).glob("*.sbatch"):
                paths = re.findall(r"configs/[A-Za-z0-9_./-]+\.json", script.read_text())
                presets.update(ROOT / path for path in paths if (ROOT / path).is_file()
                               and not path.startswith("configs/extraction/"))
        self.assertTrue(presets)
        for path in sorted(presets):
            with self.subTest(preset=path.name):
                config = json.loads(path.read_text())
                for selection in config.get("feature_selection", {}).values():
                    if selection.split("/")[0] in models:
                        self.assertIn(selection, features)
                for selection in config.get("score_sources") or []:
                    if selection.split("/")[1] in models:
                        self.assertIn(selection, scores)
        self.assertIn(KermutConfig().feature_selection["inverse_folding"], features)


class MigratedTrainingInputTests(unittest.TestCase):
    def setUp(self):
        self.fixture = migration_tests.StructureCacheMigrationTests(
            "test_all_four_cache_kinds_preserve_values_and_update_default_descriptors")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.root = self.fixture.root
        prosst = self.fixture.fixture.config
        inverse = InverseFoldingConfig(
            model_family="proteinmpnn", conditioning_mode="sequence_conditional", num_orders=10,
            seed=37, cache_id="proteinmpnn_features", output_root=prosst.output_root,
            assay_index=prosst.assay_index, structure_manifest_dir=prosst.structure_manifest_dir,
            model_dir=self.fixture.fixture.root / "proteinmpnn_models", device="cpu")
        with patch("feature_generation.inverse_runner.load_structure", side_effect=self.fixture.load_structure):
            extract(inverse, ["tiny"], adapter_factory=FakeConditionalAdapter)
        with patch("feature_generation.score_runner.load_structure", side_effect=self.fixture.load_structure):
            score(inverse, ["tiny"], adapter_factory=FakeScoreAdapter, cache_id="proteinmpnn_scores")

    def load(self, consumer, family, *, historical):
        prefix = migration_tests.migration.PREFIX if historical else ""
        inverse_prefix = prefix if family == "esm_if1" else ""
        selections = {
            "sequence": f"prosst/{prefix}prosst_features",
            "inverse_folding": f"{family}/{inverse_prefix}{family}_features",
        }
        scores = [f"scores/prosst/{prefix}prosst_scores",
                  f"scores/{family}/{inverse_prefix}{family}_scores"]
        if consumer == "kermut":
            return load_kermut(self.root, KermutConfig(
                feature_selection=selections, score_sources=scores,
                model={"use_structure_distance": False}))
        config = ExperimentConfig(
            feature_selection=selections, score_sources=scores,
            model={"enabled_branches": ["sequence_global", "inverse_folding"]},
            inverse_folding_policy="sequence_conditional" if family == "proteinmpnn" else "normalized",
            structure_consistency="per_cache")
        return load_nkn(
            self.root, feature_selection=config.feature_selection, score_sources=config.score_sources,
            enabled_branches=config.model.enabled_branches, inverse_folding_policy=config.inverse_folding_policy,
            structure_consistency=config.structure_consistency)

    def test_both_consumers_load_identical_inputs_after_migration(self):
        cases = [(consumer, family) for consumer in ("nkn", "kermut")
                 for family in ("esm_if1", "proteinmpnn")]
        original = {case: self.load(*case, historical=True) for case in cases}
        with contextlib.redirect_stdout(io.StringIO()):
            result = migration_tests.migration.migrate_assay(self.root)
        self.assertEqual(result["migrated"], 4)
        for case in cases:
            with self.subTest(consumer=case[0], inverse_model=case[1]):
                before, after = original[case], self.load(*case, historical=False)
                self.assertEqual(before.variant_ids, after.variant_ids)
                self.assertEqual(before.sequences, after.sequences)
                for name in ("targets", "scores", "global_features", "owners", "mutation_positions"):
                    torch.testing.assert_close(getattr(before, name), getattr(after, name), rtol=0, atol=0)
                self.assertEqual(before.local_features.keys(), after.local_features.keys())
                for name in before.local_features:
                    torch.testing.assert_close(before.local_features[name], after.local_features[name], rtol=0, atol=0)
                self.assertTrue(all(migration_tests.migration.PREFIX not in name for name in after.score_names))


if __name__ == "__main__":
    unittest.main()
