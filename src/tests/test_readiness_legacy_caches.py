"""Readiness can audit old cache names without rewriting or weakening integrity."""

from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from feature_generation import inverse_storage, readiness as r, score_storage
from tests.test_inverse_pipeline import structure_metadata


class LegacyReadinessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        index = self.workspace / "index.csv"
        index.write_text("DMS_id,DMS_filename,target_seq\ntiny,tiny.csv,ACD\n")
        (self.workspace / "tiny.csv").write_text("mutant,mutated_sequence,DMS_score\nA1V,VCD,1\n")
        self.inputs = r.prepare_assay(index, "tiny", singles_only=True)
        self.configs = {
            kind: replace(r.InverseFoldingConfig.load(r.CONFIG_DIR / filename),
                          assay_index=index, output_root=self.workspace / "outputs")
            for kind, filename in (("features", "proteinmpnn_v48_020_kermut.json"),
                                   ("scores", "proteinmpnn_v48_020.json"))
        }
        self.root = self.configs["features"].output_root / "tiny"
        self.structure = structure_metadata()
        self.backbone = SimpleNamespace(input_signature=self.structure["input_signature"])
        self.ids = {kind: config.cache_id if kind == "features" else r.default_cache_id(config)
                    for kind, config in self.configs.items()}
        self.leaves = {kind: self.write_cache(kind, name.removeprefix("proteingym--"))
                       for kind, name in self.ids.items()}

    def write_cache(self, kind, cache_id):
        config = self.configs[kind]
        metadata = {key: getattr(config, key) for key in
                    ("model_id", "revision", "inference_dtype", "seed")}
        metadata["dtype"] = "float32"
        provenance = {"uses_dms_labels": False, "structure_input_signature": self.structure["input_signature"]}
        if kind == "features":
            metadata.update(conditioning_mode="sequence_conditional", inference_method="conditional_probs",
                            normalization="native_21_amino_acid_tokens",
                            noncanonical_tokens="excluded_without_renormalization",
                            order_aggregation="geometric_mean_of_probabilities", num_orders=config.num_orders)
            saved = inverse_storage.make_leaf_config(cache_id, "proteinmpnn", metadata, self.structure)
            log_probs = torch.zeros(3, 20).log_softmax(-1) - 0.1
            return inverse_storage.write_inverse_cache(self.root, "proteinmpnn", cache_id, self.inputs,
                                                       saved, log_probs.exp(), log_probs, provenance)
        metadata.update(scoring_recipe_version=1, score_definition="negative_mean_sequence_nll",
                        conditioning_mode="sequence_autoregressive", inference_method="forward",
                        decoding_order="shared_random_orders", normalization="native_21_amino_acid_tokens",
                        alphabet="ACDEFGHIKLMNPQRSTVWYX", output_dtype="float32", num_orders=1)
        saved = score_storage.make_score_config(cache_id, "proteinmpnn", metadata, self.structure)
        return score_storage.write_score_cache(self.root, "proteinmpnn", cache_id, self.inputs,
                                               saved, torch.tensor([-1.]), provenance)

    def test_legacy_features_and_scores_pass_without_modifying_files(self):
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                  for path in self.root.rglob("*") if path.is_file()}
        for kind, config in self.configs.items():
            with self.subTest(kind=kind):
                report = r.validate_output(config, "tiny", self.inputs, kind, self.backbone)
                self.assertEqual(report["legacy_cache_id"], self.leaves[kind].name)
                self.assertEqual(Path(report["path"]), self.leaves[kind])
        self.assertEqual({path: (path.read_bytes(), path.stat().st_mtime_ns)
                          for path in self.root.rglob("*") if path.is_file()}, before)

    def test_current_name_takes_priority_when_both_exist(self):
        for kind, config in self.configs.items():
            current = self.write_cache(kind, self.ids[kind])
            with self.subTest(kind=kind):
                report = r.validate_output(config, "tiny", self.inputs, kind, self.backbone)
                self.assertEqual(Path(report["path"]), current)
                self.assertNotIn("legacy_cache_id", report)

    def test_incomplete_current_cache_is_not_hidden_by_legacy(self):
        for kind, config in self.configs.items():
            (self.leaves[kind].parent / self.ids[kind]).mkdir()
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "missing completion.json"):
                r.validate_output(config, "tiny", self.inputs, kind, self.backbone)

    def test_corrupt_current_cache_is_not_hidden_by_legacy(self):
        for kind, config in self.configs.items():
            current = self.write_cache(kind, self.ids[kind])
            (current / "config.json").write_bytes(b"corrupt")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                r.validate_output(config, "tiny", self.inputs, kind, self.backbone)

    def test_legacy_integrity_coverage_and_structure_checks_remain_required(self):
        for kind, config in self.configs.items():
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError, "coverage"):
                    r.validate_output(config, "tiny", SimpleNamespace(input_signature="changed"), kind, self.backbone)
                with self.assertRaisesRegex(ValueError, "different registered structure"):
                    r.validate_output(config, "tiny", self.inputs, kind, SimpleNamespace(input_signature="changed"))
                (self.leaves[kind] / "completion.json").unlink()
                with self.assertRaisesRegex(ValueError, "missing completion.json"):
                    r.validate_output(config, "tiny", self.inputs, kind, self.backbone)

    def test_legacy_name_does_not_override_a_changed_recipe(self):
        changed = replace(self.configs["features"], seed=99)
        with self.assertRaisesRegex(ValueError, "preparation config: seed"):
            r.validate_output(changed, "tiny", self.inputs, "features", self.backbone)

    def test_other_structure_sources_do_not_use_unprefixed_caches(self):
        for kind, cache_id in self.ids.items():
            other_id = cache_id.replace("proteingym--", "protssn_proteingym--", 1)
            config = SimpleNamespace(model_family="proteinmpnn", structure_source="protssn_proteingym")
            with self.subTest(kind=kind):
                self.assertEqual(r.preparation_cache_id(config, self.root, kind, other_id), other_id)


if __name__ == "__main__":
    unittest.main()
