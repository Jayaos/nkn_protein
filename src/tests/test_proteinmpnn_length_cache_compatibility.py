"""Relaxing a ProteinMPNN admission limit preserves verified cache identity."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from feature_generation.inputs import AssayInput
from feature_generation import inverse_storage, score_storage
from tests.test_inverse_pipeline import structure_metadata


class ProteinMPNNLengthCacheCompatibilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "tiny"
        self.inputs = AssayInput(
            assay_id="tiny", reference_sequence="ACD",
            rows=[
                {"variant_id": "wt", "source_row_index": 0, "mutant": "WT", "mutated_sequence": "ACD", "DMS_score": 0},
                {"variant_id": "mutant", "source_row_index": 1, "mutant": "A1V", "mutated_sequence": "VCD", "DMS_score": 1},
            ],
            source_assay_path=Path("tiny.csv"), source_assay_sha256="a" * 64,
            source_index_sha256="b" * 64, coverage={"num_variants": 2, "source_row_count": 2},
        )
        self.structure = structure_metadata()
        self.provenance = {"uses_dms_labels": False, "structure_input_signature": self.structure["input_signature"]}
        common = {
            "model_id": "v_48_020", "revision": "c" * 40, "checkpoint_sha256": "d" * 64,
            "max_length": 1022, "truncation": "reject", "inference_dtype": "float32", "seed": 37,
        }
        self.inverse_config = inverse_storage.make_leaf_config("conditional", "proteinmpnn", {
            **common, "conditioning_mode": "sequence_conditional", "inference_method": "conditional_probs",
            "normalization": "native_21_amino_acid_tokens", "noncanonical_tokens": "excluded_without_renormalization",
            "order_aggregation": "geometric_mean_of_probabilities", "num_orders": 10,
        }, self.structure)
        self.log_probs = torch.zeros(3, 20).log_softmax(-1) - 0.1
        self.inverse_leaf = inverse_storage.write_inverse_cache(
            self.root, "proteinmpnn", "conditional", self.inputs, self.inverse_config,
            self.log_probs.exp(), self.log_probs, self.provenance,
        )
        self.score_config = score_storage.make_score_config("scores", "proteinmpnn", {
            **common, "scoring_recipe_version": 1, "score_definition": "negative_mean_sequence_nll",
            "conditioning_mode": "sequence_autoregressive", "inference_method": "forward",
            "decoding_order": "shared_random_orders", "normalization": "native_21_amino_acid_tokens",
            "alphabet": "ACDEFGHIKLMNPQRSTVWYX", "output_dtype": "float32", "num_orders": 1,
        }, self.structure)
        self.scores = torch.tensor([-1., -2.])
        self.score_leaf = score_storage.write_score_cache(
            self.root, "proteinmpnn", "scores", self.inputs, self.score_config, self.scores, self.provenance,
        )

    def targets(self):
        return (
            (inverse_storage.check_existing, self.inverse_config, self.inverse_leaf, "conditional"),
            (score_storage.check_existing, self.score_config, self.score_leaf, "scores"),
        )

    def test_larger_or_unlimited_cap_reuses_both_caches_without_rewriting_files(self):
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        for check, config, leaf, cache_id in self.targets():
            for cap in (None, 2048, 3423):
                with self.subTest(cache=cache_id, cap=cap):
                    self.assertTrue(check(self.root, "proteinmpnn", cache_id, self.inputs, {**config, "max_length": cap}, resume=True))
            self.assertEqual(json.loads((leaf / "config.json").read_text())["max_length"], 1022)
        self.assertEqual({path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}, before)
        inverse_storage.validate_inverse_cache(self.root, "proteinmpnn", "conditional")
        score_storage.validate_score_cache(self.root, "proteinmpnn", "scores")

    def test_inference_and_nested_structure_changes_still_require_new_cache_ids(self):
        for check, config, _, cache_id in self.targets():
            changes = (
                {"seed": 99}, {"num_orders": config["num_orders"] + 1}, {"revision": "e" * 40},
                {"inference_dtype": "float16"}, {"truncation": "prefix"},
                {"structure": {**config["structure"], "max_length": None}},
            )
            for change in changes:
                with self.subTest(cache=cache_id, change=list(change)), self.assertRaisesRegex(ValueError, "new cache ID"):
                    check(self.root, "proteinmpnn", cache_id, self.inputs, {**config, "max_length": None, **change}, resume=True)

    def test_saved_max_length_remains_covered_by_original_integrity_signatures(self):
        for check, config, leaf, cache_id in self.targets():
            with self.subTest(cache=cache_id):
                config_path = leaf / "config.json"
                original_config = config_path.read_bytes()
                completion_path = leaf / "completion.json"
                original_completion = completion_path.read_bytes()
                config_path.write_text(json.dumps({**config, "max_length": None}), encoding="utf-8")
                # Even a resealed file checksum cannot bypass the original
                # completion/config signature (or feature payload signature).
                completion = json.loads(original_completion)
                completion["files_sha256"]["config.json"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
                completion_path.write_text(json.dumps(completion), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    check(self.root, "proteinmpnn", cache_id, self.inputs, {**config, "max_length": None}, resume=True)
                config_path.write_bytes(original_config)
                completion_path.write_bytes(original_completion)

    def test_input_coverage_and_explicit_resume_requirement_are_unchanged(self):
        changed_inputs = copy.deepcopy(self.inputs)
        changed_inputs.rows[0]["DMS_score"] = 50
        for check, config, _, cache_id in self.targets():
            relaxed = {**config, "max_length": None}
            with self.subTest(cache=cache_id), self.assertRaises(FileExistsError):
                check(self.root, "proteinmpnn", cache_id, self.inputs, relaxed)
            with self.subTest(cache=cache_id), self.assertRaisesRegex(ValueError, "inputs or coverage differ"):
                check(self.root, "proteinmpnn", cache_id, changed_inputs, relaxed, resume=True)

    def test_esm_if1_length_metadata_still_requires_exact_equality(self):
        config = inverse_storage.make_leaf_config("esm_if", "esm_if1", {
            "conditioning_mode": "reference_prefix", "max_length": 1022, "truncation": "reject",
        }, self.structure)
        log_probs = torch.zeros(3, 20).log_softmax(-1)
        inverse_storage.write_inverse_cache(self.root, "esm_if1", "esm_if", self.inputs, config, log_probs.exp(), log_probs, self.provenance)
        for cap in (None, 1021):
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, "new cache ID"):
                inverse_storage.check_existing(self.root, "esm_if1", "esm_if", self.inputs, {**config, "max_length": cap}, resume=True)

    def test_esm2_score_length_metadata_still_requires_exact_equality(self):
        config = score_storage.make_score_config("esm_scores", "esm2", {
            "scoring_recipe_version": 1, "score_definition": "sum_masked_marginal_log_odds",
            "conditioning_mode": "masked_wild_type", "inference_method": "masked_marginals",
            "normalization": "model_vocabulary", "output_dtype": "float32", "max_length": 1022,
        })
        score_storage.write_score_cache(self.root, "esm2", "esm_scores", self.inputs, config, self.scores, {"uses_dms_labels": False})
        with self.assertRaisesRegex(ValueError, "new cache ID"):
            score_storage.check_existing(self.root, "esm2", "esm_scores", self.inputs, {**config, "max_length": 1021}, resume=True)


if __name__ == "__main__":
    unittest.main()
