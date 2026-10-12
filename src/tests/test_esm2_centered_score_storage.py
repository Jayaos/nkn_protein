"""Windowed score recipes cannot masquerade as full-reference score artifacts."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from data_loading.features import read_csv
from feature_generation.inputs import prepare_assay, stable_json
from feature_generation.score_storage import (
    check_existing, make_score_config, validate_score_cache, write_score_cache,
)
from tests.test_feature_generation import write_csv


LEGACY_METADATA = {
    "model_family": "esm2", "scoring_recipe_version": 1,
    "score_definition": "sum_masked_marginal_log_odds",
    "conditioning_mode": "masked_wild_type", "inference_method": "masked_marginals",
    "normalization": "model_vocabulary", "output_dtype": "float32",
}
CENTERED_METADATA = {
    **LEGACY_METADATA, "scoring_recipe_version": 2,
    "scoring_window": "centered", "truncation": "centered", "max_length": 4,
    "window_policy": "centered_clamped", "window_indexing": "zero_based_half_open",
    "residue_aggregation": "sum_over_mutations", "masking": "one_wild_type_residue_at_a_time",
    "wild_type_subtraction": True, "noncanonical_tokens": "retained_in_normalization",
    "alphabet": "ACDEFGHIKLMNPQRSTVWY", "precision_policy": "float32_log_softmax",
}


class CenteredScoreRecipeTests(unittest.TestCase):
    def test_legacy_minimal_metadata_remains_valid_and_unchanged(self):
        result = make_score_config("full", "esm2", LEGACY_METADATA)
        self.assertNotIn("scoring_window", result)
        self.assertNotIn("truncation", result)
        self.assertNotIn("max_length", result)
        self.assertEqual(result["scoring_recipe_version"], 1)
        actual_legacy = {**LEGACY_METADATA, "truncation": "reject", "max_length": 1022}
        self.assertEqual(make_score_config("full", "esm2", actual_legacy)["max_length"], 1022)

    def test_centered_recipe_requires_complete_wt_masking_and_window_metadata(self):
        self.assertEqual(make_score_config("windowed", "esm2", CENTERED_METADATA)["scoring_recipe_version"], 2)
        for key in CENTERED_METADATA:
            if key == "model_family":  # The constructor supplies its explicit family argument.
                continue
            metadata = dict(CENTERED_METADATA)
            del metadata[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                make_score_config("windowed", "esm2", metadata)
        for change in (
            {"scoring_recipe_version": 2.0}, {"scoring_recipe_version": True}, {"scoring_recipe_version": 3},
            {"scoring_window": "full"}, {"truncation": "prefix"},
            {"window_policy": "overlapping"}, {"window_indexing": "one_based"},
            {"max_length": True}, {"max_length": 4.0}, {"max_length": 0}, {"max_length": 1023},
            {"wild_type_subtraction": 1}, {"conditioning_mode": "mutated_sequence"},
            {"masking": "all_mutations_at_once"}, {"residue_aggregation": "mean_over_mutations"},
            {"precision_policy": "float16_log_softmax"}, {"normalization": "canonical_amino_acids"},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                make_score_config("windowed", "esm2", {**CENTERED_METADATA, **change})
        for max_length in (1, 1022):
            self.assertEqual(make_score_config("windowed", "esm2", {**CENTERED_METADATA, "max_length": max_length})["max_length"], max_length)

    def test_v1_cannot_claim_centered_or_partial_windowing(self):
        for change in (
            {"scoring_window": "centered"}, {"truncation": "centered"}, {"truncation": "prefix"},
            {"window_policy": "centered_clamped"}, {"window_indexing": "zero_based_half_open"},
            {"window_policy": None}, {"scoring_recipe_version": True},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                make_score_config("full", "esm2", {**LEGACY_METADATA, **change})
        with self.assertRaises(ValueError):
            make_score_config("windowed", "esm2", {**CENTERED_METADATA, "structure": {}})


class CenteredScoreStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        index = self.root / "DMS_substitutions.csv"
        self.reference = "ACDEFGHIKL"
        write_csv(index, ["DMS_id", "DMS_filename", "target_seq", "seq_len"], [["tiny", "tiny.csv", self.reference, 10]])
        write_csv(self.root / "tiny.csv", ["variant_id", "mutant", "mutated_sequence", "DMS_score"], [
            ["wt", "WT", self.reference, 0], ["tail", "L10K", "ACDEFGHIKK", 1],
            ["middle", "F5V", "ACDEVGHIKL", 2], ["both", "L10K:F5V", "ACDEVGHIKK", 3],
        ])
        self.inputs = prepare_assay(index, "tiny")
        self.assay_root = self.root / "features" / "tiny"
        self.cache_id = "masked_v2-centered4"
        self.config = make_score_config(self.cache_id, "esm2", CENTERED_METADATA)
        self.values = torch.tensor([0., 1., 2., 3.])
        self.leaf = write_score_cache(
            self.assay_root, "esm2", self.cache_id, self.inputs, self.config,
            self.values, {"uses_dms_labels": False},
        )

    def validate(self):
        return validate_score_cache(self.assay_root, "esm2", self.cache_id)

    def rewrite_config(self, config, *, reseal_signature=True):
        config_path = self.leaf / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        completion_path = self.leaf / "completion.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["files_sha256"]["config.json"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        if reseal_signature:
            completion["config_signature"] = hashlib.sha256(stable_json(config).encode()).hexdigest()
        completion_path.write_text(json.dumps(completion), encoding="utf-8")

    def test_centered_scores_keep_full_reference_variant_and_input_identity(self):
        report = self.validate()
        self.assertEqual(report["reference_length"], 10)
        self.assertEqual(report["input_signature"], self.inputs.input_signature)
        self.assertEqual(report["num_variants"], 4)
        assay = json.loads((self.assay_root / "assay.json").read_text(encoding="utf-8"))
        self.assertEqual(assay["reference_sequence"], self.reference)
        self.assertEqual((self.assay_root / "variants.csv").read_text(encoding="utf-8"), self.inputs.variants_csv())
        self.assertEqual([row["variant_id"] for row in read_csv(self.leaf / "mean_score.csv")], list(self.inputs.variant_ids))
        self.assertEqual([float(row["score"]) for row in read_csv(self.leaf / "mean_score.csv")], self.values.tolist())

    def test_v1_and_v2_coexist_and_resume_requires_matching_window_recipe(self):
        legacy_config = make_score_config("full", "esm2", {**LEGACY_METADATA, "truncation": "reject", "max_length": 10})
        legacy = write_score_cache(self.assay_root, "esm2", "full", self.inputs, legacy_config, self.values, {"uses_dms_labels": False})
        original = {path.name: path.read_bytes() for path in legacy.iterdir()}
        for cache_id, config in ((self.cache_id, self.config), ("full", legacy_config)):
            self.assertTrue(check_existing(self.assay_root, "esm2", cache_id, self.inputs, config, resume=True))
            validate_score_cache(self.assay_root, "esm2", cache_id)
        self.assertEqual({path.name: path.read_bytes() for path in legacy.iterdir()}, original)
        with self.assertRaisesRegex(ValueError, "new cache ID"):
            check_existing(self.assay_root, "esm2", self.cache_id, self.inputs, {**self.config, "max_length": 3}, resume=True)
        with self.assertRaisesRegex(ValueError, "new cache ID"):
            check_existing(self.assay_root, "esm2", "full", self.inputs, {**self.config, "checkpoint": "full"}, resume=True)
        with self.assertRaises(FileExistsError):
            write_score_cache(self.assay_root, "esm2", self.cache_id, self.inputs, self.config, self.values, {"uses_dms_labels": False})

    def test_validator_rejects_tampering_even_after_file_checksums_are_resealed(self):
        for change in (
            {"scoring_recipe_version": 1}, {"scoring_window": "full"}, {"truncation": "reject"},
            {"window_policy": "other"}, {"window_indexing": "one_based"},
            {"max_length": 4.0}, {"max_length": True}, {"max_length": 1023},
            {"wild_type_subtraction": False}, {"checkpoint": "wrong_directory"},
        ):
            with self.subTest(change=change):
                self.rewrite_config({**self.config, **change})
                with self.assertRaises(ValueError):
                    self.validate()
        for key in ("scoring_window", "window_policy", "window_indexing", "max_length"):
            with self.subTest(missing=key):
                config = dict(self.config)
                del config[key]
                self.rewrite_config(config)
                with self.assertRaises(ValueError):
                    self.validate()

    def test_valid_recipe_change_still_requires_matching_completion_signature(self):
        self.rewrite_config({**self.config, "max_length": 3}, reseal_signature=False)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
