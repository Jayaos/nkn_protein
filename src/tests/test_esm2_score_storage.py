"""ESM2 scores are independent, immutable, and usable as GP mean inputs."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from data_loading import load_feature_bundle
from data_loading.features import read_csv
from feature_generation.inputs import AssayInput, sha256_file, stable_json
from feature_generation.score_storage import check_existing, make_score_config, validate_score_cache, write_score_cache
from feature_generation.storage import make_leaf_config, write_sequence_cache
from tests.test_inverse_pipeline import structure_metadata


class ESM2ScoreStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "tiny"
        self.inputs = AssayInput(
            assay_id="tiny", reference_sequence="ACD",
            rows=[
                {"variant_id": "wt", "source_row_index": 0, "mutant": "WT", "mutated_sequence": "ACD", "DMS_score": 0},
                {"variant_id": "first", "source_row_index": 1, "mutant": "A1V", "mutated_sequence": "VCD", "DMS_score": 1},
                {"variant_id": "double", "source_row_index": 2, "mutant": "A1V:C2D", "mutated_sequence": "VDD", "DMS_score": 2},
            ],
            source_assay_path=Path("tiny.csv"), source_assay_sha256="a" * 64,
            source_index_sha256="b" * 64, coverage={"num_variants": 3, "source_row_count": 3},
        )
        self.metadata = {
            "model_family": "esm2", "model_id": "facebook/esm2_t6_8M_UR50D", "revision": "c" * 40,
            "checkpoint_sha256": "d" * 64,
            "checkpoint_files_sha256": {"model.safetensors": "e" * 64},
            "scoring_recipe_version": 1, "score_definition": "sum_masked_marginal_log_odds",
            "conditioning_mode": "masked_wild_type", "inference_method": "masked_marginals",
            "normalization": "model_vocabulary", "output_dtype": "float32",
            "score_direction": "higher_is_more_likely", "inference_dtype": "float32",
            "masking": "one_wild_type_residue_at_a_time", "wild_type_subtraction": True,
            "residue_aggregation": "sum_over_mutations", "noncanonical_tokens": "retained_in_normalization",
        }
        self.config = make_score_config("masked_v1", "esm2", self.metadata)
        self.provenance = {"uses_dms_labels": False}
        self.scores = torch.tensor([0.0, -1.2345678, 2.7182818], dtype=torch.float32)

    @property
    def leaf(self):
        return self.root / "scores/esm2/masked_v1"

    def write(self, provenance=None):
        return write_score_cache(
            self.root, "esm2", "masked_v1", self.inputs, self.config, self.scores,
            self.provenance if provenance is None else provenance,
        )

    def rehash(self, name, config_signature=False):
        path = self.leaf / "completion.json"
        completion = json.loads(path.read_text(encoding="utf-8"))
        completion["files_sha256"][name] = sha256_file(self.leaf / name)
        if config_signature:
            config = json.loads((self.leaf / "config.json").read_text(encoding="utf-8"))
            completion["config_signature"] = hashlib.sha256(stable_json(config).encode()).hexdigest()
        path.write_text(json.dumps(completion), encoding="utf-8")

    def add_proteinmpnn(self):
        metadata = {
            "model_id": "v_48_020", "revision": "c" * 40, "checkpoint_sha256": "f" * 64,
            "scoring_recipe_version": 1, "score_definition": "negative_mean_sequence_nll",
            "conditioning_mode": "sequence_autoregressive", "inference_method": "forward",
            "num_orders": 1, "seed": 0, "decoding_order": "shared_random_orders",
            "normalization": "native_21_amino_acid_tokens", "alphabet": "ACDEFGHIKLMNPQRSTVWYX",
            "output_dtype": "float32", "score_direction": "higher_is_more_likely",
        }
        structure = structure_metadata()
        config = make_score_config("official_v1", "proteinmpnn", metadata, structure)
        provenance = {"uses_dms_labels": False, "structure_input_signature": structure["input_signature"]}
        values = torch.tensor([-1., -2., -3.], dtype=torch.float32)
        leaf = write_score_cache(self.root, "proteinmpnn", "official_v1", self.inputs, config, values, provenance)
        return leaf, config, values

    def add_embeddings(self):
        config = make_leaf_config("embeddings", {"hidden_size": 2, "model_id": self.metadata["model_id"]})
        values = torch.arange(18, dtype=torch.float32).reshape(3, 3, 2)
        return write_sequence_cache(self.root, "embeddings", self.inputs, config, values, self.provenance)

    def test_roundtrip_resume_needs_no_structures_or_representations(self):
        with patch("feature_generation.score_storage.validate_structure_metadata", side_effect=AssertionError("structure accessed")):
            self.assertFalse(check_existing(self.root, "esm2", "masked_v1", self.inputs, self.config))
            self.assertEqual(self.write(), self.leaf)
            report = validate_score_cache(self.root, "esm2")
            before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
            self.assertTrue(check_existing(self.root, "esm2", "masked_v1", self.inputs, self.config, resume=True))
            self.assertEqual(before, {path: path.read_bytes() for path in before})
        rows = read_csv(self.leaf / "mean_score.csv")
        self.assertEqual([row["variant_id"] for row in rows], list(self.inputs.variant_ids))
        self.assertEqual([float(row["score"]) for row in rows], self.scores.tolist())
        self.assertEqual(report["num_variants"], 3)
        self.assertEqual(report["score_definition"], "sum_masked_marginal_log_odds")
        self.assertEqual(report["mean_score"]["file"], "mean_score.csv")
        self.assertNotIn("structure_input_signature", report)
        self.assertNotIn("structure", self.config)
        completion = json.loads((self.leaf / "completion.json").read_text(encoding="utf-8"))
        self.assertNotIn("structure_input_signature", completion)
        self.assertEqual(completion["input_signature"], self.inputs.input_signature)
        for name in ("sequence", "structure", "inverse_folding", "structures"):
            self.assertFalse((self.root / name).exists())
        self.assertFalse((self.leaf / "zero_shot.csv").exists())
        self.assertFalse((self.leaf / "features.pt").exists())

    def test_coexisting_families_preserve_proteinmpnn_files_and_resume(self):
        proteinmpnn_leaf, config, _ = self.add_proteinmpnn()
        paths = [self.root / "assay.json", self.root / "variants.csv", *proteinmpnn_leaf.iterdir()]
        before = {path: path.read_bytes() for path in paths}
        self.write()
        self.assertEqual(before, {path: path.read_bytes() for path in paths})
        self.assertTrue(check_existing(self.root, "proteinmpnn", "official_v1", self.inputs, config, resume=True))
        self.assertEqual(validate_score_cache(self.root)["model_family"], "proteinmpnn")
        self.assertEqual(validate_score_cache(self.root, "esm2")["cache_id"], "masked_v1")
        self.assertEqual(validate_score_cache(self.root)["structure_input_signature"], config["structure"]["input_signature"])

    def test_gp_loader_selects_either_score_or_both_independently(self):
        proteinmpnn_leaf, _, proteinmpnn_scores = self.add_proteinmpnn()
        embeddings_leaf = self.add_embeddings()
        protected = {path: path.read_bytes() for path in embeddings_leaf.iterdir()}
        self.write()
        # Reordered score rows exercise joining by ID instead of file row order.
        path = self.leaf / "mean_score.csv"
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join([lines[0], *reversed(lines[1:])]) + "\n", encoding="utf-8")
        self.rehash("mean_score.csv")
        esm_source = self.leaf.relative_to(self.root).as_posix()
        mpnn_source = proteinmpnn_leaf.relative_to(self.root).as_posix()
        for sources, expected in (
            ([esm_source], self.scores[:, None]),
            ([mpnn_source, esm_source], torch.stack((proteinmpnn_scores, self.scores), dim=1)),
        ):
            with self.subTest(sources=sources):
                batch = load_feature_bundle(self.root, enabled_branches=["sequence_global"], score_sources=sources)
                self.assertEqual(batch.score_names, tuple(sources))
                self.assertEqual(batch.variant_ids, self.inputs.variant_ids)
                self.assertEqual(batch.global_features.shape, (3, 2))
                torch.testing.assert_close(batch.scores, expected.double())
                for source in sources:
                    self.assertIn(f"score:{source}.config_sha256", batch.feature_signature)
        self.assertEqual(protected, {path: path.read_bytes() for path in protected})

    def test_resume_refuses_changed_checkpoint_recipe_and_overwrite(self):
        self.write()
        with self.assertRaises(FileExistsError):
            self.write()
        with self.assertRaisesRegex(FileExistsError, "resume"):
            check_existing(self.root, "esm2", "masked_v1", self.inputs, self.config)
        for key, value in (
            ("revision", "f" * 40), ("checkpoint_sha256", "f" * 64),
            ("checkpoint_files_sha256", {"model.safetensors": "f" * 64}),
            ("inference_dtype", "bfloat16"), ("wild_type_subtraction", False),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "scoring recipe"):
                check_existing(self.root, "esm2", "masked_v1", self.inputs, {**self.config, key: value}, resume=True)
        self.assertTrue(check_existing(self.root, "esm2", "masked_v1", self.inputs, self.config, resume=True))

    def test_corrupt_scores_fail_validation_and_gp_loading_and_require_regeneration(self):
        self.add_embeddings()
        self.write()
        path = self.leaf / "mean_score.csv"
        original = path.read_bytes()
        path.write_bytes(original + b"extra,1\n")
        source = self.leaf.relative_to(self.root).as_posix()
        for action in (
            lambda: validate_score_cache(self.root, "esm2"),
            lambda: load_feature_bundle(self.root, enabled_branches=["sequence_global"], score_sources=[source]),
        ):
            with self.assertRaisesRegex(ValueError, "checksum"):
                action()
        self.assertFalse(check_existing(self.root, "esm2", "masked_v1", self.inputs, self.config, resume=True))
        path.write_bytes(original)
        (self.leaf / "completion.json").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_score_cache(self.root, "esm2")

    def test_rehashed_unsupported_recipes_and_label_provenance_fail(self):
        self.write()
        path = self.leaf / "config.json"
        original = path.read_bytes()
        path.write_text(json.dumps({**self.config, "conditioning_mode": "unmasked_wild_type"}), encoding="utf-8")
        self.rehash("config.json", config_signature=True)
        with self.assertRaisesRegex(ValueError, "recipe"):
            validate_score_cache(self.root, "esm2")
        path.write_bytes(original)
        self.rehash("config.json", config_signature=True)
        path = self.leaf / "provenance.json"
        path.write_text(json.dumps({"uses_dms_labels": True}), encoding="utf-8")
        self.rehash("provenance.json")
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_score_cache(self.root, "esm2")

    def test_invalid_config_or_label_provenance_never_publishes(self):
        for key, value in (
            ("scoring_recipe_version", 2), ("score_definition", "pseudo_log_likelihood"),
            ("conditioning_mode", "sequence_autoregressive"), ("inference_method", "forward"),
            ("normalization", "canonical_amino_acids_only"), ("output_dtype", "float16"),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "recipe"):
                make_score_config("invalid", "esm2", {**self.metadata, key: value})
        with self.assertRaisesRegex(ValueError, "structure"):
            make_score_config("invalid", "esm2", self.metadata, structure_metadata())
        with self.assertRaisesRegex(ValueError, "family"):
            make_score_config("invalid", "esm2", {**self.metadata, "model_family": "proteinmpnn"})
        with self.assertRaisesRegex(ValueError, "structure"):
            make_score_config("invalid", "esm2", {**self.metadata, "structure": {}})
        for provenance in ({}, {"uses_dms_labels": True}):
            with self.assertRaisesRegex(ValueError, "provenance"):
                self.write(provenance)
        self.assertFalse(self.leaf.exists())

    def test_esm2_cache_discovery_does_not_consider_other_families(self):
        self.add_proteinmpnn()
        self.write()
        self.assertEqual(validate_score_cache(self.root, "esm2")["cache_id"], "masked_v1")
        other = make_score_config("another_checkpoint", "esm2", {**self.metadata, "revision": "f" * 40})
        write_score_cache(self.root, "esm2", "another_checkpoint", self.inputs, other, self.scores, self.provenance)
        with self.assertRaisesRegex(ValueError, "no unique score cache"):
            validate_score_cache(self.root, "esm2")
        self.assertEqual(validate_score_cache(self.root, "esm2", "masked_v1")["num_variants"], 3)
        self.assertEqual(validate_score_cache(self.root)["cache_id"], "official_v1")


if __name__ == "__main__":
    unittest.main()
