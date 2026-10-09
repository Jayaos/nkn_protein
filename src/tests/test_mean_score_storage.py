"""Integrity and immutable-cache checks for per-variant GP mean scores."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from data_loading.features import read_csv
from feature_generation.inputs import AssayInput, sha256_file, stable_json
from feature_generation.inverse_storage import make_leaf_config, write_inverse_cache, validate_inverse_cache
from feature_generation.score_storage import check_existing, make_score_config, validate_score_cache, write_score_cache
from tests.test_inverse_pipeline import structure_metadata


class MeanScoreStorageTests(unittest.TestCase):
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
            "model_id": "v_48_020", "revision": "c" * 40, "checkpoint_sha256": "d" * 64,
            "scoring_recipe_version": 1, "score_definition": "negative_mean_sequence_nll",
            "conditioning_mode": "sequence_autoregressive", "inference_method": "forward",
            "num_orders": 10, "seed": 0, "decoding_order": "shared_random_orders",
            "normalization": "native_21_amino_acid_tokens", "alphabet": "ACDEFGHIKLMNPQRSTVWYX",
            "output_dtype": "float32", "score_direction": "higher_is_more_likely",
        }
        self.structure = structure_metadata()
        self.config = make_score_config("official_v1", "proteinmpnn", self.metadata, self.structure)
        self.provenance = {"uses_dms_labels": False, "structure_input_signature": self.structure["input_signature"]}
        self.scores = torch.tensor([-1.123456789012345, -2.0, -3.141592653589793], dtype=torch.float64)

    @property
    def leaf(self):
        return self.root / "scores/proteinmpnn/official_v1"

    def write(self, scores=None, provenance=None):
        return write_score_cache(self.root, "proteinmpnn", "official_v1", self.inputs, self.config,
                                 self.scores if scores is None else scores,
                                 self.provenance if provenance is None else provenance)

    def rehash(self, name, config_signature=False):
        path = self.leaf / "completion.json"
        completion = json.loads(path.read_text(encoding="utf-8"))
        completion["files_sha256"][name] = sha256_file(self.leaf / name)
        if config_signature:
            config = json.loads((self.leaf / "config.json").read_text(encoding="utf-8"))
            completion["config_signature"] = hashlib.sha256(stable_json(config).encode()).hexdigest()
        path.write_text(json.dumps(completion), encoding="utf-8")

    def test_roundtrip_preserves_scores_and_uses_new_descriptor(self):
        self.assertFalse(check_existing(self.root, "proteinmpnn", "official_v1", self.inputs, self.config))
        self.assertEqual(self.write(), self.leaf)
        rows = read_csv(self.leaf / "mean_score.csv")
        self.assertEqual(tuple(row["variant_id"] for row in rows), self.inputs.variant_ids)
        self.assertEqual([float(row["score"]) for row in rows], self.scores.tolist())
        report = validate_score_cache(self.root)
        self.assertEqual(report["num_variants"], 3)
        self.assertEqual(report["score_definition"], "negative_mean_sequence_nll")
        self.assertEqual(report["mean_score"]["file"], "mean_score.csv")
        self.assertFalse((self.leaf / "zero_shot.csv").exists())
        self.assertFalse((self.leaf / "features.pt").exists())
        self.assertNotIn("zero_shot", self.config)
        self.assertTrue(check_existing(self.root, "proteinmpnn", "official_v1", self.inputs, self.config, resume=True))

    def test_float32_values_are_saved_without_rounding_error(self):
        values = self.scores.float()
        self.write(scores=values)
        self.assertEqual([float(row["score"]) for row in read_csv(self.leaf / "mean_score.csv")], values.tolist())

    def test_scoring_preserves_all_representation_files_and_defaults(self):
        inverse_config = make_leaf_config("backbone", "proteinmpnn", {
            "conditioning_mode": "backbone_only", "model_id": "v_48_020", "revision": "c" * 40,
        }, self.structure)
        log_probs = torch.zeros(3, 20).log_softmax(-1)
        inverse_leaf = write_inverse_cache(self.root, "proteinmpnn", "backbone", self.inputs,
                                          inverse_config, log_probs.exp(), log_probs, self.provenance)
        paths = [self.root / "assay.json", self.root / "variants.csv", *inverse_leaf.iterdir()]
        before = {path: path.read_bytes() for path in paths}
        self.write()
        check_existing(self.root, "proteinmpnn", "official_v1", self.inputs, self.config, resume=True)
        self.assertEqual({path: path.read_bytes() for path in paths}, before)
        validate_inverse_cache(self.root)
        validate_score_cache(self.root)

    def test_resume_refuses_changed_recipe_inputs_or_overwrite(self):
        self.write()
        with self.assertRaises(FileExistsError):
            self.write()
        with self.assertRaisesRegex(FileExistsError, "resume"):
            check_existing(self.root, "proteinmpnn", "official_v1", self.inputs, self.config)
        for key, value in (("seed", 1), ("num_orders", 20), ("revision", "e" * 40)):
            with self.subTest(key=key):
                changed = {**self.config, key: value}
                with self.assertRaisesRegex(ValueError, "scoring recipe, or structure differs"):
                    check_existing(self.root, "proteinmpnn", "official_v1", self.inputs, changed, resume=True)
        changed_inputs = copy.deepcopy(self.inputs)
        changed_inputs.rows[0]["DMS_score"] = 100
        with self.assertRaisesRegex(ValueError, "inputs or coverage differ"):
            check_existing(self.root, "proteinmpnn", "official_v1", changed_inputs, self.config, resume=True)
        validate_score_cache(self.root)

    def test_invalid_tensor_and_provenance_do_not_publish(self):
        invalid = [self.scores[:2], self.scores[:, None], self.scores.half(), self.scores.long(),
                   torch.tensor([1., float("nan"), 2.]), torch.tensor([float("inf"), 1., 2.]),
                   torch.empty(3, device="meta")]
        for values in invalid:
            with self.subTest(shape=values.shape, dtype=values.dtype, device=values.device):
                with self.assertRaisesRegex(ValueError, "finite CPU"):
                    self.write(scores=values)
                self.assertFalse(self.leaf.exists())
        for provenance in ({}, {**self.provenance, "uses_dms_labels": True},
                           {**self.provenance, "structure_input_signature": "a" * 64}):
            with self.assertRaisesRegex(ValueError, "provenance"):
                self.write(provenance=provenance)
            self.assertFalse(self.leaf.exists())

    def test_missing_completion_and_corruption_block_validation_and_resume(self):
        self.write()
        original = (self.leaf / "mean_score.csv").read_bytes()
        (self.leaf / "mean_score.csv").write_bytes(original + b"extra,1\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_score_cache(self.root)
        with self.assertRaisesRegex(ValueError, "checksum"):
            check_existing(self.root, "proteinmpnn", "official_v1", self.inputs, self.config, resume=True)
        (self.leaf / "mean_score.csv").write_bytes(original)
        (self.leaf / "completion.json").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_score_cache(self.root)

    def test_rehashed_invalid_ids_and_values_are_rejected(self):
        self.write()
        invalid = [
            "variant_id,score\nwt,-1\nfirst,-2\nfirst,-3\n",
            "variant_id,score\nwt,-1\nfirst,-2\nunknown,-3\n",
            "variant_id,score\nwt,-1\nfirst,-2\n",
            "variant_id,score\nwt,-1\nfirst,nan\ndouble,-3\n",
            "variant_id,score\nwt,-1\nfirst,inf\ndouble,-3\n",
            "variant_id,score\nwt,-1\nfirst,no_number\ndouble,-3\n",
            "variant_id,score\n",
            "variant_id,fitness\nwt,-1\nfirst,-2\ndouble,-3\n",
        ]
        for content in invalid:
            with self.subTest(content=content):
                (self.leaf / "mean_score.csv").write_text(content, encoding="utf-8")
                self.rehash("mean_score.csv")
                with self.assertRaises(ValueError):
                    validate_score_cache(self.root)

    def test_csv_is_joined_by_id_and_does_not_require_row_order(self):
        self.write()
        lines = (self.leaf / "mean_score.csv").read_text(encoding="utf-8").splitlines()
        (self.leaf / "mean_score.csv").write_text("\n".join([lines[0], *lines[:0:-1]]) + "\n", encoding="utf-8")
        self.rehash("mean_score.csv")
        self.assertEqual(validate_score_cache(self.root)["num_variants"], 3)

    def test_rehashed_unsupported_recipe_and_label_using_provenance_are_rejected(self):
        self.write()
        original = (self.leaf / "config.json").read_bytes()
        changed = copy.deepcopy(self.config)
        changed["conditioning_mode"] = "backbone_only"
        (self.leaf / "config.json").write_text(json.dumps(changed), encoding="utf-8")
        self.rehash("config.json", config_signature=True)
        with self.assertRaisesRegex(ValueError, "recipe"):
            validate_score_cache(self.root)
        (self.leaf / "config.json").write_bytes(original)
        self.rehash("config.json", config_signature=True)
        (self.leaf / "provenance.json").write_text(json.dumps({**self.provenance, "uses_dms_labels": True}), encoding="utf-8")
        self.rehash("provenance.json")
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_score_cache(self.root)

    def test_failed_publication_is_not_selected_and_can_retry(self):
        with patch("feature_generation.score_storage._write_json", side_effect=OSError("disk failed")):
            with self.assertRaisesRegex(OSError, "disk failed"):
                self.write()
        self.assertFalse(self.leaf.exists())
        self.assertTrue(list(self.leaf.parent.glob(".official_v1.pending-*")))
        self.assertFalse(check_existing(self.root, "proteinmpnn", "official_v1", self.inputs, self.config, resume=True))
        self.write()
        self.assertEqual(validate_score_cache(self.root)["cache_id"], "official_v1")
        second = make_score_config("other_recipe", "proteinmpnn", self.metadata, self.structure)
        write_score_cache(self.root, "proteinmpnn", "other_recipe", self.inputs, second, self.scores, self.provenance)
        with self.assertRaisesRegex(ValueError, "no unique score cache"):
            validate_score_cache(self.root)
        self.assertEqual(validate_score_cache(self.root, cache_id="official_v1")["cache_id"], "official_v1")

    def test_config_rejects_unsafe_paths_unsupported_recipes_and_incomplete_structures(self):
        for cache_id in ("../escape", "nested/path", "CON", ".pending"):
            with self.assertRaises(ValueError):
                make_score_config(cache_id, "proteinmpnn", self.metadata, self.structure)
        for key, value in (("num_orders", 0), ("num_orders", True), ("seed", -1),
                           ("seed", True), ("normalization", "canonical_amino_acids_only"),
                           ("alphabet", "ACDEFGHIKLMNPQRSTVWY")):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                make_score_config("bad", "proteinmpnn", {**self.metadata, key: value}, self.structure)
        changed = copy.deepcopy(self.structure)
        changed["residue_mapping"][0]["backbone_atoms"].pop()
        keys = ("schema_version", "reference_sequence", "structure_sha256", "chain_id", "model_index", "residue_mapping")
        changed["input_signature"] = hashlib.sha256(stable_json({key: changed[key] for key in keys}).encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "oxygen"):
            make_score_config("bad", "proteinmpnn", self.metadata, changed)


if __name__ == "__main__":
    unittest.main()
