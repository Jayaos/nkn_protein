"""Data-contract checks using small caches and the existing repository fixture."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from data_loading import (
    AssayCatalog,
    load_feature_bundle,
    load_folds,
    parse_substitutions,
    validate_assay_coverage,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "data/features/A0A1I9GEU1_NEIME_Kennouche_2019"
INDEX = ROOT / "data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv"


def write_csv(path, header, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def make_bundle(root: Path):
    reference = "ACD"
    ids = ["WT", "A1V", "C2D", "A1V:C2D"]
    sequences = ["ACD", "VCD", "ADD", "VDD"]
    defaults = {name: {"model_family": "synthetic", "checkpoint": "v1", "path": f"{name}/synthetic/v1"} for name in ("sequence", "structure", "inverse_folding")}
    metadata = {"schema_version": 2, "layout_version": 1, "synthetic": True, "assay_id": "tiny", "reference_sequence": reference, "reference_length": 3, "variants_file": "variants.csv", "features": defaults}
    (root / "assay.json").write_text(json.dumps(metadata), encoding="utf-8")
    write_csv(root / "variants.csv", ["variant_id", "mutant", "mutated_sequence", "DMS_score", "source_row_index"], [[value, value, sequence, i + 1, i] for i, (value, sequence) in enumerate(zip(ids, sequences))])
    embeddings = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)
    common = {"schema_version": 2, "synthetic": True, "assay_id": "tiny", "reference_sequence": reference, "positions": torch.arange(1, 4)}
    order = [2, 0, 3, 1]
    probs = torch.softmax(torch.arange(20, dtype=torch.float32), dim=0).repeat(3, 1)
    payloads = {
        "sequence": {**common, "variant_ids": [ids[i] for i in order], "sequence_hashes": [hashlib.sha256(sequences[i].encode()).hexdigest() for i in order], "embeddings": embeddings[order]},
        "structure": {**common, "embeddings": torch.arange(6, dtype=torch.float32).reshape(3, 2)},
        "inverse_folding": {**common, "alphabet": "ACDEFGHIKLMNPQRSTVWY", "probabilities": probs, "log_probs": probs.log()},
    }
    for j, (name, payload) in enumerate(payloads.items()):
        leaf = root / name / "synthetic/v1"
        leaf.mkdir(parents=True)
        torch.save(payload, leaf / "features.pt")
        config = {"feature_type": name, "model_family": "synthetic", "checkpoint": "v1", "synthetic": True, "zero_shot": {"file": "zero_shot.csv", "id_column": "variant_id", "score_column": "score"}}
        (leaf / "config.json").write_text(json.dumps(config), encoding="utf-8")
        write_csv(leaf / "zero_shot.csv", ["variant_id", "score"], [[ids[i], i + 10 * j] for i in reversed(range(4))])
    return ids, sequences, embeddings


class FeatureLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ids, self.sequences, self.embedding = make_bundle(self.root)

    def test_id_joins_and_local_features_preserve_mutation_information(self):
        batch = load_feature_bundle(self.root)
        self.assertEqual(batch.variant_ids, tuple(self.ids))
        self.assertEqual(batch.scores.dtype, torch.float64)
        self.assertEqual(batch.score_names, ("sequence/synthetic/v1", "structure/synthetic/v1", "inverse_folding/synthetic/v1"))
        self.assertEqual(batch.feature_signature["pooling"], "mean")
        torch.testing.assert_close(batch.scores, torch.tensor([[i, i + 10, i + 20] for i in range(4)], dtype=torch.float64))
        torch.testing.assert_close(batch.global_features, self.embedding.double().mean(dim=1))
        self.assertEqual(batch.owners.tolist(), [1, 2, 3, 3])
        self.assertEqual(batch.mutation_positions.tolist(), [1, 2, 1, 2])
        torch.testing.assert_close(batch.local_features["sequence_site"], self.embedding[[1, 2, 3, 3], [0, 1, 0, 1]].double())
        self.assertEqual(batch.local_features["structure"].shape, (4, 2))
        self.assertEqual(batch.local_features["inverse_folding"].shape, (4, 20))

    def test_selection_reorders_packed_occurrences_and_handles_wildtype(self):
        batch = load_feature_bundle(self.root)
        chosen = batch.select([3, 0, 1])
        self.assertEqual(chosen.variant_ids, ("A1V:C2D", "WT", "A1V"))
        self.assertEqual(chosen.score_names, batch.score_names)
        self.assertEqual(chosen.feature_signature, batch.feature_signature)
        self.assertEqual(chosen.owners.tolist(), [0, 0, 2])
        self.assertEqual(chosen.mutation_positions.tolist(), [1, 2, 1])
        torch.testing.assert_close(chosen.local_features["sequence_site"], self.embedding[[3, 3, 1], [0, 1, 0]].double())
        wt = batch.select([0])
        self.assertEqual(wt.owners.numel(), 0)
        self.assertEqual(wt.local_features["inverse_folding"].shape, (0, 20))
        self.assertEqual(batch.select([]).num_variants, 0)
        with self.assertRaises(ValueError):
            batch.select([1, 1])

    def test_missing_labels_are_allowed_for_prediction(self):
        write_csv(self.root / "variants.csv", ["variant_id", "mutant", "mutated_sequence"], [[value, value, sequence] for value, sequence in zip(self.ids, self.sequences)])
        self.assertIsNone(load_feature_bundle(self.root).targets)

    def test_model_signature_ignores_variant_coverage_but_tracks_preprocessing(self):
        initial = load_feature_bundle(self.root)
        reordered = load_feature_bundle(self.root, score_sources=list(reversed(initial.score_names)))
        self.assertNotEqual(initial.score_names, reordered.score_names)
        self.assertEqual(initial.feature_signature, reordered.feature_signature)
        torch.testing.assert_close(reordered.scores, initial.scores.flip(dims=(1,)))
        config_path = self.root / "sequence/synthetic/v1/config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["zero_shot"].update(num_variants=99, variants_sha256="different-input-table")
        config_path.write_text(json.dumps(config), encoding="utf-8")
        self.assertEqual(initial.feature_signature, load_feature_bundle(self.root).feature_signature)
        config["layer"] = 7
        config_path.write_text(json.dumps(config), encoding="utf-8")
        self.assertNotEqual(initial.feature_signature, load_feature_bundle(self.root).feature_signature)

    def test_score_sources_are_explicit_and_duplicate_ids_are_rejected(self):
        batch = load_feature_bundle(self.root, score_sources=["structure/synthetic/v1"])
        self.assertEqual(batch.num_scores, 1)
        self.assertEqual(batch.score_names, ("structure/synthetic/v1",))
        torch.testing.assert_close(batch.scores[:, 0], torch.arange(10, 14, dtype=torch.float64))
        write_csv(self.root / "sequence/synthetic/v1/zero_shot.csv", ["variant_id", "score"], [["WT", 1], ["WT", 2], ["C2D", 3], ["A1V:C2D", 4]])
        with self.assertRaisesRegex(ValueError, "Score IDs"):
            load_feature_bundle(self.root)

    def test_hash_and_probability_validation(self):
        path = self.root / "sequence/synthetic/v1/features.pt"
        payload = torch.load(path, weights_only=True)
        payload["sequence_hashes"][0] = "wrong"
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "hash"):
            load_feature_bundle(self.root)
        make_root = self.root / "second"
        make_root.mkdir()
        make_bundle(make_root)
        path = make_root / "inverse_folding/synthetic/v1/features.pt"
        payload = torch.load(path, weights_only=True)
        payload["probabilities"][0, 0] = -0.1
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            load_feature_bundle(make_root)

    def test_mutation_validation(self):
        self.assertEqual(parse_substitutions("C2D:A1V", "VDD", "ACD"), (2, 1))
        self.assertEqual(parse_substitutions("WT", "ACD", "ACD"), ())
        for mutant, sequence in [("A1A", "ACD"), ("C1V", "VCD"), ("A1V:A1G", "GCD"), ("A1V", "ACD")]:
            with self.subTest(mutant=mutant), self.assertRaises(ValueError):
                parse_substitutions(mutant, sequence, "ACD")

    def test_full_assay_coverage_is_distinct_from_explicit_subset(self):
        raw = self.root / "raw"
        raw.mkdir()
        write_csv(raw / "DMS_substitutions.csv", ["DMS_id", "DMS_filename", "target_seq", "seq_len", "raw_mut_offset"], [["tiny", "tiny.csv", "ACD", 3, 100]])
        write_csv(raw / "tiny.csv", ["mutant", "mutated_sequence", "DMS_score"], [[value, sequence, i + 1] for i, (value, sequence) in enumerate(zip(self.ids, self.sequences))])
        index = raw / "DMS_substitutions.csv"
        batch = load_feature_bundle(self.root)
        catalog = AssayCatalog(index)
        self.assertEqual([v.variant_id for v in catalog.load_variants("tiny", singles_only=True)], ["A1V", "C2D"])
        self.assertEqual(validate_assay_coverage(batch, index).assay_id, "tiny")
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_assay_coverage(batch.select([1]), index)
        validate_assay_coverage(batch.select([1]), index, allow_subset=True)
        validate_assay_coverage(batch.select([1, 2]), index, singles_only=True)


class SplitTests(unittest.TestCase):
    def test_saved_generic_assignments_round_trip_and_specific_scheme_takes_precedence(self):
        variant_ids = ("a", "b", "c", "d", "e", "f")
        folds = torch.tensor([3, 0, 4, 1, 2, 0])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fold_assignments.csv"
            rows = list(zip(variant_ids, folds.tolist()))
            write_csv(path, ["variant_id", "fold"], reversed(rows))
            torch.testing.assert_close(load_folds(path, variant_ids, "random"), folds)
            write_csv(path, ["variant_id", "fold", "fold_random_5"], [[variant_id, (fold + 1) % 5, fold] for variant_id, fold in rows])
            torch.testing.assert_close(load_folds(path, variant_ids, "random"), folds)

    def test_official_folds_join_by_id_and_require_complete_scheme(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "folds.csv"
            write_csv(path, ["mutant", "fold_random_5"], [["b", 1], ["c", 2], ["a", 0]])
            self.assertEqual(load_folds(path, ["a", "b", "c"], "random", n_folds=3).tolist(), [0, 1, 2])
            with self.assertRaisesRegex(ValueError, "scheme"):
                load_folds(path, ["a", "b", "c"], "contiguous", n_folds=3)
            with self.assertRaisesRegex(ValueError, "exactly"):
                load_folds(path, ["a", "b"], "random", n_folds=3)

    def test_multiples_column_and_known_depth_exclusions_preserve_assignments(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "multiples.csv"
            write_csv(path, ["mutant", "fold_rand_multiples"], [["double", 2], ["triple", 1], ["single_b", 1], ["single_a", 0]])
            selected = ["single_a", "single_b", "double"]
            self.assertEqual(
                load_folds(path, selected, "random_multiples", n_folds=3, allowed_extra_ids=["triple"]).tolist(),
                [0, 1, 2],
            )
            with self.assertRaises(ValueError):
                load_folds(path, selected, "random_multiples", n_folds=3)
            with self.assertRaises(ValueError):
                load_folds(path, [*selected, "missing"], "random_multiples", n_folds=3, allowed_extra_ids=["triple"])
            with self.assertRaises(ValueError):
                load_folds(path, selected, "random_multiples", n_folds=3, allowed_extra_ids=["unknown"])
            # Permitting excluded IDs must not permit duplicate rows or lose a selected fold.
            write_csv(path, ["mutant", "fold_rand_multiples"], [["double", 2], ["triple", 1], ["triple", 1], ["single_b", 1], ["single_a", 0]])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_folds(path, selected, "random_multiples", n_folds=3, allowed_extra_ids=["triple"])


class ExistingFixtureTests(unittest.TestCase):
    def test_existing_fixture_is_compatible_without_rewriting_caches(self):
        batch = load_feature_bundle(FIXTURE)
        self.assertEqual(batch.num_variants, 16)
        self.assertEqual(batch.num_scores, 3)
        self.assertEqual(batch.global_features.shape, (16, 320))
        self.assertEqual(batch.local_features["structure"].shape, (16, 128))
        self.assertEqual(batch.local_features["inverse_folding"].shape, (16, 20))
        self.assertEqual(len(AssayCatalog(INDEX)), 217)
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_assay_coverage(batch, INDEX)
        validate_assay_coverage(batch, INDEX, allow_subset=True)


if __name__ == "__main__":
    unittest.main()
