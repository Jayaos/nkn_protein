"""Branch selection loads only its required caches and preserves batch semantics."""

from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import torch

from config import BRANCH_NAMES
from data_loading import load_feature_bundle
from tests.test_data import make_bundle


class OptionalBranchLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        make_bundle(self.root)

    def remove_feature_type(self, name):
        # Restrict recursive deletion to this test's temporary fixture.
        target = (self.root / name).resolve()
        if target.parent != self.root.resolve():
            raise ValueError("Refusing to remove a directory outside the test fixture")
        shutil.rmtree(target)
        metadata_path = self.root / "assay.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["features"].pop(name)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    def test_all_fifteen_combinations_preserve_selected_features(self):
        full = load_feature_bundle(self.root)
        for size in range(1, len(BRANCH_NAMES) + 1):
            for branches in combinations(BRANCH_NAMES, size):
                with self.subTest(branches=branches):
                    batch = load_feature_bundle(self.root, enabled_branches=branches)
                    self.assertEqual(set(batch.local_features), set(branches) - {"sequence_global"})
                    if "sequence_global" in branches:
                        torch.testing.assert_close(batch.global_features, full.global_features)
                    else:
                        self.assertIsNone(batch.global_features)
                    for name, values in batch.local_features.items():
                        torch.testing.assert_close(values, full.local_features[name])
                    torch.testing.assert_close(batch.owners, full.owners)
                    torch.testing.assert_close(batch.mutation_positions, full.mutation_positions)
                    score_types = {"sequence" if name.startswith("sequence_") else name for name in branches}
                    expected_scores = tuple(source for source in full.score_names if source.split("/")[0] in score_types)
                    self.assertEqual(batch.score_names, expected_scores)
                    self.assertEqual(batch.num_scores, len(score_types))
                    if branches == BRANCH_NAMES:
                        self.assertEqual(batch.feature_signature, full.feature_signature)
                    else:
                        self.assertEqual(json.loads(batch.feature_signature["enabled_branches"]), list(branches))

    def test_two_branch_loading_does_not_require_structure_directory(self):
        self.remove_feature_type("structure")
        batch = load_feature_bundle(self.root, enabled_branches=["sequence_global", "inverse_folding"])
        self.assertEqual(set(batch.local_features), {"inverse_folding"})
        self.assertEqual(batch.global_features.shape, (4, 2))
        self.assertEqual(batch.num_scores, 2)
        self.assertNotIn("structure", batch.feature_signature)
        self.assertEqual(json.loads(batch.feature_signature["enabled_branches"]), ["inverse_folding", "sequence_global"])
        with self.assertRaisesRegex(ValueError, "Missing default feature descriptor: structure"):
            load_feature_bundle(self.root)

    def test_inverse_only_does_not_require_either_embedding_directory(self):
        self.remove_feature_type("sequence")
        self.remove_feature_type("structure")
        batch = load_feature_bundle(self.root, enabled_branches=["inverse_folding"])
        self.assertIsNone(batch.global_features)
        self.assertEqual(tuple(batch.local_features), ("inverse_folding",))
        self.assertEqual(batch.score_names, ("inverse_folding/synthetic/v1",))

    def test_disabled_corrupt_cache_is_not_validated(self):
        (self.root / "inverse_folding/synthetic/v1/features.pt").write_bytes(b"invalid tensor file")
        (self.root / "structure/synthetic/v1/config.json").write_text("invalid json", encoding="utf-8")
        batch = load_feature_bundle(self.root, enabled_branches=["sequence_site"])
        self.assertEqual(tuple(batch.local_features), ("sequence_site",))
        self.assertIsNone(batch.global_features)

    def test_local_only_and_global_only_selection_and_device_transfer(self):
        for branch in BRANCH_NAMES:
            with self.subTest(branch=branch):
                batch = load_feature_bundle(self.root, enabled_branches=[branch])
                selected = batch.select([3, 0, 1]).to("cpu")
                self.assertEqual(selected.variant_ids, ("A1V:C2D", "WT", "A1V"))
                self.assertEqual(selected.owners.tolist(), [0, 0, 2])
                self.assertEqual(selected.mutation_positions.tolist(), [1, 2, 1])
                self.assertEqual(selected.device, torch.device("cpu"))
                self.assertEqual(selected.scores.dtype, torch.float64)
                self.assertEqual(selected.feature_signature, batch.feature_signature)
                for name, values in selected.local_features.items():
                    torch.testing.assert_close(values, batch.local_features[name][[2, 3, 0]])
                if branch == "sequence_global":
                    torch.testing.assert_close(selected.global_features, batch.global_features[[3, 0, 1]])
                    self.assertEqual(selected.local_features, {})
                else:
                    self.assertIsNone(selected.global_features)
                wildtype = selected.select([1])
                self.assertEqual(wildtype.owners.numel(), 0)
                self.assertEqual(selected.select([]).num_variants, 0)

    def test_explicit_feature_selection_needs_no_default_descriptor(self):
        metadata_path = self.root / "assay.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["features"] = {}
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        batch = load_feature_bundle(
            self.root,
            feature_selection={"inverse_folding": "synthetic/v1"},
            enabled_branches=["inverse_folding"],
        )
        self.assertEqual(batch.score_names, ("inverse_folding/synthetic/v1",))
        with self.assertRaisesRegex(ValueError, "Missing default feature descriptor: inverse_folding"):
            load_feature_bundle(self.root, enabled_branches=["inverse_folding"])

    def test_explicit_scores_are_independent_of_representation_branches(self):
        (self.root / "structure/synthetic/v1/features.pt").unlink()
        batch = load_feature_bundle(
            self.root,
            enabled_branches=["sequence_global"],
            score_sources=["structure/synthetic/v1"],
        )
        self.assertEqual(batch.local_features, {})
        self.assertEqual(batch.score_names, ("structure/synthetic/v1",))
        torch.testing.assert_close(batch.scores[:, 0], torch.arange(10, 14, dtype=torch.float64))
        self.assertNotIn("structure", batch.feature_signature)
        self.assertIn("score:structure/synthetic/v1.config_sha256", batch.feature_signature)

    def test_missing_selected_cache_still_fails(self):
        (self.root / "structure/synthetic/v1/features.pt").unlink()
        with self.assertRaises(FileNotFoundError):
            load_feature_bundle(self.root, enabled_branches=["structure"])

    def test_invalid_branch_selections_and_empty_feature_batch_are_rejected(self):
        for selection in ([], ["sequence"], ["sequence_global", "sequence_global"], "sequence_global"):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                load_feature_bundle(self.root, enabled_branches=selection)
        batch = load_feature_bundle(self.root, enabled_branches=["sequence_global"])
        with self.assertRaisesRegex(ValueError, "At least one"):
            replace(batch, global_features=None)
        with self.assertRaisesRegex(ValueError, "Unknown local feature"):
            replace(batch, local_features={"unknown": torch.ones((4, 1), dtype=torch.float64)})
        with self.assertRaisesRegex(ValueError, "at least one scorer"):
            replace(batch, scores=torch.empty((4, 0), dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "at least one unique"):
            load_feature_bundle(self.root, enabled_branches=["sequence_global"], score_sources=[])


if __name__ == "__main__":
    unittest.main()
