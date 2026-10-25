"""Replacement scopes, publication failures, and immutable-cache defaults."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from data_loading.features import read_csv
from feature_generation import storage
from feature_generation.score_storage import make_score_config, validate_score_cache, write_score_cache
from tests import test_mean_score_storage as fixtures


class CacheCleanupTests(unittest.TestCase):
    setUp = fixtures.MeanScoreStorageTests.setUp

    def write_sequence(self, cache_id="old", cleanup_and_save=False, resume=False):
        config = storage.make_leaf_config(cache_id, {"hidden_size": 2})
        return storage.write_sequence_cache(
            self.root, cache_id, self.inputs, config, torch.ones(3, 3, 2), {},
            cleanup_and_save=cleanup_and_save, resume=resume,
        )

    def snapshot(self):
        return {path.relative_to(self.root): path.read_bytes()
                for path in self.root.rglob("*") if path.is_file()}

    def test_default_preserves_other_recipes_and_refuses_overwrite(self):
        self.write_sequence("first")
        self.write_sequence("second")
        before = self.snapshot()
        with self.assertRaises(FileExistsError):
            self.write_sequence("first")
        self.assertEqual(self.snapshot(), before)

    def test_cleanup_replaces_all_family_recipes_and_updates_default(self):
        self.write_sequence("first")
        self.write_sequence("second")
        (self.root / "sequence/esm2/incomplete").mkdir()
        preserved = [
            self.root / "sequence/prosst/other/keep.txt",
            self.root / "inverse_folding/esm_if1/other/keep.txt",
            self.root / "scores/esm2/old/keep.txt",
            self.root.parent / "another-assay/sequence/esm2/old/keep.txt",
        ]
        for path in preserved:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"preserve")
        variants = (self.root / "variants.csv").read_bytes()
        leaf = self.write_sequence("new", cleanup_and_save=True)
        self.assertEqual(leaf, self.root / "sequence/esm2/new")
        self.assertEqual(list(leaf.parent.iterdir()), [leaf])
        report = storage.validate_sequence_cache(self.root)
        self.assertEqual(report["cache_id"], "new")
        self.assertEqual((self.root / "variants.csv").read_bytes(), variants)
        for path in preserved:
            self.assertEqual(path.read_bytes(), b"preserve")
        self.assertFalse(list(self.root.glob(".*.replacement-*")))

    def test_cleanup_repairs_corrupted_cache_at_the_same_id(self):
        leaf = self.write_sequence()
        (leaf / "features.pt").write_bytes(b"corrupted")
        self.write_sequence(cleanup_and_save=True, resume=True)
        storage.validate_sequence_cache(self.root)

    def test_resume_cleanup_preserves_current_bytes_and_timestamps(self):
        self.write_sequence("first")
        current = self.write_sequence("second")
        before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in current.iterdir()}
        config = storage.make_leaf_config("second", {"hidden_size": 2})
        self.assertTrue(storage.check_existing(
            self.root, "second", self.inputs, config, resume=True, cleanup_and_save=True,
        ))
        self.assertEqual(list(current.parent.iterdir()), [current])
        self.assertEqual({path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                          for path in current.iterdir()}, before)
        self.assertEqual(storage.validate_sequence_cache(self.root)["cache_id"], "second")

    def test_cleanup_without_resume_never_overwrites_current_or_prunes_on_error(self):
        self.write_sequence("first")
        self.write_sequence("second")
        before = self.snapshot()
        with self.assertRaises(FileExistsError):
            self.write_sequence("second", cleanup_and_save=True)
        self.assertEqual(self.snapshot(), before)

    def test_resume_repairs_invalid_current_without_cleanup_and_preserves_siblings(self):
        leaf = self.write_sequence()
        sibling = self.write_sequence("other")
        other_bytes = {path.name: path.read_bytes() for path in sibling.iterdir()}
        config = storage.make_leaf_config("old", {"hidden_size": 2})
        for missing in ("completion.json", "features.pt", "config.json"):
            with self.subTest(missing=missing):
                (leaf / missing).unlink()
                self.assertFalse(storage.check_existing(self.root, "old", self.inputs, config, resume=True))
                self.write_sequence(resume=True)
                storage.validate_sequence_cache(self.root, "old", model_family="esm2")
                self.assertEqual({path.name: path.read_bytes() for path in sibling.iterdir()}, other_bytes)

    def test_failed_repair_keeps_invalid_current_and_siblings(self):
        leaf = self.write_sequence()
        self.write_sequence("other")
        (leaf / "completion.json").unlink()
        before = self.snapshot()
        with patch("torch.save", side_effect=OSError("disk failed")):
            with self.assertRaisesRegex(OSError, "disk failed"):
                self.write_sequence(resume=True, cleanup_and_save=True)
        self.assertEqual(self.snapshot(), before)

    def test_writer_never_replaces_a_completed_current_cache_even_with_resume(self):
        self.write_sequence()
        before = self.snapshot()
        with self.assertRaises(FileExistsError):
            self.write_sequence(resume=True, cleanup_and_save=True)
        self.assertEqual(self.snapshot(), before)

    def test_cleanup_does_not_override_valid_configuration_conflicts(self):
        self.write_sequence()
        self.write_sequence("other")
        before = self.snapshot()
        changed = storage.make_leaf_config("old", {"hidden_size": 3})
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            storage.check_existing(self.root, "old", self.inputs, changed,
                                   resume=True, cleanup_and_save=True)
        self.assertEqual(self.snapshot(), before)

    def test_failed_repair_promotion_restores_invalid_current(self):
        leaf = self.write_sequence()
        self.write_sequence("other")
        (leaf / "completion.json").unlink()
        before = self.snapshot()
        rename = Path.rename

        def fail_promotion(path, target):
            if target == leaf and path.parent.name == "esm2":
                raise OSError("repair publication failed")
            return rename(path, target)

        with patch.object(Path, "rename", fail_promotion):
            with self.assertRaisesRegex(OSError, "repair publication failed"):
                self.write_sequence(resume=True, cleanup_and_save=True)
        self.assertEqual(self.snapshot(), before)

    def test_cleanup_preserves_a_default_from_another_family(self):
        self.write_sequence()
        path = self.root / "assay.json"
        metadata = json.loads(path.read_text())
        descriptor = {"model_family": "prosst", "checkpoint": "keep", "path": "sequence/prosst/keep"}
        metadata["features"]["sequence"] = descriptor
        path.write_text(json.dumps(metadata))
        self.write_sequence("new", cleanup_and_save=True)
        self.assertEqual(json.loads(path.read_text())["features"]["sequence"], descriptor)
        storage.validate_sequence_cache(self.root, "new", model_family="esm2")

    def test_cleanup_can_publish_without_any_previous_cache(self):
        self.write_sequence("new", cleanup_and_save=True)
        self.assertEqual(storage.validate_sequence_cache(self.root)["cache_id"], "new")

    def test_staging_and_validation_failures_preserve_old_outputs(self):
        self.write_sequence()
        before = self.snapshot()
        for target in ("torch.save", "feature_generation.storage.validate_sequence_cache"):
            with self.subTest(target=target), patch(target, side_effect=RuntimeError("staging failed")):
                with self.assertRaisesRegex(RuntimeError, "staging failed"):
                    self.write_sequence("new", cleanup_and_save=True)
            self.assertEqual(self.snapshot(), before)
            self.assertFalse(list(self.root.glob(".*.replacement-*")))

    def test_failed_default_publication_restores_old_family_and_metadata(self):
        self.write_sequence()
        before = self.snapshot()
        atomic_json = storage._atomic_json

        def fail_default(path, value):
            if path == self.root / "assay.json":
                raise OSError("metadata publication failed")
            return atomic_json(path, value)

        with patch.object(storage, "_atomic_json", side_effect=fail_default):
            with self.assertRaisesRegex(OSError, "metadata publication failed"):
                self.write_sequence("new", cleanup_and_save=True)
        self.assertEqual(self.snapshot(), before)
        storage.validate_sequence_cache(self.root)

    def test_failed_directory_promotion_restores_old_family(self):
        self.write_sequence()
        before = self.snapshot()
        rename = Path.rename
        destination = self.root / "sequence/esm2/new"

        def fail_promotion(path, target):
            if target == destination and path != destination:
                raise OSError("directory publication failed")
            return rename(path, target)

        with patch.object(Path, "rename", fail_promotion):
            with self.assertRaisesRegex(OSError, "directory publication failed"):
                self.write_sequence("new", cleanup_and_save=True)
        self.assertEqual(self.snapshot(), before)
        storage.validate_sequence_cache(self.root)

    def test_cleanup_rejects_redirected_family_without_touching_target(self):
        self.root.mkdir(parents=True)
        outside = self.root.parent / "external"
        outside.mkdir()
        marker = outside / "keep.txt"
        marker.write_bytes(b"preserve")
        parent = self.root / "sequence"
        parent.mkdir()
        try:
            (parent / "esm2").symlink_to(outside, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Directory symlinks unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "redirected|escapes"):
            self.write_sequence(cleanup_and_save=True)
        self.assertEqual(marker.read_bytes(), b"preserve")

    def test_score_cleanup_is_independent_of_features_and_updates_values(self):
        self.write_sequence()
        for cache_id in ("first", "second"):
            config = make_score_config(cache_id, "proteinmpnn", self.metadata, self.structure)
            write_score_cache(self.root, "proteinmpnn", cache_id, self.inputs, config, self.scores, self.provenance)
        preserved = self.snapshot()
        preserved = {path: value for path, value in preserved.items() if path.parts[0] != "scores"}
        leaf = write_score_cache(
            self.root, "proteinmpnn", "official_v1", self.inputs, self.config,
            self.scores - 1, self.provenance, cleanup_and_save=True,
        )
        self.assertEqual(list(leaf.parent.iterdir()), [leaf])
        self.assertEqual(validate_score_cache(self.root, "proteinmpnn")["cache_id"], "official_v1")
        for path, value in preserved.items():
            self.assertEqual((self.root / path).read_bytes(), value)
        self.assertEqual([float(row["score"]) for row in read_csv(leaf / "mean_score.csv")],
                         (self.scores - 1).tolist())


if __name__ == "__main__":
    unittest.main()
