"""Offline cache-name migration preserves tensors, provenance, and integrity."""

import contextlib
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from feature_generation.inverse_config import InverseFoldingConfig
from feature_generation.inverse_runner import extract as extract_inverse
from feature_generation.inverse_storage import validate_inverse_cache
from feature_generation.prosst_runner import extract as extract_prosst
from feature_generation.score_runner import score
from feature_generation.score_storage import validate_score_cache
from feature_generation.storage import _signature, validate_sequence_cache
from tests import test_prosst_pipeline as prosst_fixture
from tests.test_esm_if1_scores import adapter_for
from tests.test_prosst_scoring import FakeScoreAdapter


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sbatch_run/run_prepare_rep/migrate_structure_cache_names.py"
SPEC = importlib.util.spec_from_file_location("migrate_structure_cache_names", SCRIPT)
migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migration)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot(root):
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


class StructureCacheMigrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = prosst_fixture.ProSSTPipelineTests(
            "test_reordered_batches_preserve_variant_position_and_token_alignment")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.config = replace(
            self.fixture.config, structure_source="protssn_proteingym",
            cache_id=migration.PREFIX + "prosst_features")
        self.fixture.metadata["source_archive"] = {"source_id": "protssn_proteingym"}
        self.output_root = self.fixture.config.output_root
        self.root = self.output_root / "tiny"
        self.recipes = {
            ("sequence", "prosst"): "prosst_features",
            ("inverse_folding", "esm_if1"): "esm_if1_features",
            ("scores", "prosst"): "prosst_scores",
            ("scores", "esm_if1"): "esm_if1_scores",
        }
        self.build(["tiny"])

    def load_structure(self, manifest_path, reference_sequence, require_oxygen=True):
        structure = self.fixture.load_structure(manifest_path, reference_sequence, require_oxygen)
        structure.metadata["assay_id"] = Path(manifest_path).stem
        return structure

    def build(self, assays):
        config = self.fixture.config
        with patch("feature_generation.prosst_runner.load_structure", side_effect=self.load_structure):
            extract_prosst(config, assays, adapter_factory=prosst_fixture.FakeAdapter,
                           token_factory=self.fixture.token_factory)
        with patch("feature_generation.score_runner.load_structure", side_effect=self.load_structure):
            score(config, assays, adapter_factory=FakeScoreAdapter,
                  token_factory=self.fixture.token_factory, cache_id=migration.PREFIX + "prosst_scores")
        inverse = InverseFoldingConfig(
            model_family="esm_if1", structure_source="protssn_proteingym", device="cpu",
            cache_id=migration.PREFIX + "esm_if1_features", output_root=self.output_root,
            assay_index=config.assay_index, structure_manifest_dir=config.structure_manifest_dir,
            model_dir=self.fixture.root / "inverse_models")
        with patch("feature_generation.inverse_runner.load_structure", side_effect=self.load_structure):
            extract_inverse(inverse, assays, adapter_factory=adapter_for)
        with patch("feature_generation.score_runner.load_structure", side_effect=self.load_structure):
            score(inverse, assays, adapter_factory=adapter_for, cache_id=migration.PREFIX + "esm_if1_scores")

    def leaf(self, kind, family, *, old=True, root=None):
        recipe = self.recipes[kind, family]
        return (root or self.root) / kind / family / ((migration.PREFIX if old else "") + recipe)

    def validate_all(self, root=None, *, old=False):
        root = root or self.root
        for kind, family in self.recipes:
            cache_id = self.leaf(kind, family, old=old, root=root).name
            if kind == "sequence":
                validate_sequence_cache(root, cache_id, model_family=family)
            elif kind == "inverse_folding":
                validate_inverse_cache(root, family, cache_id)
            else:
                validate_score_cache(root, family, cache_id)

    def assert_payload_equal(self, before, after):
        self.assertEqual(set(before), set(after))
        self.assertNotEqual(before["config_signature"], after["config_signature"])
        for key, expected in before.items():
            if key == "config_signature":
                continue
            with self.subTest(payload_key=key):
                if isinstance(expected, torch.Tensor):
                    self.assertTrue(torch.equal(expected, after[key]))
                else:
                    self.assertEqual(expected, after[key])

    def test_all_four_cache_kinds_preserve_values_and_update_default_descriptors(self):
        before = {}
        for key in self.recipes:
            leaf = self.leaf(*key)
            before[key] = {
                "config": read_json(leaf / "config.json"),
                "provenance": (leaf / "provenance.json").read_bytes(),
                "values": ((leaf / "mean_score.csv").read_bytes() if key[0] == "scores"
                           else torch.load(leaf / "features.pt", weights_only=True)),
            }
        assay_before = read_json(self.root / "assay.json")
        variants_before = (self.root / "variants.csv").read_bytes()
        result = migration.migrate_assay(self.root)
        self.assertEqual(result["migrated"], 4)
        self.validate_all()
        self.assertEqual((self.root / "variants.csv").read_bytes(), variants_before)
        assay_after = read_json(self.root / "assay.json")
        self.assertEqual(assay_before["input_signature"], assay_after["input_signature"])
        for key, recipe in self.recipes.items():
            with self.subTest(kind=key):
                self.assertFalse(self.leaf(*key).exists())
                leaf = self.leaf(*key, old=False)
                config = read_json(leaf / "config.json")
                completion = read_json(leaf / "completion.json")
                self.assertEqual(config, dict(before[key]["config"], checkpoint=recipe))
                self.assertEqual(completion["cache_id"], recipe)
                self.assertEqual(completion["config_signature"], _signature(config))
                self.assertEqual((leaf / "provenance.json").read_bytes(), before[key]["provenance"])
                if key[0] == "scores":
                    self.assertEqual((leaf / "mean_score.csv").read_bytes(), before[key]["values"])
                else:
                    after = torch.load(leaf / "features.pt", weights_only=True)
                    self.assert_payload_equal(before[key]["values"], after)
                    self.assertEqual(after["config_signature"], _signature(config))
                    self.assertEqual(assay_after["features"][key[0]], {
                        "model_family": key[1], "checkpoint": recipe,
                        "path": f"{key[0]}/{key[1]}/{recipe}",
                    })

    def test_dry_run_preserves_every_file_and_timestamp(self):
        (self.root / "extraction.lock").unlink()
        before = snapshot(self.root)
        result = migration.migrate_assay(self.root, dry_run=True)
        self.assertEqual(result["planned"], 4)
        self.assertEqual(result["migrated"], 0)
        self.assertEqual(snapshot(self.root), before)

    def test_repeat_run_is_idempotent(self):
        migration.migrate_assay(self.root)
        before = snapshot(self.root)
        result = migration.migrate_assay(self.root)
        self.assertEqual(result["migrated"], 0)
        self.assertEqual(snapshot(self.root), before)
        self.validate_all()

    def test_unrelated_defaults_and_models_are_preserved(self):
        path = self.root / "assay.json"
        metadata = read_json(path)
        metadata["features"] = {
            "sequence": {"model_family": "esm2", "checkpoint": "keep", "path": "sequence/esm2/keep"},
            "inverse_folding": {"model_family": "proteinmpnn", "checkpoint": "keep",
                                "path": "inverse_folding/proteinmpnn/keep"},
        }
        path.write_text(json.dumps(metadata), encoding="utf-8")
        other = self.root / "scores/proteinmpnn" / (migration.PREFIX + "leave_this_recipe")
        other.mkdir(parents=True)
        (other / "untouched.txt").write_bytes(b"unrelated")
        migration.migrate_assay(self.root)
        self.assertEqual(read_json(path), metadata)
        self.assertEqual((other / "untouched.txt").read_bytes(), b"unrelated")

    def test_destination_collision_refuses_to_overwrite(self):
        # Even a conflict in the last target must leave earlier caches untouched.
        key = migration.TARGETS[-1]
        target = self.leaf(*key, old=False)
        target.mkdir()
        (target / "keep.txt").write_bytes(b"existing destination")
        before = snapshot(self.root)
        with self.assertRaises((FileExistsError, ValueError, RuntimeError)):
            migration.migrate_assay(self.root)
        self.assertEqual(snapshot(self.root), before)

    def test_corrupt_source_is_preserved_and_rejected(self):
        key = migration.TARGETS[-1]
        old = self.leaf(*key)
        (old / "config.json").write_bytes(b"corrupt")
        before = snapshot(self.root)
        with self.assertRaises((ValueError, RuntimeError)):
            migration.migrate_assay(self.root)
        self.assertEqual(snapshot(self.root), before)
        self.assertFalse(self.leaf(*key, old=False).exists())

    def test_failed_default_update_rolls_back_the_original_cache(self):
        before = {name: content for name, (content, _) in snapshot(self.root).items()}
        update = migration._update_default

        def fail_after_update(root, kind, family, old_id, new_id):
            update(root, kind, family, old_id, new_id)
            if old_id.startswith(migration.PREFIX):
                raise OSError("injected descriptor publication failure")

        with patch.object(migration, "_update_default", side_effect=fail_after_update):
            with self.assertRaisesRegex(OSError, "injected"):
                migration.migrate_assay(self.root)
        self.assertEqual({name: content for name, (content, _) in snapshot(self.root).items()}, before)
        self.validate_all(old=True)
        self.assertEqual(migration.migrate_assay(self.root)["migrated"], 4)
        self.validate_all()

    def test_cli_assay_filter_migrates_only_requested_assay(self):
        self.build(["other"])
        other = self.output_root / "other"
        before = snapshot(other)
        with contextlib.redirect_stdout(io.StringIO()):
            status = migration.main(["--output-root", str(self.output_root), "--assay", "tiny"])
        self.assertEqual(status, 0)
        self.assertEqual(snapshot(other), before)
        self.validate_all()

    def test_next_run_recovers_an_interrupted_publication(self):
        update = migration._update_default

        def interrupt_after_update(root, kind, family, old_id, new_id):
            update(root, kind, family, old_id, new_id)
            raise KeyboardInterrupt("simulated termination after publication")

        # Prevent in-process rollback to simulate an abrupt process termination.
        with patch.object(migration, "_update_default", side_effect=interrupt_after_update), \
                patch.object(migration, "_recover", side_effect=KeyboardInterrupt("process stopped")):
            with self.assertRaises(KeyboardInterrupt):
                migration.migrate_assay(self.root)
        pending = list(self.root.glob(migration.WORK_PREFIX + "*"))
        self.assertEqual(len(pending), 1)
        self.assertTrue((pending[0] / "old").is_dir())
        result = migration.migrate_assay(self.root)
        self.assertEqual(result["recovered"], 1)
        self.assertEqual(result["migrated"], 4)
        self.assertFalse(list(self.root.glob(migration.WORK_PREFIX + "*")))
        self.validate_all()

    def test_next_run_finishes_cleanup_of_committed_cache(self):
        remove_tree = migration.shutil.rmtree

        def interrupt_backup_removal(path, *args, **kwargs):
            if Path(path).name == "old":
                raise OSError("interrupted cleanup")
            return remove_tree(path, *args, **kwargs)

        with patch.object(migration.shutil, "rmtree", side_effect=interrupt_backup_removal):
            with self.assertRaisesRegex(OSError, "interrupted cleanup"):
                migration.migrate_assay(self.root)
        pending = list(self.root.glob(migration.WORK_PREFIX + "*"))
        self.assertEqual(len(pending), 1)
        self.assertEqual(read_json(pending[0] / "transaction.json")["phase"], "committed")
        result = migration.migrate_assay(self.root)
        self.assertEqual(result["recovered"], 1)
        self.assertEqual(result["migrated"], 3)
        self.assertFalse(list(self.root.glob(migration.WORK_PREFIX + "*")))
        self.validate_all()

    def test_initial_orphan_work_directories_are_recovered_without_touching_caches(self):
        for pending_receipt in (False, True):
            with self.subTest(pending_receipt=pending_receipt):
                work = self.root / (migration.WORK_PREFIX + "initial-orphan")
                work.mkdir()
                if pending_receipt:
                    (work / ".transaction.json.interrupted.tmp").write_bytes(b'{"version":')
                result = migration.migrate_assay(self.root)
                self.assertEqual(result["recovered"], 1)
                self.assertFalse(work.exists())
                self.validate_all()

    def test_cli_iterates_assays_and_continues_after_failure(self):
        self.build(["other"])
        key = migration.TARGETS[0]
        (self.leaf(*key, root=self.output_root / "other") / "config.json").write_bytes(b"corrupt")
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            status = migration.main(["--output-root", str(self.output_root)])
        self.assertNotEqual(status, 0)
        self.assertIn("other", output.getvalue())
        self.assertIn("tiny", output.getvalue())
        self.validate_all()
        self.assertTrue(self.leaf(*key, root=self.output_root / "other").exists())


if __name__ == "__main__":
    unittest.main()
