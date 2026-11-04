"""Repair directory-only cache renames without recomputing or trusting bad data."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import torch

from feature_generation.inputs import sha256_file
from feature_generation.storage import _signature
from tests import test_migrate_structure_cache_names as fixtures


migration = fixtures.migration
read_json = fixtures.read_json
snapshot = fixtures.snapshot


class RenamedStructureCacheRepairTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.StructureCacheMigrationTests(
            "test_all_four_cache_kinds_preserve_values_and_update_default_descriptors")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def rename_only(self, keys=None):
        for key in self.fixture.recipes if keys is None else keys:
            self.fixture.leaf(*key).rename(self.fixture.leaf(*key, old=False))

    def repair(self, **kwargs):
        return migration.migrate_assay(self.root, repair_renamed=True, **kwargs)

    def assert_unchanged_failure(self):
        for dry_run in (True, False):
            with self.subTest(dry_run=dry_run):
                before = snapshot(self.root)
                with self.assertRaises((ValueError, RuntimeError, FileExistsError, OSError)):
                    self.repair(dry_run=dry_run)
                self.assertEqual(snapshot(self.root), before)

    def test_all_four_renamed_kinds_preserve_values_provenance_and_coverage(self):
        self.rename_only()
        before = {}
        for key in self.fixture.recipes:
            leaf = self.fixture.leaf(*key, old=False)
            before[key] = {
                "config": read_json(leaf / "config.json"),
                "provenance": (leaf / "provenance.json").read_bytes(),
                "values": ((leaf / "mean_score.csv").read_bytes() if key[0] == "scores"
                           else torch.load(leaf / "features.pt", weights_only=True)),
            }
        variants = (self.root / "variants.csv").read_bytes()
        input_signature = read_json(self.root / "assay.json")["input_signature"]

        result = self.repair()

        self.assertEqual(result["migrated"], 4)
        self.fixture.validate_all()
        self.assertEqual((self.root / "variants.csv").read_bytes(), variants)
        assay = read_json(self.root / "assay.json")
        self.assertEqual(assay["input_signature"], input_signature)
        for key, cache_id in self.fixture.recipes.items():
            with self.subTest(kind=key):
                leaf = self.fixture.leaf(*key, old=False)
                self.assertFalse(self.fixture.leaf(*key).exists())
                config = read_json(leaf / "config.json")
                completion = read_json(leaf / "completion.json")
                self.assertEqual(config, dict(before[key]["config"], checkpoint=cache_id))
                self.assertEqual(completion["cache_id"], cache_id)
                self.assertEqual(completion["config_signature"], _signature(config))
                self.assertEqual((leaf / "provenance.json").read_bytes(), before[key]["provenance"])
                if key[0] == "scores":
                    self.assertEqual((leaf / "mean_score.csv").read_bytes(), before[key]["values"])
                else:
                    payload = torch.load(leaf / "features.pt", weights_only=True)
                    self.fixture.assert_payload_equal(before[key]["values"], payload)
                    self.assertEqual(payload["config_signature"], _signature(config))
                    self.assertEqual(assay["features"][key[0]], {
                        "model_family": key[1], "checkpoint": cache_id,
                        "path": f"{key[0]}/{key[1]}/{cache_id}",
                    })
        self.assertFalse(list(self.root.glob(migration.WORK_PREFIX + "*")))

    def test_dry_run_preserves_every_file_and_timestamp_without_creating_lock(self):
        self.rename_only()
        (self.root / "extraction.lock").unlink()
        before = snapshot(self.root)
        result = self.repair(dry_run=True)
        self.assertEqual(result["planned"], 4)
        self.assertEqual(result["migrated"], 0)
        self.assertEqual(snapshot(self.root), before)
        self.assertFalse((self.root / "extraction.lock").exists())
        self.assertFalse(list(self.root.glob(migration.WORK_PREFIX + "*")))

    def test_repeated_repair_is_idempotent(self):
        self.rename_only()
        self.repair()
        before = snapshot(self.root)
        result = self.repair()
        self.assertEqual(result["migrated"], 0)
        self.assertEqual(snapshot(self.root), before)
        self.fixture.validate_all()

    def test_mixed_old_migrated_and_renamed_leaves_keep_valid_cache_unchanged(self):
        migrated_key = ("scores", "prosst")
        migration._migrate_leaf(self.root, *migrated_key, self.fixture.leaf(*migrated_key).name)
        self.rename_only([("sequence", "prosst")])
        completed = self.fixture.leaf(*migrated_key, old=False)
        before = snapshot(completed)
        result = self.repair()
        self.assertEqual(result["migrated"], 3)
        self.assertEqual(snapshot(completed), before)
        self.fixture.validate_all()

    def test_existing_original_name_refuses_before_changing_siblings(self):
        key = migration.TARGETS[-1]
        shutil.copytree(self.fixture.leaf(*key), self.fixture.leaf(*key, old=False))
        self.assert_unchanged_failure()

    def test_corrupt_completion_is_rejected_before_any_repair(self):
        self.rename_only()
        path = self.fixture.leaf(*migration.TARGETS[-1], old=False) / "completion.json"
        completion = read_json(path)
        completion["cache_id"] = "unrelated_recipe"
        path.write_text(json.dumps(completion), encoding="utf-8")
        self.assert_unchanged_failure()

    def test_rehashed_wrong_feature_family_is_rejected_before_any_repair(self):
        self.rename_only()
        leaf = self.fixture.leaf("sequence", "prosst", old=False)
        config = read_json(leaf / "config.json")
        config["model_family"] = "esm2"
        (leaf / "config.json").write_text(json.dumps(config), encoding="utf-8")
        completion = read_json(leaf / "completion.json")
        completion["files_sha256"]["config.json"] = sha256_file(leaf / "config.json")
        (leaf / "completion.json").write_text(json.dumps(completion), encoding="utf-8")
        self.assert_unchanged_failure()

    def test_rehashed_payload_signature_mismatch_is_rejected_before_any_repair(self):
        self.rename_only()
        leaf = self.fixture.leaf("inverse_folding", "esm_if1", old=False)
        payload = torch.load(leaf / "features.pt", weights_only=True)
        payload["config_signature"] = "f" * 64
        torch.save(payload, leaf / "features.pt")
        completion = read_json(leaf / "completion.json")
        completion["files_sha256"]["features.pt"] = sha256_file(leaf / "features.pt")
        (leaf / "completion.json").write_text(json.dumps(completion), encoding="utf-8")
        self.assert_unchanged_failure()

    def test_failed_default_update_restores_renamed_layout_and_original_bytes(self):
        self.rename_only()
        update = migration._update_default

        def fail_after_update(root, kind, family, old_id, new_id):
            update(root, kind, family, old_id, new_id)
            if old_id.startswith(migration.PREFIX):
                raise OSError("injected repair publication failure")

        for current_defaults in (False, True):
            with self.subTest(current_defaults=current_defaults):
                if current_defaults:
                    path = self.root / "assay.json"
                    assay = read_json(path)
                    for kind in ("sequence", "inverse_folding"):
                        descriptor = assay["features"][kind]
                        descriptor["checkpoint"] = descriptor["checkpoint"].removeprefix(migration.PREFIX)
                        descriptor["path"] = f"{kind}/{descriptor['model_family']}/{descriptor['checkpoint']}"
                    path.write_text(json.dumps(assay), encoding="utf-8")
                before = {name: contents for name, (contents, _) in snapshot(self.root).items()}
                with patch.object(migration, "_update_default", side_effect=fail_after_update):
                    with self.assertRaisesRegex(OSError, "injected repair"):
                        self.repair()
                self.assertEqual({name: contents for name, (contents, _) in snapshot(self.root).items()}, before)
        self.assertEqual(self.repair()["migrated"], 4)
        self.fixture.validate_all()

    def test_next_run_recovers_interrupted_repair_publication(self):
        self.rename_only()
        update = migration._update_default

        def interrupt_after_update(root, kind, family, old_id, new_id):
            update(root, kind, family, old_id, new_id)
            raise KeyboardInterrupt("interrupted repair publication")

        with patch.object(migration, "_update_default", side_effect=interrupt_after_update), \
                patch.object(migration, "_recover", side_effect=KeyboardInterrupt("process stopped")):
            with self.assertRaises(KeyboardInterrupt):
                self.repair()
        pending = list(self.root.glob(migration.WORK_PREFIX + "*"))
        self.assertEqual(len(pending), 1)
        self.assertTrue((pending[0] / "old").is_dir())
        result = self.repair()
        self.assertEqual(result["recovered"], 1)
        self.assertEqual(result["migrated"], 4)
        self.assertFalse(list(self.root.glob(migration.WORK_PREFIX + "*")))
        self.fixture.validate_all()

    def test_next_run_finishes_cleanup_of_committed_repair(self):
        self.rename_only()
        remove_tree = migration.shutil.rmtree

        def interrupt_backup_removal(path, *args, **kwargs):
            if Path(path).name == "old":
                raise OSError("interrupted repair cleanup")
            return remove_tree(path, *args, **kwargs)

        with patch.object(migration.shutil, "rmtree", side_effect=interrupt_backup_removal):
            with self.assertRaisesRegex(OSError, "interrupted repair cleanup"):
                self.repair()
        pending = list(self.root.glob(migration.WORK_PREFIX + "*"))
        self.assertEqual(len(pending), 1)
        self.assertEqual(read_json(pending[0] / "transaction.json")["phase"], "committed")
        result = self.repair()
        self.assertEqual(result["recovered"], 1)
        self.assertEqual(result["migrated"], 3)
        self.assertFalse(list(self.root.glob(migration.WORK_PREFIX + "*")))
        self.fixture.validate_all()

    def test_cli_flag_previews_and_repairs_only_the_selected_assay(self):
        self.rename_only()
        self.fixture.build(["other"])
        other = self.fixture.output_root / "other"
        before = snapshot(self.fixture.output_root)
        arguments = ["--output-root", str(self.fixture.output_root), "--assay", "tiny", "--repair-renamed"]
        with contextlib.redirect_stdout(io.StringIO()):
            status = migration.main([*arguments, "--dry-run"])
        self.assertEqual(status, 0)
        self.assertEqual(snapshot(self.fixture.output_root), before)
        other_before = snapshot(other)
        with contextlib.redirect_stdout(io.StringIO()):
            status = migration.main(arguments)
        self.assertEqual(status, 0)
        self.assertEqual(snapshot(other), other_before)
        self.fixture.validate_all()


if __name__ == "__main__":
    unittest.main()
