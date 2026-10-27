"""Manifest recovery copies verified JSON snapshots without changing cached data."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "restore_manifests", ROOT / "sbatch_run/run_prepare_rep/restore_proteinmpnn_manifests.py")
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


class ManifestRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.outputs = self.root / "features"
        self.manifests = self.root / "structures/manifests"
        self.raw = self.root / "structures/raw/input.pdb"
        self.raw.parent.mkdir(parents=True)
        self.raw.write_bytes(b"original saved file\n")
        self.metadata = {
            "assay_id": "tiny", "schema_version": 1, "input_signature": "a" * 64,
            "structure_path": "../raw/input.pdb", "structure_sha256": recovery.sha256(self.raw),
        }
        self.leaf = self.make_cache("legacy", self.metadata)

    def make_cache(self, name, metadata):
        leaf = self.outputs / metadata["assay_id"] / "inverse_folding/proteinmpnn" / name
        leaf.mkdir(parents=True)
        config = {"model_family": "proteinmpnn", "feature_type": "inverse_folding", "checkpoint": name,
                  "structure": metadata}
        path = leaf / "config.json"
        path.write_text(json.dumps(config))
        completion = {
            "model_family": "proteinmpnn", "cache_id": name,
            "config_signature": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "structure_input_signature": metadata["input_signature"],
            "files_sha256": {"config.json": recovery.sha256(path)},
        }
        (leaf / "completion.json").write_text(json.dumps(completion))
        return leaf

    def test_preview_then_restore_exact_snapshot_and_preserve_existing_files(self):
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        status, destination = recovery.restore_one(self.outputs, self.manifests, "tiny")
        self.assertEqual(status, "RECOVERABLE")
        self.assertFalse(self.manifests.exists())
        status, restored = recovery.restore_one(self.outputs, self.manifests, "tiny", apply=True)
        self.assertEqual((status, restored), ("RESTORED", destination))
        self.assertEqual(json.loads(destination.read_text()), self.metadata)
        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents)
        self.assertEqual(recovery.restore_one(self.outputs, self.manifests, "tiny", apply=True)[0], "EXISTS")
        self.assertEqual(list(self.manifests.iterdir()), [destination])

    def test_nested_file_references_are_verified_and_preserved(self):
        for field, value in (("source", self.metadata), ("fragments", [{"structure": self.metadata}])):
            with self.subTest(field=field):
                nested = {field: value}
                recovery.verify_referenced_files(nested, self.manifests)
        self.raw.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            recovery.verify_referenced_files({"fragments": [{"structure": self.metadata}]}, self.manifests)

    def test_corrupt_snapshot_is_not_published(self):
        (self.leaf / "config.json").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "config checksum mismatch"):
            recovery.restore_one(self.outputs, self.manifests, "tiny", apply=True)
        self.assertFalse(self.manifests.exists())

    def test_conflicting_complete_snapshots_are_not_selected_arbitrarily(self):
        self.make_cache("proteingym--other", dict(self.metadata, input_signature="b" * 64))
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            recovery.restore_one(self.outputs, self.manifests, "tiny", apply=True)
        self.assertFalse(self.manifests.exists())

    def test_matching_current_and_legacy_snapshots_can_restore(self):
        self.make_cache("proteingym--legacy", self.metadata)
        self.assertEqual(recovery.restore_one(self.outputs, self.manifests, "tiny", apply=True)[0], "RESTORED")

    def test_missing_raw_file_fails_without_creating_manifest(self):
        self.raw.unlink()
        with self.assertRaises(FileNotFoundError):
            recovery.restore_one(self.outputs, self.manifests, "tiny", apply=True)
        self.assertFalse(self.manifests.exists())

    def test_other_collection_and_incomplete_snapshots_are_not_used(self):
        (self.leaf / "completion.json").unlink()
        self.make_cache("protssn_proteingym--other", self.metadata)
        with self.assertRaisesRegex(ValueError, "No completed ProteinGym"):
            recovery.restore_one(self.outputs, self.manifests, "tiny", apply=True)

    def test_concurrent_existing_destination_is_never_overwritten(self):
        self.manifests.mkdir()
        destination = self.manifests / "tiny.json"
        destination.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            recovery.write_missing(destination, self.metadata)
        self.assertEqual(destination.read_bytes(), b"keep")
        self.assertEqual(list(self.manifests.iterdir()), [destination])

    def test_batch_continues_after_failure_and_reports_nonzero(self):
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            status = recovery.main(["--output-root", str(self.outputs), "--manifest-dir", str(self.manifests),
                                    "--assay", "missing", "--assay", "tiny", "--apply"])
        self.assertEqual(status, 1)
        self.assertIn("FAILED missing", log.getvalue())
        self.assertIn("RESTORED tiny", log.getvalue())
        self.assertTrue((self.manifests / "tiny.json").is_file())

    def test_assay_cannot_escape_destination(self):
        with self.assertRaisesRegex(ValueError, "Invalid assay"):
            recovery.restore_one(self.outputs, self.manifests, "../outside", apply=True)


if __name__ == "__main__":
    unittest.main()
