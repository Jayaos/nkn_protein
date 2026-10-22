"""Checkpoint setup checks that run without PyTorch or network access."""

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "msa_runtime_setup", ROOT / "sbatch_run/run_prepare_rep/setup_msa_transformer.py",
)
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


class Response(io.BytesIO):
    def geturl(self):
        return "https://example.invalid/checkpoint.pt"


class CheckpointSetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.spec = {
            "model_id": "test", "filename": "model.pt",
            "url": "https://example.invalid/checkpoint.pt", "size_bytes": 4,
        }

    def download(self, payload=b"test"):
        with patch.object(setup.urllib.request, "urlopen", return_value=Response(payload)):
            return setup.download_checkpoint(self.root, self.spec)

    def test_complete_download_and_offline_reuse(self):
        manifest = self.download()
        self.assertEqual(manifest["sha256"], setup.sha256(self.root / "model.pt"))
        with patch.object(setup.urllib.request, "urlopen", side_effect=AssertionError("network used")):
            self.assertEqual(setup.download_checkpoint(self.root, self.spec), manifest)

    def test_corrupt_cache_is_rejected_without_overwriting(self):
        self.download()
        (self.root / "model.pt").write_bytes(b"oops")
        with self.assertRaisesRegex(ValueError, "checksum or size mismatch"):
            self.download()
        self.assertEqual((self.root / "model.pt").read_bytes(), b"oops")

    def test_wrong_download_sizes_leave_no_completed_artifact(self):
        for payload in (b"x", b"oversized"):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    self.download(payload)
                self.assertEqual(list(self.root.iterdir()), [])

    def test_existing_untracked_weights_are_not_silently_adopted(self):
        (self.root / "model.pt").write_bytes(b"test")
        with self.assertRaisesRegex(ValueError, "No complete checkpoint"):
            self.download()
        self.assertFalse((self.root / "checkpoint_manifest.json").exists())

    def test_changed_artifact_identity_is_rejected(self):
        self.download()
        manifest_path = self.root / "checkpoint_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["url"] = "https://example.invalid/different.pt"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "configured artifact"):
            setup.verify_checkpoint(self.root, self.spec)


if __name__ == "__main__":
    unittest.main()
