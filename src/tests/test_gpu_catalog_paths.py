"""Resolve moved GPU-run catalogs without replacing explicit user selections."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import large_assay_gpu as gpu


class GpuCatalogPathTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        root = patch.object(gpu, "ROOT", self.root)
        root.start()
        self.addCleanup(root.stop)
        self.preferred = self.root / "data/DMS_ProteinGym_substitutions/DMS_substitutions.csv"
        self.data_root = self.root / "data/DMS_substitutions.csv"
        self.legacy = self.root / "data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv"

    def create(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("DMS_id,DMS_filename,target_seq\n", encoding="utf-8")

    def selected(self, *options):
        return gpu._parser().parse_args([
            "--assay", "test", "--output", str(self.root / "results"), *options,
        ]).assay_index

    def test_current_catalog_wins_over_both_legacy_locations(self):
        for path in (self.legacy, self.data_root, self.preferred):
            self.create(path)
        self.assertEqual(self.selected(), self.preferred)

    def test_fallback_order_supports_both_older_layouts(self):
        self.create(self.legacy)
        self.assertEqual(self.selected(), self.legacy)
        self.create(self.data_root)
        self.assertEqual(self.selected(), self.data_root)

    def test_missing_catalog_default_points_to_current_location(self):
        self.assertEqual(self.selected(), self.preferred)

    def test_explicit_override_is_preserved_even_when_missing(self):
        self.create(self.preferred)
        override = self.root / "custom catalog.csv"
        for arguments in (("--assay-index", str(override)), (f"--assay-index={override}",)):
            with self.subTest(arguments=arguments):
                self.assertEqual(self.selected(*arguments), override)
        self.assertFalse(override.exists())


if __name__ == "__main__":
    unittest.main()
