"""Catalog discovery and table lookup without model dependencies or Slurm."""

import json
from pathlib import Path
import tempfile
import unittest

from src.assay_paths import resolve_assay_csv
from sbatch_run.run_prepare_rep.list_assays import list_assays
from . import test_nkn_sbatch as runner_tests


class AssayTablePathTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = self.root / "DMS_substitutions.csv"
        self.config = self.root / "extraction.json"
        self.config.write_text(json.dumps({"assay_index": str(self.index)}))
        self.write_index("tiny.csv")

    def write_index(self, filename):
        self.index.write_text(f"DMS_id,DMS_filename,target_seq\ntiny,{filename},ACD\n")

    def check_layout(self, directory, filename="tiny.csv"):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "tiny.csv"
        path.write_text("mutant,mutated_sequence,DMS_score\nA1C,CCD,1\n")
        self.assertEqual(resolve_assay_csv(self.index, filename), path.resolve())
        self.assertEqual(list_assays(self.config), ["tiny"])

    def test_tables_beside_catalog(self):
        self.check_layout(self.root)

    def test_tables_in_raw_substitutions(self):
        self.check_layout(self.root / "raw_substitutions")

    def test_catalog_beside_flat_proteingym_folder(self):
        self.check_layout(self.root / "DMS_ProteinGym_substitutions")

    def test_catalog_beside_proteingym_folder_containing_raw_substitutions(self):
        self.check_layout(self.root / "DMS_ProteinGym_substitutions/raw_substitutions")

    def test_explicit_relative_filename(self):
        filename = "DMS_ProteinGym_substitutions/raw_substitutions/tiny.csv"
        self.write_index(filename)
        self.check_layout(self.root / "DMS_ProteinGym_substitutions/raw_substitutions", filename)

    def test_nearby_table_takes_precedence(self):
        self.check_layout(self.root / "DMS_ProteinGym_substitutions/raw_substitutions")
        self.check_layout(self.root)

    def test_missing_table_and_directory_escape_fail(self):
        with self.assertRaisesRegex(FileNotFoundError, "searched"):
            resolve_assay_csv(self.index, "tiny.csv")
        with self.assertRaises(ValueError):
            list_assays(self.config)
        self.write_index("../tiny.csv")
        with self.assertRaisesRegex(ValueError, "inside the index directory"):
            resolve_assay_csv(self.index, "../tiny.csv")
        with self.assertRaises(ValueError):
            list_assays(self.config)


@unittest.skipUnless(runner_tests.BASH, "Bash is required for catalog dispatch checks")
class NknCatalogSbatchTests(unittest.TestCase):
    def make_fixture(self):
        fixture = runner_tests.NknSbatchTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        return fixture

    def assert_catalog(self, fixture, expected, **kwargs):
        result, records, _ = fixture.invoke(1, "singles_random", **kwargs)
        self.assertEqual(result.returncode, 0, result.stderr)
        validation = json.loads((fixture.root / "validation.json").read_text())
        self.assertEqual([record["assay_index"] for record in [validation, *records]],
                         [expected] * (1 + len(records)))

    def test_top_level_catalog_is_preferred_over_old_locations(self):
        fixture = self.make_fixture()
        old = fixture.root / "data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv"
        old.parent.mkdir(parents=True)
        old.touch()
        self.assert_catalog(fixture, "data/DMS_substitutions.csv")

    def test_older_catalog_locations_still_work(self):
        for relative in ("data/DMS_ProteinGym_substitutions/DMS_substitutions.csv",
                         "data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv"):
            with self.subTest(path=relative):
                fixture = self.make_fixture()
                (fixture.root / "data/DMS_substitutions.csv").unlink()
                path = fixture.root / relative
                path.parent.mkdir(parents=True)
                path.touch()
                self.assert_catalog(fixture, relative)

    def test_explicit_catalog_overrides_detection(self):
        fixture = self.make_fixture()
        relative = "custom catalog.csv"
        (fixture.root / relative).touch()
        self.assert_catalog(fixture, relative, assay_index=relative)

    def test_missing_catalog_fails_before_creating_results(self):
        fixture = self.make_fixture()
        (fixture.root / "data/DMS_substitutions.csv").unlink()
        result, records, _ = fixture.invoke(1, "singles_random")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ProteinGym catalog not found", result.stderr)
        self.assertEqual(records, [])
        self.assertFalse((fixture.root / "results").exists())

    def test_missing_explicit_catalog_does_not_fall_back(self):
        fixture = self.make_fixture()
        result, records, _ = fixture.invoke(1, "singles_random", assay_index="missing.csv")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ProteinGym catalog does not exist: missing.csv", result.stderr)
        self.assertEqual(records, [])
        self.assertFalse((fixture.root / "results").exists())


if __name__ == "__main__":
    unittest.main()
