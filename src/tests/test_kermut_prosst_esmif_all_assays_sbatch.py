"""Check full-catalog ProSST/ESM-IF dispatch without training or submitting jobs."""

import csv
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from . import test_kermut_small_assays_sbatch as small_tests


ROOT = small_tests.array_tests.REPOSITORY
SCRIPT = Path("sbatch_run/run_kermut/run_kermut_prosst_esmif_all_assays.sbatch")
SOURCE = (ROOT / SCRIPT).read_text(encoding="utf-8")
ASSAYS = tuple(re.findall(r'^  "([^"]+)"$', SOURCE, re.MULTILINE))
PRESET = "configs/kermut_prosst_esmif.json"


class AllAssayCatalogTests(unittest.TestCase):
    @unittest.skipUnless(small_tests.CATALOG, "Local ProteinGym catalog is required")
    def test_snapshot_includes_every_catalog_assay_in_order(self):
        with small_tests.CATALOG.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(ASSAYS, tuple(row["DMS_id"] for row in rows))
        self.assertEqual(len(ASSAYS), len(set(ASSAYS)))
        self.assertEqual(len(ASSAYS), 217)
        self.assertTrue(any(int(row["DMS_number_single_mutants"]) > 1000 for row in rows))
        self.assertIn(f"#SBATCH --array=0-{len(ASSAYS) - 1}\n", SOURCE)

    def test_resources_match_the_prosst_esmif_small_launcher(self):
        reference = (ROOT / SCRIPT.with_name(
            "run_kermut_prosst_esmif_small_assays.sbatch")).read_text(encoding="utf-8")
        pattern = r"^#SBATCH --([^=]+)=(.+)$"
        expected = dict(re.findall(pattern, reference, re.MULTILINE))
        actual = dict(re.findall(pattern, SOURCE, re.MULTILINE))
        self.assertEqual(actual.pop("job-name"), "kermut_prosst_esmif_all")
        self.assertEqual(actual.pop("array"), "0-216")
        expected.pop("job-name")
        expected.pop("array")
        self.assertEqual(actual, expected)


class AllAssayArrayTests(small_tests.SmallAssayArraySbatchTests):
    assays = ASSAYS
    script_path = SCRIPT
    default_config = PRESET

    def setUp(self):
        super().setUp()
        environment = patch.dict("os.environ", {"KERMUT_RUN_PREFIX": ""})
        environment.start()
        self.addCleanup(environment.stop)
        capture = self.root / "capture.py"
        source = capture.read_text().replace(
            "record = {", "print('PREFIX=' + os.environ.get('KERMUT_RUN_PREFIX', ''))\nrecord = {",
        )
        capture.write_text(source, encoding="utf-8")

    def test_all_five_tasks_select_the_requested_assay_and_default_config(self):
        # Override the inherited subset test: catalog coverage is checked above;
        # execute each shell validation boundary and both ends of the full list.
        for index in (0, 1, 56, 57, 99, 100, 199, 200, 209, 210, 216):
            with self.subTest(index=index):
                result, record = self.invoke(index)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(record, {"assay": ASSAYS[index], "config": PRESET,
                                          "arguments": []})
                self.assertIn("PREFIX=kermut_prosst_esmif\n", result.stdout)

    def test_help_lists_tasks_without_launching_a_runner(self):
        result, record = self.invoke(None, "--help", submit_directory=self.workspace)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNone(record)
        self.assertIn("Default array: 0-216", result.stdout)
        self.assertIn("all 217 catalog assays (no size filter)", result.stdout)
        for index, assay in enumerate(ASSAYS):
            self.assertIn(f"  {index}: {assay}\n", result.stdout)

    def test_custom_result_prefix_is_preserved(self):
        with patch.dict("os.environ", {"KERMUT_RUN_PREFIX": "kermut_custom_prosst_esmif"}):
            result, _ = self.invoke(216)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PREFIX=kermut_custom_prosst_esmif\n", result.stdout)


if __name__ == "__main__":
    unittest.main()
