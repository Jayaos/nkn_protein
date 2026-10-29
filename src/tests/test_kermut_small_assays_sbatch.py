"""Check the small-assay Kermut selection and delegation to the existing runner."""

import csv
from pathlib import Path
import re
import unittest

from . import test_nkn_array_sbatch as array_tests


SCRIPT = Path("sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn_small_assays.sbatch")
SHARED_RUNNER = Path("sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn.sbatch")
SOURCE = (array_tests.REPOSITORY / SCRIPT).read_text(encoding="utf-8")
ASSAYS = tuple(re.findall(r'^  "([^"]+)"$', SOURCE, re.MULTILINE))
CATALOG = next((array_tests.REPOSITORY / path for path in (
    "data/DMS_substitutions.csv",
    "data/DMS_ProteinGym_substitutions/DMS_substitutions.csv",
    "data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv",
) if (array_tests.REPOSITORY / path).is_file()), None)


class SmallAssayCatalogTests(unittest.TestCase):
    @unittest.skipUnless(CATALOG, "Local ProteinGym catalog is required")
    def test_snapshot_matches_inclusive_limit_in_catalog_order(self):
        with CATALOG.open(encoding="utf-8-sig", newline="") as handle:
            expected = tuple(row["DMS_id"] for row in csv.DictReader(handle)
                             if 0 < int(row["DMS_number_single_mutants"]) <= 1000)
        self.assertEqual(ASSAYS, expected)
        self.assertEqual(len(ASSAYS), len(set(ASSAYS)))
        self.assertIn(f"#SBATCH --array=0-{len(expected) - 1}\n", SOURCE)

    def test_resources_match_the_shared_launcher(self):
        shared = (array_tests.REPOSITORY / SHARED_RUNNER).read_text(encoding="utf-8")
        for name in ("account", "nodes", "ntasks", "cpus-per-task", "mem", "time", "qos"):
            with self.subTest(resource=name):
                pattern = rf"^#SBATCH --{name}=(.+)$"
                self.assertEqual(re.search(pattern, SOURCE, re.MULTILINE)[1],
                                 re.search(pattern, shared, re.MULTILINE)[1])
        self.assertIn("#SBATCH --output=%x-%A_%a.out\n", SOURCE)


class SmallAssayArraySbatchTests(array_tests.NknArraySbatchTests):
    assays = ASSAYS
    script_path = SCRIPT
    shared_runner = SHARED_RUNNER
    config_variable = "KERMUT_CONFIG"
    default_config = "configs/kermut_esm2_proteinmpnn.json"

    def test_all_five_tasks_select_the_requested_assay_and_default_config(self):
        # Exercise every task, replacing the inherited five-assay contract.
        for index, assay in enumerate(self.assays):
            with self.subTest(index=index):
                result, record = self.invoke(index)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(record, {"assay": assay, "config": self.default_config,
                                          "arguments": []})

    def test_missing_or_invalid_array_id_does_not_launch_shared_runner(self):
        for index in (None, "", "-1", str(len(self.assays)), "99999999999999999999",
                      "not-an-index", "1+1", "01", " 1"):
            with self.subTest(index=index):
                result, record = self.invoke(index)
                self.assertNotEqual(result.returncode, 0)
                self.assertIsNone(record)
                self.assertIn("SLURM_ARRAY_TASK_ID", result.stderr)

    def test_environment_config_and_all_cli_arguments_are_preserved(self):
        arguments = ("singles_random", "--parallel", "--config", "configs/custom config.json",
                     "--experiment-suffix", "comparison_v1")
        result, record = self.invoke(1, *arguments, config="configs/environment config.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(record, {"assay": self.assays[1], "config": "configs/environment config.json",
                                  "arguments": list(arguments)})

    def test_help_lists_tasks_without_launching_a_runner(self):
        result, record = self.invoke(None, "--help", submit_directory=self.workspace)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNone(record)
        self.assertIn("1-1000 single mutants (inclusive)", result.stdout)
        for index, assay in enumerate(self.assays):
            self.assertIn(f"  {index}: {assay}\n", result.stdout)


if __name__ == "__main__":
    unittest.main()
