"""Check the ESM2 small-assay array and compatibility with shared-runner callers."""

import csv
import json
import re
import shutil
import unittest

from . import test_nkn_sbatch as runner_tests
from . import test_nkn_proteinmpnn_prosst_small_assays_sbatch as prosst_tests


SOURCE = runner_tests.SCRIPT.read_text(encoding="utf-8")
ASSAYS = tuple(re.findall(r'^  "([^"]+)"$', SOURCE, re.MULTILINE))


class SmallAssayCatalogTests(unittest.TestCase):
    @unittest.skipUnless(prosst_tests.CATALOG.is_file(), "Local ProteinGym catalog is required")
    def test_selection_matches_strict_limit_and_prosst_array(self):
        with prosst_tests.CATALOG.open(encoding="utf-8-sig", newline="") as handle:
            expected = tuple(row["DMS_id"] for row in csv.DictReader(handle)
                             if 0 < int(row["DMS_number_single_mutants"]) < 1000)
        self.assertEqual(ASSAYS, expected)
        self.assertEqual(ASSAYS, prosst_tests.ASSAYS)
        self.assertEqual(len(ASSAYS), len(set(ASSAYS)))
        self.assertIn(f"#SBATCH --array=0-{len(expected) - 1}\n", SOURCE)

    def test_esm2_preset_uses_the_existing_proteinmpnn_cache(self):
        root = runner_tests.REPOSITORY
        esm2 = json.loads((root / runner_tests.NknSbatchTests.config_file).read_text())
        prosst = json.loads((root / prosst_tests.PRESET).read_text())
        cpu = json.loads((root / "configs/esm2_proteinmpnn_nkn_two_scores_additive_four_branches_noise_initial_0p5.json")
                         .read_text())
        self.assertEqual(esm2["model"], cpu["model"])
        self.assertEqual(esm2["training"], cpu["training"])
        self.assertEqual(esm2["feature_selection"]["inverse_folding"],
                         prosst["feature_selection"]["inverse_folding"])
        self.assertTrue(esm2["feature_selection"]["sequence"].startswith("esm2/"))
        self.assertEqual([source.split("/")[1] for source in esm2["score_sources"]], ["esm2", "proteinmpnn"])
        self.assertEqual(esm2["score_sources"][0], cpu["score_sources"][0])
        self.assertEqual(esm2["score_sources"][1], prosst["score_sources"][1])


@unittest.skipUnless(runner_tests.BASH, "Bash is required for Slurm orchestration tests")
class SmallAssayDispatchTests(unittest.TestCase):
    def fixture_for_assay(self, assay):
        fixture = runner_tests.NknSbatchTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        directory = fixture.root / "data/features/pretrained/singles" / assay
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "assay.json").write_text(json.dumps({"assay_id": assay, "synthetic": False}))
        return fixture

    def test_array_boundaries_select_assay_features_and_results_over_inherited_assay(self):
        for index in (0, len(ASSAYS) - 1):
            with self.subTest(index=index):
                assay = ASSAYS[index]
                fixture = self.fixture_for_assay(assay)
                result, records, _ = fixture.invoke(1, "singles_random", task_index=index, features=None)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(records), 1)
                fixture.assert_feature_scope(records, features=f"data/features/pretrained/singles/{assay}",
                                             singles_only=True)
                saved = fixture.root / "results" / assay / "nkn_esm2_proteinmpnn_singles_random_1234/config.json"
                resolved = json.loads(saved.read_text())
                requested = json.loads((runner_tests.REPOSITORY / fixture.config_file).read_text())
                self.assertEqual(set(resolved["model"]["enabled_branches"]),
                                 set(requested["model"]["enabled_branches"]))
                for name in requested["model"]["enabled_branches"]:
                    self.assertEqual(resolved["model"]["branches"][name], requested["model"]["branches"][name])
                self.assertEqual(resolved["model"]["final_widths"], [8, 4, 4, 2, 1])
                self.assertEqual(resolved["score_sources"], requested["score_sources"])
                self.assertEqual(resolved["training"]["noise_initial"], 0.5)

    def test_invalid_array_ids_fail_before_launching_workers(self):
        fixture = self.fixture_for_assay("test")
        for index in ("-1", "57", "99999999999999999999", "not-an-index", "1+1", "01", " 1"):
            with self.subTest(index=index):
                result, records, plan = fixture.invoke(1, task_index=index)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("SLURM_ARRAY_TASK_ID must be 0-56", result.stderr)
                self.assertEqual(records, [])
                self.assertIsNone(plan)

    def test_wrappers_preserve_their_own_assay_selection_and_model_config(self):
        for filename, index, config_path, prefix in (
            ("run_nkn_cpu.sbatch", 130,
             runner_tests.NknSbatchTests.config_file,
             "nkn_esm2_proteinmpnn"),
            (prosst_tests.SCRIPT.name, 56, prosst_tests.PRESET.as_posix(), "nkn_prosst_proteinmpnn"),
        ):
            with self.subTest(wrapper=filename):
                script = runner_tests.SCRIPT.parent / filename
                assays = re.findall(r'^  "([^"]+)"$', script.read_text(), re.MULTILINE)
                assay = assays[index]
                fixture = self.fixture_for_assay(assay)
                shared = fixture.root / runner_tests.SCRIPT.relative_to(runner_tests.REPOSITORY)
                shared.parent.mkdir(parents=True)
                shutil.copyfile(runner_tests.SCRIPT, shared)
                shutil.copyfile(runner_tests.REPOSITORY / config_path, fixture.root / config_path)
                fixture.script = script
                result, records, _ = fixture.invoke(1, "singles_random", task_index=index, features=None)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(records), 1)
                saved = fixture.root / "results" / assay / f"{prefix}_singles_random_1234/config.json"
                self.assertTrue(saved.is_file())
                resolved = json.loads(saved.read_text())
                requested = json.loads((fixture.root / config_path).read_text())
                self.assertEqual(resolved["feature_selection"], requested["feature_selection"])
                self.assertEqual(resolved["score_sources"], requested["score_sources"])


if __name__ == "__main__":
    unittest.main()
