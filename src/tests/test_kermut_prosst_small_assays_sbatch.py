"""Check the ProSST Kermut preset, small-assay dispatch, and saved run identity."""

import json
from pathlib import Path
import re
import shutil
import unittest
from unittest.mock import patch

from . import test_kermut_small_assays_sbatch as esm2_tests
from . import test_nkn_sbatch as runner_tests


ROOT = runner_tests.REPOSITORY
SCRIPT = Path("sbatch_run/run_kermut/run_kermut_prosst_proteinmpnn_small_assays.sbatch")
PRESET = Path("configs/kermut_prosst_proteinmpnn.json")
SOURCE = (ROOT / SCRIPT).read_text(encoding="utf-8")
ASSAYS = tuple(re.findall(r'^  "([^"]+)"$', SOURCE, re.MULTILINE))


class ProSSTPresetTests(unittest.TestCase):
    def test_only_esm2_inputs_change_in_the_preset(self):
        original = json.loads((ROOT / "configs/kermut_esm2_proteinmpnn.json").read_text())
        prosst = json.loads((ROOT / PRESET).read_text())
        reference = json.loads((ROOT / "configs/proteinmpnn_prosst_nkn.json").read_text())
        original["feature_selection"]["sequence"] = reference["feature_selection"]["sequence"]
        original["score_sources"][0] = reference["score_sources"][0]
        self.assertEqual(prosst, original)
        self.assertTrue(prosst["feature_selection"]["sequence"].startswith("prosst/"))
        self.assertEqual([source.split("/")[1] for source in prosst["score_sources"]],
                         ["prosst", "proteinmpnn"])

    def test_assay_mapping_and_slurm_resources_match_esm2(self):
        self.assertEqual(ASSAYS, esm2_tests.ASSAYS)
        pattern = r"^#SBATCH --([^=]+)=(.+)$"
        original = dict(re.findall(pattern, esm2_tests.SOURCE, re.MULTILINE))
        prosst = dict(re.findall(pattern, SOURCE, re.MULTILINE))
        self.assertEqual(prosst.pop("job-name"), "kermut_prosst_proteinmpnn_small")
        original.pop("job-name")
        self.assertEqual(prosst, original)


class ProSSTArraySbatchTests(esm2_tests.SmallAssayArraySbatchTests):
    assays = ASSAYS
    script_path = SCRIPT
    default_config = PRESET.as_posix()

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

    def test_prosst_result_prefix_is_forwarded(self):
        result, record = self.invoke(1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(record["config"], PRESET.as_posix())
        self.assertIn("PREFIX=kermut_prosst_proteinmpnn\n", result.stdout)

    def test_custom_result_prefix_is_preserved(self):
        with patch.dict("os.environ", {"KERMUT_RUN_PREFIX": "kermut_custom_prosst"}):
            result, _ = self.invoke(1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PREFIX=kermut_custom_prosst\n", result.stdout)


@unittest.skipUnless(runner_tests.BASH, "Bash is required for the shared-runner checks")
class ProSSTSharedRunnerTests(unittest.TestCase):
    def fixture(self, *, wrapper):
        # Execute the real runtime-config builder and shell scheduling, with
        # scientific workers stubbed so no model downloads or training occur.
        fixture = runner_tests.NknSbatchTests()
        fixture.script = ROOT / esm2_tests.SHARED_RUNNER
        fixture.config_file = "configs/kermut_esm2_proteinmpnn.json"
        fixture.config_variable = "KERMUT_CONFIG"
        fixture.cli_module = "baselines.kermut.cli"
        fixture.cli_file = "baselines/kermut/cli.py"
        fixture.config_sources = ("src/config.py", "baselines/kermut/config.py")
        fixture.config_type = ("baselines.kermut.config", "KermutConfig")
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        for package in ("baselines", "baselines/kermut"):
            (fixture.root / package / "__init__.py").touch()
        shutil.copyfile(ROOT / PRESET, fixture.root / PRESET)
        shared = fixture.root / esm2_tests.SHARED_RUNNER
        shared.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / esm2_tests.SHARED_RUNNER, shared)
        if wrapper:
            fixture.script = ROOT / SCRIPT
            (fixture.root / "features/assay.json").write_text(json.dumps({
                "assay_id": ASSAYS[1], "synthetic": False,
            }), encoding="utf-8")
        return fixture

    def test_wrapper_saves_prosst_config_and_result_prefix_with_same_parallel_plan(self):
        fixture = self.fixture(wrapper=True)
        with patch.dict("os.environ", {"KERMUT_RUN_PREFIX": "", "EXPERIMENT_SUFFIX": ""}):
            result, records, plan = fixture.invoke(
                15, "singles", "--parallel", "--experiment-suffix", "comparison_v1", task_index=1,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        output = (fixture.root / "results/comparison_v1" / ASSAYS[1]
                  / "kermut_prosst_proteinmpnn_singles_1234")
        saved = json.loads((output / "config.json").read_text())
        requested = json.loads((ROOT / PRESET).read_text())
        self.assertEqual(saved["feature_selection"], requested["feature_selection"])
        self.assertEqual(saved["score_sources"], requested["score_sources"])
        self.assertEqual(saved["training"], requested["training"])
        self.assertEqual(saved["evaluation"], requested["evaluation"])
        self.assertEqual(saved["num_threads"], 1)
        self.assertEqual((plan["setting_workers"], plan["fold_workers"]), (3, 5))
        self.assertEqual({record["setting"] for record in records},
                         {"singles_random", "singles_modulo", "singles_contiguous"})

    def test_shared_runner_keeps_esm2_default_results(self):
        fixture = self.fixture(wrapper=False)
        with patch.dict("os.environ", {"KERMUT_RUN_PREFIX": "", "EXPERIMENT_SUFFIX": ""}):
            result, records, _ = fixture.invoke(1, "singles_random")
        self.assertEqual(result.returncode, 0, result.stderr)
        output = fixture.root / "results/test/kermut_esm2_proteinmpnn_singles_random_1234"
        self.assertTrue((output / "config.json").is_file())
        self.assertEqual(len(records), 1)

    def test_invalid_result_prefix_fails_before_creating_results(self):
        fixture = self.fixture(wrapper=False)
        for prefix in ("../outside", "kermut_../outside", "nkn_prosst", "kermut_"):
            with self.subTest(prefix=prefix), patch.dict("os.environ", {"KERMUT_RUN_PREFIX": prefix}):
                result, records, plan = fixture.invoke(1)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("KERMUT_RUN_PREFIX", result.stderr)
                self.assertEqual(records, [])
                self.assertIsNone(plan)
                self.assertFalse((fixture.root / "results").exists())


if __name__ == "__main__":
    unittest.main()
