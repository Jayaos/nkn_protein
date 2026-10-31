"""MSA Kermut preparation recipes, array dispatch, and resolved run identity."""

import json
from pathlib import Path
import re
import shutil
import unittest
from unittest.mock import patch

from baselines.kermut.config import KermutConfig
from feature_generation.msa_config import MSAConfig
from tests import test_kermut_small_assays_sbatch as reference
from tests import test_nkn_sbatch as runner_tests


ROOT = runner_tests.REPOSITORY
SCRIPT = Path("sbatch_run/run_kermut/run_kermut_msa_proteinmpnn_small_assays.sbatch")
PRESET = Path("configs/kermut_msa_proteinmpnn.json")
SOURCE = (ROOT / SCRIPT).read_text(encoding="utf-8")
ASSAYS = tuple(re.findall(r'^  "([^"]+)"$', SOURCE, re.MULTILINE))


class MSAPresetTests(unittest.TestCase):
    def test_msa_inputs_match_preparation_and_keep_kermut_settings(self):
        original = json.loads((ROOT / "configs/kermut_esm2_proteinmpnn.json").read_text())
        msa = MSAConfig.load(ROOT / "configs/extraction/msa_transformer.json")
        original["feature_selection"]["sequence"] = f"msa_transformer/{msa.cache_id}"
        original["score_sources"][0] = f"scores/msa_transformer/{msa.score_cache_id}"
        self.assertEqual(json.loads((ROOT / PRESET).read_text()), original)
        config = KermutConfig.load(ROOT / PRESET)
        brca2 = config.for_assay("BRCA2_HUMAN_Erwood_2022_HEK293T")
        self.assertFalse(brca2.model.use_structure_distance)
        self.assertEqual(brca2.score_sources, (original["score_sources"][0],))

    def test_assay_mapping_and_resources_match_reference(self):
        self.assertEqual(ASSAYS, reference.ASSAYS)
        self.assertEqual(len(ASSAYS), 57)
        pattern = r"^#SBATCH --([^=]+)=(.+)$"
        expected = dict(re.findall(pattern, reference.SOURCE, re.MULTILINE))
        actual = dict(re.findall(pattern, SOURCE, re.MULTILINE))
        expected["job-name"] = "kermut_msa_proteinmpnn_small"
        self.assertEqual(actual, expected)


@unittest.skipUnless(runner_tests.BASH, "Bash is required for launcher tests")
class MSALauncherTests(unittest.TestCase):
    def test_dispatch_overrides_help_and_invalid_task(self):
        fixture = reference.SmallAssayArraySbatchTests()
        fixture.script_path = SCRIPT
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        with patch.dict("os.environ", {"KERMUT_RUN_PREFIX": ""}):
            for index in (0, 6, 56):
                with self.subTest(index=index):
                    result, record = fixture.invoke(index)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(record, {"assay": ASSAYS[index], "config": PRESET.as_posix(), "arguments": []})
            arguments = ("singles_random", "--parallel", "--config", "configs/custom config.json")
            result, record = fixture.invoke(1, *arguments, config="configs/environment.json")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(record["config"], "configs/environment.json")
            self.assertEqual(record["arguments"], list(arguments))
            result, record = fixture.invoke(57)
            self.assertNotEqual(result.returncode, 0)
            self.assertIsNone(record)
            result, record = fixture.invoke(None, "--help", submit_directory=fixture.workspace)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIsNone(record)
            self.assertIn(PRESET.as_posix(), result.stdout)
            self.assertIn("MSA Transformer", result.stdout)

    def test_shared_runner_saves_msa_config_and_distinct_result_prefix(self):
        fixture = runner_tests.NknSbatchTests()
        fixture.script = ROOT / SCRIPT
        fixture.config_file = "configs/kermut_esm2_proteinmpnn.json"
        fixture.config_variable = "KERMUT_CONFIG"
        fixture.cli_module = "baselines.kermut.cli"
        fixture.cli_file = "baselines/kermut/cli.py"
        fixture.config_sources = ("src/config.py", "baselines/kermut/config.py")
        fixture.config_type = ("baselines.kermut.config", "KermutConfig")
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        shutil.copyfile(ROOT / PRESET, fixture.root / PRESET)
        for package in ("baselines", "baselines/kermut"):
            (fixture.root / package / "__init__.py").touch()
        shared = fixture.root / reference.SHARED_RUNNER
        shared.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / reference.SHARED_RUNNER, shared)
        (fixture.root / "features/assay.json").write_text(json.dumps({
            "assay_id": ASSAYS[1], "synthetic": False,
        }))
        with patch.dict("os.environ", {"KERMUT_RUN_PREFIX": "", "EXPERIMENT_SUFFIX": ""}):
            result, records, plan = fixture.invoke(
                15, "singles", "--parallel", "--experiment-suffix", "comparison_v1", task_index=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = fixture.root / "results/comparison_v1" / ASSAYS[1] / "kermut_msa_proteinmpnn_singles_1234"
        saved = json.loads((output / "config.json").read_text())
        requested = json.loads((ROOT / PRESET).read_text())
        for key in ("feature_selection", "score_sources", "training", "evaluation", "structure_manifest_dir"):
            self.assertEqual(saved[key], requested[key])
        self.assertEqual((plan["setting_workers"], plan["fold_workers"], saved["num_threads"]), (3, 5, 1))
        self.assertEqual({record["setting"] for record in records},
                         {"singles_random", "singles_modulo", "singles_contiguous"})


if __name__ == "__main__":
    unittest.main()
