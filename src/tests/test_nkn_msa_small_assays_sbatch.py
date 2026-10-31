"""MSA NKN cache recipes, array dispatch, and shared-runner configuration."""

import json
from pathlib import Path
import re
import shutil
import unittest
from unittest.mock import patch

from config import ExperimentConfig
from feature_generation.msa_config import MSAConfig
from tests import test_nkn_array_sbatch as array_tests
from tests import test_nkn_sbatch as runner_tests


ROOT = runner_tests.REPOSITORY
SCRIPT = Path("sbatch_run/run_nkn/run_nkn_msa_proteinmpnn_small_assays.sbatch")
SHARED = Path("sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn_small_assays.sbatch")
PRESET = Path("configs/msa_transformer_proteinmpnn_nkn_four_branches_noise_initial_0p5.json")
REFERENCE_PRESET = Path("configs/esm2_proteinmpnn_nkn_four_branches_noise_initial_0p5.json")
SOURCE = (ROOT / SCRIPT).read_text(encoding="utf-8")
ASSAYS = tuple(re.findall(r'^  "([^"]+)"$', SOURCE, re.MULTILINE))


class MSAPresetTests(unittest.TestCase):
    def test_msa_inputs_match_preparation_and_preserve_reference_nkn(self):
        reference = json.loads((ROOT / REFERENCE_PRESET).read_text())
        msa = MSAConfig.load(ROOT / "configs/extraction/msa_transformer.json")
        reference["feature_selection"]["sequence"] = f"msa_transformer/{msa.cache_id}"
        reference["score_sources"][0] = f"scores/msa_transformer/{msa.score_cache_id}"
        self.assertEqual(json.loads((ROOT / PRESET).read_text()), reference)
        config = ExperimentConfig.load(ROOT / PRESET)
        self.assertEqual(set(config.model.enabled_branches),
                         {"sequence_site", "sequence_global", "inverse_folding", "structure_distance"})
        self.assertEqual(config.training.noise_initial, 0.5)
        brca2 = config.for_assay("BRCA2_HUMAN_Erwood_2022_HEK293T")
        self.assertNotIn("structure_distance", brca2.model.enabled_branches)
        self.assertEqual(list(brca2.score_sources), [reference["score_sources"][0]])

    def test_assay_mapping_and_resources_match_reference_launcher(self):
        reference = (ROOT / SHARED).read_text()
        self.assertEqual(ASSAYS, tuple(re.findall(r'^  "([^"]+)"$', reference, re.MULTILINE)))
        self.assertEqual(len(ASSAYS), 57)
        pattern = r"^#SBATCH --([^=]+)=(.+)$"
        expected = dict(re.findall(pattern, reference, re.MULTILINE))
        expected["job-name"] = "nkn_msa_proteinmpnn_small_assays"
        self.assertEqual(dict(re.findall(pattern, SOURCE, re.MULTILINE)), expected)


@unittest.skipUnless(runner_tests.BASH, "Bash is required for launcher tests")
class MSALauncherTests(unittest.TestCase):
    def test_dispatch_overrides_help_and_invalid_task(self):
        fixture = array_tests.NknArraySbatchTests()
        fixture.script_path = SCRIPT
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
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

    def fixture(self):
        fixture = runner_tests.NknSbatchTests()
        fixture.script = ROOT / SCRIPT
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        shutil.copyfile(ROOT / PRESET, fixture.root / PRESET)
        shared = fixture.root / SHARED
        shared.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / SHARED, shared)
        (fixture.root / "features/assay.json").write_text(json.dumps({
            "assay_id": ASSAYS[1], "synthetic": False,
        }))
        return fixture

    def test_shared_runner_saves_msa_config_and_distinct_result_prefix(self):
        fixture = self.fixture()
        with patch.dict("os.environ", {"EXPERIMENT_SUFFIX": "", "NKN_OUTPUT_ROOT": ""}):
            result, records, plan = fixture.invoke(
                15, "singles", "--parallel", "--experiment-suffix", "comparison_v1", task_index=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = fixture.root / "results/comparison_v1" / ASSAYS[1] / "nkn_msa_proteinmpnn_singles_1234"
        saved = ExperimentConfig.load(output / "config.json")
        requested = ExperimentConfig.load(ROOT / PRESET)
        for key in ("feature_selection", "score_sources", "model", "training", "evaluation",
                    "structure_manifest_dir", "inverse_folding_policy"):
            self.assertEqual(getattr(saved, key), getattr(requested, key))
        self.assertEqual((plan["setting_workers"], plan["fold_workers"], saved.num_threads), (3, 5, 1))
        self.assertEqual({record["setting"] for record in records},
                         {"singles_random", "singles_modulo", "singles_contiguous"})

    def test_explicit_config_overrides_environment_and_wrapper_default(self):
        fixture = self.fixture()
        with patch.dict("os.environ", {"EXPERIMENT_SUFFIX": "", "NKN_OUTPUT_ROOT": ""}):
            result, records, _ = fixture.invoke(
                15, "singles_random", "--config", REFERENCE_PRESET.as_posix(),
                config_environment=PRESET.as_posix(), task_index=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = fixture.root / "results" / ASSAYS[1] / "nkn_msa_proteinmpnn_singles_random_1234"
        saved = ExperimentConfig.load(output / "config.json")
        requested = ExperimentConfig.load(ROOT / REFERENCE_PRESET)
        self.assertEqual(saved.feature_selection, requested.feature_selection)
        self.assertEqual(saved.score_sources, requested.score_sources)
        self.assertEqual(len(records), 1)


if __name__ == "__main__":
    unittest.main()
