"""MSA/ESM-IF1 NKN cache recipes, array dispatch, and shared-runner configuration."""

import json
from pathlib import Path
import re
import shutil
import unittest
from unittest.mock import patch

from config import ExperimentConfig
from feature_generation.msa_config import MSAConfig
from feature_generation.inverse_config import InverseFoldingConfig
from feature_generation.score_runner import default_cache_id
from tests import test_nkn_array_sbatch as array_tests
from tests import test_nkn_sbatch as runner_tests


ROOT = runner_tests.REPOSITORY
SCRIPT = Path("sbatch_run/run_nkn/run_nkn_msa_esmif_small_assays.sbatch")
SHARED = Path("sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn_small_assays.sbatch")
PRESET = Path("configs/msa_transformer_esm_if1_nkn_four_branches_noise_initial_0p5.json")
REFERENCE_PRESET = Path("configs/esm2_proteinmpnn_nkn_four_branches_noise_initial_0p5.json")
SOURCE = (ROOT / SCRIPT).read_text(encoding="utf-8")
ASSAYS = tuple(re.findall(r'^  "([^"]+)"$', SOURCE, re.MULTILINE))


class MSAPresetTests(unittest.TestCase):
    def test_msa_inputs_match_preparation_and_preserve_reference_nkn(self):
        reference = json.loads((ROOT / REFERENCE_PRESET).read_text())
        msa = MSAConfig.load(ROOT / "configs/extraction/msa_transformer.json")
        reference["feature_selection"]["sequence"] = f"msa_transformer/{msa.cache_id}"
        reference["score_sources"][0] = f"scores/msa_transformer/{msa.score_cache_id}"
        inverse = InverseFoldingConfig.load(ROOT / "configs/extraction/esm_if1_142m.json")
        reference["feature_selection"]["inverse_folding"] = f"esm_if1/{inverse.cache_id}"
        reference["score_sources"][1] = f"scores/esm_if1/{default_cache_id(inverse)}"
        reference["inverse_folding_policy"] = "normalized"
        reference["structure_manifest_dir"] = inverse.structure_manifest_dir.relative_to(ROOT).as_posix()
        self.assertEqual(json.loads((ROOT / PRESET).read_text()), reference)
        config = ExperimentConfig.load(ROOT / PRESET)
        self.assertEqual(set(config.model.enabled_branches),
                         {"sequence_site", "sequence_global", "inverse_folding", "structure_distance"})
        self.assertEqual(config.training.noise_initial, 0.5)
        brca2 = config.for_assay("BRCA2_HUMAN_Erwood_2022_HEK293T")
        self.assertNotIn("structure_distance", brca2.model.enabled_branches)
        self.assertEqual(list(brca2.score_sources), reference["score_sources"])
        self.assertEqual(config.structure_consistency, "shared")
        self.assertEqual(config.structure_manifest_dir, "data/structures/protssn_proteingym/manifests")

    def test_assay_mapping_and_resources_match_reference_launcher(self):
        reference = (ROOT / SHARED).read_text()
        self.assertEqual(ASSAYS, tuple(re.findall(r'^  "([^"]+)"$', reference, re.MULTILINE)))
        self.assertEqual(len(ASSAYS), 57)
        pattern = r"^#SBATCH --([^=]+)=(.+)$"
        expected = dict(re.findall(pattern, reference, re.MULTILINE))
        expected["job-name"] = "nkn_msa_esmif_small_assays"
        self.assertEqual(dict(re.findall(pattern, SOURCE, re.MULTILINE)), expected)

    def test_selected_cache_names_match_the_local_prepared_example(self):
        config = ExperimentConfig.load(ROOT / PRESET)
        selections = [f"{kind}/{selection}" for kind, selection in config.feature_selection.items()]
        selections.extend(config.score_sources)
        checked = 0
        for assay in (ROOT / "data/features/pretrained/singles").glob("*"):
            if not all((assay / selection / "config.json").is_file() for selection in selections):
                continue
            for selection in selections:
                with self.subTest(assay=assay.name, selection=selection):
                    kind, family, checkpoint = selection.split("/")
                    saved = json.loads((assay / selection / "config.json").read_text())
                    self.assertEqual((saved["feature_type"], saved["model_family"], saved["checkpoint"]),
                                     (kind, family, checkpoint))
                    if family == "esm_if1":
                        self.assertEqual(saved["structure"]["source_archive"]["source_id"], "protssn_proteingym")
                        if kind == "inverse_folding":
                            self.assertEqual(saved["normalization"], "canonical_amino_acids_only")
                            self.assertEqual(saved["conditioning_mode"], "reference_prefix")
            checked += 1
        if not checked:
            self.skipTest("No complete local MSA/ESM-IF1 example is available")


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

    def fixture(self, index=1):
        fixture = runner_tests.NknSbatchTests()
        fixture.script = ROOT / SCRIPT
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        shutil.copyfile(ROOT / PRESET, fixture.root / PRESET)
        shared = fixture.root / SHARED
        shared.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / SHARED, shared)
        (fixture.root / "features/assay.json").write_text(json.dumps({
            "assay_id": ASSAYS[index], "synthetic": False,
        }))
        return fixture

    def test_shared_runner_saves_msa_config_and_distinct_result_prefix(self):
        fixture = self.fixture()
        with patch.dict("os.environ", {"EXPERIMENT_SUFFIX": "", "NKN_OUTPUT_ROOT": ""}):
            result, records, plan = fixture.invoke(
                15, "singles", "--parallel", "--experiment-suffix", "comparison_v1", task_index=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = fixture.root / "results/comparison_v1" / ASSAYS[1] / "nkn_msa_esmif_singles_1234"
        saved = ExperimentConfig.load(output / "config.json")
        requested = ExperimentConfig.load(ROOT / PRESET)
        for key in ("feature_selection", "score_sources", "model", "training", "evaluation",
                    "structure_manifest_dir", "inverse_folding_policy", "structure_consistency"):
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
        output = fixture.root / "results" / ASSAYS[1] / "nkn_msa_esmif_singles_random_1234"
        saved = ExperimentConfig.load(output / "config.json")
        requested = ExperimentConfig.load(ROOT / REFERENCE_PRESET)
        self.assertEqual(saved.feature_selection, requested.feature_selection)
        self.assertEqual(saved.score_sources, requested.score_sources)
        self.assertEqual(len(records), 1)

    def test_brca2_runtime_retains_both_model_scores(self):
        index = ASSAYS.index("BRCA2_HUMAN_Erwood_2022_HEK293T")
        fixture = self.fixture(index)
        with patch.dict("os.environ", {"EXPERIMENT_SUFFIX": "", "NKN_OUTPUT_ROOT": ""}):
            result, records, _ = fixture.invoke(15, "singles_random", task_index=index)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = fixture.root / "results" / ASSAYS[index] / "nkn_msa_esmif_singles_random_1234"
        saved = ExperimentConfig.load(output / "config.json")
        requested = ExperimentConfig.load(ROOT / PRESET)
        self.assertEqual(saved.score_sources, requested.score_sources)
        self.assertEqual([source.split("/")[1] for source in saved.score_sources], ["msa_transformer", "esm_if1"])
        self.assertNotIn("structure_distance", saved.model.enabled_branches)
        self.assertEqual(len(records), 1)


if __name__ == "__main__":
    unittest.main()
