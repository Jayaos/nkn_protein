"""Check ProSST/ProteinMPNN cache selection and small-assay array dispatch."""

import csv
import json
from pathlib import Path
import re
import shutil
import unittest

from . import test_nkn_array_sbatch as array_tests
from . import test_nkn_sbatch as runner_tests
from src.feature_generation.inverse_config import InverseFoldingConfig
from src.feature_generation.prosst_config import ProSSTConfig


SCRIPT = Path("sbatch_run/run_nkn/run_nkn_prosst_proteinmpnn_small_assays.sbatch")
PRESET = Path("configs/proteinmpnn_prosst_nkn_four_branches_noise_initial_0p5.json")
CATALOG = next((array_tests.REPOSITORY / path for path in (
    "data/DMS_substitutions.csv",
    "data/DMS_ProteinGym_substitutions/DMS_substitutions.csv",
    "data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv",
) if (array_tests.REPOSITORY / path).is_file()), array_tests.REPOSITORY / "data/DMS_substitutions.csv")
SOURCE = (array_tests.REPOSITORY / SCRIPT).read_text(encoding="utf-8")
ASSAYS = tuple(re.findall(r'^  "([^"]+)"$', SOURCE, re.MULTILINE))


class SmallAssayCatalogTests(unittest.TestCase):
    @unittest.skipUnless(CATALOG.is_file(), "Local ProteinGym catalog is required")
    def test_snapshot_matches_inclusive_single_mutant_limit_in_catalog_order(self):
        with CATALOG.open(encoding="utf-8-sig", newline="") as handle:
            expected = tuple(row["DMS_id"] for row in csv.DictReader(handle)
                             if 0 < int(row["DMS_number_single_mutants"]) <= 1000)
        self.assertEqual(ASSAYS, expected)
        self.assertEqual(len(ASSAYS), len(set(ASSAYS)))
        self.assertIn(f"#SBATCH --array=0-{len(expected) - 1}\n", SOURCE)


class SmallAssayArraySbatchTests(array_tests.NknArraySbatchTests):
    assays = ASSAYS
    script_path = SCRIPT
    default_config = PRESET.as_posix()

    def test_all_five_tasks_select_the_requested_assay_and_default_config(self):
        # Override the legacy five-assay fixture contract for the entire subset.
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

    def test_help_lists_all_tasks_without_launching_a_runner(self):
        result, record = self.invoke(None, "--help", submit_directory=self.workspace)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNone(record)
        self.assertIn("1-1000 single mutants (inclusive)", result.stdout)
        for index, assay in enumerate(self.assays):
            self.assertIn(f"  {index}: {assay}\n", result.stdout)


class ProteinMPNNProSSTPresetTests(unittest.TestCase):
    def test_feature_selection_matches_shipped_preparation_recipes(self):
        root = array_tests.REPOSITORY
        config = json.loads((root / PRESET).read_text(encoding="utf-8"))
        prosst = ProSSTConfig.load(root / "configs/extraction/prosst_2048_singles.json")
        inverse = InverseFoldingConfig.load(root / "configs/extraction/proteinmpnn_v48_020_kermut.json")
        self.assertEqual(config["feature_selection"], {
            "sequence": f"prosst/{prosst.cache_id}",
            # Existing ProteinMPNN caches predate the structure-source prefix.
            "inverse_folding": f"proteinmpnn/{inverse.cache_id.removeprefix('proteingym--')}",
        })
        cpu = json.loads((root / "configs/esm2_proteinmpnn_nkn_two_scores_additive_four_branches_noise_initial_0p5.json")
                         .read_text(encoding="utf-8"))
        self.assertEqual(config["model"], cpu["model"])
        self.assertEqual(config["training"], cpu["training"])
        self.assertEqual(config["inverse_folding_policy"], "sequence_conditional")
        self.assertEqual(config["structure_consistency"], "per_cache")
        self.assertEqual((prosst.site_embeddings, prosst.global_window), ("centered", "non_overlapping"))
        self.assertEqual([source.split("/")[1] for source in config["score_sources"]],
                         ["prosst", "proteinmpnn"])
        original = json.loads((root / "configs/proteinmpnn_prosst_nkn.json").read_text(encoding="utf-8"))
        self.assertEqual(config["score_sources"], original["score_sources"])

    def test_selected_paths_match_local_saved_cache_metadata(self):
        root = array_tests.REPOSITORY
        config = json.loads((root / PRESET).read_text(encoding="utf-8"))
        cache_root = root / "data/features/pretrained/singles"
        selections = [f"{kind}/{selection}"
                      for kind, selection in config["feature_selection"].items()]
        selections.extend(config["score_sources"])
        checked = 0
        for selection in selections:
            kind, family, checkpoint = selection.split("/")
            # Inspect only locally available families; the full HPC catalog is
            # not required. The selected recipe must exist when that family does.
            for family_root in cache_root.glob(f"*/{kind}/{family}"):
                with self.subTest(assay=family_root.parent.parent.name, selection=selection):
                    path = family_root / checkpoint / "config.json"
                    self.assertTrue(path.is_file(), f"Selected cache does not exist: {path}")
                    metadata = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual(metadata["checkpoint"], checkpoint)
                    self.assertEqual(metadata["model_family"], family)
                    self.assertEqual(metadata["feature_type"], kind)
                    checked += 1
        if not checked:
            self.skipTest("No local pretrained caches are available")

    @unittest.skipUnless(runner_tests.BASH, "Bash is required for the shared-runner check")
    def test_shared_runner_resolves_the_new_preset(self):
        fixture = runner_tests.NknSbatchTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        shutil.copyfile(array_tests.REPOSITORY / PRESET, fixture.root / PRESET)
        result, records, _ = fixture.invoke(15, "singles_random", "--config", PRESET.as_posix())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(records), 1)
        runtime_path = next((fixture.root / "results").rglob("config.json"))
        resolved = json.loads(runtime_path.read_text())
        requested = json.loads((array_tests.REPOSITORY / PRESET).read_text())
        self.assertEqual(resolved["feature_selection"], requested["feature_selection"])
        self.assertEqual(resolved["score_sources"], requested["score_sources"])
        self.assertEqual(resolved["structure_consistency"], "per_cache")
        self.assertEqual(set(resolved["model"]["enabled_branches"]),
                         set(requested["model"]["enabled_branches"]))
        for name in requested["model"]["enabled_branches"]:
            self.assertEqual(resolved["model"]["branches"][name], requested["model"]["branches"][name])
        self.assertEqual(resolved["model"]["final_widths"], requested["model"]["final_widths"])
        self.assertEqual(resolved["training"]["noise_initial"], 0.5)


if __name__ == "__main__":
    unittest.main()
