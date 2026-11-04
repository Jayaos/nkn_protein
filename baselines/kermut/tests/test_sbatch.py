"""Run Kermut's shell orchestration with the same scheduling checks as NKN."""

import json
import os
from pathlib import Path
import re
import shutil
import unittest
from unittest.mock import patch

from src.tests import test_nkn_sbatch as sbatch_tests


class KermutSbatchTests(sbatch_tests.NknSbatchTests):
    script = sbatch_tests.REPOSITORY / 'sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn.sbatch'
    config_file = 'configs/kermut_esm2_proteinmpnn.json'
    config_variable = 'KERMUT_CONFIG'
    cli_module = 'baselines.kermut.cli'
    cli_file = 'baselines/kermut/cli.py'
    config_sources = ('src/config.py', 'baselines/kermut/config.py')
    config_type = ('baselines.kermut.config', 'KermutConfig')

    def setUp(self):
        super().setUp()
        # Load the real configuration without importing the scientific runtime
        # from package __init__, since shell orchestration uses cheap stubs.
        for package in ('baselines', 'baselines/kermut'):
            (self.root / package / '__init__.py').touch()


@unittest.skipUnless(sbatch_tests.BASH, "Bash is required for output-root checks")
class KermutOutputRootTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict("os.environ", {
            "KERMUT_OUTPUT_ROOT": "", "KERMUT_RUN_PREFIX": "", "EXPERIMENT_SUFFIX": "",
            "NKN_OUTPUT_ROOT": "nkn-output-must-not-affect-kermut",
        })
        environment.start()
        self.addCleanup(environment.stop)
        self.fixture = KermutSbatchTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        # Record worker destinations as well as the real runtime-config artifacts.
        helper = self.fixture.root / "helper.py"
        source = helper.read_text().replace(
            "record = {'setting': setting, 'command': args[2],",
            "record = {'setting': setting, 'command': args[2], "
            "'output': options['--output'], 'config': options['--config'],",
        )
        helper.write_text(source, encoding="utf-8")

    def saved_run(self, root, *, assay="test", selection="singles_random",
                  prefix="kermut_esm2_proteinmpnn", suffix=None):
        root = Path(root)
        if not root.is_absolute():
            root = self.fixture.root / root
        if suffix:
            root /= suffix
        return root / assay / f"{prefix}_{selection}_1234"

    @staticmethod
    def shell_path(path):
        # The production launcher runs on Linux; use Git Bash's POSIX spelling
        # when testing on Windows, including sandbox-specific /tmp mounts.
        if os.name != "nt":
            return path.as_posix()
        return sbatch_tests.subprocess.check_output(
            [sbatch_tests.BASH, "-c", 'printf "%s/%s\\n" "$(pwd -P)" "$1"', "bash", path.name],
            cwd=path.parent, text=True,
        ).strip()

    def assert_saved_run(self, run, records):
        self.assertTrue((run / "config.json").is_file())
        self.assertTrue((run / "execution_plan.json").is_file())
        self.assertTrue((run / "folds/singles.csv").is_file())
        for record in records:
            for key, expected in (("config", run / "config.json"),
                                  ("output", run / "cv" / record["setting"])):
                actual = Path(record[key])
                if not actual.is_absolute():
                    actual = self.fixture.root / actual
                self.assertEqual(actual.resolve(), expected.resolve())

    def test_default_root_remains_results_and_ignores_nkn_environment(self):
        result, records, _ = self.fixture.invoke(1, "singles_random")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(records), 1)
        self.assert_saved_run(self.saved_run("results"), records)
        self.assertFalse((self.fixture.root / "nkn-output-must-not-affect-kermut").exists())

    def test_environment_root_with_spaces_composes_suffix_and_parallel_outputs(self):
        root = self.fixture.root / "environment output"
        with patch.dict("os.environ", {"KERMUT_OUTPUT_ROOT": self.shell_path(root)}):
            result, records, _ = self.fixture.invoke(
                15, "singles", "--parallel", "--experiment-suffix", "comparison")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(records), 3)
        run = self.saved_run(root, selection="singles", suffix="comparison")
        self.assert_saved_run(run, records)
        plan = json.loads((run / "execution_plan.json").read_text())
        self.assertEqual((plan["setting_workers"], plan["fold_workers"]), (3, 5))
        self.assertEqual({path.name for path in (run / "logs").glob("*.log")},
                         {record["setting"] + ".log" for record in records})
        self.assertFalse((self.fixture.root / "results").exists())

    def test_cli_relative_root_overrides_environment_and_uses_repository_root(self):
        nested = self.fixture.root / "nested submission"
        nested.mkdir()
        original_run = sbatch_tests.subprocess.run

        def run_from_nested(*args, **kwargs):
            kwargs["env"]["SLURM_SUBMIT_DIR"] = nested.as_posix()
            kwargs["cwd"] = nested
            return original_run(*args, **kwargs)

        with patch.dict("os.environ", {"KERMUT_OUTPUT_ROOT": "unused-environment-root"}), \
                patch.object(sbatch_tests.subprocess, "run", side_effect=run_from_nested):
            result, records, _ = self.fixture.invoke(
                1, "--output-root", "relative outputs", "singles_random",
                "--experiment-suffix", "comparison")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_saved_run(self.saved_run("relative outputs", suffix="comparison"), records)
        self.assertFalse((nested / "relative outputs").exists())
        self.assertFalse((self.fixture.root / "unused-environment-root").exists())

    def test_equals_form_accepts_an_absolute_root(self):
        root = self.fixture.root / "absolute outputs"
        with patch.dict("os.environ", {"KERMUT_OUTPUT_ROOT": "unused-environment-root"}):
            result, records, _ = self.fixture.invoke(
                1, "singles_random", f"--output-root={self.shell_path(root)}")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_saved_run(self.saved_run(root), records)
        self.assertFalse((self.fixture.root / "unused-environment-root").exists())

    def test_missing_empty_or_option_values_fail_before_creating_outputs(self):
        for arguments in (("--output-root",), ("--output-root", ""),
                          ("--output-root=",), ("--output-root", "--parallel")):
            with self.subTest(arguments=arguments):
                result, records, _ = self.fixture.invoke(1, *arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--output-root requires a directory path", result.stderr)
                self.assertEqual(records, [])
                self.assertFalse((self.fixture.root / "results").exists())
                self.assertFalse((self.fixture.root / "validation.json").exists())

    def test_existing_run_directory_is_not_overwritten(self):
        arguments = ("singles_random", "--output-root", "custom outputs")
        first, _, _ = self.fixture.invoke(1, *arguments)
        self.assertEqual(first.returncode, 0, first.stderr)
        run = self.saved_run("custom outputs")
        config = (run / "config.json").read_bytes()
        sentinel = run / "keep.txt"
        sentinel.write_text("existing result")
        for path in (self.fixture.root / "records").glob("*.json"):
            path.unlink()
        second, records, _ = self.fixture.invoke(1, *arguments)
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(records, [])
        self.assertEqual((run / "config.json").read_bytes(), config)
        self.assertEqual(sentinel.read_text(), "existing result")

    def test_all_assay_prosst_esmif_wrapper_forwards_output_root(self):
        wrapper = sbatch_tests.REPOSITORY / "sbatch_run/run_kermut/run_kermut_prosst_esmif_all_assays.sbatch"
        assay = re.findall(r'^  "([^"]+)"$', wrapper.read_text(), re.MULTILINE)[0]
        shared = self.fixture.root / self.fixture.script.relative_to(sbatch_tests.REPOSITORY)
        shared.parent.mkdir(parents=True)
        shutil.copyfile(self.fixture.script, shared)
        preset = Path("configs/kermut_prosst_esmif.json")
        shutil.copyfile(sbatch_tests.REPOSITORY / preset, self.fixture.root / preset)
        (self.fixture.root / "features/assay.json").write_text(
            json.dumps({"assay_id": assay, "synthetic": False}), encoding="utf-8")
        self.fixture.script = wrapper
        result, records, _ = self.fixture.invoke(
            1, "singles_random", "--output-root", "wrapper outputs",
            "--experiment-suffix", "comparison", task_index=0)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(records), 1)
        run = self.saved_run("wrapper outputs", assay=assay, prefix="kermut_prosst_esmif",
                             suffix="comparison")
        self.assert_saved_run(run, records)
        resolved = json.loads((run / "config.json").read_text())
        requested = json.loads((self.fixture.root / preset).read_text())
        self.assertEqual(resolved["feature_selection"], requested["feature_selection"])
        self.assertEqual(resolved["score_sources"], requested["score_sources"])
