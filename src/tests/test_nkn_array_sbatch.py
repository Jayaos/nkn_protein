"""Exercise assay-array dispatch without Slurm, feature extraction, or training."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = Path("sbatch_run/run_nkn/run_nkn_assays.sbatch")
SHARED_RUNNER = Path("sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch")
DEFAULT_CONFIG = "configs/esm2_proteinmpnn_nkn_two_scores.json"
ASSAYS = (
    "PIN1_HUMAN_Tsuboyama_2023_1I6C",
    "VKOR1_HUMAN_Chiasson_2020_activity",
    "BLAT_ECOLX_Jacquier_2013",
    "A0A1I9GEU1_NEIME_Kennouche_2019",
    "FKBP3_HUMAN_Tsuboyama_2023_2KFV",
)


def available_bash():
    if os.name == "nt":
        for candidate in (Path("C:/Program Files/Git/usr/bin/bash.exe"),
                          Path("C:/Program Files/Git/bin/bash.exe")):
            if candidate.is_file():
                return str(candidate)
        return None  # Windows' bash.exe may be an unconfigured WSL launcher.
    return shutil.which("bash")


BASH = available_bash()


@unittest.skipUnless(BASH, "Bash is required for Slurm array orchestration tests")
class NknArraySbatchTests(unittest.TestCase):
    assays = ASSAYS
    script_path = SCRIPT
    shared_runner = SHARED_RUNNER
    config_variable = "NKN_CONFIG"
    default_config = DEFAULT_CONFIG

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.root = self.workspace / "repository with spaces"
        (self.root / self.script_path.parent).mkdir(parents=True)
        self.script = self.root / self.script_path
        shutil.copyfile(REPOSITORY / self.script_path, self.script)
        self.record_path = self.root / "dispatch.json"
        (self.root / self.shared_runner).write_text(textwrap.dedent("""\
            #!/bin/bash
            exec "$TEST_PYTHON" "$TEST_ROOT/capture.py" "$@"
            """), encoding="utf-8", newline="\n")
        (self.root / "capture.py").write_text(textwrap.dedent("""\
            import json
            import os
            from pathlib import Path
            import sys

            record = {"assay": os.environ.get("ASSAY"),
                      "config": os.environ.get(os.environ["TEST_CONFIG_VARIABLE"]),
                      "arguments": sys.argv[1:]}
            (Path(os.environ["TEST_ROOT"]) / "dispatch.json").write_text(json.dumps(record))
            raise SystemExit(int(os.environ.get("TEST_RUNNER_EXIT", "0")))
            """), encoding="utf-8")
        self.bootstrap = self.root / "bootstrap.sh"
        self.bootstrap.write_text('export PATH="/usr/bin:/bin:$PATH"\n',
                                  encoding="utf-8", newline="\n")

    def invoke(self, task_index, *arguments, config=None, submit_directory=None,
               use_submit_directory=True, runner_exit=0):
        self.record_path.unlink(missing_ok=True)
        environment = os.environ.copy()
        for name in ("SLURM_ARRAY_TASK_ID", "SLURM_SUBMIT_DIR", self.config_variable):
            environment.pop(name, None)
        environment.update(
            BASH_ENV=self.bootstrap.as_posix(), TEST_ROOT=self.root.as_posix(),
            TEST_PYTHON=Path(sys.executable).as_posix(),
            TEST_RUNNER_EXIT=str(runner_exit),
            TEST_CONFIG_VARIABLE=self.config_variable,
            ASSAY="inherited-assay-must-not-override-array-selection",
        )
        if task_index is not None:
            environment["SLURM_ARRAY_TASK_ID"] = str(task_index)
        if config is not None:
            environment[self.config_variable] = config
        if use_submit_directory:
            environment["SLURM_SUBMIT_DIR"] = (submit_directory or self.root).as_posix()
            cwd = self.workspace
        else:
            cwd = submit_directory or self.root
        result = subprocess.run(
            [BASH, str(self.script), *arguments], cwd=cwd, env=environment,
            capture_output=True, text=True, timeout=20,
        )
        record = json.loads(self.record_path.read_text()) if self.record_path.exists() else None
        return result, record

    def test_all_five_tasks_select_the_requested_assay_and_default_config(self):
        self.assertIn("#SBATCH --array=0-4", self.script.read_text())
        for index, assay in enumerate(self.assays):
            with self.subTest(index=index):
                result, record = self.invoke(index)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(record, {"assay": assay, "config": self.default_config,
                                          "arguments": []})

    def test_missing_or_invalid_array_id_does_not_launch_shared_runner(self):
        for index in (None, "", "-1", "5", "not-an-index", "1+1", "01", " 1"):
            with self.subTest(index=index):
                result, record = self.invoke(index)
                self.assertNotEqual(result.returncode, 0)
                self.assertIsNone(record)
                self.assertTrue(result.stderr.strip())

    def test_environment_config_and_all_cli_arguments_are_preserved(self):
        arguments = ("singles_random", "--parallel", "--config", "configs/custom config.json",
                     "--held-out-interval", "100")
        result, record = self.invoke(1, *arguments, config="configs/environment config.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(record, {"assay": self.assays[1], "config": "configs/environment config.json",
                                  "arguments": list(arguments)})

    def test_empty_config_environment_uses_two_score_default(self):
        result, record = self.invoke(0, config="")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(record["config"], self.default_config)

    def test_submission_from_nested_directory_finds_repository(self):
        nested = self.root / self.script_path.parent / "nested directory"
        nested.mkdir()
        result, record = self.invoke(3, "singles", "--parallel", submit_directory=nested)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(record["assay"], self.assays[3])
        self.assertEqual(record["arguments"], ["singles", "--parallel"])

    def test_manual_execution_uses_working_directory_when_submit_directory_is_unset(self):
        result, record = self.invoke(4, submit_directory=self.root / self.script_path.parent,
                                     use_submit_directory=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(record["assay"], self.assays[4])

    def test_shared_runner_failure_is_propagated(self):
        result, record = self.invoke(2, runner_exit=9)
        self.assertEqual(result.returncode, 9, result.stderr)
        self.assertEqual(record["assay"], self.assays[2])


if __name__ == "__main__":
    unittest.main()
