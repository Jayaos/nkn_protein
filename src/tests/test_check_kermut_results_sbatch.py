"""Result-checker dispatch, path handling, and failure propagation without Slurm."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

from tests.test_prepare_sbatch import BASH, REPOSITORY


SCRIPT = Path("sbatch_run/run_kermut/check_kermut_results.sbatch")
HELPER = "sbatch_run/run_kermut/check_kermut_results.py"


@unittest.skipUnless(BASH, "Bash is required for result-checker launcher tests")
class CheckKermutResultsSbatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.root = self.workspace / "repository with spaces"
        (self.root / SCRIPT.parent).mkdir(parents=True)
        self.script = self.root / SCRIPT
        shutil.copyfile(REPOSITORY / SCRIPT, self.script)
        (self.root / HELPER).touch()
        self.nested = self.root / "nested directory"
        self.nested.mkdir()
        (self.root / "etc/profile.d").mkdir(parents=True)
        self.bootstrap = self.workspace / "bootstrap.sh"
        self.bootstrap.write_text(textwrap.dedent('''\
            export PATH="/usr/bin:/bin:$PATH"
            conda() {
              printf '%s\\n' "$*" >> "$TEST_ROOT/conda-calls.txt"
              if [[ "$1" == info ]]; then
                printf '%s\\n' "$TEST_ROOT"
              elif [[ "$1" == activate ]]; then
                export TEST_ACTIVE_ENV="$2"
              fi
            }
            '''), encoding="utf-8", newline="\n")
        (self.root / "etc/profile.d/conda.sh").write_text(
            'python() { "$TEST_PYTHON" "$TEST_ROOT/record.py" "$@"; }\n',
            encoding="utf-8", newline="\n",
        )
        (self.root / "record.py").write_text(textwrap.dedent('''\
            import json, os, sys
            from pathlib import Path
            record = {
                'args': sys.argv[1:], 'cwd': str(Path.cwd()),
                'environment': {key: os.environ.get(key) for key in (
                    'TEST_ACTIVE_ENV', 'CUDA_VISIBLE_DEVICES', 'PYTHONNOUSERSITE',
                    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'ASSAY_INDEX')},
            }
            Path('checker-call.json').write_text(json.dumps(record))
            raise SystemExit(int(os.environ.get('TEST_EXIT_CODE', '0')))
            '''), encoding="utf-8")

    def invoke(self, *arguments, exit_code=0, direct=False, outside=False):
        record_path = self.root / "checker-call.json"
        record_path.unlink(missing_ok=True)
        env = os.environ.copy()
        env.pop("SLURM_SUBMIT_DIR", None)
        env.update(BASH_ENV=self.bootstrap.as_posix(), TEST_ROOT=self.root.as_posix(),
                   TEST_PYTHON=Path(sys.executable).as_posix(), TEST_EXIT_CODE=str(exit_code),
                   CUDA_VISIBLE_DEVICES="0", ASSAY_INDEX="custom catalog.csv")
        if not direct:
            env["SLURM_SUBMIT_DIR"] = (self.workspace if outside else self.nested).as_posix()
        result = subprocess.run([BASH, str(self.script), *arguments], env=env, cwd=self.workspace,
                                capture_output=True, text=True, timeout=30)
        record = json.loads(record_path.read_text()) if record_path.is_file() else None
        return result, record

    def test_results_path_spaces_and_options_are_forwarded_from_repository(self):
        arguments = ("results/experiment with spaces", "--assay-index", "catalog with spaces.csv",
                     "--run-prefix", "kermut_custom", "--settings", "singles_random", "singles_modulo",
                     "--report", "reports/result audit.json")
        result, record = self.invoke(*arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(record["args"], ["-u", HELPER, *arguments])
        self.assertEqual(Path(record["cwd"]), self.root)
        self.assertEqual(record["environment"], {
            "TEST_ACTIVE_ENV": "nkn_protein", "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "ASSAY_INDEX": "custom catalog.csv",
        })

    def test_direct_bash_execution_discovers_repository_from_script(self):
        result, record = self.invoke("results/experiment", direct=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(record["cwd"]), self.root)
        self.assertEqual(record["args"], ["-u", HELPER, "results/experiment"])

    def test_help_does_not_require_repository_or_conda(self):
        (self.root / HELPER).unlink()
        for argument in ("--help", "-h"):
            with self.subTest(argument=argument):
                result, record = self.invoke(argument, outside=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("RESULTS_DIR", result.stdout)
                self.assertIn("kermut_prosst_esmif", result.stdout)
                self.assertIsNone(record)
        self.assertFalse((self.root / "conda-calls.txt").exists())

    def test_missing_results_argument_fails_before_environment_setup(self):
        result, record = self.invoke()
        self.assertEqual(result.returncode, 2)
        self.assertIn("RESULTS_DIR", result.stderr)
        self.assertIsNone(record)
        self.assertFalse((self.root / "conda-calls.txt").exists())

    def test_submission_outside_repository_is_rejected(self):
        result, record = self.invoke("results/experiment", outside=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("repository or a subdirectory", result.stderr)
        self.assertIsNone(record)
        self.assertFalse((self.root / "conda-calls.txt").exists())

    def test_helper_completion_and_input_failures_reach_slurm(self):
        for status in (1, 2):
            with self.subTest(status=status):
                result, record = self.invoke("results/experiment", exit_code=status)
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertIsNotNone(record)


if __name__ == "__main__":
    unittest.main()
