"""Exercise plotting-job setup and argument forwarding without Slurm or Conda."""

import argparse
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
SCRIPT = REPOSITORY / "sbatch_run/run_plotting/plot_nkn_calibration.sbatch"


def available_bash():
    if os.name == "nt":
        for candidate in (Path("C:/Program Files/Git/usr/bin/bash.exe"),
                          Path("C:/Program Files/Git/bin/bash.exe")):
            if candidate.is_file():
                return str(candidate)
        return None
    return shutil.which("bash")


BASH = available_bash()


@unittest.skipUnless(BASH, "Bash is required for plotting launcher tests")
class PlottingSbatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.root = self.workspace / "repository with spaces"
        for directory in ("src/plotting", "src/experiments", "sbatch_run/run_plotting",
                          "etc/profile.d", "job temp"):
            (self.root / directory).mkdir(parents=True)
        (self.root / "src/plotting/calibration.py").touch()
        (self.root / "src/experiments/cli.py").touch()
        self.script = self.root / SCRIPT.relative_to(REPOSITORY)
        shutil.copyfile(SCRIPT, self.script)
        self.bootstrap = self.workspace / "bootstrap.sh"
        self.bootstrap.write_text(textwrap.dedent("""\
            export PATH="/usr/bin:/bin:$PATH"
            conda() {
              if [[ "${TEST_NO_CONDA:-}" == 1 ]]; then
                echo 'Conda must not run for help' >&2
                return 19
              fi
              if [[ "$1" == info ]]; then printf '%s\\n' "$TEST_ROOT"; fi
              if [[ "$1" == activate && "$2" != nkn_protein ]]; then return 20; fi
            }
            """), encoding="utf-8", newline="\n")
        (self.root / "etc/profile.d/conda.sh").write_text(textwrap.dedent("""\
            python() { "$TEST_PYTHON" "$TEST_ROOT/record.py" "$@"; }
            """), encoding="utf-8", newline="\n")
        (self.root / "record.py").write_text(textwrap.dedent("""\
            import json
            import os
            from pathlib import Path
            import sys

            args = sys.argv[1:]
            assert args[:3] == ['-u', '-m', 'plotting.calibration']
            names = ('MPLBACKEND', 'MPLCONFIGDIR', 'XDG_CACHE_HOME',
                     'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'CUDA_VISIBLE_DEVICES', 'PYTHONNOUSERSITE')
            record = {'args': args[3:], 'cwd': str(Path.cwd()),
                      'environment': {name: os.environ[name] for name in names}}
            (Path(os.environ['TEST_ROOT']) / 'record.json').write_text(json.dumps(record))
            raise SystemExit(int(os.environ.get('TEST_CHILD_STATUS', '0')))
            """), encoding="utf-8")

    def invoke(self, *arguments, outside=False, submit=True, environment=None):
        env = os.environ.copy()
        for name in ("PLOT_RESULTS_DIR", "PLOT_OUTPUT_DIR", "SLURM_SUBMIT_DIR",
                     "TEST_NO_CONDA", "TEST_CHILD_STATUS"):
            env.pop(name, None)
        env.update(BASH_ENV=self.bootstrap.as_posix(), TEST_ROOT=self.root.as_posix(),
                   TEST_PYTHON=Path(sys.executable).as_posix(),
                   SLURM_TMPDIR=(self.root / "job temp").as_posix())
        if submit:
            env["SLURM_SUBMIT_DIR"] = (self.root / "src/plotting").as_posix()
        env.update(environment or {})
        result = subprocess.run(
            [BASH, str(self.script), *arguments], env=env,
            cwd=self.workspace if outside else self.root,
            capture_output=True, text=True, timeout=20,
        )
        record_path = self.root / "record.json"
        return result, json.loads(record_path.read_text()) if record_path.exists() else None

    def test_defaults_use_saved_results_and_headless_job_cache(self):
        result, record = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(record["args"], ["--results-dir", "results", "--output-dir", "plots"])
        self.assertEqual(Path(record["cwd"]), self.root)
        env = record["environment"]
        self.assertEqual(env["MPLBACKEND"], "Agg")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "PYTHONNOUSERSITE"):
            self.assertEqual(env[key], "1")
        self.assertIn("nkn-plot.", env["MPLCONFIGDIR"])
        self.assertIn("nkn-plot.", env["XDG_CACHE_HOME"])
        self.assertEqual(list((self.root / "job temp").iterdir()), [])

    def test_explicit_paths_override_environment_and_preserve_other_arguments(self):
        arguments = ["--results-dir", "CLI results", "--output-dir=CLI plots",
                     "--assay", "assay A", "--run-dir", "a run with spaces",
                     "--formats", "png", "pdf", "--dpi", "240"]
        result, record = self.invoke(*arguments, environment={
            "PLOT_RESULTS_DIR": "environment results", "PLOT_OUTPUT_DIR": "environment plots",
        })
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(record["args"], ["--results-dir", "environment results",
                                         "--output-dir", "environment plots", *arguments])
        parser = argparse.ArgumentParser()
        parser.add_argument("--results-dir")
        parser.add_argument("--output-dir")
        selected, _ = parser.parse_known_args(record["args"])
        self.assertEqual((selected.results_dir, selected.output_dir), ("CLI results", "CLI plots"))

    def test_manual_execution_finds_repository_from_script_location(self):
        result, record = self.invoke(outside=True, submit=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(record["cwd"]), self.root)

    def test_help_runs_without_repository_or_conda(self):
        (self.root / "src/plotting/calibration.py").unlink()
        result, record = self.invoke("--help", outside=True, submit=False,
                                     environment={"TEST_NO_CONDA": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PLOT_RESULTS_DIR", result.stdout)
        self.assertIn("--run-dir", result.stdout)
        self.assertIsNone(record)

    def test_python_failure_is_propagated_and_cache_is_cleaned(self):
        result, record = self.invoke(environment={"TEST_CHILD_STATUS": "7"})
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIsNotNone(record)
        self.assertEqual(list((self.root / "job temp").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
