"""Exercise comparison-job setup and argument forwarding without Slurm/Conda."""

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


def available_bash():
    if os.name == "nt":
        for path in (Path("C:/Program Files/Git/usr/bin/bash.exe"),
                     Path("C:/Program Files/Git/bin/bash.exe")):
            if path.is_file():
                return str(path)
        return None
    return shutil.which("bash")


BASH = available_bash()


@unittest.skipUnless(BASH, "Bash is required for plotting launcher tests")
class PredictiveVarianceSbatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.root = self.workspace / "repository with spaces"
        for directory in ("src/plotting", "src/experiments", "sbatch_run/run_plotting",
                          "etc/profile.d", "job temp"):
            (self.root / directory).mkdir(parents=True)
        (self.root / "src/plotting/predictive_variance.py").touch()
        (self.root / "src/experiments/cli.py").touch()
        relative = Path("sbatch_run/run_plotting/plot_predictive_variance.sbatch")
        self.script = self.root / relative
        shutil.copyfile(REPOSITORY / relative, self.script)
        self.bootstrap = self.workspace / "bootstrap.sh"
        self.bootstrap.write_text(textwrap.dedent("""\
            export PATH="/usr/bin:/bin:$PATH"
            conda() {
              if [[ "${TEST_NO_CONDA:-}" == 1 ]]; then return 19; fi
              if [[ "$1" == info ]]; then printf '%s\\n' "$TEST_ROOT"; fi
              if [[ "$1" == activate && "$2" != nkn_protein ]]; then return 20; fi
            }
            """), encoding="utf-8", newline="\n")
        (self.root / "etc/profile.d/conda.sh").write_text(
            'python() { "$TEST_PYTHON" "$TEST_ROOT/record.py" "$@"; }\n',
            encoding="utf-8", newline="\n")
        (self.root / "record.py").write_text(textwrap.dedent("""\
            import json
            import os
            from pathlib import Path
            import sys

            assert sys.argv[1:4] == ['-u', '-m', 'plotting.predictive_variance']
            names = ('MPLBACKEND', 'MPLCONFIGDIR', 'XDG_CACHE_HOME',
                     'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'CUDA_VISIBLE_DEVICES', 'PYTHONNOUSERSITE')
            record = {'args': sys.argv[4:], 'cwd': str(Path.cwd()),
                      'environment': {name: os.environ[name] for name in names}}
            (Path(os.environ['TEST_ROOT']) / 'record.json').write_text(json.dumps(record))
            raise SystemExit(int(os.environ.get('TEST_CHILD_STATUS', '0')))
            """), encoding="utf-8")

    def invoke(self, *arguments, submit=True, environment=None):
        env = os.environ.copy()
        for name in ("PLOT_KERMUT_RESULTS_DIR", "PLOT_NKN_RESULTS_DIR", "PLOT_OUTPUT_DIR",
                     "SLURM_SUBMIT_DIR", "TEST_NO_CONDA", "TEST_CHILD_STATUS"):
            env.pop(name, None)
        env.update(BASH_ENV=self.bootstrap.as_posix(), TEST_ROOT=self.root.as_posix(),
                   TEST_PYTHON=Path(sys.executable).as_posix(),
                   SLURM_TMPDIR=(self.root / "job temp").as_posix())
        if submit:
            env["SLURM_SUBMIT_DIR"] = (self.root / "src/plotting").as_posix()
        env.update(environment or {})
        result = subprocess.run([BASH, str(self.script), *arguments], env=env,
                                cwd=self.workspace, capture_output=True, text=True, timeout=20)
        record_path = self.root / "record.json"
        return result, json.loads(record_path.read_text()) if record_path.exists() else None

    def test_defaults_use_headless_cpu_environment_and_clean_cache(self):
        result, record = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(record["args"], ["--kermut-results-dir", "results",
                                         "--nkn-results-dir", "results",
                                         "--output-dir", "plots/predictive_variance"])
        self.assertEqual(Path(record["cwd"]), self.root)
        env = record["environment"]
        self.assertEqual(env["MPLBACKEND"], "Agg")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONNOUSERSITE"):
            self.assertEqual(env[key], "1")
        for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME"):
            self.assertIn("variance-plot.", env[key])
        self.assertEqual(list((self.root / "job temp").iterdir()), [])

    def test_cli_paths_override_environment_and_preserve_selection_arguments(self):
        arguments = ["--kermut-results-dir", "Kermut CLI results",
                     "--nkn-results-dir=NKN CLI results", "--output-dir=CLI plots",
                     "--kermut-run-dir", "Kermut selected run", "--nkn-run-dir", "NKN selected run",
                     "--assay", "assay A", "--variance-kind", "latent", "--formats", "png", "svg"]
        result, record = self.invoke(*arguments, environment={
            "PLOT_KERMUT_RESULTS_DIR": "Kermut environment", "PLOT_NKN_RESULTS_DIR": "NKN environment",
            "PLOT_OUTPUT_DIR": "environment plots",
        })
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(record["args"][-len(arguments):], arguments)
        parser = argparse.ArgumentParser()
        for name in ("kermut-results-dir", "nkn-results-dir", "output-dir"):
            parser.add_argument(f"--{name}")
        selected, _ = parser.parse_known_args(record["args"])
        self.assertEqual((selected.kermut_results_dir, selected.nkn_results_dir, selected.output_dir),
                         ("Kermut CLI results", "NKN CLI results", "CLI plots"))

    def test_environment_defaults_and_manual_root_discovery(self):
        result, record = self.invoke(submit=False, environment={
            "PLOT_KERMUT_RESULTS_DIR": "Kermut environment", "PLOT_NKN_RESULTS_DIR": "NKN environment",
            "PLOT_OUTPUT_DIR": "environment plots",
        })
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(record["cwd"]), self.root)
        self.assertEqual(record["args"], ["--kermut-results-dir", "Kermut environment",
                                         "--nkn-results-dir", "NKN environment",
                                         "--output-dir", "environment plots"])

    def test_help_does_not_require_repository_or_conda(self):
        (self.root / "src/plotting/predictive_variance.py").unlink()
        result, record = self.invoke("--help", submit=False, environment={"TEST_NO_CONDA": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--kermut-run-dir", result.stdout)
        self.assertIn("--nkn-results-dir", result.stdout)
        self.assertIn("PLOT_KERMUT_RESULTS_DIR", result.stdout)
        self.assertIn("training-standardized", result.stdout)
        self.assertIsNone(record)

    def test_python_failure_propagates_and_cache_is_cleaned(self):
        result, record = self.invoke(environment={"TEST_CHILD_STATUS": "7"})
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIsNotNone(record)
        self.assertEqual(list((self.root / "job temp").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
