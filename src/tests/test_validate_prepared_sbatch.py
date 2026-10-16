"""Readiness launcher dispatch, failure reporting, and CPU-only execution."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import unittest

from tests.test_prepare_sbatch import BASH, PREPARATION, REPOSITORY, PreparationFixture


@unittest.skipUnless(BASH, "Bash is required for readiness launcher tests")
class ValidatePreparedSbatchTests(PreparationFixture):
    def setUp(self):
        super().setUp()
        self.script = self.root / PREPARATION / "validate_prepared.sbatch"
        shutil.copyfile(REPOSITORY / PREPARATION / self.script.name, self.script)
        (self.root / "src/feature_generation/readiness.py").touch()
        self.bootstrap = self.workspace / "bootstrap.sh"
        self.bootstrap.write_text(textwrap.dedent('''\
            export PATH="/usr/bin:/bin:$PATH"
            conda() {
              if [[ "$1" == info ]]; then printf '%s\\n' "$TEST_ROOT"; fi
            }
            '''), encoding="utf-8", newline="\n")
        (self.root / "etc/profile.d/conda.sh").write_text(
            'python() { "$TEST_PYTHON" "$TEST_ROOT/record.py" "$@"; }\n',
            encoding="utf-8", newline="\n",
        )
        (self.root / "record.py").write_text(textwrap.dedent('''\
            import json, os, runpy, sys
            from pathlib import Path
            args = [arg for arg in sys.argv[1:] if arg != '-u']
            if args[0].endswith('list_assays.py'):
                sys.stdout.reconfigure(newline='\\n')
                sys.argv = args
                runpy.run_path(args[0], run_name='__main__')
            assert args[:2] == ['-m', 'feature_generation.readiness'], args
            assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
            Path('calls.jsonl').write_text(json.dumps(args))
            print('fixture asset audit')
            raise SystemExit(1 if os.environ.get('TEST_FAIL_ASSAY') else 0)
            '''), encoding="utf-8")

    def invoke(self, *, failed_assay="", arguments=()):
        env = os.environ.copy()
        env.update(BASH_ENV=self.bootstrap.as_posix(), TEST_ROOT=self.root.as_posix(),
                   TEST_PYTHON=Path(sys.executable).as_posix(), TEST_FAIL_ASSAY=failed_assay,
                   SLURM_SUBMIT_DIR=(self.root / "src/feature_generation").as_posix(), SLURM_JOB_ID="1234")
        return subprocess.run([BASH, str(self.script), *arguments], env=env, cwd=self.workspace,
                              capture_output=True, text=True, timeout=90)

    def test_launches_asset_audit_without_nkn_config_or_reports(self):
        result = self.invoke(arguments=("--esm2-config", "configs/extraction/custom config.json"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fixture asset audit", result.stdout)
        self.assertFalse((self.root / "results").exists())
        self.assertEqual(json.loads((self.root / "calls.jsonl").read_text()),
                         ["-m", "feature_generation.readiness", "--esm2-config", "configs/extraction/custom config.json"])

    def test_propagates_asset_audit_failure_to_slurm(self):
        result = self.invoke(failed_assay="fixture")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse((self.root / "results").exists())


if __name__ == "__main__":
    unittest.main()
