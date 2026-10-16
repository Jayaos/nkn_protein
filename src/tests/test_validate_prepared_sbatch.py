"""Readiness launcher dispatch, failure reporting, and CPU-only execution."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import unittest

from tests.test_prepare_sbatch import ASSAYS, BASH, PREPARATION, REPOSITORY, PreparationFixture


@unittest.skipUnless(BASH, "Bash is required for readiness launcher tests")
class ValidatePreparedSbatchTests(PreparationFixture):
    def setUp(self):
        super().setUp()
        self.script = self.root / PREPARATION / "validate_prepared.sbatch"
        shutil.copyfile(REPOSITORY / PREPARATION / self.script.name, self.script)
        (self.root / "src/experiments").mkdir()
        (self.root / "src/experiments/cli.py").touch()
        config_name = "esm2_proteinmpnn_nkn_two_scores_additive_four_branches_noise_initial_0p5.json"
        shutil.copyfile(REPOSITORY / "configs" / config_name, self.root / "configs" / config_name)
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
            assert args[:3] == ['-m', 'experiments.cli', 'validate'], args
            assert args[args.index('--device') + 1] == 'cpu'
            assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
            assert '--full-assay' in args and '--singles-only' in args
            assert '--no-output' in args and '--output' not in args
            assay = Path(args[args.index('--features') + 1]).name
            with Path('calls.jsonl').open('a') as handle:
                handle.write(json.dumps({'assay': assay, 'args': args}) + '\\n')
            print('fixture validation: ' + assay)
            if assay == os.environ.get('TEST_FAIL_ASSAY'):
                print('fixture missing cache', file=sys.stderr)
                raise SystemExit(1)
            '''), encoding="utf-8")

    def invoke(self, *, failed_assay="", arguments=()):
        env = os.environ.copy()
        env.update(BASH_ENV=self.bootstrap.as_posix(), TEST_ROOT=self.root.as_posix(),
                   TEST_PYTHON=Path(sys.executable).as_posix(), TEST_FAIL_ASSAY=failed_assay,
                   SLURM_SUBMIT_DIR=(self.root / "src/experiments").as_posix(), SLURM_JOB_ID="1234")
        return subprocess.run([BASH, str(self.script), *arguments], env=env, cwd=self.workspace,
                              capture_output=True, text=True, timeout=90)

    def test_success_validates_entire_catalog_and_prints_without_reports(self):
        alternate = self.root / "configs/custom config.json"
        alternate.write_text("{}")
        result = self.invoke(arguments=("--config", "configs/custom config.json"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / "results").exists())
        for i, assay in enumerate(ASSAYS):
            self.assertIn(f"{i}\t{assay}\tREADY", result.stdout)
            self.assertIn(f"fixture validation: {assay}", result.stdout)
        self.assertIn("Ready: 5 / 5; failed: 0", result.stdout)
        calls = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]
        self.assertEqual([call["assay"] for call in calls], ASSAYS)
        for call in calls:
            self.assertEqual(call["args"][call["args"].index("--config") + 1], "configs/custom config.json")

    def test_failed_assay_keeps_checking_and_returns_failure(self):
        result = self.invoke(failed_assay=ASSAYS[1])
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse((self.root / "results").exists())
        self.assertIn("Ready: 4 / 5; failed: 1", result.stdout)
        self.assertIn(f"1\t{ASSAYS[1]}\tFAILED", result.stdout)
        self.assertIn("fixture missing cache", result.stderr)
        self.assertIn(f"{len(ASSAYS) - 1}\t{ASSAYS[-1]}\tREADY", result.stdout)

    def test_missing_catalog_input_fails_before_validating(self):
        (self.raw / f"{ASSAYS[0]}.csv").unlink()
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "calls.jsonl").exists())
        self.assertFalse((self.root / "results").exists())
        self.assertNotIn("Ready:", result.stdout)


if __name__ == "__main__":
    unittest.main()
