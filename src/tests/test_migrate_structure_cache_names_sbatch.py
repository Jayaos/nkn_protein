"""Migration launcher dispatch, argument forwarding, and CPU-only execution."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import unittest

from tests.test_prepare_sbatch import BASH, PREPARATION, REPOSITORY, PreparationFixture


@unittest.skipUnless(BASH, "Bash is required for migration launcher tests")
class MigrateStructureCacheNamesSbatchTests(PreparationFixture):
    def setUp(self):
        super().setUp()
        self.script = self.root / PREPARATION / "migrate_structure_cache_names.sbatch"
        shutil.copyfile(REPOSITORY / PREPARATION / self.script.name, self.script)
        (self.root / "src/feature_generation/storage.py").touch()
        (self.root / PREPARATION / "migrate_structure_cache_names.py").touch()
        self.bootstrap = self.workspace / "bootstrap.sh"
        self.bootstrap.write_text(textwrap.dedent('''\
            export PATH="/usr/bin:/bin:$PATH"
            conda() {
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
                    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
            }
            Path('migration-call.json').write_text(json.dumps(record))
            print('fixture cache migration')
            raise SystemExit(int(os.environ.get('TEST_EXIT_CODE', '0')))
            '''), encoding="utf-8")

    def invoke(self, *arguments, exit_code=0):
        env = os.environ.copy()
        env.update(BASH_ENV=self.bootstrap.as_posix(), TEST_ROOT=self.root.as_posix(),
                   TEST_PYTHON=Path(sys.executable).as_posix(), TEST_EXIT_CODE=str(exit_code),
                   SLURM_SUBMIT_DIR=(self.root / "src/feature_generation").as_posix(),
                   SLURM_JOB_ID="1234", CUDA_VISIBLE_DEVICES="0")
        result = subprocess.run([BASH, str(self.script), *arguments], env=env, cwd=self.workspace,
                                capture_output=True, text=True, timeout=90)
        path = self.root / "migration-call.json"
        return result, json.loads(path.read_text()) if path.is_file() else None

    def test_defaults_apply_with_project_environment_and_cpu_only(self):
        result, record = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fixture cache migration", result.stdout)
        self.assertEqual(record["args"], [
            "-u", "sbatch_run/run_prepare_rep/migrate_structure_cache_names.py"])
        self.assertEqual(Path(record["cwd"]), self.root)
        self.assertEqual(record["environment"], {
            "TEST_ACTIVE_ENV": "nkn_protein", "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        })

    def test_forwards_custom_root_assays_and_dry_run(self):
        arguments = ("--output-root", "data/prepared outputs/singles", "--assay",
                     "BLAT_ECOLX_Jacquier_2013", "--assay", "another_assay", "--dry-run")
        result, record = self.invoke(*arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(record["args"], [
            "-u", "sbatch_run/run_prepare_rep/migrate_structure_cache_names.py", *arguments])

    def test_propagates_helper_failure_to_slurm(self):
        result, record = self.invoke(exit_code=23)
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertIsNotNone(record)


if __name__ == "__main__":
    unittest.main()
