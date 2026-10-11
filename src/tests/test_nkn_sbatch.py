"""Execute the Slurm script with cheap workers to check its CPU scheduling."""

import json
from importlib import import_module
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch"


def available_bash():
    if os.name == "nt":
        for candidate in (Path("C:/Program Files/Git/usr/bin/bash.exe"),
                          Path("C:/Program Files/Git/bin/bash.exe")):
            if candidate.is_file():
                return str(candidate)
        return None  # Windows' bash.exe may be an unconfigured WSL launcher.
    return shutil.which("bash")


BASH = available_bash()


@unittest.skipUnless(BASH, "Bash is required for Slurm orchestration tests")
class NknSbatchTests(unittest.TestCase):
    script = SCRIPT
    config_file = "configs/esm2_proteinmpnn_nkn.json"
    config_variable = "NKN_CONFIG"
    cli_module = "experiments.cli"
    cli_file = "src/experiments/cli.py"
    config_sources = ("src/config.py",)
    config_type = ("src.config", "ExperimentConfig")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for directory in ("src", "configs", "features", "etc/profile.d", "records"):
            (self.root / directory).mkdir(parents=True)
        (self.root / self.cli_file).parent.mkdir(parents=True, exist_ok=True)
        (self.root / self.cli_file).touch()
        for relative_path in (*self.config_sources, self.config_file):
            target = self.root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPOSITORY / relative_path, target)
        (self.root / "src/torch.py").write_text('__version__ = "test stub"\n', encoding="utf-8")
        (self.root / "features/assay.json").write_text(
            json.dumps({"assay_id": "test", "synthetic": False}), encoding="utf-8",
        )
        (self.root / "folds.csv").write_text("mutant,fold_random_5\nA1C,0\n", encoding="utf-8")
        self.bootstrap = self.root / "bootstrap.sh"
        self.bootstrap.write_text(textwrap.dedent("""\
            export PATH="/usr/bin:/bin:$PATH"
            conda() {
              if [[ "$1" == info ]]; then printf '%s\\n' "$TEST_ROOT"; fi
            }
            """), encoding="utf-8", newline="\n")
        (self.root / "etc/profile.d/conda.sh").write_text(textwrap.dedent("""\
            python() { "$TEST_PYTHON" "$TEST_ROOT/helper.py" "$@"; }
            """), encoding="utf-8", newline="\n")
        (self.root / "helper.py").write_text(textwrap.dedent("""\
            import json
            import os
            from pathlib import Path
            import sys
            import time

            args = sys.argv[1:]
            assert args.pop(0) == '-u'
            root = Path(os.environ['TEST_ROOT'])
            if args[0] == '-':
                source = sys.stdin.read()
                if 'from dataclasses import replace' in source:
                    sys.path.insert(0, str(root / 'src'))
                    sys.argv = args
                    exec(compile(source, '<runtime-config>', 'exec'))
                else:
                    # Scientific preflight is covered separately; preserve its
                    # output files so this exercises real shell orchestration.
                    output = Path(args[1]).parent / 'folds'
                    output.mkdir()
                    for name in ('singles', 'multiples'):
                        (output / f'{name}.csv').write_text('snapshot')
                raise SystemExit(0)
            assert args[:2] == ['-m', os.environ['TEST_CLI_MODULE']]
            if args[2] == 'validate':
                raise SystemExit(0)
            options = {}
            for index, value in enumerate(args):
                if value.startswith('--') and index + 1 < len(args):
                    options[value] = args[index + 1]
            setting = options['--setting']
            record = {'setting': setting, 'command': args[2],
                      'fold_workers': int(options.get('--fold-workers', '1')),
                      'threads': int(os.environ['OMP_NUM_THREADS']),
                      'start': time.monotonic()}
            assert os.environ['MKL_NUM_THREADS'] == os.environ['OMP_NUM_THREADS']
            assert os.environ['OPENBLAS_NUM_THREADS'] == os.environ['OMP_NUM_THREADS']
            time.sleep(0.5)
            record['end'] = time.monotonic()
            (root / 'records' / f'{setting}.json').write_text(json.dumps(record))
            raise SystemExit(9 if setting == os.environ.get('TEST_FAIL_SETTING') else 0)
            """), encoding="utf-8")

    def invoke(self, cpus, *arguments, fail_setting=None, config_environment=None):
        environment = os.environ.copy()
        for name in ("FOLDS_CSV", self.config_variable, "TEST_FAIL_SETTING", "SLURM_CPUS_PER_TASK"):
            environment.pop(name, None)
        environment.update(
            BASH_ENV=self.bootstrap.as_posix(), TEST_ROOT=self.root.as_posix(),
            TEST_PYTHON=Path(sys.executable).as_posix(), ASSAY="test",
            FEATURES="features", SINGLES_FOLDS_CSV="folds.csv", MULTIPLES_FOLDS_CSV="folds.csv",
            SLURM_SUBMIT_DIR=self.root.as_posix(), SLURM_JOB_ID="1234",
            TEST_CLI_MODULE=self.cli_module,
        )
        if cpus is not None:
            environment['SLURM_CPUS_PER_TASK'] = str(cpus)
        if fail_setting:
            environment["TEST_FAIL_SETTING"] = fail_setting
        if config_environment is not None:
            environment[self.config_variable] = config_environment
        result = subprocess.run([BASH, str(self.script), *arguments], cwd=self.root, env=environment,
                                capture_output=True, text=True, timeout=30)
        records = [json.loads(path.read_text()) for path in (self.root / "records").glob("*.json")]
        plans = list((self.root / "results").rglob("execution_plan.json"))
        plan = json.loads(plans[0].read_text()) if plans else None
        return result, records, plan

    @staticmethod
    def peak_concurrency(records):
        events = [(record['start'], 1) for record in records]
        events += [(record['end'], -1) for record in records]
        active = peak = 0
        for _, change in sorted(events):
            active += change
            peak = max(peak, active)
        return peak

    def test_fifteen_cpus_run_three_settings_with_five_single_threaded_folds(self):
        result, records, plan = self.invoke(15, "singles", "--parallel")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(records), 3)
        self.assertEqual(self.peak_concurrency(records), 3)
        self.assertEqual({(r['fold_workers'], r['threads']) for r in records}, {(5, 1)})
        self.assertEqual((plan['allocated_cpus'], plan['setting_workers']), (15, 3))
        self.assertIn('Completed.', result.stdout)

    def test_default_remains_serial_with_all_threads_per_fit(self):
        result, records, plan = self.invoke(4)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(records), 3)
        self.assertEqual(self.peak_concurrency(records), 1)
        self.assertEqual({(r['fold_workers'], r['threads']) for r in records}, {(1, 4)})
        self.assertFalse(plan['parallel'])

    def test_default_cpu_request_and_manual_fallback_are_fifteen(self):
        self.assertIn('#SBATCH --cpus-per-task=15', self.script.read_text())
        result, records, plan = self.invoke(None, 'singles', '--parallel')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(plan['allocated_cpus'], 15)
        self.assertEqual({(r['fold_workers'], r['threads']) for r in records}, {(5, 1)})

    def test_config_flag_overrides_environment_and_preserves_settings(self):
        config = json.loads((self.root / self.config_file).read_text())
        config['evaluation']['coverage_levels'] = [0.8]
        config['num_threads'] = 9
        selected = self.root / 'selected config.json'
        selected.write_text(json.dumps(config), encoding='utf-8')
        result, records, _ = self.invoke(
            15, 'singles', '--parallel', '--config', str(selected),
            config_environment='missing-environment-config.json',
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        runtime = next((self.root / 'results').rglob('config.json'))
        saved = json.loads(runtime.read_text())
        config.update(device='cpu', num_threads=1)
        # Configuration classes fill in defaults and normalize branch order.
        module, name = self.config_type
        config_class = getattr(import_module(module), name)
        expected = json.loads(json.dumps(config_class.from_dict(config).to_dict()))
        self.assertEqual(saved, expected)
        self.assertEqual({r['threads'] for r in records}, {1})

    def test_five_cpus_on_one_setting_provide_five_fold_workers(self):
        result, records, _ = self.invoke(5, "singles_random", "--parallel")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([(r['fold_workers'], r['threads']) for r in records], [(5, 1)])

    def test_small_all_allocation_bounds_concurrent_settings_and_keeps_fit_route(self):
        result, records, plan = self.invoke(2, "all", "--parallel")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(records), 5)
        self.assertEqual(self.peak_concurrency(records), 2)
        self.assertEqual(plan['fold_workers'], 1)
        self.assertEqual({(r['fold_workers'], r['threads']) for r in records}, {(1, 1)})
        extrapolation = next(r for r in records if r['setting'] == 'single_to_double')
        self.assertEqual(extrapolation['command'], 'fit')

    def test_child_failure_makes_job_fail_without_completion_message(self):
        result, records, _ = self.invoke(15, "singles", "--parallel", fail_setting="singles_modulo")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(records), 3)
        self.assertIn('singles_modulo failed (exit 9)', result.stderr)
        self.assertNotIn('Completed.', result.stdout)

    def test_invalid_cpu_count_fails_before_launching_workers(self):
        result, records, plan = self.invoke(0, "--parallel")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(records, [])
        self.assertIsNone(plan)
        self.assertIn('positive integer', result.stderr)


if __name__ == '__main__':
    unittest.main()
