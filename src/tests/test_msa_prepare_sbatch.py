"""Exercise the MSA launcher and catalog selection without a cluster or model."""

import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

from tests.test_prepare_sbatch import ASSAYS, BASH, PREPARATION, REPOSITORY, PreparationFixture


class MSASelectionTests(PreparationFixture):
    def setUp(self):
        super().setUp()
        self.config = self.root / "configs/extraction/msa_transformer.json"
        self.config.write_text(json.dumps({"assay_index": str(self.index)}))

    def invoke(self, *args):
        environment = os.environ.copy()
        environment.pop("SLURM_ARRAY_TASK_ID", None)
        return subprocess.run([sys.executable, str(REPOSITORY / PREPARATION / "prepare_msa_transformer.py"),
                               "--config", str(self.config), *args], env=environment,
                              capture_output=True, text=True, timeout=30)

    def test_count_entire_catalog_and_subset_without_torch(self):
        for args, expected in (((), len(ASSAYS)), (("--assay", ASSAYS[-1]), 1)):
            result = self.invoke("--count", *args)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(int(result.stdout), expected)

    def test_reject_bad_array_index_unknown_and_duplicate_assays(self):
        for args in (("--task-index", "-1"), ("--task-index", str(len(ASSAYS))),
                     ("--assay", "missing"), ("--assay", ASSAYS[0], "--assay", ASSAYS[0])):
            result = self.invoke(*args)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("error:", result.stderr)

    def test_array_and_subset_route_to_the_selected_assay(self):
        spec = importlib.util.spec_from_file_location("msa_prepare_launcher", REPOSITORY / PREPARATION / "prepare_msa_transformer.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        values = json.loads(self.config.read_text())
        values.update(cache_id="test", score_cache_id="test-scores")
        self.config.write_text(json.dumps(values))
        helper_path = str(REPOSITORY / PREPARATION)
        with patch.object(sys, "path", [helper_path, *sys.path]), \
                patch.dict(os.environ, {"SLURM_ARRAY_TASK_ID": "1"}), \
                patch("setup_msa_transformer.validate_runtime", return_value=({}, object(), object())), \
                patch("feature_generation.msa_runner.MSAInference"), \
                patch("feature_generation.msa_runner.prepare", return_value=[{"status": "completed"}]) as run:
            self.assertEqual(launcher.main(["--config", str(self.config), "--resume"]), 0)
            self.assertEqual(run.call_args.args[1], [ASSAYS[1]])
            self.assertTrue(run.call_args.kwargs["resume"])
            self.assertEqual(launcher.main(["--config", str(self.config), "--assay", ASSAYS[-1],
                                           "--assay", ASSAYS[0]]), 0)
            self.assertEqual(run.call_args.args[1], [ASSAYS[0]])


@unittest.skipUnless(BASH, "Bash required")
class MSABatchTests(PreparationFixture):
    def setUp(self):
        super().setUp()
        self.script = self.root / PREPARATION / "prepare_msa_transformer.sbatch"
        shutil.copyfile(REPOSITORY / PREPARATION / self.script.name, self.script)
        (self.root / "src/feature_generation/msa_runner.py").touch()
        (self.root / "env_msa_transformer.yaml").touch()
        self.bootstrap = self.workspace / "bootstrap.sh"
        self.bootstrap.write_text(textwrap.dedent('''\
            export PATH="/usr/bin:/bin:$PATH"
            conda() {
              if [[ "$1" == info ]]; then printf '%s\\n' "$TEST_ROOT"; fi
              if [[ "$1" == activate && "$2" != nkn_msa_transformer ]]; then return 20; fi
            }
            '''), encoding="utf-8", newline="\n")
        (self.root / "etc/profile.d/conda.sh").write_text(
            'python() { "$TEST_PYTHON" "$TEST_ROOT/record.py" "$@"; }\n', encoding="utf-8", newline="\n")
        (self.root / "record.py").write_text(textwrap.dedent('''\
            import json, os, sys
            from pathlib import Path
            record = dict(args=sys.argv[1:], cwd=str(Path.cwd()), task=os.environ.get('SLURM_ARRAY_TASK_ID'),
                          config=os.environ.get('MSA_TRANSFORMER_CONFIG'), threads=os.environ['OMP_NUM_THREADS'],
                          no_user_site=os.environ['PYTHONNOUSERSITE'])
            (Path(os.environ['TEST_ROOT']) / 'calls.json').write_text(json.dumps(record))
            raise SystemExit(int(os.environ.get('TEST_EXIT', '0')))
            '''), encoding="utf-8")

    def invoke(self, *args, fail=False):
        environment = os.environ.copy()
        environment.update(BASH_ENV=self.bootstrap.as_posix(), TEST_ROOT=self.root.as_posix(),
                           TEST_PYTHON=Path(sys.executable).as_posix(), SLURM_ARRAY_TASK_ID="2",
                           SLURM_SUBMIT_DIR=(self.root / "src/feature_generation").as_posix(),
                           SLURM_CPUS_PER_TASK="6", MSA_TRANSFORMER_CONFIG="env config.json",
                           TEST_EXIT="7" if fail else "0")
        result = subprocess.run([BASH, str(self.script), *args], env=environment, cwd=self.workspace,
                                capture_output=True, text=True, timeout=30)
        record = json.loads((self.root / "calls.json").read_text())
        return result, record

    def test_spooled_job_finds_root_and_forwards_config_subset_and_array(self):
        result, record = self.invoke("--config", "custom config.json", "--assay", ASSAYS[0])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(record["cwd"]), self.root)
        self.assertEqual(record["args"], ["-u", "sbatch_run/run_prepare_rep/prepare_msa_transformer.py", "--resume",
                                          "--config", "custom config.json", "--assay", ASSAYS[0]])
        self.assertEqual((record["task"], record["config"], record["threads"], record["no_user_site"]),
                         ("2", "env config.json", "6", "1"))

    def test_inference_failure_propagates_to_slurm(self):
        result, _ = self.invoke(fail=True)
        self.assertEqual(result.returncode, 7)


if __name__ == "__main__":
    unittest.main()
