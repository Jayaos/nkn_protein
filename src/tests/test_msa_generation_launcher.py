"""Exercise output snapshotting and Slurm routing without submitting real jobs."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "sbatch_run/run_prepare_rep/generate_msa_weights.sbatch"
BASH = (shutil.which("bash") if os.name != "nt"
        else next((str(path) for path in [Path("C:/Program Files/Git/bin/bash.exe")]
                   if path.is_file()), None))


class OutputSnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reference = self.root / "reference.csv"
        self.reference.write_bytes(b"DMS_id\r\nA\r\nB\r\n")
        self.output = self.root / "data/msa/DMS_msa_weights_generated"

    def snapshot(self, check=True):
        source = LAUNCHER.read_text().split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        return subprocess.run(
            [sys.executable, "-B", "-", str(self.reference),
             str(ROOT / "configs/setup/msa_generation.json"), str(self.output)],
            input=source, text=True, capture_output=True, check=check,
        )

    def test_snapshot_saves_directly_in_requested_output_directory(self):
        result = self.snapshot()
        count, directory = result.stdout.strip().split("\t")
        output = Path(directory)
        self.assertEqual(count, "2")
        self.assertEqual(output, self.output.resolve())
        self.assertEqual((output / "DMS_substitutions.csv").read_bytes(), self.reference.read_bytes())
        config = json.loads((output / "config.json").read_text())
        self.assertEqual(Path(config["reference_csv"]), output / "DMS_substitutions.csv")
        self.assertEqual(Path(config["output_folder"]), output / "assays")
        self.assertNotIn("weights_folder", config)

    def test_existing_output_is_not_overwritten_by_resubmission(self):
        self.snapshot()
        original = {path.name: path.read_bytes() for path in self.output.iterdir()}
        self.reference.write_text("DMS_id\nC\n")
        result = self.snapshot(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output directory is not empty", result.stderr)
        self.assertEqual(original, {path.name: path.read_bytes() for path in self.output.iterdir()})


@unittest.skipUnless(BASH, "Bash required for mock scheduler checks")
class LauncherRoutingTests(unittest.TestCase):
    def run_launcher(self, *args):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        output = root / "data/msa/DMS_msa_weights_generated"
        output.mkdir(parents=True)
        (output / "config.json").write_text("{}")
        bash_path = root.as_posix()
        if os.name == "nt":
            bash_path = "/" + bash_path[0].lower() + bash_path[2:]
        body = LAUNCHER.read_text().split("mode=submit\n", 1)[1]
        harness = "fixture_root=" + shlex.quote(bash_path) + "\n" + r'''
set -euo pipefail
project_root="$fixture_root"
submission_dir="$fixture_root"
output_dir_test="$fixture_root/data/msa/DMS_msa_weights_generated"
OMP_NUM_THREADS=8
SLURM_ARRAY_TASK_ID=42
mode=submit
python() {
  if [[ "$*" == *--generate-only* ]]; then
    printf 'worker:%s\n' "$*"
  elif [[ "$*" == *generated_msa.py* ]]; then
    printf 'summary:%s\n' "$*"
  else
    printf '217\t%s\n' "$output_dir_test"
  fi
}
sbatch() {
  printf '<%s>' "$@" >> "$fixture_root/submissions.log"
  printf '\n' >> "$fixture_root/submissions.log"
  if [[ "$*" == *afterany:* ]]; then printf '202;phoenix\n'; else printf '101;phoenix\n'; fi
}
''' + "mode=submit\n" + body
        arguments = [bash_path + "/data/msa/DMS_msa_weights_generated" if arg == "OUTPUT" else arg for arg in args]
        result = subprocess.run([BASH, "-s", "--", *arguments], input=harness,
                                text=True, capture_output=True, check=True)
        log = root / "submissions.log"
        return result.stdout, log.read_text() if log.exists() else ""

    def test_default_launches_entire_catalog_and_summary_after_failures(self):
        output, submissions = self.run_launcher()
        self.assertIn("<--array=0-216%4>", submissions)
        self.assertIn("<--dependency=afterany:101>", submissions)
        self.assertIn("<--cpus-per-task=8>", submissions)
        self.assertIn("<--worker>", submissions)
        self.assertIn("<--summarize>", submissions)
        self.assertEqual(len(submissions.splitlines()), 2)
        self.assertIn("Generation array: 101 (217 assays", output)

    def test_worker_generates_the_requested_catalog_row(self):
        output, submissions = self.run_launcher("--worker", "--output-dir", "OUTPUT")
        self.assertIn("worker:", output)
        self.assertIn("--generate-only --assay-index 42 --cpus 8", output)
        self.assertEqual(submissions, "")

    def test_summary_mode_does_not_launch_generation(self):
        output, submissions = self.run_launcher("--summarize")
        self.assertIn("summary:", output)
        self.assertEqual(submissions, "")


if __name__ == "__main__":
    unittest.main()
