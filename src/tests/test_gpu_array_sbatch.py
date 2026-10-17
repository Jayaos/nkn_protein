"""Exercise the array launcher without a Slurm installation or GPU allocation."""

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

from experiments.large_assay_gpu import _parser


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sbatch_run/run_nkn/run_nkn_gpu.sbatch"
BASH = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
DEFAULT_ASSAYS = (
    "A4_HUMAN_Seuma_2022",
    "BRCA2_HUMAN_Erwood_2022_HEK293T",
    "FKBP3_HUMAN_Tsuboyama_2023_2KFV",
)
FOURTH_ASSAY = "S22A1_HUMAN_Yee_2023_activity"


@unittest.skipUnless(BASH and Path(BASH).is_file(), "Bash required")
class HumanAssayArrayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "src/experiments").mkdir(parents=True)
        (self.root / "src/experiments/large_assay_gpu.py").touch()
        self.subdir = self.root / "submission"
        self.subdir.mkdir()
        capture = self.root / "capture.py"
        capture.write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['TEST_RECORD']).write_text(json.dumps({"
            "'args':sys.argv[1:], 'cuda':os.environ.get('CUDA_VISIBLE_DEVICES'),"
            "'threads':os.environ['OMP_NUM_THREADS']}))\n", encoding="utf-8",
        )
        interpreter = self.root / "python-wrapper"
        interpreter.write_text('#!/bin/bash\nexec "$TEST_PYTHON" "$TEST_CAPTURE" "$@"\n',
                               encoding="utf-8", newline="\n")
        interpreter.chmod(0o755)
        self.record = self.root / "record.json"
        self.env = dict(
            os.environ, NKN_PYTHON=interpreter.as_posix(),
            SLURM_SUBMIT_DIR=self.subdir.as_posix(), SLURM_CPUS_PER_TASK="8",
            SLURM_JOB_ID="4242", SLURM_ARRAY_JOB_ID="4200", CUDA_VISIBLE_DEVICES="3",
            NKN_OUTPUT_ROOT=(self.root / "results").as_posix(),
            TEST_PYTHON=Path(sys.executable).as_posix(), TEST_CAPTURE=capture.as_posix(),
            TEST_RECORD=self.record.as_posix(),
        )
        self.env.pop("NKN_CONFIG", None)
        self.env.pop("EXPERIMENT_SUFFIX", None)

    def invoke(self, index, *arguments, script=None):
        # The production list is user-editable; use an explicit supported fixture.
        if script is None:
            script = self.editable_script(DEFAULT_ASSAYS)
        return subprocess.run(
            [BASH, script.as_posix(), *arguments], cwd=self.subdir,
            env=dict(self.env, SLURM_ARRAY_TASK_ID=str(index)),
            text=True, capture_output=True, timeout=30,
        )

    def editable_script(self, assays):
        """Exercise the documented workflow: edit only the list in the file."""
        contents = SCRIPT.read_text(encoding="utf-8")
        replacement = "assays=(\n" + "".join(f"  {shlex.quote(assay)}\n" for assay in assays) + ")"
        contents, replacements = re.subn(r"(?ms)^assays=\(.*?\)", lambda _: replacement,
                                         contents, count=1)
        self.assertEqual(replacements, 1)
        script = self.root / "editable.sbatch"
        script.write_text(contents, encoding="utf-8", newline="\n")
        return script

    def test_all_tasks_dispatch_selected_real_assays_with_both_backends(self):
        outputs = set()
        for index, assay in enumerate(DEFAULT_ASSAYS):
            for backend in ("cholesky", "bbmm"):
                with self.subTest(index=index, backend=backend):
                    result = self.invoke(
                        index, "singles_random", "--gp-computation-backend", backend,
                        "--max-steps", "2", "--min-steps", "0",
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    captured = json.loads(self.record.read_text())
                    self.assertEqual(captured["args"][:3], ["-u", "-m", "experiments.large_assay_gpu"])
                    parsed = _parser().parse_args(captured["args"][3:])
                    self.assertEqual(parsed.assay, assay)
                    self.assertEqual(parsed.selection, "singles_random")
                    self.assertEqual(parsed.gp_computation_backend, backend)
                    self.assertEqual(parsed.max_steps, 2)
                    self.assertEqual(parsed.min_steps, 0)
                    self.assertEqual(parsed.num_threads, 8)
                    self.assertEqual(parsed.config, Path(
                        "configs/esm2_proteinmpnn_nkn_two_scores_additive_four_branches_noise_initial_1p0.json"
                    ))
                    self.assertEqual(parsed.output, self.root / "results" / assay
                                     / f"nkn_esm2_proteinmpnn_singles_random_gpu_4242_{index}")
                    outputs.add(parsed.output)
                    self.assertEqual(captured["cuda"], "3")
                    self.assertEqual(captured["threads"], "8")
        self.assertEqual(len(outputs), 3)

    def test_invalid_tasks_and_managed_overrides_fail_before_dispatch(self):
        cases = [(index, ()) for index in ("-1", "3", "01", "invalid", "9" * 100, "18446744073709551616")]
        cases += [("0", args) for args in (
            ("--output", "shared"), ("--output=shared",),
            ("--device", "cpu"), ("--device=cpu",),
            ("--assay", FOURTH_ASSAY), (f"--assay={FOURTH_ASSAY}",),
        )]
        for index, args in cases:
            with self.subTest(index=index, args=args):
                result = self.invoke(index, *args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("error:", result.stderr)
                self.assertFalse(self.record.exists())

    def test_custom_list_dispatches_fourth_supported_assay(self):
        script = self.editable_script([*DEFAULT_ASSAYS, FOURTH_ASSAY])
        result = self.invoke(3, "singles_random", "--max-steps", "2", "--min-steps", "0",
                             script=script)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        captured = json.loads(self.record.read_text())
        self.assertEqual(captured["args"][:3], ["-u", "-m", "experiments.large_assay_gpu"])
        parsed = _parser().parse_args(captured["args"][3:])
        self.assertEqual(parsed.assay, FOURTH_ASSAY)
        self.assertEqual(parsed.output, self.root / "results" / FOURTH_ASSAY
                         / "nkn_esm2_proteinmpnn_singles_random_gpu_4242_3")

    def test_default_selection_and_environment_config_are_forwarded(self):
        self.env["NKN_CONFIG"] = "configs/custom.json"
        result = self.invoke(0)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        captured = json.loads(self.record.read_text())
        parsed = _parser().parse_args(captured["args"][3:])
        self.assertEqual(parsed.selection, "singles")
        self.assertEqual(parsed.config, Path("configs/custom.json"))
        self.assertEqual(parsed.assay, DEFAULT_ASSAYS[0])
        self.assertEqual(parsed.output, self.root / "results" / DEFAULT_ASSAYS[0]
                         / "nkn_esm2_proteinmpnn_singles_gpu_4242_0")

    def test_results_group_is_consumed_without_changing_selection_or_backend(self):
        cases = (
            (("gpu_bbmm_test",), "gpu_bbmm_test", "singles"),
            (("gpu_bbmm_test", "singles_random"), "gpu_bbmm_test", "singles_random"),
            (("singles_modulo", "--experiment-suffix", "named"), "named", "singles_modulo"),
            (("--experiment-suffix=named",), "named", "singles"),
            (("gpu_bbmm_test", "--experiment-suffix", "override"), "override", "singles"),
            ((), "from_env", "singles"),
        )
        self.env["EXPERIMENT_SUFFIX"] = "from_env"
        for arguments, group, selection in cases:
            with self.subTest(arguments=arguments):
                result = self.invoke(0, *arguments, "--gp-computation-backend", "bbmm")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                captured = json.loads(self.record.read_text())
                parsed = _parser().parse_args(captured["args"][3:])
                self.assertEqual(parsed.selection, selection)
                self.assertEqual(parsed.gp_computation_backend, "bbmm")
                self.assertEqual(parsed.output, self.root / "results" / group / DEFAULT_ASSAYS[0]
                                 / f"nkn_esm2_proteinmpnn_{selection}_gpu_4242_0")

    def test_invalid_group_paths_fail_before_dispatch(self):
        for group in (".", "..", "../outside", "path/name", "path\\name"):
            with self.subTest(group=group):
                result = self.invoke(0, group, "--gp-computation-backend", "bbmm")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("single folder name", result.stderr)
                self.assertFalse(self.record.exists())

    def test_help_reflects_edited_list_without_initializing_python(self):
        assays = [*DEFAULT_ASSAYS, FOURTH_ASSAY]
        script = self.editable_script(assays)
        result = self.invoke("invalid", "--help", script=script)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Configured human assays: 4", result.stdout)
        self.assertIn("--array=0-3", result.stdout)
        for index, assay in enumerate(assays):
            self.assertIn(f"{index}: {assay}", result.stdout)
        self.assertFalse(self.record.exists())

    def test_invalid_lists_fail_before_interpreter_dispatch(self):
        valid = DEFAULT_ASSAYS[0]
        lists = [[], [valid, valid], [valid, ""], [valid, "../outside"],
                 [valid, "path/name"], [valid, "two words"],
                 [valid, "UNSUPPORTED_HUMAN_example_2026"]]
        for assays in lists:
            with self.subTest(assays=assays):
                script = self.editable_script(assays)
                result = self.invoke(0, "singles_random", script=script)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("error:", result.stderr)
                self.assertFalse(self.record.exists())


if __name__ == "__main__":
    unittest.main()
