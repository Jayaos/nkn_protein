"""Check the real-assay GPU launch contract without fitting a large model."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import csv
from importlib.util import find_spec
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from data_loading import FeatureBatch
from experiments import large_assay_gpu as gpu


ROOT = Path(__file__).resolve().parents[2]
FOLDS = ROOT / "data/DMS_ProteinGym_substitutions/cv_folds_singles_substitutions" / f"{gpu.ASSAY}.csv"
SCRIPT = ROOT / "sbatch_run/run_nkn/run_nkn_gpu.sbatch"
BASH = str(Path("C:/Program Files/Git/bin/bash.exe")) if os.name == "nt" else shutil.which("bash")


class LargeAssayGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        raw = ROOT / "data/DMS_ProteinGym_substitutions/raw_substitutions" / f"{gpu.ASSAY}.csv"
        with raw.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        cls.ids = tuple(row.get("variant_id") or row["mutant"] for row in rows)
        positions = [int(re.fullmatch(r"[A-Z](\d+)[A-Z]", row["mutant"]).group(1)) for row in rows]
        n = len(rows)
        # Only IDs/positions come from the human assay; numerical values are
        # dummy tensors. The fitting call is mocked in these orchestration tests.
        cls.batch = FeatureBatch(
            assay_id=gpu.ASSAY, variant_ids=cls.ids,
            targets=torch.arange(n, dtype=torch.float64),
            scores=torch.ones(n, 2, dtype=torch.float64),
            global_features=torch.ones(n, 1, dtype=torch.float64),
            local_features={}, owners=torch.arange(n),
            mutation_positions=torch.tensor(positions),
        )

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.features = self.root / "features"
        self.features.mkdir()
        self.metadata = self.features / "assay.json"
        self.metadata.write_text(json.dumps({"assay_id": gpu.ASSAY, "synthetic": False}))
        self.output = self.root / "output"

    def invoke(self, *extra, cuda=True):
        args = [
            "--features", str(self.features), "--folds", str(FOLDS),
            "--output", str(self.output), "--num-threads", "1",
            "--max-steps", "2", "--min-steps", "0", *extra,
        ]
        stream = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(stream))
            stack.enter_context(redirect_stderr(stream))
            stack.enter_context(patch.object(gpu.torch.cuda, "is_available", return_value=cuda))
            for name in ("synchronize", "reset_peak_memory_stats"):
                stack.enter_context(patch.object(gpu.torch.cuda, name))
            for name in ("max_memory_allocated", "max_memory_reserved"):
                stack.enter_context(patch.object(gpu.torch.cuda, name, return_value=1024))
            stack.enter_context(patch.object(gpu.torch.cuda, "get_device_name", return_value="test GPU"))
            stack.enter_context(patch.object(gpu.torch.cuda, "get_device_properties",
                                            return_value=SimpleNamespace(total_memory=80 * 2**30)))
            stack.enter_context(patch.object(gpu, "validate_device_parity", return_value={"status": "passed"}))
            stack.enter_context(patch.object(gpu, "validate_backend_parity", return_value={"status": "passed"}))
            loader = stack.enter_context(patch.object(gpu, "load_feature_bundle", return_value=self.batch))
            coverage = stack.enter_context(patch.object(gpu, "validate_assay_coverage"))
            fit = stack.enter_context(patch.object(gpu, "run_cv"))
            status = gpu.main(args)
        return status, stream.getvalue(), loader, coverage, fit

    def test_full_human_assay_and_published_folds_reach_serial_cuda_runner(self):
        self.assertEqual(len(self.ids), 10094)
        self.assertEqual(len(set(self.ids)), 10094)
        with (ROOT / "data/DMS_substitutions.csv").open(newline="", encoding="utf-8") as handle:
            record = next(row for row in csv.DictReader(handle) if row["DMS_id"] == gpu.ASSAY)
        self.assertEqual(record["source_organism"], "Homo sapiens")
        self.assertEqual(int(record["DMS_number_single_mutants"]), len(self.ids))
        status, log, loader, coverage, fit = self.invoke()
        self.assertEqual(status, 0, log)
        self.assertEqual(loader.call_args.kwargs["device"], "cpu")
        self.assertTrue(coverage.call_args.kwargs["singles_only"])
        self.assertEqual(fit.call_count, 3)
        for call, setting in zip(fit.call_args_list, gpu.SETTINGS):
            batch, folds, config, output = call.args
            self.assertEqual(batch.num_variants, 10094)
            self.assertEqual(set(folds.tolist()), set(range(5)))
            self.assertEqual(config.device, "cuda")
            self.assertEqual(config.training.max_steps, 2)
            self.assertEqual(config.training.inference_backend, "cholesky")
            self.assertIsNone(config.training.max_train_variants)
            self.assertEqual(call.kwargs["setting"], setting)
            self.assertEqual(call.kwargs["fold_workers"], 1)
            self.assertEqual(call.kwargs["split_source"], "provided")
        self.assertEqual((self.output / "folds/singles.csv").read_bytes(), FOLDS.read_bytes())
        manifest = json.loads((self.output / "fold_manifest.json").read_text())
        for counts in manifest["held_out_counts"].values():
            self.assertEqual(sum(counts), 10094)
        self.assertEqual(len(json.loads((self.output / "benchmark.json").read_text())["settings"]), 3)

    @unittest.skipUnless(find_spec("linear_operator"), "optional linear_operator dependency required")
    def test_bbmm_controls_reach_all_settings_and_saved_config(self):
        status, log, _, _, fit = self.invoke(
            "--inference-backend", "bbmm", "--bbmm-cg-tolerance", "0.0001",
            "--bbmm-max-cg-iterations", "200", "--bbmm-num-trace-samples", "16",
            "--bbmm-max-lanczos-iterations", "40", "--bbmm-preconditioner-size", "10",
            "--bbmm-probe-seed", "37",
        )
        self.assertEqual(status, 0, log)
        self.assertEqual(fit.call_count, 3)
        expected = {
            "inference_backend": "bbmm", "bbmm_cg_tolerance": 0.0001,
            "bbmm_max_cg_iterations": 200, "bbmm_num_trace_samples": 16,
            "bbmm_max_lanczos_iterations": 40, "bbmm_preconditioner_size": 10,
            "bbmm_probe_seed": 37,
        }
        for call in fit.call_args_list:
            training = call.args[2].training
            for name, value in expected.items():
                self.assertEqual(getattr(training, name), value)
            self.assertEqual(call.args[0].num_variants, 10094)
            self.assertEqual(call.kwargs["fold_workers"], 1)
        saved = json.loads((self.output / "config.json").read_text())["training"]
        for name, value in expected.items():
            self.assertEqual(saved[name], value)
        self.assertIn("bbmm", log.lower())
        self.assertEqual(json.loads((self.output / "backend_parity.json").read_text())["status"], "passed")
        for filename in ("execution_plan.json", "benchmark.json"):
            metadata = json.loads((self.output / filename).read_text())
            self.assertEqual(metadata["inference_backend"], "bbmm")
            self.assertEqual(metadata["inference"]["bbmm_probe_seed"], 37)

    def test_invalid_later_split_stops_before_any_fit(self):
        broken = self.root / "broken_folds.csv"
        broken.write_text(FOLDS.read_text().replace("fold_contiguous_5", "missing_contiguous"))
        status, log, _, _, fit = self.invoke("--folds", str(broken))
        self.assertEqual(status, 2)
        self.assertIn("fold_contiguous_5", log)
        fit.assert_not_called()
        self.assertTrue((self.output / "failure.json").is_file())

    def test_identity_and_no_cuda_fail_before_loading(self):
        self.metadata.write_text(json.dumps({"assay_id": "wrong", "synthetic": False}))
        status, log, loader, _, fit = self.invoke()
        self.assertEqual(status, 2)
        self.assertIn("Expected real prepared assets", log)
        loader.assert_not_called()
        fit.assert_not_called()
        self.assertFalse(self.output.exists())
        status, log, loader, _, fit = self.invoke(cuda=False)
        self.assertEqual(status, 2)
        self.assertIn("CUDA is unavailable", log)
        loader.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_existing_results_are_preserved(self):
        self.output.mkdir()
        marker = self.output / "marker"
        marker.write_text("keep")
        status, log, loader, _, fit = self.invoke()
        self.assertEqual(status, 2)
        self.assertIn("already exists", log)
        self.assertEqual(marker.read_text(), "keep")
        loader.assert_not_called()
        fit.assert_not_called()

    @unittest.skipUnless(BASH and Path(BASH).is_file(), "Bash required")
    def test_sbatch_dispatch_preserves_gpu_and_selects_real_assay_output(self):
        (self.root / "src/experiments").mkdir(parents=True)
        (self.root / "src/experiments/large_assay_gpu.py").touch()
        subdir = self.root / "submission"
        subdir.mkdir()
        capture = self.root / "capture.py"
        capture.write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['TEST_RECORD']).write_text(json.dumps({"
            "'args':sys.argv[1:], 'cuda':os.environ.get('CUDA_VISIBLE_DEVICES'),"
            "'threads':os.environ['OMP_NUM_THREADS']}))\n"
        )
        interpreter = self.root / "python-wrapper"
        interpreter.write_text('#!/bin/bash\nexec "$TEST_PYTHON" "$TEST_CAPTURE" "$@"\n', newline="\n")
        interpreter.chmod(0o755)
        record = self.root / "record.json"
        env = dict(os.environ, NKN_PYTHON=interpreter.as_posix(), SLURM_SUBMIT_DIR=subdir.as_posix(),
                   SLURM_CPUS_PER_TASK="8", SLURM_ARRAY_TASK_ID="0", SLURM_JOB_ID="4242",
                   CUDA_VISIBLE_DEVICES="3", TEST_PYTHON=Path(sys.executable).as_posix(),
                   TEST_CAPTURE=capture.as_posix(), TEST_RECORD=record.as_posix())
        overrides = [
            [],
            ["--inference-backend", "bbmm", "--bbmm-cg-tolerance", "0.0001",
             "--bbmm-max-cg-iterations", "200", "--bbmm-num-trace-samples", "16",
             "--bbmm-max-lanczos-iterations", "40", "--bbmm-preconditioner-size", "10",
             "--bbmm-probe-seed", "37", "--output", "results/custom_bbmm"],
        ]
        for extra in overrides:
            with self.subTest(extra=extra):
                forwarded = ["singles_random", "--max-steps", "2", "--min-steps", "0", *extra]
                result = subprocess.run(
                    [BASH, SCRIPT.as_posix(), *forwarded],
                    cwd=subdir, env=env, text=True, capture_output=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                captured = json.loads(record.read_text())
                self.assertEqual(captured["args"][:3], ["-u", "-m", "experiments.large_assay_gpu"])
                self.assertEqual(captured["args"][-len(forwarded):], forwarded)
                output = captured["args"][captured["args"].index("--output") + 1]
                self.assertIn(gpu.ASSAY, output)
                self.assertTrue(output.endswith("singles_random_gpu_4242"))
                parsed = gpu._parser().parse_args(captured["args"][3:])
                self.assertEqual(parsed.output, Path("results/custom_bbmm") if extra else Path(output))
                if extra:
                    self.assertEqual(parsed.inference_backend, "bbmm")
                    self.assertEqual(parsed.bbmm_probe_seed, 37)
                    self.assertEqual(parsed.bbmm_num_trace_samples, 16)
                    self.assertEqual(parsed.bbmm_cg_tolerance, 0.0001)
                self.assertEqual(captured["cuda"], "3")
                self.assertEqual(captured["threads"], "8")


if __name__ == "__main__":
    unittest.main()
