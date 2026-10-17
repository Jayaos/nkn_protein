"""Published-fold launcher checks with synthetic inputs and mocked GPU fitting."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from config import ExperimentConfig
from data_loading import FeatureBatch
from experiments import large_assay_gpu as gpu


ROOT = Path(__file__).resolve().parents[2]


class PopulatedFoldLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.features = self.root / "features"
        self.features.mkdir()
        self.assay = "synthetic_fold_protocol"
        (self.features / "assay.json").write_text(json.dumps(
            {"assay_id": self.assay, "synthetic": False}), encoding="utf-8")
        self.output = self.root / "output"
        self.batch = FeatureBatch(
            assay_id=self.assay, variant_ids=tuple(f"variant_{i}" for i in range(8)),
            targets=torch.arange(8, dtype=torch.float64),
            scores=torch.ones(8, 2, dtype=torch.float64), local_features={},
            global_features=torch.ones(8, 1, dtype=torch.float64),
            owners=torch.arange(8), mutation_positions=torch.arange(4).repeat_interleave(2) + 1,
        )
        self.assignments = {
            "singles_random": [0, 1, 2, 3, 0, 1, 2, 3],
            "singles_modulo": [0, 0, 1, 1, 2, 2, 3, 3],
            "singles_contiguous": [0, 0, 2, 2, 4, 4, 4, 4],
        }
        self.folds = self.root / "published.csv"
        self.write_folds()

    def write_folds(self):
        with self.folds.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["variant_id", "fold_random_5", "fold_modulo_5", "fold_contiguous_5"])
            for index, variant in enumerate(self.batch.variant_ids):
                writer.writerow([variant, *(self.assignments[s][index] for s in gpu.SETTINGS)])

    def invoke_gpu(self):
        stream = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(stream))
            stack.enter_context(redirect_stderr(stream))
            stack.enter_context(patch.object(gpu.torch.cuda, "is_available", return_value=True))
            for name in ("synchronize", "reset_peak_memory_stats"):
                stack.enter_context(patch.object(gpu.torch.cuda, name))
            for name in ("max_memory_allocated", "max_memory_reserved"):
                stack.enter_context(patch.object(gpu.torch.cuda, name, return_value=1024))
            stack.enter_context(patch.object(gpu.torch.cuda, "get_device_name", return_value="mock"))
            stack.enter_context(patch.object(gpu.torch.cuda, "get_device_properties",
                                            return_value=SimpleNamespace(total_memory=1024)))
            stack.enter_context(patch.object(gpu, "load_feature_bundle", return_value=self.batch))
            stack.enter_context(patch.object(gpu, "validate_assay_coverage"))
            fit = stack.enter_context(patch.object(gpu, "run_cv"))
            status = gpu.main([
                "--assay", self.assay, "--features", str(self.features), "--folds", str(self.folds),
                "--output", str(self.output), "--max-steps", "2", "--min-steps", "0",
            ])
        return status, stream.getvalue(), fit

    def test_gpu_preserves_ids_and_records_actual_fit_counts(self):
        status, log, fit = self.invoke_gpu()
        self.assertEqual(status, 0, log)
        self.assertEqual(fit.call_count, 3)
        plan = json.loads((self.output / "execution_plan.json").read_text())
        manifest = json.loads((self.output / "fold_manifest.json").read_text())
        self.assertEqual((self.output / "folds/singles.csv").read_bytes(), self.folds.read_bytes())
        for call, setting in zip(fit.call_args_list, gpu.SETTINGS):
            expected = self.assignments[setting]
            ids = sorted(set(expected))
            self.assertEqual(call.args[1].tolist(), expected)
            self.assertEqual(plan["settings"][setting]["fits"], len(ids))
            self.assertEqual(plan["settings"][setting]["fold_ids"], ids)
            self.assertEqual(manifest["fold_ids"][setting], ids)
            self.assertIn(f"Running {len(ids)}-fold", log)

    def test_gpu_rejects_invalid_later_scheme_before_any_fit(self):
        self.assignments["singles_contiguous"] = [0] * 8
        self.write_folds()
        status, log, fit = self.invoke_gpu()
        self.assertEqual(status, 2, log)
        self.assertIn("at least two populated folds", log)
        fit.assert_not_called()

    def test_cpu_preflight_records_populated_folds_and_caps_workers(self):
        script = ROOT / "sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch"
        preflight = re.findall(r"<<'PY'\n(.*?)\nPY", script.read_text(), re.DOTALL)[-1]
        self.output.mkdir()
        config_path = self.output / "config.json"
        config_path.write_text(json.dumps(ExperimentConfig().to_dict()))
        plan_path = self.output / "execution_plan.json"
        plan_path.write_text(json.dumps({"settings": {
            s: {"fits": None, "fold_workers": 5} for s in gpu.SETTINGS}}))
        argv = ["-", str(config_path), str(self.features), str(self.folds), "unused.csv",
                "2024", "0.8", *gpu.SETTINGS]
        with patch.object(sys, "argv", argv), \
                patch("data_loading.load_feature_bundle", return_value=self.batch), \
                redirect_stdout(io.StringIO()):
            exec(compile(preflight, str(script), "exec"), {"__name__": "__main__"})
        plan = json.loads(plan_path.read_text())
        manifest = json.loads((self.output / "fold_manifest.json").read_text())
        for setting in gpu.SETTINGS:
            ids = sorted(set(self.assignments[setting]))
            self.assertEqual(plan["settings"][setting],
                             {"fits": len(ids), "fold_workers": len(ids), "fold_ids": ids})
            saved = manifest["sources"]["singles"]["settings"][setting]
            self.assertEqual(saved["fold_ids"], ids)
            self.assertEqual(saved["n_folds"], len(ids))
        self.assertEqual((self.output / "folds/singles.csv").read_bytes(), self.folds.read_bytes())


if __name__ == "__main__":
    unittest.main()
