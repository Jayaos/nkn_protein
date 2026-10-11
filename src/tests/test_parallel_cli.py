"""CLI resource validation and forwarding for CPU fold workers."""

import contextlib
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.cli import main
from test_pipeline import small_config, synthetic_batch


class ParallelCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = small_config()
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config.to_dict()), encoding="utf-8")
        self.features = self.root / "features"
        self.features.mkdir()
        (self.features / "assay.json").write_text('{"synthetic": true}', encoding="utf-8")
        self.batch = synthetic_batch()
        self.batch = self.batch.select(torch.bincount(self.batch.owners) == 1)
        self.output = self.root / "output"
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())

    def invoke(self, *extra, cv_runner=None):
        errors = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
            try:
                status = main([
                    "cv", "--features", str(self.features), "--config", str(self.config_path),
                    "--output", str(self.output), "--setting", "singles_random",
                    "--folds", str(self.root / "folds.csv"), *extra,
                ], cv_runner=cv_runner)
            except SystemExit as error:
                status = error.code
        return status, errors.getvalue()

    def test_forwards_workers_and_records_manifest(self):
        def complete(*args, **kwargs):
            self.output.mkdir()
            return {"n_folds": 3}

        with patch("experiments.cli.load_feature_bundle", return_value=self.batch), \
                patch("experiments.cli.load_folds", return_value=torch.tensor([0, 1, 2, 0, 1, 2])), \
                patch("experiments.cli.run_cv", side_effect=complete) as runner, \
                patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "3"}):
            status, errors = self.invoke("--fold-workers", "3")
        self.assertEqual(status, 0, errors)
        self.assertEqual(runner.call_args.kwargs["fold_workers"], 3)
        manifest = json.loads((self.output / "input_manifest.json").read_text())
        self.assertEqual(manifest["fold_workers"], 3)

    def test_invalid_worker_counts_fail_before_loading(self):
        for value in ("0", "-1", "4", "1.5"):
            with self.subTest(value=value), patch("experiments.cli.load_feature_bundle") as loader:
                status, errors = self.invoke("--fold-workers", value)
                self.assertEqual(status, 2, errors)
                loader.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_parallel_cuda_rejected_before_loading(self):
        with patch("experiments.cli.load_feature_bundle") as loader:
            status, errors = self.invoke("--fold-workers", "2", "--device", "cuda")
        self.assertEqual(status, 2)
        self.assertIn("require device cpu", errors)
        loader.assert_not_called()

    def test_slurm_guard_counts_threads_per_worker(self):
        config = replace(self.config, num_threads=2)
        self.config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
        with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "5"}), \
                patch("experiments.cli.load_feature_bundle") as loader:
            status, errors = self.invoke("--fold-workers", "3")
        self.assertEqual(status, 2)
        self.assertIn("exceeds SLURM_CPUS_PER_TASK", errors)
        loader.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_serial_custom_runner_does_not_receive_new_keyword(self):
        def serial_runner(batch, folds, config, output, *, scheme, setting, split_source, progress):
            Path(output).mkdir()
            return {"n_folds": 3}

        with patch("experiments.cli.load_feature_bundle", return_value=self.batch), \
                patch("experiments.cli.load_folds", return_value=torch.tensor([0, 1, 2, 0, 1, 2])):
            status, errors = self.invoke(cv_runner=serial_runner)
        self.assertEqual(status, 0, errors)


if __name__ == "__main__":
    unittest.main()
