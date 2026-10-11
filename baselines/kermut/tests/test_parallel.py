"""Kermut must retain its model and fitting protocol in spawned CV workers."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from baselines.kermut.runner import run_cv
from baselines.kermut.training import FittedKermut
from baselines.kermut.tests.fixtures import make_cache, read_csv, small_config, synthetic_batch, write_csv


ROOT = Path(__file__).resolve().parents[3]


class ParallelKermutTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.batch = synthetic_batch()
        self.config = small_config()
        self.folds = torch.arange(20) % 10 // 2
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())

    def test_spawned_singles_preserve_both_precisions_and_final_checkpoints(self):
        batch = self.batch.select(range(10))
        for dtype in ("float32", "float64"):
            with self.subTest(dtype=dtype):
                config = replace(self.config, training=replace(self.config.training, dtype=dtype))
                serial, parallel = self.root / f"serial_{dtype}", self.root / f"parallel_{dtype}"
                expected = run_cv(batch, self.folds[:10], config, serial, setting="singles_random")
                actual = run_cv(batch, self.folds[:10], config, parallel,
                                setting="singles_random", fold_workers=2)
                self.assertEqual(actual, expected)
                self.assertEqual(read_csv(parallel / "predictions.csv"), read_csv(serial / "predictions.csv"))
                for fold in range(5):
                    directory = parallel / f"fold_{fold}"
                    reference = serial / f"fold_{fold}"
                    for filename in ("history.json", "restart_summaries.json", "fit_summary.json", "metrics.json"):
                        self.assertEqual(json.loads((directory / filename).read_text()),
                                         json.loads((reference / filename).read_text()))
                    state = torch.load(directory / "checkpoint.pt", weights_only=True)
                    self.assertEqual(state["model_type"], "kermut")
                    fitted = FittedKermut.load(directory / "checkpoint.pt")
                    expected_fit = FittedKermut.load(reference / "checkpoint.pt")
                    self.assertEqual(fitted.training_config.dtype, dtype)
                    self.assertEqual(fitted.train.scores.dtype, getattr(torch, dtype))
                    self.assertEqual(fitted.history[-1]["step"], config.training.n_steps)
                    self.assertEqual(fitted.restart_summaries[0]["checkpoint_selection"], "final_iterate")
                    for name, value in fitted.model.state_dict().items():
                        torch.testing.assert_close(value, expected_fit.model.state_dict()[name], rtol=0, atol=0)
                    training = batch.select(self.folds[:10] != fold)
                    torch.testing.assert_close(fitted.transform.target_scale,
                                               training.targets.to(getattr(torch, dtype)).std(unbiased=True))
                    torch.testing.assert_close(fitted.train.scores,
                                               training.scores.to(getattr(torch, dtype)), rtol=0, atol=0)
                    execution = json.loads((directory / "execution.json").read_text())
                    self.assertNotEqual(execution["pid"], os.getpid())
                    self.assertEqual(execution["num_threads"], 1)
                    self.assertEqual(execution["num_interop_threads"], 1)

    def test_mixed_parallel_reuses_fits_for_both_evaluation_domains(self):
        expected = run_cv(self.batch, self.folds, self.config, self.root / "serial_mixed",
                          scheme="random_multiples", setting="mixed_random_singles")
        output = self.root / "parallel_mixed"
        actual = run_cv(self.batch, self.folds, self.config, output,
                        scheme="random_multiples", setting="mixed_random_singles", fold_workers=2)
        self.assertEqual(actual, expected)
        self.assertEqual(set(actual["setting_metrics"]), {"mixed_random_singles", "mixed_random_doubles"})
        self.assertEqual(len(list(output.glob("fold_*/checkpoint.pt"))), 5)
        self.assertEqual(len(read_csv(output / "predictions.csv")), 20)

    def test_worker_failure_retains_kermut_error_and_no_aggregate(self):
        config = replace(self.config, training=replace(self.config.training, max_train_variants=2))
        output = self.root / "failed"
        with self.assertRaisesRegex(ValueError, "dense-GP limit"):
            run_cv(self.batch, self.folds, config, output,
                   scheme="random_multiples", setting="mixed_random_singles", fold_workers=2)
        failure = json.loads((output / "failure.json").read_text())
        self.assertEqual(failure["phase"], "parallel_cv")
        self.assertIn("dense-GP limit", failure["error"])
        self.assertTrue((output / f"fold_{failure['fold']}" / "failure.json").is_file())
        self.assertFalse((output / "metrics.json").exists())

    def test_cli_loads_real_cache_format_and_spawns_kermut_workers(self):
        features, config, batch, index = make_cache(self.root / "cache")
        config = replace(config, training=replace(config.training, dtype="float32"))
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
        folds = self.root / "folds.csv"
        assignments = torch.arange(20) % 10 // 2
        write_csv(folds, ["mutant", "fold_random_5"],
                  [[variant, int(fold)] for variant, fold in zip(batch.variant_ids, assignments)])
        output = self.root / "cli"
        environment = os.environ.copy()
        environment.update(PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT))),
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                           SLURM_CPUS_PER_TASK="2")
        result = subprocess.run([
            sys.executable, "-m", "baselines.kermut.cli", "cv", "--features", str(features),
            "--config", str(config_path), "--output", str(output), "--setting", "singles_random",
            "--folds", str(folds), "--fold-workers", "2", "--assay-index", str(index),
            "--full-assay", "--quiet",
        ], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((output / "execution.json").read_text())["fold_workers"], 2)
        self.assertEqual(json.loads((output / "input_manifest.json").read_text())["fold_workers"], 2)
        self.assertEqual(json.loads((output / "metrics.json").read_text())["n_variants"], 10)
        for fold in range(5):
            summary = json.loads((output / f"fold_{fold}/fit_summary.json").read_text())
            self.assertEqual(summary["model"], "kermut")
            self.assertEqual(summary["dtype"], "float32")
            self.assertEqual(summary["optimizer_updates"], 2)


if __name__ == "__main__":
    unittest.main()
