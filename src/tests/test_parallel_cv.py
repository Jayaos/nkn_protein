"""Spawned CPU folds preserve serial results and report worker failures."""

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import threading
from types import FunctionType
import unittest
from unittest.mock import patch

import torch

from experiments import run_cv
from gp.training import FittedGP, fit_gp
from test_pipeline import read_predictions, small_config, synthetic_batch


def importable_fit(training, model_config, training_config, *, progress=None):
    """Expose whether the selected callback executes inside each child."""
    if progress is not None:
        progress({"event": "custom_fit", "pid": os.getpid()})
    return fit_gp(training, model_config, training_config, progress=progress)


class ParallelCVTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.batch = synthetic_batch()
        self.config = small_config()
        self.folds = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

    def test_spawned_folds_match_serial_and_keep_local_artifacts(self):
        serial = self.root / "serial"
        parallel = self.root / "parallel"
        expected = run_cv(self.batch, self.folds, self.config, serial)
        events = []
        callback_threads = []

        def progress(event):
            # A local closure must work without being pickled into a child.
            callback_threads.append(threading.get_ident())
            events.append(event)

        old_environment = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
        actual = run_cv(self.batch, self.folds, self.config, parallel, fold_workers=2, progress=progress)
        self.assertEqual(expected, actual)
        self.assertEqual(read_predictions(serial / "predictions.csv"), read_predictions(parallel / "predictions.csv"))
        self.assertEqual([fold["fold"] for fold in actual["fold_metrics"]], [0, 1, 2])
        self.assertEqual({event["fold"] for event in events if event.get("event") == "fold_start"}, {0, 1, 2})
        self.assertEqual({event["fold"] for event in events if event.get("event") == "fold_complete"}, {0, 1, 2})
        self.assertEqual({event["fold"] for event in events if "step" in event}, {0, 1, 2})
        self.assertEqual(set(callback_threads), {threading.get_ident()})
        self.assertEqual(old_environment, {name: os.environ.get(name) for name in old_environment})
        pids = set()
        for fold in range(3):
            folder = parallel / f"fold_{fold}"
            execution = json.loads((folder / "execution.json").read_text())
            self.assertNotEqual(execution["pid"], os.getpid())
            pids.add(execution["pid"])
            self.assertEqual(execution["num_threads"], 1)
            self.assertEqual(execution["num_interop_threads"], 1)
            self.assertEqual(set(execution["thread_environment"].values()), {"1"})
            for filename in ("history.json", "restart_summaries.json", "fit_summary.json", "metrics.json"):
                self.assertEqual(json.loads((folder / filename).read_text()), json.loads((serial / f"fold_{fold}" / filename).read_text()))
            training = FittedGP.load(folder / "checkpoint.pt")
            self.assertEqual(set(training.train.variant_ids), {variant for index, variant in enumerate(self.batch.variant_ids) if self.folds[index] != fold})
            torch.testing.assert_close(training.transform.target_mean, self.batch.targets[self.folds != fold].mean())
            self.assertTrue((folder / "progress.jsonl").is_file())
        # Spawn startup varies; tiny folds may finish before every worker is
        # ready. Every fit must use a child and the pool must obey its limit.
        self.assertGreaterEqual(len(pids), 1)
        self.assertLessEqual(len(pids), 2)
        self.assertEqual(json.loads((parallel / "execution.json").read_text())["start_method"], "spawn")

    def test_importable_fit_runs_in_workers_and_matches_serial_local_callback(self):
        calls = []

        def local_fit(*args, **kwargs):
            calls.append(os.getpid())
            return importable_fit(*args, **kwargs)

        serial = self.root / "custom_serial"
        expected = run_cv(self.batch, self.folds, self.config, serial, fit_function=local_fit)
        self.assertEqual(calls, [os.getpid()] * self.config.evaluation.n_folds)
        parallel = self.root / "custom_parallel"
        events = []
        actual = run_cv(
            self.batch, self.folds, self.config, parallel,
            fit_function=importable_fit, fold_workers=2, progress=events.append,
        )
        self.assertEqual(expected, actual)
        self.assertEqual(read_predictions(serial / "predictions.csv"), read_predictions(parallel / "predictions.csv"))
        custom_events = [event for event in events if event.get("event") == "custom_fit"]
        self.assertEqual(sorted(event["fold"] for event in custom_events), list(range(self.config.evaluation.n_folds)))
        for event in custom_events:
            self.assertNotEqual(event["pid"], os.getpid())
            execution = json.loads((parallel / f"fold_{event['fold']}" / "execution.json").read_text())
            self.assertEqual(event["pid"], execution["pid"])

    def test_nonconsecutive_observed_folds_match_serial_and_cap_workers(self):
        config = replace(self.config, evaluation=replace(self.config.evaluation, n_folds=5))
        folds = torch.tensor([4, 0, 2, 4, 0, 2, 4, 0])
        serial = self.root / "observed_serial"
        parallel = self.root / "observed_parallel"
        expected = run_cv(self.batch, folds, config, serial)
        events = []
        actual = run_cv(self.batch, folds, config, parallel, fold_workers=5, progress=events.append)
        self.assertEqual(actual, expected)
        self.assertEqual(actual["n_folds"], 3)
        self.assertEqual(actual["fold_ids"], [0, 2, 4])
        self.assertEqual([fold["fold"] for fold in actual["fold_metrics"]], [0, 2, 4])
        self.assertEqual(read_predictions(serial / "predictions.csv"), read_predictions(parallel / "predictions.csv"))
        self.assertEqual([int(row["fold"]) for row in read_predictions(parallel / "predictions.csv")], folds.tolist())
        self.assertEqual({event["fold"] for event in events if event.get("event") == "fold_complete"}, {0, 2, 4})
        self.assertEqual(json.loads((parallel / "execution.json").read_text())["fold_workers"], 3)
        self.assertEqual({path.name for path in parallel.glob("fold_*") if path.is_dir()}, {"fold_0", "fold_2", "fold_4"})
        for fold in (0, 2, 4):
            execution = json.loads((parallel / f"fold_{fold}" / "execution.json").read_text())
            self.assertNotEqual(execution["pid"], os.getpid())
            fitted = FittedGP.load(parallel / f"fold_{fold}" / "checkpoint.pt")
            expected_train = {variant for index, variant in enumerate(self.batch.variant_ids) if folds[index] != fold}
            self.assertEqual(set(fitted.train.variant_ids), expected_train)

    def test_invalid_parallel_options_fail_before_output(self):
        for index, workers in enumerate((0, -1, 4, 1.5, True, "2")):
            output = self.root / f"invalid_{index}"
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "fold_workers"):
                run_cv(self.batch, self.folds, self.config, output, fold_workers=workers)
            self.assertFalse(output.exists())
        output = self.root / "gpu"
        with self.assertRaisesRegex(ValueError, "device='cpu'"):
            run_cv(self.batch, self.folds, replace(self.config, device="cuda"), output, fold_workers=2)
        self.assertFalse(output.exists())
        output = self.root / "custom_fit"
        with self.assertRaisesRegex(ValueError, "custom fit_function"):
            run_cv(self.batch, self.folds, self.config, output, fold_workers=2, fit_function=lambda *args: None)
        self.assertFalse(output.exists())

    def test_nonimportable_custom_fit_fails_before_output(self):
        def local_fit(*args, **kwargs):
            return importable_fit(*args, **kwargs)

        duplicate = FunctionType(importable_fit.__code__, importable_fit.__globals__, importable_fit.__name__)
        missing_module = FunctionType(importable_fit.__code__, importable_fit.__globals__, importable_fit.__name__)
        missing_module.__module__ = "_nonexistent_parallel_fit_module"
        for index, callback in enumerate((local_fit, duplicate, missing_module, object())):
            output = self.root / f"custom_invalid_{index}"
            with self.subTest(callback=callback), self.assertRaisesRegex(ValueError, "custom fit_function"):
                run_cv(self.batch, self.folds, self.config, output, fold_workers=2, fit_function=callback)
            self.assertFalse(output.exists())

    def test_custom_fit_serialization_failure_precedes_output(self):
        output = self.root / "custom_unserializable"
        with patch("experiments.runner.multiprocessing.reduction.ForkingPickler.dumps", side_effect=TypeError("cannot pickle callback")):
            with self.assertRaisesRegex(ValueError, "custom fit_function") as raised:
                run_cv(self.batch, self.folds, self.config, output, fold_workers=2, fit_function=importable_fit)
        self.assertIsInstance(raised.exception.__cause__, TypeError)
        self.assertFalse(output.exists())

    def test_worker_failure_propagates_and_retains_evidence(self):
        output = self.root / "failure"
        config = replace(self.config, training=replace(self.config.training, max_train_variants=2))
        with self.assertRaisesRegex(ValueError, "dense-GP limit"):
            run_cv(self.batch, self.folds, config, output, fold_workers=2)
        failure = json.loads((output / "failure.json").read_text())
        self.assertEqual(failure["phase"], "parallel_cv")
        fold_failure = json.loads((output / f"fold_{failure['fold']}" / "failure.json").read_text())
        self.assertEqual(fold_failure["phase"], "training")
        self.assertIn("dense-GP limit", fold_failure["error"])
        self.assertFalse((output / "metrics.json").exists())

    def test_parent_callback_interrupt_cancels_long_training(self):
        output = self.root / "interrupt"
        config = replace(self.config, training=replace(self.config.training, max_steps=100000, patience=100000))

        def stop(event):
            if "step" in event:
                raise KeyboardInterrupt("stop parallel test")

        with self.assertRaisesRegex(KeyboardInterrupt, "stop parallel test"):
            run_cv(self.batch, self.folds, config, output, fold_workers=2, progress=stop)
        self.assertTrue((output / "failure.json").is_file())
        self.assertFalse((output / "metrics.json").exists())
        self.assertTrue(list(output.glob("fold_*/cancelled.json")))


if __name__ == "__main__":
    unittest.main()
