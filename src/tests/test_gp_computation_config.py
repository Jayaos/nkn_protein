"""Backend naming and saved-config compatibility without GPU dependencies."""

import argparse
from contextlib import redirect_stderr
from dataclasses import asdict
import io
from pathlib import Path
import runpy
import unittest

from config import ExperimentConfig, TrainingConfig


class GPComputationConfigTests(unittest.TestCase):
    def test_saved_backend_names_load_and_serialize_with_new_name(self):
        for backend in ("cholesky", "bbmm"):
            for key in ("inference_backend", "gp_computation_backend"):
                with self.subTest(backend=backend, key=key):
                    values = {key: backend}
                    training = TrainingConfig.from_dict(values)
                    config = ExperimentConfig.from_dict({"training": values})
                    self.assertEqual(training.gp_computation_backend, backend)
                    self.assertEqual(config.training, training)
                    self.assertEqual(config.to_dict()["training"]["gp_computation_backend"], backend)
                    self.assertNotIn("inference_backend", asdict(training))
                    self.assertEqual(values, {key: backend})

    def test_conflicting_saved_backend_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            TrainingConfig.from_dict({"inference_backend": "bbmm",
                                      "gp_computation_backend": "cholesky"})

    def test_backend_cli_override_and_old_option_rejection(self):
        # Load the shared argument helpers without gp.__init__ importing PyTorch.
        helpers = runpy.run_path(str(Path(__file__).resolve().parents[1] / "gp/inference.py"))
        parser = argparse.ArgumentParser()
        helpers["add_inference_arguments"](parser)
        self.assertIn("training and prediction", parser.format_help())
        for backend in ("cholesky", "bbmm"):
            args = parser.parse_args(["--gp-computation-backend", backend])
            self.assertEqual(args.gp_computation_backend, backend)
        args = parser.parse_args(["--gp-computation-backend", "cholesky"])
        resolved = helpers["resolve_training_config"](
            TrainingConfig(gp_computation_backend="bbmm"), args)
        self.assertEqual(resolved.gp_computation_backend, "cholesky")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--inference-backend", "bbmm"])


if __name__ == "__main__":
    unittest.main()
