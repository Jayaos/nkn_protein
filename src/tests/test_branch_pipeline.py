"""Branch-subset training and CLI checks using only temporary synthetic caches."""

import contextlib
from dataclasses import replace
import io
from itertools import combinations
import json
from pathlib import Path
import tempfile
import unittest

import torch

from config import DEFAULT_BRANCHES
from data_loading import load_feature_bundle
from experiments import run_cv, run_extrapolation, run_predict
from experiments.cli import main
from gp.training import FittedGP, fit_gp
from test_data import make_bundle, write_csv
from test_pipeline import read_predictions, small_config, synthetic_batch


def select_branches(batch, names):
    return replace(
        batch,
        global_features=batch.global_features if "sequence_global" in names else None,
        local_features={name: values for name, values in batch.local_features.items() if name in names},
    )


def branch_config(names):
    config = small_config()
    return replace(config, model=replace(config.model, enabled_branches=tuple(names)))


def partial_bundle(root, names):
    """Leave only required representation tensors; score CSVs are independent."""
    ids, sequences, _ = make_bundle(root)
    required = {"sequence" if name.startswith("sequence_") else name for name in names}
    path = root / "assay.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["features"] = {name: value for name, value in metadata["features"].items() if name in required}
    path.write_text(json.dumps(metadata), encoding="utf-8")
    for name in {"sequence", "structure", "inverse_folding"} - required:
        (root / name / "synthetic/v1/features.pt").unlink()
    return ids, sequences


class BranchPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.batch = synthetic_batch()
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)

    def test_every_nonempty_subset_fits_predicts_and_round_trips(self):
        for count in range(1, len(DEFAULT_BRANCHES) + 1):
            for names in combinations(DEFAULT_BRANCHES, count):
                with self.subTest(branches=names):
                    config = branch_config(names)
                    batch = select_branches(self.batch, names)
                    fitted = fit_gp(batch.select(range(6)), config.model, config.training)
                    test = replace(batch.select([6, 7]), targets=None)
                    before = fitted.predict(test)
                    self.assertEqual(tuple(fitted.model.kernel.branches), names)
                    self.assertEqual(set(fitted.train.local_features), set(names) - {"sequence_global"})
                    self.assertEqual(fitted.train.global_features is None, "sequence_global" not in names)
                    self.assertTrue(torch.isfinite(before.mean).all())
                    self.assertTrue(torch.isfinite(before.observation_covariance).all())
                    self.assertTrue((before.observation_covariance.diagonal() > 0).all())
                    self.assertTrue(any(row["step"] == 2 for row in fitted.history))
                    path = self.root / ("-".join(names) + ".pt")
                    fitted.save(path)
                    restored = FittedGP.load(path)
                    self.assertEqual(restored.model_config.enabled_branches, names)
                    self.assertEqual(tuple(restored.model.kernel.branches), names)
                    self.assertEqual(restored.train.global_features is None, "sequence_global" not in names)
                    after = restored.predict(test)
                    torch.testing.assert_close(after.mean, before.mean, rtol=1e-12, atol=1e-12)
                    torch.testing.assert_close(
                        after.observation_covariance, before.observation_covariance, rtol=1e-12, atol=1e-12
                    )

    def test_legacy_full_branch_checkpoint_keeps_predictions(self):
        config = small_config()
        fitted = fit_gp(self.batch.select(range(6)), config.model, config.training)
        test = replace(self.batch.select([6, 7]), targets=None)
        before = fitted.predict(test)
        path = self.root / "legacy.pt"
        fitted.save(path)
        state = torch.load(path, weights_only=True)
        del state["model_config"]["enabled_branches"]
        torch.save(state, path)
        restored = FittedGP.load(path)
        self.assertEqual(restored.model_config.enabled_branches, DEFAULT_BRANCHES)
        after = restored.predict(test)
        torch.testing.assert_close(after.mean, before.mean, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(after.observation_covariance, before.observation_covariance, rtol=1e-12, atol=1e-12)

    def test_prediction_config_cannot_change_fitted_branches(self):
        config = branch_config(("sequence_global",))
        batch = select_branches(self.batch, config.model.enabled_branches)
        output = self.root / "fit"
        run_extrapolation(batch, config, output)
        mismatched = branch_config(("inverse_folding",))
        with self.assertRaisesRegex(ValueError, "branches.*checkpoint"):
            run_predict(replace(batch, targets=None), output / "checkpoint.pt", mismatched, self.root / "rejected")
        self.assertFalse((self.root / "rejected").exists())
        fitted = FittedGP.load(output / "checkpoint.pt")
        wrong_features = select_branches(self.batch, mismatched.model.enabled_branches)
        with self.assertRaises(ValueError):
            fitted.predict(replace(wrong_features, targets=None))

    def test_inverse_only_cross_validation_keeps_inputs_and_reload_predictions(self):
        config = branch_config(("inverse_folding",))
        batch = select_branches(self.batch, config.model.enabled_branches)
        folds = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        output = self.root / "inverse_cv"
        report = run_cv(batch, folds, config, output, split_source="provided")
        self.assertEqual(report["n_folds"], 3)
        self.assertEqual(report["n_variants"], batch.num_variants)
        for fold in range(3):
            directory = output / f"fold_{fold}"
            fitted = FittedGP.load(directory / "checkpoint.pt")
            self.assertIsNone(fitted.train.global_features)
            self.assertEqual(set(fitted.train.local_features), {"inverse_folding"})
            test = batch.select(folds == fold)
            self.assertTrue(set(test.variant_ids).isdisjoint(fitted.train.variant_ids))
            prediction = fitted.predict(replace(test, targets=None))
            rows = read_predictions(directory / "predictions.csv")
            expected = torch.tensor([float(row["mean"]) for row in rows], dtype=torch.float64)
            torch.testing.assert_close(prediction.mean, expected, rtol=1e-12, atol=1e-12)


class BranchExperimentCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)

    def invoke(self, arguments):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            status = main([str(value) for value in arguments])
        return status, output.getvalue(), errors.getvalue()

    def test_partial_caches_validate_fit_and_predict_with_independent_scores(self):
        cases = (
            (("sequence_global",), "inverse_folding/synthetic/v1"),
            (("inverse_folding",), "sequence/synthetic/v1"),
            (("inverse_folding", "sequence_global"), "inverse_folding/synthetic/v1"),
        )
        for names, score_source in cases:
            with self.subTest(branches=names):
                directory = self.root / "-".join(names)
                feature_root = directory / "features"
                feature_root.mkdir(parents=True)
                ids, sequences = partial_bundle(feature_root, names)
                config = replace(branch_config(names), score_sources=[score_source])
                config_path = directory / "config.json"
                config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
                for command in ("validate", "fit"):
                    split_options = ["--setting", "single_to_double"] if command == "fit" else []
                    status, output, errors = self.invoke([
                        command, "--features", feature_root, "--config", config_path,
                        "--output", directory / command, "--quiet", *split_options,
                    ])
                    self.assertEqual(status, 0, errors)
                    summary = json.loads(output)
                    if command == "validate":
                        self.assertTrue(summary["valid"])
                        self.assertEqual(summary["n_score_sources"], 1)
                        self.assertEqual(summary["enabled_branches"], list(names))
                        self.assertEqual(summary["global_feature_dimension"], 2 if "sequence_global" in names else None)
                        self.assertEqual(set(summary["local_feature_dimensions"]), set(names) - {"sequence_global"})
                fitted = FittedGP.load(directory / "fit/checkpoint.pt")
                self.assertEqual(fitted.model_config.enabled_branches, names)
                self.assertEqual(fitted.train.score_names, (score_source,))
                self.assertEqual(set(fitted.train.local_features), set(names) - {"sequence_global"})
                self.assertEqual(fitted.train.global_features is None, "sequence_global" not in names)
                # Predict without labels or an explicit config: recover the saved branch selection.
                write_csv(feature_root / "variants.csv", ["variant_id", "mutant", "mutated_sequence"],
                          [[value, value, sequence] for value, sequence in zip(ids, sequences)])
                status, output, errors = self.invoke([
                    "predict", "--features", feature_root, "--checkpoint", directory / "fit/checkpoint.pt",
                    "--output", directory / "predict", "--quiet",
                ])
                self.assertEqual(status, 0, errors)
                self.assertEqual(json.loads(output)["n_variants"], len(ids))
                rows = read_predictions(directory / "predict/predictions.csv")
                self.assertEqual([row["variant_id"] for row in rows], ids)
                self.assertTrue(all(row["target"] == "" for row in rows))
                expected = fitted.predict(load_feature_bundle(
                    feature_root, score_sources=[score_source], enabled_branches=names
                ))
                actual = torch.tensor([float(row["mean"]) for row in rows], dtype=torch.float64)
                torch.testing.assert_close(actual, expected.mean, rtol=1e-12, atol=1e-12)

    def test_branch_selection_still_requires_a_mean_score(self):
        features = self.root / "features"
        features.mkdir()
        partial_bundle(features, ("sequence_global",))
        config = replace(branch_config(("sequence_global",)), score_sources=[])
        path = self.root / "config.json"
        path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
        status, _, errors = self.invoke([
            "validate", "--features", features, "--config", path, "--output", self.root / "rejected",
        ])
        self.assertEqual(status, 2)
        self.assertIn("mean score", errors)
        self.assertFalse((self.root / "rejected").exists())


if __name__ == "__main__":
    unittest.main()
