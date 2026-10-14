"""Validation selects checkpoints without using outer-test labels or groups."""

from dataclasses import replace
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from config import TrainingConfig
from data_loading import FeatureBatch
from experiments import run_cv, run_extrapolation
from experiments.cli import _load_config, _parser, main
from experiments.runner import _validate_fold_assignments
from gp.linalg import NumericalError
from gp.model import ExactProteinGP
from gp.training import FittedGP, fit_gp
from test_pipeline import read_predictions, small_config, synthetic_batch


def five_fold_batch():
    generator = torch.Generator().manual_seed(43)
    return FeatureBatch(
        assay_id="validation_test", variant_ids=tuple(f"v{i}" for i in range(20)),
        targets=torch.randn(20, dtype=torch.float64, generator=generator),
        scores=torch.randn(20, 2, dtype=torch.float64, generator=generator),
        global_features=torch.randn(20, 4, dtype=torch.float64, generator=generator),
        local_features={}, owners=torch.arange(20),
        mutation_positions=torch.arange(1, 6).repeat_interleave(4),
        reference_sequence="ACDEF", score_names=("first", "second"),
    )


class ValidationStoppingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = small_config()
        batch = synthetic_batch()
        self.train = batch.select(range(4))
        self.validation = batch.select([4, 5])
        self.test = batch.select([6, 7])
        self.training = replace(self.config.training, stopping_criterion="validation_spearman",
                                max_steps=10, min_steps=0, patience=2, validation_interval=2)

    def fit(self, training=None, **kwargs):
        return fit_gp(self.train, self.config.model, training or self.training,
                      validation=kwargs.pop("validation", self.validation), **kwargs)

    def test_restores_highest_validation_checkpoint_even_when_nll_keeps_improving(self):
        snapshots = []
        original = ExactProteinGP.posterior_mean

        def record(model, train, y, validation):
            self.assertFalse(torch.is_grad_enabled())
            self.assertIsNone(validation.targets)
            snapshots.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            return original(model, train, y, validation)

        with patch.object(ExactProteinGP, "posterior_mean", record), \
                patch("gp.training.spearman", side_effect=[.2, .8, .7]):
            fitted = self.fit()
        self.assertEqual(fitted.history[-1]["step"], 4)
        self.assertEqual(fitted.selected_step, 2)
        self.assertEqual(fitted.best_validation_spearman, .8)
        self.assertLess(fitted.history[4]["nll"], fitted.history[2]["nll"])
        self.assertEqual(fitted.selected_nll_per_variant, fitted.history[2]["nll_per_variant"])
        self.assertEqual(fitted.best_nll_per_variant, fitted.history[4]["nll_per_variant"])
        for key, value in snapshots[1].items():
            torch.testing.assert_close(fitted.model.state_dict()[key], value, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            fitted.save(path)
            restored = FittedGP.load(path)
            self.assertEqual(restored.fit_summary(4), fitted.fit_summary(4))
            torch.testing.assert_close(restored.predict(self.test).mean, fitted.predict(self.test).mean)

    def test_patience_counts_steps_ties_do_not_reset_and_step_zero_can_win(self):
        with patch("gp.training.spearman", return_value=.5):
            fitted = self.fit(replace(self.training, patience=3))
        self.assertEqual(fitted.selected_step, 0)
        self.assertEqual([r["step"] for r in fitted.history if "validation_spearman" in r], [0, 2, 4])
        self.assertEqual(fitted.history[-1]["step"], 4)  # next check after 3 elapsed steps
        with patch("gp.training.spearman", return_value=.5):
            delayed = self.fit(replace(self.training, min_steps=5))
        self.assertEqual(delayed.selected_step, 0)
        self.assertEqual(delayed.history[-1]["step"], 6)

    def test_default_patience_stops_at_first_nonimproving_100_step_check(self):
        cfg = replace(self.training, max_steps=200, validation_interval=100, patience=100)
        with patch("gp.training.spearman", return_value=.5):
            fitted = self.fit(cfg)
        self.assertEqual(fitted.history[-1]["step"], 100)
        self.assertEqual(fitted.selected_step, 0)

    def test_restart_selection_uses_validation_and_final_off_grid_step_is_evaluated(self):
        cfg = replace(self.training, max_steps=3, seeds=(0, 1), lengthscale_multipliers=(1., 2.))
        with patch("gp.training.spearman", side_effect=[.1, .3, .4, .2, .7, .6]):
            fitted = self.fit(cfg)
        self.assertEqual(fitted.selected_seed, 1)
        self.assertEqual(fitted.selected_step, 2)
        self.assertEqual(fitted.best_validation_spearman, .7)
        for seed in cfg.seeds:
            self.assertEqual([r["step"] for r in fitted.history
                              if r["seed"] == seed and "validation_spearman" in r], [0, 2, 3])

    def test_outer_test_labels_do_not_change_fit_and_validation_does_not_change_gradients(self):
        cfg = replace(self.training, max_steps=4, min_steps=10, held_out_interval=2)
        fitted = self.fit(cfg, held_out=self.test)
        changed = self.fit(cfg, held_out=replace(self.test, targets=-self.test.targets))
        self.assertEqual(fitted.restart_summaries, changed.restart_summaries)
        for key, value in fitted.model.state_dict().items():
            torch.testing.assert_close(value, changed.model.state_dict()[key], rtol=0, atol=0)
        inverted_validation = self.fit(cfg, held_out=self.test,
                                       validation=replace(self.validation, targets=-self.validation.targets))
        self.assertEqual([r["nll"] for r in fitted.history], [r["nll"] for r in inverted_validation.history])
        torch.testing.assert_close(fitted.transform.target_mean, self.train.targets.mean())

    def test_invalid_validation_and_undefined_scores_fail_explicitly(self):
        for validation, message in [(None, "separate validation"), (self.train, "overlap"),
                                    (replace(self.validation, targets=torch.ones(2)), "nonconstant")]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.fit(validation=validation)
        with self.assertRaisesRegex(ValueError, "overlap validation"):
            self.fit(replace(self.training, held_out_interval=2), held_out=self.validation)
        with patch("gp.training.spearman", return_value=None), \
                self.assertRaisesRegex(NumericalError, "No defined validation Spearman"):
            self.fit()
        # The NLL mode never inspects an unused validation batch.
        self.fit(replace(self.training, stopping_criterion="training_nll"),
                 validation=replace(self.validation, targets=None))

    def test_defaults_validation_and_cli_overrides(self):
        defaults = TrainingConfig()
        self.assertEqual((defaults.stopping_criterion, defaults.min_steps, defaults.patience,
                          defaults.validation_interval), ("training_nll", 0, 100, 100))
        for field, value in [("min_steps", -1), ("min_steps", True), ("patience", 0),
                             ("validation_interval", 0), ("validation_interval", 1.5),
                             ("stopping_criterion", "test_spearman")]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                replace(defaults, **{field: value})
        args = _parser().parse_args([
            "cv", "--features", "unused", "--output", "unused", "--folds", "unused",
            "--setting", "singles_random", "--stopping-criterion", "validation_spearman",
            "--validation-interval", "50", "--patience", "200", "--min-steps", "0",
        ])
        cfg = _load_config(args)
        self.assertEqual((cfg.training.stopping_criterion, cfg.training.validation_interval,
                          cfg.training.patience, cfg.training.min_steps), ("validation_spearman", 50, 200, 0))

    def test_five_fold_partitions_preserve_positions_and_parallel_matches_serial(self):
        batch = five_fold_batch()
        folds = torch.arange(5).repeat_interleave(4)
        config = replace(self.config,
                         model=replace(self.config.model, enabled_branches=("sequence_global",)),
                         training=replace(self.training, max_steps=3, held_out_interval=1),
                         evaluation=replace(self.config.evaluation, n_folds=5))
        # Validation must accept original positional fold assignments, not resplit variants.
        _validate_fold_assignments(batch, folds, config, "modulo")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            serial = run_cv(batch, folds, config, root / "serial", scheme="contiguous", setting="singles_contiguous")
            parallel = run_cv(batch, folds, config, root / "parallel", scheme="contiguous",
                              setting="singles_contiguous", fold_workers=2)
            self.assertEqual(serial, parallel)
            for fold in range(5):
                one, two = (root / mode / f"fold_{fold}" for mode in ("serial", "parallel"))
                for filename in ("validation_spearman.csv", "held_out_spearman.csv", "partition_assignments.csv",
                                 "partition_summary.json", "fit_summary.json", "history.json", "metrics.json"):
                    self.assertEqual((one / filename).read_text(), (two / filename).read_text(), filename)
                partition = json.loads((one / "partition_summary.json").read_text())
                self.assertEqual(partition["validation_fold"], (fold + 1) % 5)
                self.assertEqual((partition["n_train"], partition["n_validation"], partition["n_test"]), (12, 4, 4))
                assignment = read_predictions(one / "partition_assignments.csv")
                for group in range(5):
                    self.assertEqual(len({r["partition"] for r in assignment if int(r["fold"]) == group}), 1)
                fitted = FittedGP.load(one / "checkpoint.pt")
                expected = batch.select((folds != fold) & (folds != (fold + 1) % 5))
                self.assertEqual(fitted.train.variant_ids, expected.variant_ids)
                torch.testing.assert_close(fitted.transform.target_mean, expected.targets.mean())
                observed = read_predictions(one / "validation_spearman.csv")
                self.assertEqual(fitted.best_validation_spearman, max(float(r["validation_spearman"]) for r in observed))

    def test_unsupported_protocols_and_invalid_grouping_rejected_before_output(self):
        batch = five_fold_batch()
        config = replace(self.config, training=self.training,
                         evaluation=replace(self.config.evaluation, n_folds=5))
        folds = torch.arange(5).repeat_interleave(4)
        bad_folds = folds.clone()
        bad_folds[0], bad_folds[4] = 1, 0
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            with self.assertRaisesRegex(ValueError, "multiple positional folds"):
                run_cv(batch, bad_folds, config, output, scheme="modulo")
            with self.assertRaisesRegex(ValueError, "five-fold"):
                run_cv(synthetic_batch(), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
                       replace(self.config, training=self.training), output)
            with self.assertRaisesRegex(ValueError, "single_to_double"):
                run_extrapolation(synthetic_batch(), config, output)
            self.assertFalse(output.exists())

    def test_cli_runs_validation_mode_and_records_scores_when_quiet(self):
        batch = five_fold_batch()
        cfg = replace(self.config,
                      model=replace(self.config.model, enabled_branches=("sequence_global",)),
                      evaluation=replace(self.config.evaluation, n_folds=5))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "assay.json").write_text('{"synthetic": true}')
            config_path = root / "config.json"
            config_path.write_text(json.dumps(cfg.to_dict()))
            folds = root / "folds.csv"
            folds.write_text("mutant,fold_modulo_5\n" + "".join(f"v{i},{i // 4}\n" for i in range(20)))
            output = root / "output"
            errors = io.StringIO()
            with patch("experiments.cli.load_feature_bundle", return_value=batch), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
                status = main([
                    "cv", "--features", str(root), "--config", str(config_path), "--output", str(output),
                    "--folds", str(folds), "--setting", "singles_modulo", "--quiet",
                    "--stopping-criterion", "validation_spearman", "--validation-interval", "1", "--min-steps", "0",
                ])
            self.assertEqual(status, 0, errors.getvalue())
            resolved = json.loads((output / "resolved_config.json").read_text())
            self.assertEqual(resolved["training"]["stopping_criterion"], "validation_spearman")
            self.assertEqual(resolved["training"]["min_steps"], 0)
            for fold in range(5):
                path = output / f"fold_{fold}"
                self.assertTrue((path / "validation_spearman.csv").is_file())
                self.assertFalse((path / "held_out_spearman.csv").exists())

    def test_legacy_training_nll_checkpoint_still_loads(self):
        fitted = self.fit(replace(self.training, stopping_criterion="training_nll", max_steps=2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            fitted.save(path)
            state = torch.load(path, weights_only=True)
            for name in ("best_validation_spearman", "selected_step", "selected_nll_per_variant"):
                state.pop(name)
            for name in ("stopping_criterion", "validation_interval"):
                state["training_config"].pop(name)
            torch.save(state, path)
            restored = FittedGP.load(path)
            self.assertEqual(restored.training_config.stopping_criterion, "training_nll")
            torch.testing.assert_close(restored.predict(self.test).mean, fitted.predict(self.test).mean)


if __name__ == "__main__":
    unittest.main()
