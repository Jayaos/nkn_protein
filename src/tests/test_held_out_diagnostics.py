"""Held-out monitoring observes current predictions without influencing fitting."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from config import TrainingConfig
from evaluation import spearman
from experiments import run_cv, run_extrapolation
from gp.linalg import NumericalError
from gp.model import ExactProteinGP
from gp.training import FittedGP, fit_gp
from test_pipeline import read_predictions, small_config, synthetic_batch


class HeldOutDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.batch = synthetic_batch()
        self.train = self.batch.select(range(5))
        self.test = self.batch.select(range(5, 8))
        self.config = small_config()
        self.training = replace(self.config.training, max_steps=5, held_out_interval=2,
                                seeds=(0, 1), lengthscale_multipliers=(1., 2.))

    def fit(self, cfg=None, test=None):
        return fit_gp(self.train, self.config.model, cfg or self.training,
                      held_out=self.test if test is None else test)

    def assert_same_fit(self, first, second):
        self.assertEqual(first.restart_summaries, second.restart_summaries)
        self.assertEqual(first.selected_seed, second.selected_seed)
        self.assertEqual(first.best_nll_per_variant, second.best_nll_per_variant)
        for name, value in first.model.state_dict().items():
            torch.testing.assert_close(value, second.model.state_dict()[name], rtol=0, atol=0)
        for left, right in zip(first.history, second.history):
            self.assertEqual({k: v for k, v in left.items() if not k.startswith(('held_out_', 'n_held_out'))},
                             {k: v for k, v in right.items() if not k.startswith(('held_out_', 'n_held_out'))})
        self.assertEqual(len(first.history), len(second.history))

    def test_frequency_restarts_and_label_independence(self):
        disabled = self.fit(replace(self.training, held_out_interval=0))
        enabled = self.fit()
        reversed_labels = self.fit(test=replace(self.test, targets=-self.test.targets))
        self.assert_same_fit(disabled, enabled)
        self.assert_same_fit(enabled, reversed_labels)
        self.assertFalse(any('held_out_spearman' in row for row in disabled.history))
        for seed in self.training.seeds:
            self.assertEqual([r['step'] for r in enabled.history
                              if r['seed'] == seed and 'held_out_spearman' in r], [0, 2, 4, 5])
        for left, right in zip(enabled.history, reversed_labels.history):
            if 'held_out_spearman' in left:
                self.assertAlmostEqual(left['held_out_spearman'], -right['held_out_spearman'])

    def test_mean_only_matches_independent_conditioning_and_has_no_test_covariance(self):
        fitted = self.fit()
        test = fitted.transform.apply(replace(self.test, targets=None))
        model, train, y = fitted.model, fitted.train, fitted.standardized_targets
        with torch.no_grad():
            kernel = model.kernel(train)
            _, jitter = model.training_factor(train)
            noisy = kernel + (model.noise + jitter) * torch.eye(train.num_variants)
            cross = model.kernel(test, train)
            expected = model.mean(test) + cross @ torch.linalg.solve(noisy, y - model.mean(train))
            with patch.object(model.kernel, 'forward', wraps=model.kernel.forward) as forward:
                actual = model.posterior_mean(train, y, test)
            self.assertEqual(forward.call_count, 2)  # train/train and train/test only
            torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)

    def test_diagnostics_match_current_checkpoint_not_best_checkpoint(self):
        original = ExactProteinGP.posterior_mean
        observed = []

        def record(model, train, y, test):
            self.assertFalse(torch.is_grad_enabled())
            self.assertIsNone(test.targets)
            mean = original(model, train, y, test)
            observed.append(spearman(self.test.targets, mean))
            return mean

        with patch.object(ExactProteinGP, 'posterior_mean', record):
            fitted = self.fit()
        self.assertEqual(observed, [r['held_out_spearman'] for r in fitted.history if 'held_out_spearman' in r])
        # A run ending at a checkpoint must reproduce that checkpoint's diagnostic.
        cfg = replace(self.training, max_steps=2, seeds=(0,), lengthscale_multipliers=(1.,))
        short = self.fit(cfg)
        self.assertEqual(short.history[-1]['held_out_spearman'], observed[1])

    def test_early_stop_undefined_correlations_and_numerical_failure_are_diagnostic_only(self):
        cfg = replace(self.training, min_steps=3, patience=2, tolerance=1e6)
        baseline = self.fit(replace(cfg, held_out_interval=0))
        constant = self.fit(cfg, replace(self.test, targets=torch.ones_like(self.test.targets)))
        self.assert_same_fit(baseline, constant)
        self.assertEqual([r['step'] for r in constant.history
                          if r['seed'] == 0 and 'held_out_spearman' in r], [0, 2, 3])
        self.assertTrue(all(r['held_out_spearman'] is None for r in constant.history if 'held_out_spearman' in r))
        with patch.object(ExactProteinGP, 'posterior_mean', side_effect=NumericalError('diagnostic failure')):
            failed_diagnostic = self.fit(cfg)
        self.assert_same_fit(baseline, failed_diagnostic)
        self.assertEqual(failed_diagnostic.history[0]['held_out_error'], 'diagnostic failure')
        json.dumps(failed_diagnostic.history, allow_nan=False)

    def test_interval_validation_and_required_disjoint_labels(self):
        for value in (-1, True, 1.5, '100'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'held_out_interval'):
                TrainingConfig(held_out_interval=value)
        with self.assertRaisesRegex(ValueError, 'observed targets'):
            fit_gp(self.train, self.config.model, self.training)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.fit(test=self.train)
        with self.assertRaisesRegex(ValueError, 'observed targets'):
            self.fit(test=replace(self.test, targets=None))
        # Disabled mode does not even inspect held-out labels.
        self.fit(replace(self.training, held_out_interval=0), replace(self.test, targets=None))

    def test_serial_and_parallel_csv_histories_and_checkpoints_match(self):
        config = replace(self.config, training=replace(self.training, seeds=(0,), lengthscale_multipliers=(1.,)))
        folds = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            serial = run_cv(self.batch, folds, config, root / 'serial')
            parallel = run_cv(self.batch, folds, config, root / 'parallel', fold_workers=2)
            self.assertEqual(serial, parallel)
            for fold in range(3):
                one, two = (root / mode / f'fold_{fold}' for mode in ('serial', 'parallel'))
                self.assertEqual((one / 'held_out_spearman.csv').read_text(),
                                 (two / 'held_out_spearman.csv').read_text())
                csv_rows = read_predictions(one / 'held_out_spearman.csv')
                self.assertEqual([int(r['step']) for r in csv_rows], [0, 2, 4, 5])
                checkpoint = FittedGP.load(one / 'checkpoint.pt')
                history = json.loads((one / 'history.json').read_text())
                self.assertEqual(history, checkpoint.history)
                self.assertEqual(checkpoint.training_config.held_out_interval, 2)
                self.assertTrue(all(int(r['n_held_out']) == int((folds == fold).sum()) for r in csv_rows))

    def test_extrapolation_and_mixed_diagnostics_use_requested_held_out_domain(self):
        config = replace(self.config, training=replace(self.config.training, held_out_interval=1))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_extrapolation(self.batch, config, root / 'extrapolation')
            rows = read_predictions(root / 'extrapolation/held_out_spearman.csv')
            self.assertTrue(all(int(r['n_held_out']) == 2 for r in rows))
            folds = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
            run_cv(self.batch, folds, config, root / 'mixed', scheme='random_multiples',
                   setting='mixed_random_doubles')
            for fold in range(3):
                rows = read_predictions(root / f'mixed/fold_{fold}/held_out_spearman.csv')
                self.assertTrue(all(int(r['n_held_out']) == (2 if fold == 1 else 0) for r in rows))
                if fold != 1:
                    self.assertTrue(all(r['held_out_spearman'] == '' for r in rows))


if __name__ == '__main__':
    unittest.main()
