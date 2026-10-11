"""Kermut model adapters for the shared splits, evaluation and artifact writers."""

from experiments import runner as shared

from .training import FittedKermut, fit_gp


def run_cv(batch, folds, config, output_dir, **kwargs):
    return shared.run_cv(batch, folds, config, output_dir, fit_function=fit_gp, **kwargs)


def run_extrapolation(batch, config, output_dir, **kwargs):
    return shared.run_extrapolation(batch, config, output_dir, fit_function=fit_gp, **kwargs)


def run_predict(batch, checkpoint, config, output_dir):
    return shared.run_predict(
        batch, checkpoint, config, output_dir, load_function=FittedKermut.load,
    )
