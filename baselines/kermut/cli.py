"""Run with python -m baselines.kermut.cli from the repository root."""

from experiments.cli import main as shared_main

from .config import KermutConfig
from .data import load_feature_bundle
from .runner import run_cv, run_extrapolation, run_predict


def main(argv=None):
    return shared_main(
        argv, config_type=KermutConfig, feature_loader=load_feature_bundle,
        cv_runner=run_cv, extrapolation_runner=run_extrapolation,
        prediction_runner=run_predict, prog="python -m baselines.kermut.cli",
    )


if __name__ == "__main__":
    raise SystemExit(main())
