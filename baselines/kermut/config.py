"""JSON settings for the independent Kermut baseline.

Defaults follow the publication's reproduction implementation for optimization
and use this project's existing frozen ESM2/ProteinMPNN caches for inputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
import re
from typing import Mapping

from config import EvaluationConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_SELECTION = {
    "sequence": "esm2/esm2_t6_8M_UR50D-layer6-c731040fcd8d",
    "inverse_folding": "proteinmpnn/v_48_020-backbone_only-8907e6671bfb",
}
DEFAULT_SCORE_SOURCES = (
    "scores/esm2/esm2_t6_8M_UR50D-masked_marginals-float32-c731040fcd8d",
)


def _finite(name, value, *, positive=False, nonnegative=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _positive_int(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _settings(cls, value, name):
    if isinstance(value, cls):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    unknown = set(value) - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {name} settings: {sorted(unknown)}")
    try:
        return cls(**dict(value))
    except (TypeError, OverflowError) as error:
        raise ValueError(f"Invalid {name} settings: {error}") from error


def _cache_path(value, name, depth):
    if not isinstance(value, str) or "\\" in value or ":" in value:
        raise ValueError(f"{name} must be a relative cache path using '/' separators")
    parts = value.split("/")
    if len(parts) != depth or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"{name} must contain exactly {depth} nonempty relative path components")


@dataclass
class KermutModelConfig:
    """Initial unconstrained parameters, transformed inside the kernel."""

    raw_h_scale: float = 1.0
    raw_h_lengthscale: float = 0.1
    raw_d_lengthscale: float = 0.1
    raw_p_lengthscale: float = 0.1
    raw_alpha: float = 0.5
    raw_sequence_lengthscale: float = 0.0

    def __post_init__(self):
        for item in fields(self):
            _finite(item.name, getattr(self, item.name))

    @property
    def enabled_branches(self):
        # The shared experiment runner checks feature availability with this
        # interface. Structural coordinates are added by our own data adapter.
        return ("inverse_folding", "sequence_global")


@dataclass
class KermutTrainingConfig:
    n_steps: int = 150
    learning_rate: float = 0.1
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    epsilon: float = 1e-8
    seed: int = 2024
    dtype: str = "float32"
    raw_noise: float = 0.0
    noise_floor: float = 1e-4
    noise_prior_scale: float = 0.1
    jitter_initial: float = 1e-6
    jitter_max: float = 1e-3
    max_train_variants: int | None = None

    def __post_init__(self):
        _positive_int("n_steps", self.n_steps)
        for name in ("learning_rate", "epsilon", "noise_floor", "noise_prior_scale",
                     "jitter_initial", "jitter_max"):
            _finite(name, getattr(self, name), positive=True)
        _finite("weight_decay", self.weight_decay, nonnegative=True)
        _finite("raw_noise", self.raw_noise)
        if self.jitter_max < self.jitter_initial:
            raise ValueError("jitter_max must be at least jitter_initial")
        if not isinstance(self.betas, (list, tuple)) or len(self.betas) != 2:
            raise ValueError("betas must contain two AdamW coefficients")
        for beta in self.betas:
            _finite("betas", beta, nonnegative=True)
            if beta >= 1:
                raise ValueError("betas must lie in [0, 1)")
        self.betas = tuple(self.betas)
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be an integer in [0, 2**32 - 1]")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")
        if self.max_train_variants is not None:
            _positive_int("max_train_variants", self.max_train_variants)


@dataclass
class KermutConfig:
    model: KermutModelConfig = field(default_factory=KermutModelConfig)
    training: KermutTrainingConfig = field(default_factory=KermutTrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    feature_selection: dict = field(default_factory=lambda: dict(DEFAULT_FEATURE_SELECTION))
    score_sources: tuple = DEFAULT_SCORE_SOURCES
    structure_manifest_dir: str = "data/structures/manifests"
    pooling: str = "mean"
    device: str = "cpu"
    num_threads: int = 4

    def __post_init__(self):
        for name, cls in (("model", KermutModelConfig), ("training", KermutTrainingConfig),
                          ("evaluation", EvaluationConfig)):
            setattr(self, name, _settings(cls, getattr(self, name), name))
        # Also reject booleans/nonfinite values in shared evaluation settings.
        for level in self.evaluation.coverage_levels:
            _finite("coverage_levels", level, positive=True)
        if not isinstance(self.feature_selection, Mapping):
            raise ValueError("feature_selection must be a JSON object")
        if set(self.feature_selection) - set(DEFAULT_FEATURE_SELECTION):
            raise ValueError("Kermut feature_selection supports only sequence and inverse_folding")
        self.feature_selection = dict(self.feature_selection)
        for name, value in self.feature_selection.items():
            _cache_path(value, f"feature_selection.{name}", 2)
        if not isinstance(self.score_sources, (list, tuple)) or not self.score_sources:
            raise ValueError("score_sources must be a nonempty list of mean score cache paths")
        for value in self.score_sources:
            _cache_path(value, "score_sources", 3)
        if len(set(self.score_sources)) != len(self.score_sources):
            raise ValueError("score_sources must not contain duplicates")
        self.score_sources = tuple(self.score_sources)
        if not isinstance(self.structure_manifest_dir, str) or not self.structure_manifest_dir.strip():
            raise ValueError("structure_manifest_dir must be a nonempty path string")
        if self.pooling != "mean":
            raise ValueError("Kermut supports mean pooling of cached residue embeddings")
        if not isinstance(self.device, str) or re.fullmatch(r"cpu|cuda(?::\d+)?|mps", self.device) is None:
            raise ValueError("device must be cpu, cuda, cuda:<index>, or mps")
        _positive_int("num_threads", self.num_threads)

    @property
    def resolved_structure_manifest_dir(self):
        path = Path(self.structure_manifest_dir).expanduser()
        return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return _settings(cls, values, "Kermut")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
