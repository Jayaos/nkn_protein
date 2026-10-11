"""Validated, serializable experiment settings independent of pretrained models."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


VECTOR_PRIMITIVES = ("cosine", "rbf", "matern32")
IF_PRIMITIVES = ("hellinger_exp", "hellinger_rbf", "bhattacharyya")
BRANCH_NAMES = ("inverse_folding", "structure", "sequence_site", "sequence_global")


def validate_enabled_branches(values):
    """Return a nonempty branch selection in stable model channel order."""
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("enabled_branches must be a nonempty list or tuple of branch names")
    if any(not isinstance(name, str) for name in values):
        raise ValueError("enabled_branches must contain branch names as strings")
    if len(set(values)) != len(values):
        raise ValueError("enabled_branches must not contain duplicate names")
    unknown = set(values) - set(BRANCH_NAMES)
    if unknown:
        raise ValueError(f"Unknown enabled branches: {sorted(unknown)}; choose from {BRANCH_NAMES}")
    return tuple(name for name in BRANCH_NAMES if name in values)


def validate_widths(widths):
    widths = tuple(widths)
    if not widths or len(widths) % 2 != 1 or widths[-1] != 1:
        raise ValueError("NKN widths must alternate linear/product layers and end with linear width 1")
    if any(type(w) is not int or w < 1 for w in widths):
        raise ValueError("NKN widths must be positive integers")
    for i in range(1, len(widths), 2):
        if widths[i - 1] % 2 or widths[i] != widths[i - 1] // 2:
            raise ValueError("Product width must be half the preceding even linear width")
    return widths


@dataclass
class BranchConfig:
    copies: dict = field(default_factory=lambda: {name: 1 for name in VECTOR_PRIMITIVES})
    widths: tuple = (8, 4, 4, 2, 1)

    def __post_init__(self):
        self.widths = validate_widths(self.widths)
        self.copies = dict(self.copies)
        if not self.copies or any(type(n) is not int or n < 0 for n in self.copies.values()):
            raise ValueError("Primitive copy counts must be nonnegative integers")
        if sum(self.copies.values()) < 1:
            raise ValueError("Each branch needs at least one primitive")


def _branches():
    return {name: BranchConfig(copies={p: 1 for p in (IF_PRIMITIVES if name == "inverse_folding" else VECTOR_PRIMITIVES)})
            for name in BRANCH_NAMES}


@dataclass
class ModelConfig:
    branches: dict = field(default_factory=_branches)
    final_widths: tuple = (8, 4, 4, 2, 1)
    enabled_branches: tuple = BRANCH_NAMES

    def __post_init__(self):
        self.enabled_branches = validate_enabled_branches(self.enabled_branches)
        self.final_widths = validate_widths(self.final_widths)
        unknown = set(self.branches) - set(BRANCH_NAMES)
        if unknown:
            raise ValueError(f"Unknown branch configurations: {sorted(unknown)}")
        # Omitted hyperparameters keep their defaults, including for disabled branches.
        defaults = _branches()
        branches = {**defaults, **self.branches}
        self.branches = {
            k: v if isinstance(v, BranchConfig) else BranchConfig(**{**asdict(defaults[k]), **v})
            for k, v in branches.items()
        }
        for name, branch in self.branches.items():
            allowed = IF_PRIMITIVES if name == "inverse_folding" else VECTOR_PRIMITIVES
            if set(branch.copies) - set(allowed):
                raise ValueError(f"Unsupported primitives for {name}: {set(branch.copies) - set(allowed)}")


@dataclass
class TrainingConfig:
    max_steps: int = 5000
    min_steps: int = 1000
    patience: int = 500
    tolerance: float = 1e-4
    learning_rate: float = 1e-3
    betas: tuple = (0.9, 0.999)
    epsilon: float = 1e-8
    seeds: tuple = (0,)
    lengthscale_multipliers: tuple = (1.0,)
    noise_initial: float = 0.01
    noise_floor: float = 1e-4
    jitter_initial: float = 1e-6
    jitter_max: float = 1e-3
    max_train_variants: int | None = None

    def __post_init__(self):
        for name in ("max_steps", "min_steps", "patience"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "epsilon", "noise_floor", "jitter_initial", "jitter_max"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        if not math.isfinite(self.noise_initial) or self.noise_initial <= self.noise_floor:
            raise ValueError("Initial noise must exceed its positive lower bound")
        if self.jitter_max < self.jitter_initial:
            raise ValueError("jitter_max must be at least jitter_initial")
        self.betas = tuple(self.betas)
        if len(self.betas) != 2 or any(not 0 <= b < 1 for b in self.betas):
            raise ValueError("Adam betas must be in [0,1)")
        self.seeds, self.lengthscale_multipliers = tuple(self.seeds), tuple(self.lengthscale_multipliers)
        if not self.seeds or len(self.seeds) != len(self.lengthscale_multipliers):
            raise ValueError("Each restart seed needs a lengthscale multiplier")
        if any(type(s) is not int or s < 0 for s in self.seeds):
            raise ValueError("Seeds must be nonnegative integers")
        if any(not math.isfinite(m) or m <= 0 for m in self.lengthscale_multipliers):
            raise ValueError("Lengthscale multipliers must be positive and finite")
        if self.max_train_variants is not None and (type(self.max_train_variants) is not int or self.max_train_variants < 1):
            raise ValueError("max_train_variants must be positive or null")


@dataclass
class EvaluationConfig:
    coverage_levels: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    n_bins: int = 10
    n_folds: int = 5
    pair_block_size: int = 512

    def __post_init__(self):
        self.coverage_levels = tuple(self.coverage_levels)
        if not self.coverage_levels or any(not 0 < q < 1 for q in self.coverage_levels):
            raise ValueError("Coverage levels must lie strictly between zero and one")
        if len(set(self.coverage_levels)) != len(self.coverage_levels):
            raise ValueError("Coverage levels must be distinct")
        for name in ("n_bins", "n_folds", "pair_block_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_folds < 2:
            raise ValueError("Cross-validation requires at least two folds")


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    feature_selection: dict = field(default_factory=dict)
    score_sources: list | None = None
    inverse_folding_policy: str = "normalized"
    pooling: str = "mean"
    device: str = "cpu"
    num_threads: int = 1

    def __post_init__(self):
        for name, cls in (("model", ModelConfig), ("training", TrainingConfig), ("evaluation", EvaluationConfig)):
            if not isinstance(getattr(self, name), cls):
                setattr(self, name, cls(**getattr(self, name)))
        if type(self.num_threads) is not int or self.num_threads < 1:
            raise ValueError("num_threads must be a positive integer")
        if self.pooling != "mean":
            raise ValueError("The current cache adapter supports mean pooling; supply custom pooled features through FeatureBatch")
        if self.inverse_folding_policy not in ("normalized", "sequence_conditional"):
            raise ValueError("inverse_folding_policy must be 'normalized' or 'sequence_conditional'")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**values)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
