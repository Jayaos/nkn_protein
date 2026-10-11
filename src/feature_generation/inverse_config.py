"""Pinned native inverse-folding recipes; single-chain reference distributions."""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import re

from .config import PROJECT_ROOT

PRESETS = json.loads(Path(__file__).with_name("inverse_artifacts.json").read_text(encoding="utf-8"))
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class InverseFoldingConfig:
    feature_type: str = "inverse_folding"
    model_family: str = "proteinmpnn"
    model_id: str | None = None
    revision: str | None = None
    cache_id: str | None = None
    model_dir: Path | None = None
    output_root: Path = Path("data/features/pretrained")
    assay_index: Path = Path("data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv")
    structure_manifest_dir: Path = Path("data/structures/manifests")
    conditioning_mode: str | None = None
    device: str = "auto"
    inference_dtype: str = "float32"
    output_dtype: str = "float32"
    max_length: int = 1022
    seed: int = 0
    num_orders: int = 10

    def __post_init__(self):
        if self.feature_type != "inverse_folding":
            raise ValueError("feature_type must be inverse_folding")
        if not isinstance(self.model_family, str) or self.model_family not in PRESETS:
            raise ValueError(f"Supported inverse folding model_family values: {', '.join(PRESETS)}")
        preset = PRESETS[self.model_family]
        for name in ("model_id", "revision"):
            value = getattr(self, name)
            if value is None:
                setattr(self, name, preset[name])
            elif value != preset[name]:
                raise ValueError(f"{name} must be {preset[name]!r} for the pinned {self.model_family} recipe")
        if self.conditioning_mode is None:
            self.conditioning_mode = preset["conditioning_mode"]
        allowed_modes = ("backbone_only", "sequence_conditional") if self.model_family == "proteinmpnn" else (preset["conditioning_mode"],)
        if self.conditioning_mode not in allowed_modes:
            raise ValueError(f"conditioning_mode must be one of {allowed_modes} for {self.model_family}")
        if type(self.num_orders) is not int or self.num_orders < 1:
            raise ValueError("num_orders must be a positive integer")
        if self.cache_id is None:
            self.cache_id = f"{self.model_id.split('/')[-1]}-{self.conditioning_mode}"
            if self.conditioning_mode == "sequence_conditional":
                self.cache_id += f"-orders{self.num_orders}-seed{self.seed}"
            self.cache_id += f"-{self.revision[:12]}"
        if not isinstance(self.cache_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.cache_id):
            raise ValueError("cache_id must be a safe single directory name")
        reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
        if self.cache_id.endswith(".") or self.cache_id.split(".")[0].upper() in reserved:
            raise ValueError("cache_id must be a portable directory name")
        if self.model_dir is None:
            self.model_dir = Path("data/models") / self.model_family
        for name in ("model_dir", "output_root", "assay_index", "structure_manifest_dir"):
            value = getattr(self, name)
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"{name} must be a nonempty path")
            path = Path(value).expanduser()
            setattr(self, name, (path if path.is_absolute() else PROJECT_ROOT / path).resolve())
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        if self.inference_dtype != "float32" or self.output_dtype != "float32":
            raise ValueError("Native inverse folding inference_dtype and output_dtype must be float32")
        if type(self.max_length) is not int or not 1 <= self.max_length <= 1022:
            raise ValueError("max_length must be an integer between 1 and 1022; truncation is disabled")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer between 0 and 2**32-1")

    @classmethod
    def load(cls, path: str | Path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Inverse folding configuration must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown inverse folding configuration keys: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict:
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}
