"""Pinned frozen ProSST sequence representations conditioned on a fixed backbone."""

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import re

from .config import PROJECT_ROOT

PRESET = json.loads(Path(__file__).with_name("prosst_artifacts.json").read_text(encoding="utf-8"))


@dataclass
class ProSSTConfig:
    feature_type: str = "sequence"
    model_family: str = "prosst"
    model_id: str = PRESET["model_id"]
    revision: str = PRESET["revision"]
    cache_id: str | None = None
    model_dir: Path = Path("data/models/prosst")
    output_root: Path = Path("data/features/pretrained")
    assay_index: Path = Path("data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv")
    structure_manifest_dir: Path = Path("data/structures/manifests")
    structure_token_dir: Path = Path("data/structures/tokens/prosst")
    layer: int = 12
    device: str = "auto"
    inference_dtype: str = "float32"
    output_dtype: str = "float32"
    max_length: int = 1022
    max_tokens_per_batch: int = 512
    max_output_gib: float = 4.0
    quantizer_batch_size: int = 32

    def __post_init__(self):
        if self.feature_type != "sequence" or self.model_family != "prosst":
            raise ValueError("ProSST requires feature_type=sequence and model_family=prosst")
        for name in ("model_id", "revision"):
            if getattr(self, name) != PRESET[name]:
                raise ValueError(f"{name} must be {PRESET[name]!r} for the pinned ProSST recipe")
        if type(self.layer) is not int or not 0 <= self.layer <= PRESET["num_layers"]:
            raise ValueError("layer must be an integer between 0 and 12")
        if self.cache_id is None:
            self.cache_id = f"ProSST-2048-layer{self.layer}-fixed-backbone-{self.revision[:12]}"
        if not isinstance(self.cache_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.cache_id):
            raise ValueError("cache_id must be a safe single directory name")
        reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
        if self.cache_id.endswith(".") or self.cache_id.split(".")[0].upper() in reserved:
            raise ValueError("cache_id must be a portable directory name")
        for name in ("model_dir", "output_root", "assay_index", "structure_manifest_dir", "structure_token_dir"):
            value = getattr(self, name)
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"{name} must be a nonempty path")
            path = Path(value).expanduser()
            setattr(self, name, (path if path.is_absolute() else PROJECT_ROOT / path).resolve())
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        if self.inference_dtype != "float32" or self.output_dtype != "float32":
            raise ValueError("ProSST inference_dtype and output_dtype must be float32")
        # An operational memory guard, not a claim of a hard architecture limit.
        for name in ("max_length", "max_tokens_per_batch", "quantizer_batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.max_output_gib) not in (int, float) or not math.isfinite(self.max_output_gib) or self.max_output_gib <= 0:
            raise ValueError("max_output_gib must be finite and positive")

    @classmethod
    def load(cls, path: str | Path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("ProSST configuration must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown ProSST configuration keys: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict:
        return {name: str(value) if isinstance(value, Path) else value for name, value in asdict(self).items()}
