"""Strict configuration for independently cached pretrained representations."""

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PRESETS = {
    "facebook/esm2_t6_8M_UR50D": {
        "num_layers": 6, "hidden_size": 320,
        "revision": "c731040fcd8d73dceaa04b0a8e6329b345b0f5df",
        "weights_sha256": "24c5fa474c48f3b754b86efe752d5f189d2bcd88190fa2270fc92b2ef3034189",
    },
    "facebook/esm2_t33_650M_UR50D": {
        "num_layers": 33, "hidden_size": 1280,
        "revision": "08e4846e537177426273712802403f7ba8261b6c",
        "weights_sha256": "a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0",
    },
}


@dataclass
class ExtractionConfig:
    model_id: str = "facebook/esm2_t33_650M_UR50D"
    revision: str | None = None
    cache_id: str | None = None
    model_dir: Path = Path("data/models/esm2")
    output_root: Path = Path("data/features/pretrained")
    assay_index: Path = Path("data/DMS_ProteinGym_substitutions/DMS_substitutions.csv")
    layer: int | None = None
    device: str = "auto"
    max_tokens_per_batch: int = 1024
    max_length: int = 1022
    inference_dtype: str = "float32"
    output_dtype: str = "float32"
    max_output_gib: float = 4.0

    def __post_init__(self):
        if not isinstance(self.model_id, str) or self.model_id not in MODEL_PRESETS:
            raise ValueError(f"Supported model_id values: {', '.join(MODEL_PRESETS)}")
        preset = MODEL_PRESETS[self.model_id]
        if self.revision is None:
            self.revision = preset["revision"]
        if not isinstance(self.revision, str) or re.fullmatch(r"[0-9a-f]{40}", self.revision) is None:
            raise ValueError("revision must be an exact lowercase 40-character Hugging Face commit SHA")
        if self.layer is None:
            self.layer = preset["num_layers"]
        if type(self.layer) is not int or not 0 <= self.layer <= preset["num_layers"]:
            raise ValueError(f"layer must be between 0 and {preset['num_layers']}")
        if self.cache_id is None:
            self.cache_id = f"{self.model_id.split('/')[-1]}-layer{self.layer}-{self.revision[:12]}"
        if not isinstance(self.cache_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.cache_id) is None:
            raise ValueError("cache_id must be a safe single directory name")
        if self.cache_id.endswith(".") or self.cache_id.split(".")[0].upper() in {
            "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))
        }:
            raise ValueError("cache_id must be a portable directory name")
        for name in ("model_dir", "output_root", "assay_index"):
            value = getattr(self, name)
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"{name} must be a nonempty path")
            path = Path(value).expanduser()
            setattr(self, name, (path if path.is_absolute() else PROJECT_ROOT / path).resolve())
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        for name in ("max_tokens_per_batch", "max_length"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_length > 1022:
            raise ValueError("This extractor supports at most 1022 residues; it never truncates")
        if self.inference_dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("inference_dtype must be float32, float16, or bfloat16")
        if self.output_dtype != "float32":
            raise ValueError("output_dtype must be float32")
        if type(self.max_output_gib) not in (int, float) or not math.isfinite(self.max_output_gib) or self.max_output_gib <= 0:
            raise ValueError("max_output_gib must be finite and positive")

    @classmethod
    def load(cls, path: str | Path):
        with Path(path).open(encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("Extraction configuration must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown extraction configuration keys: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict:
        return {name: str(value) if isinstance(value, Path) else value for name, value in asdict(self).items()}
