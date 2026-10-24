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
    assay_index: Path = Path("data/DMS_ProteinGym_substitutions/DMS_substitutions.csv")
    structure_manifest_dir: Path = Path("data/structures/manifests")
    structure_token_dir: Path = Path("data/structures/tokens/prosst")
    layer: int = 12
    device: str = "auto"
    inference_dtype: str = "float32"
    output_dtype: str = "float32"
    max_length: int = 2046
    max_tokens_per_batch: int = 2048
    pooling: str = "none"
    global_window: str = "prefix"
    truncation: str = "reject"
    site_embeddings: str = "none"
    scoring_window: str = "full"
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
        if self.pooling not in ("none", "mean"):
            raise ValueError("pooling must be none or mean")
        if self.global_window not in ("prefix", "non_overlapping"):
            raise ValueError("global_window must be prefix or non_overlapping")
        if self.truncation not in ("reject", "none"):
            raise ValueError("ProSST truncation must be reject or none")
        if self.global_window == "non_overlapping":
            if self.pooling != "mean" or self.truncation != "none":
                raise ValueError("non_overlapping global windows require mean pooling and truncation=none")
        elif self.pooling != "none" or self.truncation != "reject":
            raise ValueError("Full-residue ProSST extraction requires pooling=none and truncation=reject")
        if self.site_embeddings not in ("none", "centered"):
            raise ValueError("site_embeddings must be none or centered")
        if self.site_embeddings == "centered" and self.pooling != "mean":
            raise ValueError("centered site embeddings require mean pooling")
        if self.scoring_window not in ("full", "centered"):
            raise ValueError("scoring_window must be full or centered")
        if self.cache_id is None:
            self.cache_id = f"ProSST-2048-layer{self.layer}-fixed-backbone-{self.revision[:12]}"
            if self.global_window == "non_overlapping":
                self.cache_id += f"-mean-all-nonoverlap{self.max_length}"
            if self.site_embeddings == "centered":
                self.cache_id += "-centered-sites"
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
        # Residues per input; the batch budget separately includes BOS and EOS.
        for name in ("max_length", "max_tokens_per_batch", "quantizer_batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_length > 2046:
            raise ValueError("ProSST supports at most 2046 residues per model input (2048 tokens including BOS/EOS)")
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
