"""ProteinGym MSA sampling/scoring and the NKN embedding cache contract."""

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
import re

from .config import PROJECT_ROOT
from .inputs import safe_component


LEGACY_SCORE_RECIPE = {
    "scoring_recipe_version": 1,
    "score_definition": "sum_masked_marginal_log_odds",
    "conditioning_mode": "masked_wild_type_query_and_fixed_homologs",
    "inference_method": "masked_marginals",
    "normalization": "model_vocabulary",
    "output_dtype": "float32",
    "scoring_window": "centered",
    "masking": "one_query_residue_at_a_time",
    "wild_type_subtraction": True,
    "residue_aggregation": "sum_over_mutations",
}

SAMPLING = "proteingym_sequence_reweighting_with_replacement"
EMBEDDING_RECIPE = {
    "msa_recipe_version": 3,
    "global_region": "msa_interval",
    "global_accumulation_dtype": "float64",
    "embedding_homolog_filter": "drop_all_gap_window_rows",
    "empty_embedding_window": "query_only",
}
SCORE_RECIPE = {
    **LEGACY_SCORE_RECIPE,
    "scoring_recipe_version": 2,
    "output_dtype": "float64",
    "scoring_window": "proteingym_optimal_tokens",
    "scoring_token_window": 1024,
    "scoring_coverage": "msa_interval",
    "score_aggregation": "arithmetic_mean_over_sampling_seeds",
    "window_policy": "proteingym_get_optimal_window",
    "window_indexing": "zero_based_half_open_tokens",
    "window_sequence_length": "msa_residues_plus_two",
}


@dataclass(frozen=True)
class MSAConfig:
    cache_id: str
    score_cache_id: str
    assay_index: Path = Path("data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv")
    msa_root: Path = Path("data/msa/DMS_msa_weights_generated")
    msa_reference_csv: Path = Path("data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv")
    runtime_dir: Path = Path("data/models/msa_transformer")
    runtime_config: Path = Path("configs/setup/msa_transformer_runtime.json")
    output_root: Path = Path("data/features/pretrained/singles")
    model_family: str = "msa_transformer"
    max_length: int = 1022
    max_depth: int = 400
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    sampling: str = SAMPLING
    scoring_token_window: int = 1024
    max_tokens_per_msa: int = 16384
    max_output_gib: float = 32.0
    uncovered_positions: str = "full_query_gap_padded_homologs"
    device: str = "cuda"

    def __post_init__(self):
        safe_component(self.cache_id, "cache ID")
        safe_component(self.score_cache_id, "score cache ID")
        for name in ("assay_index", "msa_root", "msa_reference_csv", "runtime_dir", "runtime_config", "output_root"):
            value = Path(getattr(self, name)).expanduser()
            object.__setattr__(self, name, (PROJECT_ROOT / value).resolve())
        for name, maximum in (("max_length", 1022), ("max_depth", 1024), ("max_tokens_per_msa", 2**31)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in 1..{maximum}")
        if (not isinstance(self.seeds, (list, tuple)) or not self.seeds
                or any(type(seed) is not int or seed < 0 for seed in self.seeds)
                or len(set(self.seeds)) != len(self.seeds)):
            raise ValueError("seeds must be a nonempty sequence of unique nonnegative integers")
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if self.sampling != SAMPLING:
            raise ValueError("Use ProteinGym weighted sampling with replacement")
        if type(self.scoring_token_window) is not int or self.scoring_token_window != 1024:
            raise ValueError("ProteinGym scoring requires a 1024-token window")
        if not isinstance(self.max_output_gib, (int, float)) or not math.isfinite(self.max_output_gib) or self.max_output_gib <= 0:
            raise ValueError("max_output_gib must be finite and positive")
        if self.model_family != "msa_transformer" or self.device not in ("cpu", "cuda"):
            raise ValueError("Expected msa_transformer and device cpu/cuda")
        if self.uncovered_positions != "full_query_gap_padded_homologs":
            raise ValueError("This recipe requires the full query and gap-padded homologs")

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict) or set(values) - {field.name for field in fields(cls)}:
            raise ValueError("Unknown MSA preparation configuration fields")
        return cls(**values)


def validate_msa_metadata(config):
    """Require an explicit, reproducible alignment and sampling identity."""
    expected = {
        "model_id": "esm_msa1b_t12_100M_UR50S", "layer": 12, "hidden_size": 768,
        "inference_dtype": "float32",
    }
    if any(type(config.get(key)) is not type(value) or config.get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported MSA Transformer metadata")
    for name, maximum in (("max_length", 1022), ("max_depth", 1024)):
        if type(config.get(name)) is not int or not 1 <= config[name] <= maximum:
            raise ValueError(f"Invalid MSA {name}")
    for key in ("checkpoint_sha256", "msa_manifest_sha256"):
        if not isinstance(config.get(key), str) or not re.fullmatch("[0-9a-f]{64}", config[key]):
            raise ValueError(f"Missing MSA fingerprint: {key}")
    version = config.get("msa_recipe_version")
    if type(version) is not int or version not in (1, 2, 3):
        raise ValueError("Unsupported MSA recipe version")
    if version == 3:
        if config.get("feature_type") != "sequence" or any(
                type(config.get(key)) is not type(value) or config.get(key) != value
                for key, value in EMBEDDING_RECIPE.items()):
            raise ValueError("Invalid MSA-region embedding recipe")
        start, end = config.get("global_region_start"), config.get("global_region_end")
        columns = config.get("msa_covered_positions")
        if (type(start) is not int or type(end) is not int or not 0 <= start < end
                or not isinstance(columns, list) or len(columns) != end - start
                or any(type(position) is not int or position != start + i + 1
                       for i, position in enumerate(columns))):
            raise ValueError("Global embedding region must match contiguous MSA coverage")
    if version == 1:
        # Previously prepared caches remain readable under their original IDs.
        expected = {
            "uncovered_positions": "full_query_gap_padded_homologs",
            "sampling": "weighted_without_replacement_sorted_query_first",
            "window_policy": "centered_clamped", "window_indexing": "zero_based_half_open",
        }
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError("Invalid legacy MSA recipe")
        if type(config.get("seed")) is not int or config["seed"] < 0:
            raise ValueError("Invalid MSA sampling seed")
        indices = config.get("sampled_row_indices")
        if (not isinstance(indices, list) or not indices or indices[0] != 0
                or any(type(i) is not int or i < 0 for i in indices)
                or indices != sorted(set(indices)) or len(indices) > config["max_depth"]):
            raise ValueError("Invalid sampled MSA row indices")
        return
    if config.get("sampling") != SAMPLING or config.get("sampling_rng") != "python_random_choices":
        raise ValueError("MSA ensemble requires ProteinGym's weighted replacement sampler")
    seeds, samples = config.get("seeds"), config.get("sampled_row_indices")
    if (not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds) or "seed" in config):
        raise ValueError("Invalid MSA ensemble seeds")
    total = config.get("msa_total_rows")
    if type(total) is not int or total < 1:
        raise ValueError("Missing MSA source row count")
    if not isinstance(samples, list) or len(samples) != len(seeds):
        raise ValueError("Expected one sampled row list per seed")
    for indices in samples:
        if (not isinstance(indices, list) or len(indices) != (config["max_depth"] if total > 1 else 1)
                or indices[0] != 0 or any(type(i) is not int or not 0 <= i < total for i in indices)
                or 0 in indices[1:]):
            raise ValueError("Invalid ProteinGym sampled row indices")
    if config["feature_type"] == "sequence":
        if (config.get("embedding_aggregation") != "arithmetic_mean_over_sampling_seeds"
                or config.get("uncovered_positions") != "full_query_gap_padded_homologs"):
            raise ValueError("Missing NKN embedding ensemble/coverage policy")
