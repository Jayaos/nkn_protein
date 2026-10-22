"""Frozen ESM2 masked-marginal mutation scores for the GP mean function."""

import torch

from .esm2 import ESM2Adapter
from .masked_marginals import CANONICAL_ALPHABET, MaskedMarginalScoring
from ..config import MODEL_PRESETS



class ESM2ScoreAdapter(MaskedMarginalScoring, ESM2Adapter):
    """Score substitutions in singly masked WT context, without fitness labels."""

    def score_metadata(self) -> dict:
        metadata = self.metadata()
        # A scoring recipe has no chosen representation layer or pooling method.
        for name in (
            "checkpoint", "layer", "embedding_dim", "hidden_size", "pooling", "includes_special_tokens",
            "site_embeddings", "cache_layout", "site_context", "window_policy", "window_indexing",
            "global_window", "global_window_aggregation",
        ):
            metadata.pop(name, None)
        metadata.update(
            feature_type="scores", model_family="esm2", model_class="EsmForMaskedLM",
            score_definition="sum_masked_marginal_log_odds",
            conditioning_mode="masked_wild_type", inference_method="masked_marginals",
            scoring_recipe_version=1, normalization="model_vocabulary",
            score_direction="higher_is_more_likely", output_dtype="float32",
            residue_aggregation="sum_over_mutations",
            masking="one_wild_type_residue_at_a_time", wild_type_subtraction=True,
            noncanonical_tokens="retained_in_normalization",
            alphabet=CANONICAL_ALPHABET, precision_policy="float32_log_softmax",
            # Full-context scoring keeps its original recipe and cache identity,
            # independently of the selected embedding policy.
            truncation="reject",
        )
        if self.config.scoring_window == "centered":
            metadata.update(
                scoring_recipe_version=2, scoring_window="centered", truncation="centered",
                window_policy="centered_clamped", window_indexing="zero_based_half_open",
            )
        return metadata

    def load(self) -> None:
        if self._model is not None:
            return
        device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if device == "cpu" and self.config.inference_dtype == "float16":
            raise ValueError("Use float32 or bfloat16 for CPU scoring")
        if device == "cuda" and self.config.inference_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA device does not support bfloat16; choose float32 or float16")
        path = self._checkpoint()
        try:
            from transformers import EsmForMaskedLM, EsmTokenizer
        except ImportError as exc:
            raise ImportError("Install the dependencies in env.yaml before scoring sequences") from exc
        tokenizer = EsmTokenizer.from_pretrained(str(path), local_files_only=True)
        # HF instantiates modules on CPU before loading their weights. Do not
        # consume the caller's random stream, even when loading fails.
        with torch.random.fork_rng(devices=[]):
            model, loading_info = EsmForMaskedLM.from_pretrained(
                str(path), local_files_only=True, use_safetensors=True,
                torch_dtype=getattr(torch, self.config.inference_dtype), output_loading_info=True,
            )
        problems = {
            name: loading_info.get(name) for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            if loading_info.get(name)
        }
        if problems:
            raise ValueError(f"ESM2 scoring requires fully pretrained model and language-model head weights: {problems}")
        preset = MODEL_PRESETS[self.config.model_id]
        if model.config.hidden_size != self.hidden_size or model.config.num_hidden_layers != preset["num_layers"]:
            raise ValueError("Local model architecture does not match the selected ESM2 checkpoint")
        if tokenizer.mask_token_id != model.config.mask_token_id:
            raise ValueError("ESM2 model and tokenizer disagree on the mask token")
        self._device = torch.device(device)
        model.requires_grad_(False).eval().to(self._device)
        self._tokenizer, self._model = tokenizer, model
