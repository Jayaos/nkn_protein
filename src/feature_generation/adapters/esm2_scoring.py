"""Frozen ESM2 masked-marginal mutation scores for the GP mean function."""

from collections.abc import Iterator, Sequence

import torch
from torch import Tensor

from data_loading.sequence_windows import centered_window_bounds

from .esm2 import ALPHABET, ESM2Adapter
from ..config import MODEL_PRESETS


CANONICAL_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


class ESM2ScoreAdapter(ESM2Adapter):
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

    def _reference_tokens(self, reference_sequence: str):
        tokens = self._tokenizer(
            [reference_sequence], add_special_tokens=True, padding=True, truncation=False,
            return_attention_mask=True, return_special_tokens_mask=True, return_tensors="pt",
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        special_mask = tokens["special_tokens_mask"]
        expected_shape = (1, len(reference_sequence) + 2)
        if (
            tuple(input_ids.shape) != expected_shape or attention_mask.shape != input_ids.shape
            or special_mask.shape != input_ids.shape or input_ids.dtype != torch.long
            or not torch.all(attention_mask == 1) or not torch.all((special_mask == 0) | (special_mask == 1))
            or int(special_mask.sum()) != 2
        ):
            raise ValueError("Tokenizer masks must align with the complete reference and its BOS/EOS tokens")
        residue_positions = (~special_mask[0].bool()).nonzero().flatten()
        residue_ids = input_ids[0, residue_positions].tolist()
        if self._tokenizer.convert_ids_to_tokens(residue_ids) != list(reference_sequence):
            raise ValueError("Tokenizer residues do not match the complete reference sequence")
        amino_acid_ids = self._tokenizer.convert_tokens_to_ids(list(CANONICAL_ALPHABET))
        vocab_size = self._model.config.vocab_size
        mask_id = self._tokenizer.mask_token_id
        if (
            not isinstance(amino_acid_ids, list) or len(amino_acid_ids) != 20
            or any(type(index) is not int or not 0 <= index < vocab_size for index in amino_acid_ids)
            or len(set(amino_acid_ids)) != 20
            or self._tokenizer.convert_ids_to_tokens(amino_acid_ids) != list(CANONICAL_ALPHABET)
            or type(mask_id) is not int or not 0 <= mask_id < vocab_size or mask_id in amino_acid_ids
            or self._tokenizer.convert_ids_to_tokens([mask_id]) != [self._tokenizer.mask_token]
            or mask_id not in self._tokenizer.all_special_ids
            or any(index in self._tokenizer.all_special_ids for index in amino_acid_ids)
            or torch.any(input_ids < 0) or torch.any(input_ids >= vocab_size)
        ):
            raise ValueError("Tokenizer must provide distinct canonical amino-acid tokens and a valid special mask token")
        return input_ids, attention_mask, residue_positions, amino_acid_ids

    def iter_scores(
        self, reference_sequence: str, sequences: Sequence[str], *, batch_size: int = 8,
    ) -> Iterator[tuple[list[int], Tensor]]:
        """Sum mutant-minus-WT log probabilities at mutated sites only.

        Each required site is evaluated once, with only that WT site masked.
        Multiple mutations share this WT context; their contributions are
        additive. Centered mode limits each WT context to a clamped window
        around that site. Scores retain the native model vocabulary normalization.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(reference_sequence, str) or not reference_sequence or set(reference_sequence) - ALPHABET:
            raise ValueError("Reference must be a nonempty canonical amino-acid sequence")
        length = len(reference_sequence)
        if length > self.config.max_length and self.config.scoring_window == "full":
            raise ValueError(f"Reference has {length} residues, exceeding max_length={self.config.max_length}; truncation is disabled")
        represented_length = min(length, self.config.max_length)
        if represented_length + 2 > self.config.max_tokens_per_batch:
            raise ValueError("One reference sequence exceeds max_tokens_per_batch, including BOS/EOS tokens")
        if isinstance(sequences, (str, bytes)):
            raise ValueError("sequences must be a nonempty collection of full-length sequences")
        try:
            sequences = list(sequences)
        except TypeError as exc:
            raise ValueError("sequences must be a nonempty collection of full-length sequences") from exc
        if not sequences:
            raise ValueError("sequences must be a nonempty collection of full-length sequences")
        mutations = []
        for index, sequence in enumerate(sequences):
            if not isinstance(sequence, str) or len(sequence) != length or set(sequence) - ALPHABET:
                raise ValueError(f"Sequence {index} must contain exactly {length} canonical amino acids; only substitutions are supported")
            mutations.append([(site, aa) for site, aa in enumerate(sequence) if aa != reference_sequence[site]])
        sites = sorted({site for changes in mutations for site, _ in changes})
        self.load()
        # Short references share one tokenization in both scoring modes. Long
        # references are tokenized window by window, never as a complete input.
        reference_tokens = self._reference_tokens(reference_sequence) if length <= self.config.max_length else None
        masked_batch_size = min(batch_size, self.config.max_tokens_per_batch // (represented_length + 2))
        log_odds = torch.empty((len(sites), 20), dtype=torch.float32)
        site_rows = {site: row for row, site in enumerate(sites)}
        aa_columns = {aa: column for column, aa in enumerate(CANONICAL_ALPHABET)}
        for start in range(0, len(sites), masked_batch_size):
            batch_sites = sites[start:start + masked_batch_size]
            count = len(batch_sites)
            if reference_tokens is not None:
                input_ids, attention_mask, residue_positions, amino_acid_ids = reference_tokens
                positions = residue_positions[batch_sites].to(self._device)
                masked_ids = input_ids.to(self._device).expand(count, -1).clone()
                batch_attention = attention_mask.to(self._device).expand(count, -1)
            else:
                window_tokens = {}
                token_rows, attention_rows, local_positions = [], [], []
                for site in batch_sites:
                    bounds = centered_window_bounds(site + 1, length, self.config.max_length)
                    if bounds not in window_tokens:
                        window_tokens[bounds] = self._reference_tokens(reference_sequence[bounds[0]:bounds[1]])
                    input_ids, attention_mask, residue_positions, amino_acid_ids = window_tokens[bounds]
                    token_rows.append(input_ids)
                    attention_rows.append(attention_mask)
                    local_positions.append(int(residue_positions[site - bounds[0]]))
                masked_ids = torch.cat(token_rows).to(self._device)
                batch_attention = torch.cat(attention_rows).to(self._device)
                positions = torch.tensor(local_positions, device=self._device)
            rows = torch.arange(count, device=self._device)
            masked_ids[rows, positions] = self._tokenizer.mask_token_id
            with torch.inference_mode():
                output = self._model(
                    input_ids=masked_ids,
                    attention_mask=batch_attention, return_dict=True,
                )
                logits = output.logits
                if tuple(logits.shape) != (count, represented_length + 2, self._model.config.vocab_size):
                    raise ValueError("ESM2 returned an unexpected masked-language-model logits shape")
                masked_logits = logits[rows, positions].to(torch.float32)
                if not torch.isfinite(masked_logits).all():
                    raise ValueError("ESM2 returned nonfinite masked-language-model logits")
                native_log_probs = masked_logits.log_softmax(dim=-1)
                canonical_log_probs = native_log_probs[:, amino_acid_ids]
                wt_columns = [aa_columns[reference_sequence[site]] for site in batch_sites]
                values = canonical_log_probs - canonical_log_probs[rows, wt_columns].unsqueeze(-1)
                if not torch.isfinite(values).all():
                    raise ValueError("ESM2 returned nonfinite masked-marginal log-odds")
                log_odds[start:start + count] = values.to(device="cpu", dtype=torch.float32)
                del output, logits
        for start in range(0, len(sequences), batch_size):
            batch_mutations = mutations[start:start + batch_size]
            values = torch.zeros(len(batch_mutations), dtype=torch.float32)
            for row, changes in enumerate(batch_mutations):
                if changes:
                    values[row] = log_odds[
                        [site_rows[site] for site, _ in changes], [aa_columns[aa] for _, aa in changes],
                    ].sum()
            if not torch.isfinite(values).all():
                raise ValueError("ESM2 returned nonfinite variant scores")
            yield list(range(start, start + len(batch_mutations))), values
