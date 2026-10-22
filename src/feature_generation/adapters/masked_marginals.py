"""Shared masked-WT log-odds calculation for amino-acid language models."""

from collections.abc import Iterator, Sequence

import torch
from torch import Tensor

from data_loading.sequence_windows import centered_window_bounds

CANONICAL_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
ALPHABET = frozenset(CANONICAL_ALPHABET)


class MaskedMarginalScoring:
    @property
    def scoring_window(self):
        return getattr(self.config, "scoring_window", "full")

    def _masked_forward(self, input_ids, attention_mask):
        return self._model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

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
        if length > self.config.max_length and self.scoring_window == "full":
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
                output = self._masked_forward(masked_ids, batch_attention)
                logits = output.logits
                if tuple(logits.shape) != (count, represented_length + 2, self._model.config.vocab_size):
                    raise ValueError("Masked language model returned an unexpected masked-language-model logits shape")
                masked_logits = logits[rows, positions].to(torch.float32)
                if not torch.isfinite(masked_logits).all():
                    raise ValueError("Masked language model returned nonfinite masked-language-model logits")
                native_log_probs = masked_logits.log_softmax(dim=-1)
                canonical_log_probs = native_log_probs[:, amino_acid_ids]
                wt_columns = [aa_columns[reference_sequence[site]] for site in batch_sites]
                values = canonical_log_probs - canonical_log_probs[rows, wt_columns].unsqueeze(-1)
                if not torch.isfinite(values).all():
                    raise ValueError("Masked language model returned nonfinite masked-marginal log-odds")
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
                raise ValueError("Masked language model returned nonfinite variant scores")
            yield list(range(start, start + len(batch_mutations))), values
