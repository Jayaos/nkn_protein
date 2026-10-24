"""ProSST masked-WT log odds conditioned on the fixed reference backbone."""

from .masked_marginals import CANONICAL_ALPHABET, MaskedMarginalScoring
from .prosst import ProSSTAdapter


class ProSSTScoreAdapter(MaskedMarginalScoring, ProSSTAdapter):
    def score_metadata(self):
        metadata = self.metadata()
        for name in ("checkpoint", "layer", "embedding_dim", "hidden_size", "pooling",
                     "includes_special_tokens", "amino_acid_masking", "site_embeddings",
                     "cache_layout", "site_context", "window_policy", "window_indexing",
                     "global_window", "global_window_aggregation", "structure_window"):
            metadata.pop(name, None)
        metadata.update(
            feature_type="scores", score_definition="sum_masked_marginal_log_odds",
            conditioning_mode="masked_wild_type_and_fixed_reference_structure",
            inference_method="masked_marginals", scoring_recipe_version=1,
            normalization="model_vocabulary", score_direction="higher_is_more_likely",
            output_dtype="float32", residue_aggregation="sum_over_mutations",
            masking="one_wild_type_residue_at_a_time", wild_type_subtraction=True,
            noncanonical_tokens="retained_in_normalization", alphabet=CANONICAL_ALPHABET,
            precision_policy="float32_log_softmax", scoring_window="full", truncation="reject",
            structure_masking="none",
        )
        metadata["max_length_policy"] = "configured_memory_guard"
        if self.config.scoring_window == "centered":
            metadata.update(
                scoring_recipe_version=2, scoring_window="centered", truncation="centered",
                window_policy="centered_clamped", window_indexing="zero_based_half_open",
                max_length_policy="residue_window", structure_window="aligned_full_reference_tokens",
            )
        return metadata

    def load(self):
        super().load()
        if self._tokenizer.mask_token_id != self._model.config.mask_token_id:
            raise ValueError("ProSST model and tokenizer disagree on the mask token")

    def iter_scores(self, reference_sequence, sequences, *, batch_size=8):
        # Validate full-reference alignment before slicing either input stream.
        self._validate_sequences([reference_sequence], windowed=self.config.scoring_window == "centered")
        yield from super().iter_scores(reference_sequence, sequences, batch_size=batch_size)

    def _reference_tokens(self, reference_sequence):
        result = super()._reference_tokens(reference_sequence)
        input_ids, _, residue_positions, _ = result
        if (input_ids[0, 0].item() != self._tokenizer.cls_token_id
            or input_ids[0, -1].item() != self._tokenizer.eos_token_id
            or residue_positions.tolist() != list(range(1, len(reference_sequence) + 1))):
            raise ValueError("ProSST amino-acid and structure tokens are not aligned")
        return result

    def _masked_forward(self, input_ids, attention_mask):
        return self._masked_window_forward(
            input_ids, attention_mask, [(0, self._structure_tokens.numel())] * input_ids.shape[0],
        )

    def _masked_window_forward(self, input_ids, attention_mask, bounds):
        structure_ids = self._structure_window_ids(bounds)
        if structure_ids.shape != input_ids.shape:
            raise ValueError("ProSST masked sequence and structure windows must align")
        return self._model(
            input_ids=input_ids, attention_mask=attention_mask,
            ss_input_ids=structure_ids.to(self._device),
            return_dict=True,
        )
