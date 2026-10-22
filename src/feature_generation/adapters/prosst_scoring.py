"""ProSST masked-WT log odds conditioned on the fixed reference backbone."""

import torch

from .masked_marginals import CANONICAL_ALPHABET, MaskedMarginalScoring
from .prosst import ProSSTAdapter


class ProSSTScoreAdapter(MaskedMarginalScoring, ProSSTAdapter):
    def score_metadata(self):
        metadata = self.metadata()
        for name in ("checkpoint", "layer", "embedding_dim", "hidden_size", "pooling",
                     "includes_special_tokens", "amino_acid_masking"):
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
        return metadata

    def load(self):
        super().load()
        if self._tokenizer.mask_token_id != self._model.config.mask_token_id:
            raise ValueError("ProSST model and tokenizer disagree on the mask token")

    def iter_scores(self, reference_sequence, sequences, *, batch_size=8):
        # Validate full-reference alignment and memory guards before model I/O.
        list(self._batches([reference_sequence]))
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
        structure_ids = torch.cat((torch.tensor([1]), self._structure_tokens + 3, torch.tensor([2])))
        return self._model(
            input_ids=input_ids, attention_mask=attention_mask,
            ss_input_ids=structure_ids.unsqueeze(0).expand(input_ids.shape[0], -1).to(self._device),
            return_dict=True,
        )
