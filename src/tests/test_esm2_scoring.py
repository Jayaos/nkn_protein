"""Masked-marginal ESM2 score semantics with independent token/model fixtures."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from feature_generation.adapters.esm2_scoring import CANONICAL_ALPHABET, ESM2ScoreAdapter
from feature_generation.config import ExtractionConfig


VOCABULARY = [
    "<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P",
    "K", "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>",
]


class Tokenizer:
    mask_token_id = 32
    mask_token = "<mask>"
    all_special_ids = [0, 1, 2, 3, 32]

    def __init__(self):
        self.calls = []
        self.corrupt = None

    def convert_tokens_to_ids(self, tokens):
        return [VOCABULARY.index(token) for token in tokens]

    def convert_ids_to_tokens(self, indices):
        return [VOCABULARY[index] for index in indices]

    def __call__(self, sequences, **kwargs):
        self.calls.append((list(sequences), kwargs))
        ids = torch.tensor([[0, *self.convert_tokens_to_ids(sequence), 2] for sequence in sequences])
        special = torch.zeros_like(ids)
        special[:, [0, -1]] = 1
        attention = torch.ones_like(ids)
        if self.corrupt == "residues":
            ids[:, 1] = VOCABULARY.index("W")
        elif self.corrupt == "special":
            special[:, 1] = 1
        elif self.corrupt == "shape":
            attention = attention[:, :-1]
        elif self.corrupt == "padding":
            attention[:, -1] = 0
        return {"input_ids": ids, "attention_mask": attention, "special_tokens_mask": special}


class MaskedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.calls = []
        self.outputs = []
        self.config = SimpleNamespace(hidden_size=320, num_hidden_layers=6, vocab_size=33, mask_token_id=32)

    def forward(self, input_ids, attention_mask, return_dict):
        assert not torch.is_grad_enabled()
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        # Sequence context and masked position both affect the target logits.
        context = (input_ids * torch.arange(1, input_ids.shape[1] + 1, device=input_ids.device)).sum(-1)
        tokens = torch.arange(33, dtype=torch.float32, device=input_ids.device)
        positions = torch.arange(input_ids.shape[1], dtype=torch.float32, device=input_ids.device)
        logits = (context[:, None, None] % 17 + positions[None, :, None]) * tokens[None, None, :] / 61
        logits[..., 24] = 30  # Noncanonical X dominates the softmax normalization.
        self.outputs.append(logits.clone())
        return SimpleNamespace(logits=logits)


def adapter(*, max_tokens=20, model=None, **kwargs):
    value = ESM2ScoreAdapter(ExtractionConfig(
        model_id="facebook/esm2_t6_8M_UR50D", device="cpu", max_tokens_per_batch=max_tokens, **kwargs,
    ))
    value._device = torch.device("cpu")
    value._tokenizer = Tokenizer()
    value._model = model if model is not None else MaskedModel()
    value.load = Mock()
    return value


def scores(value, sequences, reference="ACD", **kwargs):
    return torch.cat([batch for _, batch in value.iter_scores(reference, sequences, **kwargs)])


class ESM2ScoringTests(unittest.TestCase):
    def test_masked_marginal_formula_wild_type_and_multiple_mutations(self):
        value = adapter()
        batches = list(value.iter_scores("ACD", ["ACD", "CCD", "AED", "CED", "ACY"], batch_size=3))
        result = torch.cat([batch for _, batch in batches])
        self.assertEqual([indices for indices, _ in batches], [[0, 1, 2], [3, 4]])
        self.assertEqual(result[0].item(), 0)
        # Site 0/1/2 each appears once in the only model call. Expected scores
        # follow the publication's individual full-vocabulary log-softmax ratios.
        log_probs = value._model.outputs[0].float().log_softmax(-1)
        first = log_probs[0, 1, VOCABULARY.index("C")] - log_probs[0, 1, VOCABULARY.index("A")]
        second = log_probs[1, 2, VOCABULARY.index("E")] - log_probs[1, 2, VOCABULARY.index("C")]
        third = log_probs[2, 3, VOCABULARY.index("Y")] - log_probs[2, 3, VOCABULARY.index("D")]
        torch.testing.assert_close(result, torch.stack([first * 0, first, second, first + second, third]), rtol=0, atol=0)
        for _, batch in batches:
            self.assertEqual(batch.dtype, torch.float32)
            self.assertEqual(batch.device.type, "cpu")
            self.assertFalse(batch.requires_grad)
            self.assertTrue(batch.is_contiguous())

    def test_only_unique_mutated_sites_are_masked_in_wild_type_context(self):
        value = adapter(max_tokens=10)
        sequences = ["ACD", "CCD", "AED", "CED", "ACY", "WCD"]
        scores(value, sequences, batch_size=8)
        calls = value._model.calls
        self.assertEqual([tuple(ids.shape) for ids, _ in calls], [(2, 5), (1, 5)])
        expected_wt = torch.tensor([0, 5, 23, 13, 2])
        site = 1
        for ids, attention in calls:
            self.assertTrue(torch.all(attention == 1))
            for row in ids:
                self.assertEqual(int((row == 32).sum()), 1)
                self.assertEqual(row[site], 32)
                repaired = row.clone()
                repaired[site] = expected_wt[site]
                torch.testing.assert_close(repaired, expected_wt)
                site += 1
        self.assertEqual(value._tokenizer.calls[0][0], ["ACD"])
        self.assertFalse(value._tokenizer.calls[0][1]["truncation"])

    def test_batch_size_limit_and_token_budget_produce_same_scores(self):
        first, second = adapter(max_tokens=20), adapter(max_tokens=10)
        sequences = ["WCD", "AED", "ACY", "WEY"]
        torch.testing.assert_close(scores(first, sequences, batch_size=1), scores(second, sequences, batch_size=8), rtol=0, atol=0)
        self.assertEqual(len(first._model.calls), 3)
        self.assertEqual(len(second._model.calls), 2)

    def test_only_wild_type_needs_no_forward_pass(self):
        value = adapter()
        torch.testing.assert_close(scores(value, ["ACD", "ACD"]), torch.zeros(2))
        self.assertEqual(value._model.calls, [])

    def test_invalid_inputs_fail_before_loading(self):
        for bad in ([], None, "ACD", [None], ["AC"], ["ACDE"], ["ACX"], ["acd"], ["ACD", "A-D"]):
            value = adapter()
            with self.subTest(sequences=bad), self.assertRaises(ValueError):
                scores(value, bad)
            value.load.assert_not_called()
        for ref in (None, "", "ACX", "acd"):
            value = adapter()
            with self.subTest(reference=ref), self.assertRaises(ValueError):
                scores(value, ["ACD"], reference=ref)
            value.load.assert_not_called()
        for count in (0, True, 1.5):
            value = adapter()
            with self.subTest(batch_size=count), self.assertRaisesRegex(ValueError, "batch_size"):
                scores(value, ["ACD"], batch_size=count)
            value.load.assert_not_called()
        for value, message in ((adapter(max_tokens=4), "BOS/EOS"), (adapter(max_length=2), "max_length")):
            with self.assertRaisesRegex(ValueError, message):
                scores(value, ["ACD"])
            value.load.assert_not_called()

    def test_tokenizer_alignment_and_mask_identity_fail_before_forward(self):
        for corruption in ("residues", "special", "shape", "padding"):
            value = adapter()
            value._tokenizer.corrupt = corruption
            with self.subTest(corruption=corruption), self.assertRaisesRegex(ValueError, "Tokenizer"):
                scores(value, ["WCD"])
            self.assertFalse(value._model.calls)
        for mask_id in (None, 5, 99):
            value = adapter()
            value._tokenizer.mask_token_id = mask_id
            with self.subTest(mask=mask_id), self.assertRaisesRegex(ValueError, "mask token"):
                scores(value, ["WCD"])
            self.assertFalse(value._model.calls)
        value = adapter()
        value._tokenizer.convert_tokens_to_ids = lambda tokens: [5] * len(tokens)
        with self.assertRaises(ValueError):
            scores(value, ["WCD"])
        self.assertFalse(value._model.calls)

    def test_bad_model_outputs_are_rejected(self):
        for logits in (torch.zeros(1, 5, 20), torch.zeros(1, 4, 33), torch.full((1, 5, 33), float("nan")),
                       torch.full((1, 5, 33), float("inf"))):
            value = adapter()
            value._model.forward = Mock(return_value=SimpleNamespace(logits=logits))
            with self.subTest(shape=tuple(logits.shape)), self.assertRaisesRegex(ValueError, "logits"):
                scores(value, ["WCD"])

    def test_loading_is_offline_frozen_and_preserves_rng(self):
        value = ESM2ScoreAdapter(ExtractionConfig(model_id="facebook/esm2_t6_8M_UR50D", device="cpu"))
        model, tokenizer = MaskedModel(), Tokenizer()
        def construct(*args, **kwargs):
            torch.rand(9)  # Constructor would otherwise advance the global RNG.
            return model, {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
        hf = SimpleNamespace(EsmForMaskedLM=SimpleNamespace(from_pretrained=Mock(side_effect=construct)),
                             EsmTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)))
        rng = torch.get_rng_state().clone()
        with patch.object(value, "_checkpoint", return_value=Path("verified")), patch.dict(sys.modules, {"transformers": hf}):
            value.load()
            value.load()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(hf.EsmForMaskedLM.from_pretrained.call_count, 1)
        options = hf.EsmForMaskedLM.from_pretrained.call_args.kwargs
        self.assertTrue(options["local_files_only"])
        self.assertTrue(options["use_safetensors"])
        self.assertTrue(options["output_loading_info"])
        self.assertEqual(options["torch_dtype"], torch.float32)

    def test_missing_head_mismatched_weights_and_wrong_architecture_are_rejected(self):
        issues = [
            {"missing_keys": ["lm_head.dense.weight"]},
            {"mismatched_keys": [("lm_head.bias", (32,), (33,))]},
            {"unexpected_keys": ["unknown.weight"]}, {"error_msgs": ["bad checkpoint"]},
        ]
        for info in issues:
            value = ESM2ScoreAdapter(ExtractionConfig(model_id="facebook/esm2_t6_8M_UR50D", device="cpu"))
            hf = SimpleNamespace(EsmForMaskedLM=SimpleNamespace(from_pretrained=Mock(return_value=(MaskedModel(), info))),
                                 EsmTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=Tokenizer())))
            with self.subTest(info=info), patch.object(value, "_checkpoint", return_value=Path("verified")), patch.dict(sys.modules, {"transformers": hf}):
                with self.assertRaisesRegex(ValueError, "fully pretrained"):
                    value.load()
                self.assertIsNone(value._model)
        for field, bad in (("hidden_size", 42), ("num_hidden_layers", 2), ("mask_token_id", 31)):
            value = ESM2ScoreAdapter(ExtractionConfig(model_id="facebook/esm2_t6_8M_UR50D", device="cpu"))
            model = MaskedModel()
            setattr(model.config, field, bad)
            hf = SimpleNamespace(EsmForMaskedLM=SimpleNamespace(from_pretrained=Mock(return_value=(model, {}))),
                                 EsmTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=Tokenizer())))
            with self.subTest(field=field), patch.object(value, "_checkpoint", return_value=Path("verified")), patch.dict(sys.modules, {"transformers": hf}):
                with self.assertRaises(ValueError):
                    value.load()

    def test_unavailable_device_and_incompatible_precision_fail_before_checkpoint_loading(self):
        for device, dtype, available, bf16 in (("cuda", "float32", False, False), ("cpu", "float16", False, False),
                                             ("cuda", "bfloat16", True, False)):
            value = ESM2ScoreAdapter(ExtractionConfig(model_id="facebook/esm2_t6_8M_UR50D", device=device, inference_dtype=dtype))
            with patch.object(value, "_checkpoint") as checkpoint, patch("torch.cuda.is_available", return_value=available), patch("torch.cuda.is_bf16_supported", return_value=bf16):
                with self.assertRaises(ValueError):
                    value.load()
                checkpoint.assert_not_called()

    def test_metadata_has_score_semantics_and_no_representation_layer(self):
        value = adapter()
        original = {"checkpoint": "embedding-only", "layer": 6, "embedding_dim": 320, "hidden_size": 320,
                    "pooling": "none", "includes_special_tokens": False, "model_class": "EsmModel",
                    "model_id": value.config.model_id, "revision": value.config.revision,
                    "checkpoint_files_sha256": {"model.safetensors": "verified"}, "inference_dtype": "float32", "max_length": 1022}
        with patch.object(value, "metadata", return_value=dict(original)):
            metadata = value.score_metadata()
        expected = dict(feature_type="scores", model_family="esm2", model_class="EsmForMaskedLM",
                        score_definition="sum_masked_marginal_log_odds", conditioning_mode="masked_wild_type",
                        inference_method="masked_marginals", normalization="model_vocabulary", output_dtype="float32",
                        residue_aggregation="sum_over_mutations", masking="one_wild_type_residue_at_a_time",
                        wild_type_subtraction=True, noncanonical_tokens="retained_in_normalization",
                        score_direction="higher_is_more_likely", scoring_recipe_version=1)
        for name, expected_value in expected.items():
            self.assertEqual(metadata[name], expected_value)
        for name in ("checkpoint", "layer", "embedding_dim", "hidden_size", "pooling", "includes_special_tokens"):
            self.assertNotIn(name, metadata)
        self.assertEqual(metadata["checkpoint_files_sha256"], original["checkpoint_files_sha256"])
        self.assertEqual(metadata["alphabet"], CANONICAL_ALPHABET)


if __name__ == "__main__":
    unittest.main()
