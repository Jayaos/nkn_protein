"""Centered masked-marginal scoring retains WT context and original sites."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from feature_generation.adapters.esm2_scoring import ESM2ScoreAdapter
from tests.test_esm2_scoring import VOCABULARY, adapter, scores


def mutated(reference, changes):
    sequence = list(reference)
    for position, amino_acid in changes.items():
        sequence[position - 1] = amino_acid
    return "".join(sequence)


class CenteredScoringTests(unittest.TestCase):
    def scorer(self, **kwargs):
        kwargs.setdefault("max_length", 6)
        kwargs.setdefault("max_tokens", 24)
        return adapter(scoring_window="centered", **kwargs)

    def test_windows_clamp_at_ends_and_mask_exactly_one_wild_type_site(self):
        reference = "ACDEFGHIKLMN"
        changes = {1: "V", 6: "D", 12: "K"}
        value = self.scorer()
        sequences = [reference, mutated(reference, changes)]
        result = scores(value, sequences, reference, batch_size=8)
        self.assertEqual(result[0], 0)
        windows = [reference[:6], reference[2:8], reference[6:]]
        self.assertEqual([call[0][0] for call in value._tokenizer.calls], windows)
        self.assertEqual(len(value._model.calls), 1)
        ids, attention = value._model.calls[0]
        self.assertEqual(ids.shape, (3, 8))
        torch.testing.assert_close(attention, torch.ones_like(attention))
        for row, window, local_position in zip(ids, windows, (1, 4, 6)):
            expected = torch.tensor([0, *value._tokenizer.convert_tokens_to_ids(window), 2])
            expected[local_position] = value._tokenizer.mask_token_id
            torch.testing.assert_close(row, expected)
            self.assertEqual(int((row == value._tokenizer.mask_token_id).sum()), 1)
            self.assertEqual(int(row[0]), 0)
            self.assertEqual(int(row[-1]), 2)

    def test_distant_sites_are_evaluated_once_and_scores_remain_additive(self):
        reference = "ACDEFGHIKLMNPQRSTVWY" * 150
        first = mutated(reference, {1000: "V"})
        second = mutated(reference, {2300: "K"})
        both = mutated(reference, {1000: "V", 2300: "K"})
        value = self.scorer(max_length=1022, max_tokens=2048)
        sequences = [reference, first, second, both, mutated(reference, {2300: "D"})]
        batches = list(value.iter_scores(reference, sequences, batch_size=2))
        result = torch.cat([batch for _, batch in batches])
        self.assertEqual([indices for indices, _ in batches], [[0, 1], [2, 3], [4]])
        self.assertEqual([call[0][0] for call in value._tokenizer.calls], [
            reference[488:1510], reference[1788:2810],
        ])
        self.assertEqual(len(value._model.calls), 1)
        ids, _ = value._model.calls[0]
        self.assertEqual(ids.shape, (2, 1024))
        self.assertTrue(torch.all(ids[:, 512] == value._tokenizer.mask_token_id))
        self.assertTrue(torch.all((ids == value._tokenizer.mask_token_id).sum(dim=1) == 1))
        torch.testing.assert_close(result[3], result[1] + result[2], rtol=0, atol=0)
        self.assertEqual(result[0], 0)
        logits = value._model.outputs[0].float().log_softmax(-1)
        for result_index, window_row, target in ((1, 0, "V"), (2, 1, "K"), (4, 1, "D")):
            expected = logits[window_row, 512, VOCABULARY.index(target)] - logits[window_row, 512, VOCABULARY.index("Y")]
            torch.testing.assert_close(result[result_index], expected, rtol=0, atol=0)

    def test_nearby_mutations_do_not_enter_each_others_wild_type_context(self):
        reference = "ACDEFGHIKLMN"
        first = mutated(reference, {6: "D"})
        second = mutated(reference, {7: "V"})
        both = mutated(reference, {6: "D", 7: "V"})
        value = self.scorer()
        result = scores(value, [first, second, both], reference, batch_size=8)
        torch.testing.assert_close(result[2], result[0] + result[1], rtol=0, atol=0)
        rows = value._model.calls[0][0]
        for row, wild_type_window in zip(rows, (reference[2:8], reference[3:9])):
            expected = torch.tensor([0, *value._tokenizer.convert_tokens_to_ids(wild_type_window), 2])
            expected[4] = value._tokenizer.mask_token_id
            torch.testing.assert_close(row, expected)
        # The second target is still WT H in the first target's window.
        self.assertEqual(int(rows[0, 5]), VOCABULARY.index("H"))
        self.assertEqual(int(rows[1, 3]), VOCABULARY.index("G"))

    def test_short_sequences_and_exact_window_length_match_legacy_scores(self):
        for reference in ("ACD", "ACDEFG"):
            sequences = [reference, mutated(reference, {1: "V"}), mutated(reference, {len(reference): "K"}),
                         mutated(reference, {1: "V", len(reference): "K"})]
            with self.subTest(length=len(reference)):
                legacy = adapter(max_length=6, max_tokens=24)
                centered = self.scorer()
                torch.testing.assert_close(scores(legacy, sequences, reference), scores(centered, sequences, reference), rtol=0, atol=0)
                self.assertEqual(len(centered._tokenizer.calls), 1)
                self.assertEqual(centered._tokenizer.calls[0][0], [reference])
                for (left, _), (right, _) in zip(legacy._model.calls, centered._model.calls):
                    torch.testing.assert_close(left, right)

    def test_1023_tail_uses_shifted_window_and_keeps_original_residue_identity(self):
        reference = "A" * 1022 + "C"
        value = self.scorer(max_length=1022, max_tokens=1024)
        result = scores(value, [mutated(reference, {1023: "K"})], reference)
        self.assertEqual(value._tokenizer.calls[0][0], [reference[1:]])
        ids, _ = value._model.calls[0]
        self.assertEqual(ids.shape, (1, 1024))
        self.assertEqual(int(ids[0, 1022]), value._tokenizer.mask_token_id)
        self.assertEqual(int(ids[0, 1023]), 2)
        log_probs = value._model.outputs[0].log_softmax(-1)
        expected = log_probs[0, 1022, VOCABULARY.index("K")] - log_probs[0, 1022, VOCABULARY.index("C")]
        torch.testing.assert_close(result[0], expected, rtol=0, atol=0)

    def test_token_budget_and_requested_batch_size_do_not_change_scores(self):
        reference = "ACDEFGHIKLMN"
        sequences = [mutated(reference, {1: "V"}), mutated(reference, {6: "D"}),
                     mutated(reference, {12: "K"}), mutated(reference, {1: "V", 6: "D", 12: "K"})]
        singleton = self.scorer(max_tokens=24)
        limited = self.scorer(max_tokens=16)
        wide = self.scorer(max_tokens=24)
        expected = scores(singleton, sequences, reference, batch_size=1)
        torch.testing.assert_close(scores(limited, sequences, reference, batch_size=8), expected, rtol=0, atol=0)
        torch.testing.assert_close(scores(wide, sequences, reference, batch_size=8), expected, rtol=0, atol=0)
        self.assertEqual([ids.shape[0] for ids, _ in singleton._model.calls], [1, 1, 1])
        self.assertEqual([ids.shape[0] for ids, _ in limited._model.calls], [2, 1])
        self.assertEqual([ids.shape[0] for ids, _ in wide._model.calls], [3])
        self.assertTrue(all(ids.numel() <= 16 for ids, _ in limited._model.calls))

    def test_full_vocabulary_normalization_precedes_canonical_selection(self):
        reference = "ACDEFGHIKLMN"
        value = self.scorer()
        observed_widths = []
        original_softmax = torch.Tensor.log_softmax

        def capture_softmax(tensor, *args, **kwargs):
            observed_widths.append(tensor.shape[-1])
            return original_softmax(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "log_softmax", capture_softmax):
            result = scores(value, [mutated(reference, {6: "D"})], reference)
        self.assertEqual(observed_widths, [len(VOCABULARY)])
        logits = value._model.outputs[0][0, 4].float()
        native = logits.log_softmax(-1)
        self.assertEqual(int(native.argmax()), VOCABULARY.index("X"))
        expected = native[VOCABULARY.index("D")] - native[VOCABULARY.index("G")]
        torch.testing.assert_close(result[0], expected, rtol=0, atol=0)

    def test_long_wild_type_only_has_zero_scores_and_no_model_passes(self):
        reference = "A" * 3423
        value = self.scorer(max_length=1022, max_tokens=1024)
        torch.testing.assert_close(scores(value, [reference, reference], reference), torch.zeros(2))
        self.assertEqual(value._model.calls, [])
        self.assertEqual(value._tokenizer.calls, [])

    def test_all_sequences_and_budget_are_validated_before_weights_load(self):
        reference = "ACDEFGHIKLMN"
        for sequences in ([reference + "A"], [reference[:-1] + "X"], [reference, None]):
            value = self.scorer()
            with self.subTest(sequences=sequences), self.assertRaises(ValueError):
                scores(value, sequences, reference)
            value.load.assert_not_called()
        value = self.scorer(max_tokens=7)
        with self.assertRaisesRegex(ValueError, "BOS/EOS"):
            scores(value, [mutated(reference, {12: "K"})], reference)
        value.load.assert_not_called()
        for corruption in ("residues", "special", "shape", "padding"):
            value = self.scorer()
            value._tokenizer.corrupt = corruption
            with self.subTest(corruption=corruption), self.assertRaisesRegex(ValueError, "Tokenizer"):
                scores(value, [mutated(reference, {12: "K"})], reference)
            self.assertFalse(value._model.calls)

    def test_centered_recipe_is_distinct_and_embedding_choices_do_not_change_it(self):
        centered = self.scorer()
        legacy = ESM2ScoreAdapter(replace(centered.config, scoring_window="full"))
        recipes = []
        for value in (legacy, centered):
            value._manifest = {"files_sha256": {"model.safetensors": "a" * 64}}
            with patch.object(value, "_checkpoint"), patch("feature_generation.adapters.esm2.version", return_value="test"):
                recipes.append(value.score_metadata())
        self.assertEqual(recipes[0]["scoring_recipe_version"], 1)
        self.assertEqual(recipes[0]["truncation"], "reject")
        self.assertNotIn("scoring_window", recipes[0])
        changed = {
            "scoring_recipe_version": 2, "scoring_window": "centered", "truncation": "centered",
            "window_policy": "centered_clamped", "window_indexing": "zero_based_half_open",
        }
        self.assertEqual(recipes[1], {**recipes[0], **changed})
        compact = ESM2ScoreAdapter(replace(centered.config, pooling="mean", truncation="prefix", site_embeddings="centered"))
        compact._manifest = centered._manifest
        with patch.object(compact, "_checkpoint"), patch("feature_generation.adapters.esm2.version", return_value="test"):
            self.assertEqual(compact.score_metadata(), recipes[1])


if __name__ == "__main__":
    unittest.main()
