"""Full-sequence means use disjoint bounded windows with residue weighting."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from feature_generation.adapters.esm2 import ESM2Adapter
from feature_generation.adapters.esm2_scoring import ESM2ScoreAdapter
from feature_generation.config import ExtractionConfig
from tests.test_esm2_centered_adapter import ContextModel, WindowTokenizer, expected_residues
from tests.test_feature_generation import FakeTokenizer


def expected_global(sequence, width, layer=6):
    vectors = torch.cat([
        expected_residues(sequence[start:start + width], layer)
        for start in range(0, len(sequence), width)
    ])
    return vectors.double().mean(dim=0).float()


class NonOverlappingAdapterTests(unittest.TestCase):
    def adapter(self, **overrides):
        values = dict(
            model_id="facebook/esm2_t6_8M_UR50D", device="cpu", pooling="mean",
            global_window="non_overlapping", truncation="none", site_embeddings="centered",
            max_length=4, max_tokens_per_batch=12,
        )
        values.update(overrides)
        value = ESM2Adapter(ExtractionConfig(**values))
        value._model, value._tokenizer = ContextModel(), WindowTokenizer()
        value._device = torch.device("cpu")
        return value

    def test_global_windows_cover_each_residue_once_and_weight_short_final_chunk(self):
        sequence = "ACDEFGHIKL"
        value = self.adapter()
        index, global_vector, sites = list(value.iter_site_embeddings([sequence], [()]))[0]
        windows = [window for batch, _ in value._tokenizer.calls for window in batch]
        self.assertEqual(windows, ["ACDE", "FGHI", "KL"])
        self.assertEqual("".join(windows), sequence)
        self.assertEqual(index, 0)
        self.assertEqual(sites.shape, (0, 320))
        torch.testing.assert_close(global_vector, expected_global(sequence, 4), rtol=0, atol=0)
        incorrectly_unweighted = torch.stack([expected_residues(window).mean(dim=0) for window in windows]).mean(dim=0)
        self.assertFalse(torch.allclose(global_vector, incorrectly_unweighted))
        self.assertEqual([shape for shape, _ in value._model.calls], [(2, 6), (1, 4)])

    def test_site_windows_share_exact_global_bounds_but_do_not_change_global_mean(self):
        sequence = "ACDEFGHIKL"
        value = self.adapter()
        _, global_vector, sites = list(value.iter_site_embeddings([sequence], [(1, 6, 10)]))[0]
        windows = [window for batch, _ in value._tokenizer.calls for window in batch]
        self.assertEqual(windows, [sequence[:4], sequence[4:8], sequence[8:], sequence[3:7], sequence[6:10]])
        self.assertEqual(windows.count(sequence[:4]), 1)
        expected_sites = torch.stack([
            expected_residues(sequence[:4])[0], expected_residues(sequence[3:7])[2],
            expected_residues(sequence[6:10])[3],
        ])
        torch.testing.assert_close(sites, expected_sites, rtol=0, atol=0)
        torch.testing.assert_close(global_vector, expected_global(sequence, 4), rtol=0, atol=0)
        without_sites = self.adapter()
        no_site_global = list(without_sites.iter_site_embeddings([sequence], [()]))[0][1]
        torch.testing.assert_close(global_vector, no_site_global, rtol=0, atol=0)

    def test_tail_mutation_affects_full_global_but_not_legacy_prefix_global(self):
        sequences = ["A" * 8 + "CC", "A" * 8 + "CK"]
        value = self.adapter()
        full = list(value.iter_site_embeddings(sequences, [(), (10,)]))
        self.assertFalse(torch.equal(full[0][1], full[1][1]))
        for output, sequence in zip(full, sequences):
            torch.testing.assert_close(output[1], expected_global(sequence, 4), rtol=0, atol=0)
        prefix = self.adapter(global_window="prefix", truncation="prefix")
        legacy = list(prefix.iter_site_embeddings(sequences, [(), (10,)]))
        torch.testing.assert_close(legacy[0][1], legacy[1][1], rtol=0, atol=0)
        torch.testing.assert_close(full[1][2], legacy[1][2], rtol=0, atol=0)

    def test_short_sequences_match_prefix_mode_exactly_and_reuse_site_input(self):
        for sequence in ("ACD", "ACDE"):
            for layer in (0, 3, 6):
                with self.subTest(sequence=sequence, layer=layer):
                    value = self.adapter(layer=layer)
                    prefix = self.adapter(layer=layer, global_window="prefix", truncation="prefix")
                    positions = [(1, len(sequence))]
                    full_result = list(value.iter_site_embeddings([sequence], positions))[0]
                    prefix_result = list(prefix.iter_site_embeddings([sequence], positions))[0]
                    torch.testing.assert_close(full_result[1], prefix_result[1], rtol=0, atol=0)
                    torch.testing.assert_close(full_result[2], prefix_result[2], rtol=0, atol=0)
                    self.assertEqual([call[0] for call in value._tokenizer.calls], [[sequence]])

    def test_1023_boundary_uses_single_tail_residue_for_global_and_centered_tail_site(self):
        sequence = "A" * 1022 + "V"
        value = self.adapter(max_length=1022, max_tokens_per_batch=1024)
        output = list(value.iter_site_embeddings([sequence], [(1023,)]))[0]
        self.assertEqual([call[0][0] for call in value._tokenizer.calls], [sequence[:1022], "V", sequence[1:]])
        torch.testing.assert_close(output[1], expected_global(sequence, 1022), rtol=0, atol=0)
        torch.testing.assert_close(output[2][0], expected_residues(sequence[1:])[-1], rtol=0, atol=0)
        self.assertEqual([shape for shape, _ in value._model.calls], [(1, 1024), (1, 3), (1, 1024)])

    def test_variants_preserve_order_and_every_model_batch_obeys_token_budget(self):
        sequences = ["ACDEFGHIKL", "VCDEFGHIKL", "ACDEFGHIKV"]
        positions = [(), (1,), (10,)]
        value = self.adapter()
        outputs = list(value.iter_site_embeddings(sequences, positions))
        self.assertEqual([index for index, _, _ in outputs], [0, 1, 2])
        for output, sequence, requested_sites in zip(outputs, sequences, positions):
            torch.testing.assert_close(output[1], expected_global(sequence, 4), rtol=0, atol=0)
            self.assertEqual(output[2].shape, (len(requested_sites), 320))
            for tensor in output[1:]:
                self.assertEqual(tensor.dtype, torch.float32)
                self.assertEqual(tensor.device.type, "cpu")
                self.assertFalse(tensor.requires_grad)
                self.assertTrue(torch.isfinite(tensor).all())
        for (count, tokens), grad_enabled in value._model.calls:
            self.assertLessEqual(tokens, 6)
            self.assertLessEqual(count * tokens, 12)
            self.assertFalse(grad_enabled)
        for windows, kwargs in value._tokenizer.calls:
            self.assertEqual(len(set(map(len, windows))), 1)
            self.assertFalse(kwargs["truncation"])

    def test_special_tokens_and_padding_never_contribute_to_global_sum(self):
        sequence = "ACDEFGHIKL"
        value = self.adapter(max_tokens_per_batch=14)
        value._tokenizer = FakeTokenizer()  # Extra non-special padding tests the attention mask.
        output = list(value.iter_site_embeddings([sequence], [()]))[0]
        torch.testing.assert_close(output[1], expected_global(sequence, 4), rtol=0, atol=0)
        self.assertEqual([shape for shape, _ in value._model.calls], [(2, 7), (1, 5)])

    def test_global_only_iterator_uses_the_same_full_sequence_recipe(self):
        sequences = ["ACDEFGHIKL", "VCDEFGHIKL"]
        value = self.adapter(site_embeddings="none")
        outputs = list(value.iter_embeddings(sequences))
        self.assertEqual([indices for indices, _ in outputs], [[0], [1]])
        torch.testing.assert_close(torch.cat([tensor for _, tensor in outputs]), torch.stack([
            expected_global(sequence, 4) for sequence in sequences
        ]), rtol=0, atol=0)
        self.assertEqual([window for batch, _ in value._tokenizer.calls for window in batch], [
            "ACDE", "FGHI", "KL", "VCDE", "FGHI", "KL",
        ])

    def test_new_config_identity_and_metadata_are_explicit(self):
        legacy = ExtractionConfig()
        full = ExtractionConfig(pooling="mean", truncation="none", global_window="non_overlapping")
        compact = ExtractionConfig(pooling="mean", truncation="none", global_window="non_overlapping", site_embeddings="centered")
        self.assertEqual(full.cache_id, legacy.cache_id + "-mean-all-nonoverlap1022")
        self.assertEqual(compact.cache_id, full.cache_id + "-centered-sites")
        for config in (legacy, full, compact):
            value = ESM2Adapter(config)
            value._manifest = {"files_sha256": {"model.safetensors": "a" * 64}}
            with patch.object(value, "_checkpoint"), patch("feature_generation.adapters.esm2.version", return_value="test"):
                metadata = value.metadata()
            if config.global_window == "prefix":
                self.assertNotIn("global_window", metadata)
                self.assertNotIn("global_window_aggregation", metadata)
            else:
                self.assertEqual(metadata["global_window"], "non_overlapping")
                self.assertEqual(metadata["global_window_aggregation"], "length_weighted_mean")
                self.assertEqual(metadata["truncation"], "none")
        for overrides in (
            {"global_window": "overlapping"}, {"global_window": True},
            {"truncation": "none"}, {"global_window": "non_overlapping"},
            {"global_window": "non_overlapping", "pooling": "mean", "truncation": "prefix"},
            {"global_window": "non_overlapping", "pooling": "none", "truncation": "none"},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                ExtractionConfig(**overrides)

    def test_full_and_centered_score_recipes_are_unchanged_by_global_window_policy(self):
        full_embedding = self.adapter().config
        for scoring_window in ("full", "centered"):
            with self.subTest(scoring_window=scoring_window):
                configs = [
                    replace(full_embedding, scoring_window=scoring_window),
                    replace(full_embedding, scoring_window=scoring_window, global_window="prefix", truncation="prefix"),
                ]
                recipes = []
                for config in configs:
                    value = ESM2ScoreAdapter(config)
                    value._manifest = {"files_sha256": {"model.safetensors": "a" * 64}}
                    with patch.object(value, "_checkpoint"), patch("feature_generation.adapters.esm2.version", return_value="test"):
                        recipes.append(value.score_metadata())
                self.assertEqual(recipes[0], recipes[1])
                self.assertNotIn("global_window", recipes[0])
                self.assertNotIn("global_window_aggregation", recipes[0])


if __name__ == "__main__":
    unittest.main()
