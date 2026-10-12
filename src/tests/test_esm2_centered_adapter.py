"""Centered site extraction uses bounded, unmasked variant windows."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from feature_generation.adapters.esm2 import ESM2Adapter
from feature_generation.adapters.esm2_scoring import ESM2ScoreAdapter
from feature_generation.config import ExtractionConfig


class WindowTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, sequences, **kwargs):
        self.calls.append((list(sequences), kwargs))
        ids = torch.tensor([[1, *map(ord, sequence), 2] for sequence in sequences])
        return {
            "input_ids": ids, "attention_mask": torch.ones_like(ids),
            "special_tokens_mask": ((ids == 1) | (ids == 2)).long(),
        }

    def convert_ids_to_tokens(self, ids):
        return list(map(chr, ids))


class ContextModel(torch.nn.Module):
    """Values expose residue identity, local position, context and layer."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_ids, attention_mask, output_hidden_states, return_dict):
        self.calls.append((tuple(input_ids.shape), torch.is_grad_enabled()))
        context = (input_ids * (input_ids > 2)).sum(dim=1, keepdim=True) * 10
        positions = torch.arange(input_ids.shape[1])
        base = (input_ids * 100 + positions + context).float().unsqueeze(-1).expand(-1, -1, 320)
        states = tuple(base + layer * 10000 for layer in range(7))
        return SimpleNamespace(last_hidden_state=states[-1], hidden_states=states)


def expected_residues(sequence, layer=6):
    context = sum(map(ord, sequence)) * 10
    return torch.tensor([
        [ord(aa) * 100 + position + 1 + context + layer * 10000] * 320
        for position, aa in enumerate(sequence)
    ], dtype=torch.float32)


class CenteredAdapterTests(unittest.TestCase):
    def adapter(self, **overrides):
        values = dict(
            model_id="facebook/esm2_t6_8M_UR50D", device="cpu",
            pooling="mean", truncation="prefix", site_embeddings="centered",
            max_length=1022, max_tokens_per_batch=1024,
        )
        values.update(overrides)
        adapter = ESM2Adapter(ExtractionConfig(**values))
        adapter._model, adapter._tokenizer = ContextModel(), WindowTokenizer()
        adapter._device = torch.device("cpu")
        return adapter

    def test_short_sequences_reuse_full_input_for_global_and_all_sites(self):
        sequences = ["ACDE", "VCDE", "ACDV", "VCDV"]
        positions = [(), (1,), (4,), (1, 4)]
        for layer in (0, 3, 6):
            with self.subTest(layer=layer):
                adapter = self.adapter(layer=layer, max_tokens_per_batch=18)
                outputs = list(adapter.iter_site_embeddings(sequences, positions))
                self.assertEqual([index for index, _, _ in outputs], [0, 1, 2, 3])
                self.assertEqual([len(call[0]) for call in adapter._tokenizer.calls], [3, 1])
                for (index, global_vector, site_vectors), sequence, sites in zip(outputs, sequences, positions):
                    residue_vectors = expected_residues(sequence, layer)
                    torch.testing.assert_close(global_vector, residue_vectors.mean(dim=0))
                    torch.testing.assert_close(site_vectors, residue_vectors[[p - 1 for p in sites]])
                    self.assertEqual(tuple(global_vector.shape), (320,))
                    self.assertEqual(tuple(site_vectors.shape), (len(sites), 320))
                    for tensor in (global_vector, site_vectors):
                        self.assertEqual(tensor.dtype, torch.float32)
                        self.assertEqual(tensor.device.type, "cpu")
                        self.assertFalse(tensor.requires_grad)
                self.assertTrue(all(not grad_enabled for _, grad_enabled in adapter._model.calls))

    def test_distant_mutations_get_exact_centered_variant_windows(self):
        sequence = "A" * 999 + "V" + "A" * 1299 + "K" + "A" * 700
        adapter = self.adapter()
        index, global_vector, site_vectors = list(adapter.iter_site_embeddings([sequence], [(1000, 2300)]))[0]
        self.assertEqual(index, 0)
        windows = [call[0][0] for call in adapter._tokenizer.calls]
        self.assertEqual(windows, [sequence[:1022], sequence[488:1510], sequence[1788:2810]])
        self.assertEqual(windows[1][511], "V")
        self.assertEqual(windows[2][511], "K")
        self.assertNotIn("K", windows[1])
        self.assertNotIn("V", windows[2])
        torch.testing.assert_close(global_vector, expected_residues(sequence[:1022]).mean(dim=0))
        torch.testing.assert_close(site_vectors, torch.stack([
            expected_residues(windows[1])[511], expected_residues(windows[2])[511],
        ]))
        self.assertEqual([shape for shape, _ in adapter._model.calls], [(1, 1024)] * 3)

    def test_both_sequence_ends_clamp_and_identical_windows_are_reused(self):
        sequence = "VC" + "A" * 1496 + "DE"
        positions = (1500, 1, 2, 1499)
        adapter = self.adapter(max_tokens_per_batch=2048)
        _, global_vector, site_vectors = list(adapter.iter_site_embeddings([sequence], [positions]))[0]
        self.assertEqual(len(adapter._tokenizer.calls), 1)
        self.assertEqual(adapter._tokenizer.calls[0][0], [sequence[:1022], sequence[-1022:]])
        prefix, suffix = expected_residues(sequence[:1022]), expected_residues(sequence[-1022:])
        torch.testing.assert_close(global_vector, prefix.mean(dim=0))
        torch.testing.assert_close(site_vectors, torch.stack([suffix[-1], prefix[0], prefix[1], suffix[-2]]))

    def test_1022_and_1023_boundary_keeps_tail_site_but_prefix_mean_stays_same(self):
        adapter = self.adapter()
        short = "A" * 1021 + "C"
        short_result = list(adapter.iter_site_embeddings([short], [(1022,)]))[0]
        self.assertEqual([call[0] for call in adapter._tokenizer.calls], [[short]])
        torch.testing.assert_close(short_result[2][0], expected_residues(short)[-1])
        adapter = self.adapter()
        long_variants = [short + "D", short + "V"]
        outputs = list(adapter.iter_site_embeddings(long_variants, [(1023,), (1023,)]))
        self.assertEqual([call[0][0] for call in adapter._tokenizer.calls], [
            short, long_variants[0][1:], short, long_variants[1][1:],
        ])
        torch.testing.assert_close(outputs[0][1], outputs[1][1])
        self.assertFalse(torch.equal(outputs[0][2], outputs[1][2]))
        for output, sequence in zip(outputs, long_variants):
            torch.testing.assert_close(output[2][0], expected_residues(sequence[1:])[-1])

    def test_site_context_includes_other_mutations_inside_its_window(self):
        first = "A" * 999 + "V" + "A" * 2000
        nearby = first[:1049] + "D" + first[1050:]
        distant = first[:2299] + "K" + first[2300:]
        adapter = self.adapter()
        outputs = list(adapter.iter_site_embeddings([first, nearby, distant], [(1000,), (1000, 1050), (1000, 2300)]))
        self.assertFalse(torch.equal(outputs[0][2][0], outputs[1][2][0]))
        torch.testing.assert_close(outputs[0][2][0], outputs[2][2][0])
        windows = [sequence for sequences, _ in adapter._tokenizer.calls for sequence in sequences]
        self.assertIn(nearby[488:1510], windows)
        self.assertIn("D", nearby[488:1510])

    def test_wild_type_only_needs_prefix_and_empty_site_matrix(self):
        adapter = self.adapter()
        output = list(adapter.iter_site_embeddings(["A" * 3423], [()]))[0]
        self.assertEqual(output[2].shape, (0, 320))
        self.assertEqual(adapter._tokenizer.calls[0][0], ["A" * 1022])
        self.assertEqual(len(adapter._model.calls), 1)

    def test_full_sequence_and_positions_validate_before_loading_weights(self):
        invalid_inputs = (
            (["A" * 1022 + "X"], [(1023,)]),
            (["ACD", "ACDE"], [(1,), (1,)]),
            ([], []), (["ACD"], []),
            (["ACD"], [(0,)]), (["ACD"], [(4,)]),
            (["ACD"], [(True,)]), (["ACD"], [(1.0,)]),
            (["ACD"], [(1, 1)]), (["ACD"], ["1"]),
        )
        for sequences, sites in invalid_inputs:
            with self.subTest(lengths=[len(sequence) for sequence in sequences], sites=sites):
                adapter = self.adapter()
                with patch.object(adapter, "load", side_effect=AssertionError("validation must come first")):
                    with self.assertRaises(ValueError):
                        list(adapter.iter_site_embeddings(sequences, sites))
        adapter = self.adapter(max_tokens_per_batch=1023)
        with patch.object(adapter, "load", side_effect=AssertionError("token budget must be checked first")):
            with self.assertRaisesRegex(ValueError, "max_tokens_per_batch"):
                list(adapter.iter_site_embeddings(["A" * 3423], [(2300,)]))

    def test_config_and_metadata_keep_legacy_recipes_separate(self):
        residue = ExtractionConfig()
        prefix = ExtractionConfig(pooling="mean", truncation="prefix")
        centered = ExtractionConfig(pooling="mean", truncation="prefix", site_embeddings="centered")
        self.assertEqual(prefix.cache_id, residue.cache_id + "-mean-first1022")
        self.assertEqual(centered.cache_id, prefix.cache_id + "-centered-sites")
        extra = {
            "site_embeddings": "centered", "cache_layout": "mutation_sites_v1",
            "site_context": "mutated_sequence", "window_policy": "centered_clamped",
            "window_indexing": "zero_based_half_open",
        }
        score_recipes = []
        for config in (residue, prefix, centered):
            adapter = ESM2ScoreAdapter(config)
            adapter._manifest = {"files_sha256": {"model.safetensors": "a" * 64}}
            with patch.object(adapter, "_checkpoint"), patch("feature_generation.adapters.esm2.version", return_value="test"):
                metadata = adapter.metadata()
                if config.site_embeddings == "centered":
                    self.assertEqual({key: metadata[key] for key in extra}, extra)
                else:
                    self.assertTrue(set(extra).isdisjoint(metadata))
                score_recipes.append(adapter.score_metadata())
        self.assertEqual(score_recipes[0], score_recipes[1])
        self.assertEqual(score_recipes[0], score_recipes[2])
        self.assertTrue(set(extra).isdisjoint(score_recipes[2]))
        for overrides in (
            {"site_embeddings": "centered"}, {"site_embeddings": "all"},
            {"site_embeddings": True},
            {"site_embeddings": "centered", "pooling": "mean", "truncation": "prefix", "max_length": 1023},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                ExtractionConfig(**overrides)

    def test_centered_embedding_mode_does_not_enable_long_reference_scoring(self):
        config = self.adapter(max_length=3).config
        adapter = ESM2ScoreAdapter(config)
        with patch.object(adapter, "load", side_effect=AssertionError("long score input must be rejected")):
            with self.assertRaisesRegex(ValueError, "truncation is disabled"):
                list(adapter.iter_scores("ACDE", ["VCDE"]))
        config = replace(config, site_embeddings="none")
        with self.assertRaisesRegex(ValueError, "site_embeddings=centered"):
            list(ESM2Adapter(config).iter_site_embeddings(["ACD"], [(1,)]))


if __name__ == "__main__":
    unittest.main()
