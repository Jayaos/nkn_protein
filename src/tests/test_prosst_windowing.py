"""Synthetic checks for full-coverage ProSST windows, alignment, and caches."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from data_loading import load_feature_bundle
from data_loading.sequence_windows import mean_sequence_recipe
from feature_generation.adapters.prosst import ProSSTAdapter
from feature_generation.adapters.prosst_scoring import ProSSTScoreAdapter
from feature_generation.prosst_config import ProSSTConfig
from feature_generation.prosst_runner import extract
from feature_generation.score_runner import default_cache_id, score
from feature_generation.score_storage import _validate_config, validate_score_cache
from feature_generation.storage import validate_sequence_cache
from tests.test_prosst_backend import FakeBackbone, FakeTokenizer
from tests.test_prosst_scoring import Model, Tokenizer, VOCAB
from tests import test_prosst_pipeline as fixtures


def window_config(**overrides):
    values = dict(device="cpu", max_length=4, max_tokens_per_batch=12,
                  pooling="mean", global_window="non_overlapping", truncation="none",
                  site_embeddings="centered", scoring_window="centered")
    values.update(overrides)
    return ProSSTConfig(**values)


def ready(config=None, *, scoring=False):
    adapter = (ProSSTScoreAdapter if scoring else ProSSTAdapter)(config or window_config())
    adapter._device = torch.device("cpu")
    adapter._tokenizer = Tokenizer() if scoring else FakeTokenizer()
    adapter._model = Model() if scoring else SimpleNamespace(prosst=FakeBackbone())
    adapter._manifest = {"files_sha256": {"model/test": "a" * 64}, "quantizer_revision": "b" * 40}
    adapter._checkpoint = lambda: Path(".")
    adapter.set_structure_tokens(torch.arange(9))
    return adapter


class ProSSTWindowTests(unittest.TestCase):
    def test_presets_and_cache_ids(self):
        root = Path(__file__).resolve().parents[2]
        for name in ("prosst_2048.json", "prosst_2048_singles.json"):
            config = ProSSTConfig.load(root / "configs/extraction" / name)
            self.assertEqual((config.max_length, config.max_tokens_per_batch), (2046, 2048))
            self.assertEqual(config.scoring_window, "centered")
            self.assertTrue(config.cache_id.endswith("mean-all-nonoverlap2046-centered-sites"))
            self.assertTrue(default_cache_id(config).endswith("centered2046"))
        for overrides in ({"max_length": 2047}, {"pooling": "none"}, {"truncation": "reject"},
                          {"site_embeddings": "invalid"}, {"scoring_window": "invalid"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                window_config(**overrides)
        self.assertNotEqual(default_cache_id(window_config()), default_cache_id(ProSSTConfig()))

    def test_global_tail_weighting_centered_sites_and_aligned_window_batches(self):
        adapter = ready()
        sequence = "ACDEFGHIK"
        positions = [1, 4, 5, 9]
        [(index, global_value, sites)] = list(adapter.iter_site_embeddings([sequence], [positions]))
        self.assertEqual(index, 0)
        # Fake backbone values are AA ID + (raw structure ID + 3) + layer 12.
        residues = torch.tensor([FakeTokenizer.alphabet.index(aa) + 18 + i
                                 for i, aa in enumerate(sequence)], dtype=torch.float32)
        torch.testing.assert_close(global_value, residues.mean().expand(768))
        torch.testing.assert_close(sites[:, 0], residues[torch.tensor(positions) - 1])
        expected_bounds = [(0, 4), (4, 8), (8, 9), (1, 5), (2, 6), (5, 9)]
        rows = [(aa, ss) for call in adapter._model.prosst.calls
                for aa, ss in zip(call["input_ids"], call["ss_input_ids"])]
        self.assertEqual(len(rows), len(expected_bounds))  # first site reuses global window
        for (aa, ss), (start, end) in zip(rows, expected_bounds):
            self.assertEqual(FakeTokenizer().convert_ids_to_tokens(aa[1:-1].tolist()), list(sequence[start:end]))
            self.assertEqual(ss.tolist(), [1, *range(start + 3, end + 3), 2])
        self.assertTrue(all(call["input_ids"].numel() <= 12 for call in adapter._model.prosst.calls))

    def test_short_inputs_equal_full_reference_and_include_variant_changes(self):
        full = ready(ProSSTConfig(device="cpu"))
        windowed = ready()
        for adapter in (full, windowed):
            adapter.set_structure_tokens(torch.tensor([31, 60, 2047]))
        sequences = ["ACD", "AYD"]
        full_values = torch.cat([v for _, v in full.iter_embeddings(sequences)])
        results = list(windowed.iter_site_embeddings(sequences, [[], [2]]))
        for i, global_value, sites in results:
            torch.testing.assert_close(global_value, full_values[i].mean(0), rtol=0, atol=0)
            torch.testing.assert_close(sites, full_values[i, [] if i == 0 else [1]], rtol=0, atol=0)
        self.assertFalse(torch.equal(results[0][1], results[1][1]))

    def test_2048_boundary_special_tokens_tail_and_budget(self):
        for length in (2045, 2046, 2047, 2048):
            adapter = ready(window_config(max_length=2046, max_tokens_per_batch=2048))
            adapter.set_structure_tokens(torch.arange(length) % 2048)
            [(index, global_value, sites)] = list(adapter.iter_site_embeddings(["A" * length], [[length]]))
            self.assertEqual(tuple(sites.shape), (1, 768))
            widths = [call["input_ids"].shape[1] for call in adapter._model.prosst.calls]
            self.assertEqual(widths, [length + 2] if length <= 2046 else [2048, length - 2046 + 2, 2048])
            self.assertLessEqual(max(widths), 2048)
        adapter = ready(window_config(max_length=2046, max_tokens_per_batch=2047))
        adapter.set_structure_tokens(torch.zeros(2046, dtype=torch.long))
        adapter.load = Mock(side_effect=AssertionError("must validate before model I/O"))
        with self.assertRaisesRegex(ValueError, "BOS/EOS"):
            list(adapter.iter_embeddings(["A" * 2046]))
        adapter.load.assert_not_called()

    def test_scoring_pairs_each_masked_row_with_its_own_structure_window(self):
        reference = "ACDEFGHIK"
        sequences = [reference, "CCDEFGHIK", "ACDEYGHIK", "ACDEFGHIY", "CCDEYGHIK"]
        adapter = ready(scoring=True)
        first = torch.cat([v for _, v in adapter.iter_scores(reference, sequences, batch_size=8)])
        rows = [(aa, ss) for aa_batch, ss_batch in adapter._model.calls for aa, ss in zip(aa_batch, ss_batch)]
        self.assertEqual(len(rows), 3)
        for (aa, ss), (start, end), local in zip(rows, [(0, 4), (2, 6), (5, 9)], [1, 3, 4]):
            expected = [1, *[VOCAB.index(x) for x in reference[start:end]], 2]
            expected[local] = 3
            self.assertEqual(aa.tolist(), expected)
            self.assertEqual(ss.tolist(), [1, *range(start + 3, end + 3), 2])
        expected = [0., 6 / 17, 10 * (VOCAB.index("Y") - VOCAB.index("F")) / 17,
                    14 * (VOCAB.index("Y") - VOCAB.index("K")) / 17]
        torch.testing.assert_close(first, torch.tensor([*expected, expected[1] + expected[2]]))
        second = torch.cat([v for _, v in adapter.iter_scores(reference, sequences, batch_size=1)])
        torch.testing.assert_close(first, second)

    def test_short_scores_match_full_and_scoring_policy_is_independent(self):
        full, centered = ready(ProSSTConfig(device="cpu"), scoring=True), ready(scoring=True)
        for adapter in (full, centered):
            adapter.set_structure_tokens(torch.arange(3))
        a = torch.cat([v for _, v in full.iter_scores("ACD", ["VCD", "AYD"])])
        b = torch.cat([v for _, v in centered.iter_scores("ACD", ["VCD", "AYD"])])
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        centered._structure_tokens = torch.arange(9)
        centered.config.scoring_window = "full"
        with self.assertRaisesRegex(ValueError, "max_length"):
            list(centered.iter_scores("ACDEFGHIK", ["CCDEFGHIK"]))
        # Centered scores also work with legacy full-residue embedding settings.
        standalone = ready(ProSSTConfig(max_length=4, scoring_window="centered"), scoring=True)
        self.assertEqual(len(list(standalone.iter_scores("ACDEFGHIK", ["CCDEFGHIK"]))), 1)

    def test_family_specific_window_limits(self):
        adapter = ready(window_config(max_length=2046))
        metadata = adapter.metadata()
        self.assertEqual(mean_sequence_recipe(metadata)["max_length"], 2046)
        for family in ("esm2", "msa_transformer"):
            with self.subTest(family=family), self.assertRaisesRegex(ValueError, "1022"):
                mean_sequence_recipe({**metadata, "model_family": family})


class ProSSTWindowPipelineTests(unittest.TestCase):
    setUp = fixtures.ProSSTPipelineTests.setUp
    load_structure = fixtures.ProSSTPipelineTests.load_structure
    token_factory = fixtures.ProSSTPipelineTests.token_factory

    def test_compact_cache_resume_gp_loading_and_full_structure_reuse(self):
        config = replace(self.config, max_length=2, cache_id=None, pooling="mean",
                         global_window="non_overlapping", truncation="none",
                         site_embeddings="centered", scoring_window="centered")
        result = extract(config, ["tiny"], adapter_factory=ready, token_factory=self.token_factory)[0]
        self.assertEqual(result["shape"], [4, 768])
        self.assertEqual(result["site_shape"], [4, 768])
        self.assertEqual(len(self.quantizations), 1)
        validate_sequence_cache(config.output_root / "tiny", config.cache_id, model_family="prosst")
        with patch("feature_generation.score_runner.load_structure", side_effect=self.load_structure):
            scored = score(config, ["tiny"], adapter_factory=lambda c: ready(c, scoring=True),
                           token_factory=self.token_factory)[0]
        batch = load_feature_bundle(config.output_root / "tiny", enabled_branches=["sequence_global", "sequence_site"],
                                    score_sources=[scored["score_source"]])
        self.assertEqual(batch.variant_ids, ("wt", "first", "second", "double"))
        self.assertEqual(tuple(batch.global_features.shape), (4, 768))
        self.assertEqual(tuple(batch.local_features["sequence_site"].shape), (4, 768))
        resumed = extract(config, ["tiny"], adapter_factory=ready, token_factory=self.token_factory, resume=True)
        self.assertEqual(resumed[0]["status"], "skipped")
        with self.assertRaisesRegex(ValueError, "differs"):
            extract(replace(config, max_length=1), ["tiny"], adapter_factory=ready,
                    token_factory=self.token_factory, resume=True)

    def test_centered_score_cache_and_recipe_validation(self):
        config = replace(self.config, max_length=2, scoring_window="centered")
        with patch("feature_generation.score_runner.load_structure", side_effect=self.load_structure):
            options = dict(adapter_factory=lambda c: ready(c, scoring=True), token_factory=self.token_factory)
            result = score(config, ["tiny"], **options)[0]
            report = validate_score_cache(config.output_root / "tiny", "prosst", default_cache_id(config))
            self.assertEqual(report["num_variants"], 4)
            self.assertEqual(score(config, ["tiny"], resume=True, **options)[0]["status"], "skipped")
        import json
        metadata = json.loads((Path(result["path"]) / "config.json").read_text())
        self.assertEqual(metadata["scoring_recipe_version"], 2)
        for override in ({"max_length": 2047}, {"scoring_window": "full"},
                         {"structure_window": "prefix"}, {"scoring_recipe_version": 1}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                _validate_config({**metadata, **override}, "prosst", default_cache_id(config))


if __name__ == "__main__":
    unittest.main()
