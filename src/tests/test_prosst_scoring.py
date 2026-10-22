"""Synthetic checks for structure-conditioned masked marginal scores."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from data_loading import load_feature_bundle
from feature_generation.adapters.prosst_scoring import ProSSTScoreAdapter
from feature_generation.prosst_config import ProSSTConfig
from feature_generation.prosst_runner import extract
from feature_generation.score_runner import score, default_cache_id
from feature_generation.score_storage import validate_score_cache
from tests import test_prosst_pipeline as fixtures
from tests.test_prosst_pipeline import FakeAdapter

VOCAB = ["<pad>", "<cls>", "<eos>", "<mask>", *"ACDEFGHIKLMNPQRSTVWY", "X"]


class Tokenizer:
    pad_token_id, cls_token_id, eos_token_id, mask_token_id = 0, 1, 2, 3
    mask_token = "<mask>"
    all_special_ids = [0, 1, 2, 3]

    def convert_tokens_to_ids(self, tokens):
        return [VOCAB.index(token) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [VOCAB[index] for index in ids]

    def __call__(self, sequences, **kwargs):
        ids = torch.tensor([[1, *self.convert_tokens_to_ids(sequence), 2] for sequence in sequences])
        special = torch.zeros_like(ids)
        special[:, [0, -1]] = 1
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids), special_tokens_mask=special)


class Model:
    config = SimpleNamespace(vocab_size=len(VOCAB), mask_token_id=3)

    def __init__(self):
        self.calls, self.outputs = [], []

    def __call__(self, input_ids, attention_mask, ss_input_ids, return_dict):
        assert not torch.is_grad_enabled()
        self.calls.append((input_ids.clone(), ss_input_ids.clone()))
        logits = (input_ids + ss_input_ids).float().unsqueeze(-1) * torch.arange(len(VOCAB)) / 17
        self.outputs.append(logits.clone())
        return SimpleNamespace(logits=logits)


def ready_adapter(**kwargs):
    adapter = ProSSTScoreAdapter(ProSSTConfig(device="cpu", max_tokens_per_batch=10, **kwargs))
    adapter._device = torch.device("cpu")
    adapter._tokenizer, adapter._model = Tokenizer(), Model()
    adapter.set_structure_tokens(torch.tensor([0, 12, 2047]))
    return adapter


class ProSSTScoringTests(unittest.TestCase):
    def test_formula_masking_structure_alignment_and_token_budget(self):
        adapter = ready_adapter()
        batches = list(adapter.iter_scores("ACD", ["ACD", "CCD", "AED", "CED", "ACY"], batch_size=4))
        values = torch.cat([values for _, values in batches])
        self.assertEqual([indices for indices, _ in batches], [[0, 1, 2, 3], [4]])
        self.assertEqual([len(ids) for ids, _ in adapter._model.calls], [2, 1])
        logits = torch.cat(adapter._model.outputs)
        probabilities = logits.log_softmax(-1)
        expected = []
        for row, position, wt, mutant in [(0, 1, "A", "C"), (1, 2, "C", "E"), (2, 3, "D", "Y")]:
            expected.append(probabilities[row, position, VOCAB.index(mutant)] - probabilities[row, position, VOCAB.index(wt)])
        torch.testing.assert_close(values, torch.stack([torch.tensor(0.), expected[0], expected[1], expected[0] + expected[1], expected[2]]))
        all_ids = torch.cat([ids for ids, _ in adapter._model.calls])
        torch.testing.assert_close(all_ids, torch.tensor([[1, 3, 5, 6, 2], [1, 4, 3, 6, 2], [1, 4, 5, 3, 2]]))
        for _, structure in adapter._model.calls:
            torch.testing.assert_close(structure, torch.tensor([[1, 3, 15, 2050, 2]]).expand(len(structure), -1))

    def test_structure_changes_scores_and_batch_size_does_not(self):
        adapter = ready_adapter()
        first = torch.cat([v for _, v in adapter.iter_scores("ACD", ["CCD", "AED"], batch_size=1)])
        second = torch.cat([v for _, v in adapter.iter_scores("ACD", ["CCD", "AED"], batch_size=8)])
        torch.testing.assert_close(first, second)
        adapter.set_structure_tokens(torch.tensor([10, 20, 30]))
        changed = torch.cat([v for _, v in adapter.iter_scores("ACD", ["CCD", "AED"])])
        self.assertFalse(torch.allclose(first, changed))

    def test_invalid_inputs_and_nonfinite_logits(self):
        for reference, sequences in [("AC", ["CC"]), ("ACD", ["AC"]), ("ACD", ["AXD"])]:
            with self.subTest(reference=reference, sequences=sequences), self.assertRaises(ValueError):
                list(ready_adapter().iter_scores(reference, sequences))
        adapter = ready_adapter(max_length=2)
        with self.assertRaisesRegex(ValueError, "max_length"):
            list(adapter.iter_scores("ACD", ["CCD"]))
        adapter = ready_adapter()
        adapter._model = lambda **kwargs: SimpleNamespace(logits=torch.full((1, 5, len(VOCAB)), float("nan")))
        adapter._model.config = Model.config
        adapter.load = lambda: None
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            list(adapter.iter_scores("ACD", ["CCD"]))


class FakeScoreAdapter(FakeAdapter):
    score_metadata = ProSSTScoreAdapter.score_metadata

    def iter_scores(self, reference, sequences, *, batch_size):
        assert not torch.is_grad_enabled()
        yield list(range(len(sequences))), torch.tensor([
            sum(ord(b) - ord(a) for a, b in zip(reference, sequence)) for sequence in sequences
        ], dtype=torch.float32)


class ProSSTScorePipelineTests(unittest.TestCase):
    setUp = fixtures.ProSSTPipelineTests.setUp
    load_structure = fixtures.ProSSTPipelineTests.load_structure
    token_factory = fixtures.ProSSTPipelineTests.token_factory

    def run_scores(self, config=None, **kwargs):
        with patch("feature_generation.score_runner.load_structure", side_effect=self.load_structure):
            return score(config or self.config, ["tiny"], adapter_factory=FakeScoreAdapter,
                         token_factory=self.token_factory, **kwargs)

    def test_cache_resume_embedding_independence_and_gp_loading(self):
        result = self.run_scores()[0]
        report = validate_score_cache(self.config.output_root / "tiny", "prosst")
        self.assertEqual(report["num_variants"], 4)
        self.assertEqual(self.run_scores(resume=True)[0]["status"], "skipped")
        changed_layer = replace(self.config, layer=8, cache_id="another-embedding")
        self.assertEqual(default_cache_id(changed_layer), default_cache_id(self.config))
        # Fake metadata includes a layer, as the real adapter does; scoring removes it.
        self.assertEqual(self.run_scores(changed_layer, resume=True)[0]["status"], "skipped")
        extract(self.config, ["tiny"], adapter_factory=FakeAdapter, token_factory=self.token_factory)
        batch = load_feature_bundle(self.config.output_root / "tiny", enabled_branches=["sequence_global"],
                                    score_sources=[result["score_source"]])
        self.assertEqual(batch.variant_ids, ("wt", "first", "second", "double"))
        self.assertTrue(torch.isfinite(batch.scores).all())

    def test_changed_tokens_are_rejected_on_resume(self):
        self.run_scores()
        self.tokens = torch.tensor([1, 2, 3])
        with self.assertRaisesRegex(ValueError, "differs"):
            self.run_scores(resume=True)

    def test_tampered_score_file_is_rejected(self):
        result = self.run_scores()[0]
        from pathlib import Path
        (Path(result["path"]) / "mean_score.csv").write_text("variant_id,score\nwt,123\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_score_cache(self.config.output_root / "tiny", "prosst")
