"""Windowed ESM2 mean scores retain full-assay identity and independent caches."""

import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from data_loading import load_feature_bundle
from feature_generation.adapters.esm2 import ESM2Adapter
from feature_generation.config import ExtractionConfig
from feature_generation.runner import extract
from feature_generation.score_runner import default_cache_id, score
from feature_generation.score_storage import validate_score_cache
from tests.test_esm2_centered_pipeline import FakeCenteredAdapter
from tests.test_esm2_score_pipeline import FakeESM2ScoreAdapter
from tests.test_feature_generation import write_csv


class FakeCenteredScoreAdapter(FakeESM2ScoreAdapter):
    """An additive local-position oracle, deliberately yielding reversed rows."""

    def score_metadata(self):
        metadata = super().score_metadata()
        if self.config.scoring_window == "centered":
            metadata.update(
                scoring_recipe_version=2, scoring_window="centered", truncation="centered",
                window_policy="centered_clamped", window_indexing="zero_based_half_open",
                max_length=self.config.max_length, residue_aggregation="sum_over_mutations",
                masking="one_wild_type_residue_at_a_time", wild_type_subtraction=True,
                noncanonical_tokens="retained_in_normalization", alphabet="ACDEFGHIKLMNPQRSTVWY",
                precision_policy="float32_log_softmax",
            )
        return metadata

    def iter_scores(self, reference_sequence, sequences, *, batch_size=8):
        if self.config.scoring_window != "centered":
            yield from super().iter_scores(reference_sequence, sequences, batch_size=batch_size)
            return
        if torch.is_grad_enabled():
            raise AssertionError("Scoring must disable gradients")
        self.score_calls.append((reference_sequence, tuple(sequences), batch_size))
        width = min(len(reference_sequence), self.config.max_length)
        scores = []
        for sequence in sequences:
            value = 0.0
            for index, (wild_type, mutant) in enumerate(zip(reference_sequence, sequence)):
                if wild_type != mutant:
                    start = max(0, min(index - width // 2, len(reference_sequence) - width))
                    value += (index - start + 1) * (ord(mutant) - ord(wild_type)) / 100
            scores.append(value)
        for stop in range(len(sequences), 0, -batch_size):
            indices = list(range(max(0, stop - batch_size), stop))[::-1]
            yield indices, torch.tensor([scores[index] for index in indices], dtype=torch.float32)


class CenteredScorePipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.index = self.raw / "DMS_substitutions.csv"
        self.reference = "A" * 3000
        self.ids = ("z_wt", "m_first", "a_second", "x_double")
        self.mutants = ("WT", "A1000V", "A2300K", "A1000V:A2300K")
        self.sequences = (
            self.reference,
            "A" * 999 + "V" + "A" * 2000,
            "A" * 2299 + "K" + "A" * 700,
            "A" * 999 + "V" + "A" * 1299 + "K" + "A" * 700,
        )
        self.write_assay()
        self.config = ExtractionConfig(
            model_id="facebook/esm2_t6_8M_UR50D", model_dir=self.root / "models",
            output_root=self.root / "features", assay_index=self.index, device="cpu",
            pooling="mean", truncation="prefix", site_embeddings="centered",
            scoring_window="centered", max_length=1022, max_tokens_per_batch=1024,
        )
        self.adapters = []

    @property
    def assay_root(self):
        return self.config.output_root / "tiny"

    @property
    def leaf(self):
        return self.assay_root / "scores" / "esm2" / default_cache_id(self.config)

    def write_assay(self):
        write_csv(self.index, ["DMS_id", "DMS_filename", "target_seq", "seq_len"], [
            ["tiny", "tiny.csv", self.reference, len(self.reference)],
        ])
        write_csv(self.raw / "tiny.csv", ["variant_id", "mutant", "mutated_sequence", "DMS_score"], [
            [variant_id, mutant, sequence, index / 2]
            for index, (variant_id, mutant, sequence) in enumerate(zip(self.ids, self.mutants, self.sequences))
        ])

    def factory(self, config):
        adapter = FakeCenteredScoreAdapter(config)
        self.adapters.append(adapter)
        return adapter

    def run_score(self, **kwargs):
        return score(self.config, ["tiny"], adapter_factory=self.factory, **kwargs)

    def score_rows(self, leaf=None):
        with ((self.leaf if leaf is None else leaf) / "mean_score.csv").open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def test_long_assay_scores_keep_original_order_and_load_with_compact_embeddings(self):
        events = []
        result = self.run_score(batch_size=2, progress=events.append)[0]
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["score_source"].endswith("-centered1022"))
        expected = torch.tensor([0., 107.52, 51.2, 158.72], dtype=torch.float32)
        rows = self.score_rows()
        self.assertEqual([row["variant_id"] for row in rows], list(self.ids))
        torch.testing.assert_close(torch.tensor([float(row["score"]) for row in rows]), expected)
        self.assertEqual(self.adapters[0].score_calls, [(self.reference, self.sequences, 2)])
        self.assertEqual(next(event["num_masked_positions"] for event in events if event["status"] == "scoring"), 2)
        metadata = json.loads((self.leaf / "config.json").read_text())
        self.assertEqual(metadata["scoring_recipe_version"], 2)
        self.assertEqual(metadata["scoring_window"], "centered")
        self.assertEqual(metadata["max_length"], 1022)
        report = validate_score_cache(self.assay_root, model_family="esm2", cache_id=default_cache_id(self.config))
        self.assertEqual(report["num_variants"], 4)
        extract(self.config, ["tiny"], adapter_factory=FakeCenteredAdapter)
        batch = load_feature_bundle(
            self.assay_root, enabled_branches=["sequence_global", "sequence_site"],
            feature_selection={"sequence": f"esm2/{self.config.cache_id}"}, score_sources=[result["score_source"]],
        )
        self.assertEqual(batch.variant_ids, self.ids)
        self.assertEqual(batch.reference_sequence, self.reference)
        self.assertEqual(batch.sequences, self.sequences)
        self.assertEqual(batch.mutation_positions.tolist(), [1000, 2300, 1000, 2300])
        torch.testing.assert_close(batch.scores[:, 0], expected.double())

    def test_resume_ignores_embedding_choices_but_rejects_changed_window_recipe(self):
        self.run_score()
        original_leaf = self.leaf
        original_config = (original_leaf / "config.json").read_bytes()
        self.config = replace(self.config, cache_id="other-embeddings", layer=1,
                              pooling="none", truncation="reject", site_embeddings="none")
        with patch.object(FakeCenteredScoreAdapter, "iter_scores", side_effect=AssertionError("must reuse score cache")):
            self.assertEqual(self.run_score(resume=True)[0]["status"], "skipped")
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 1)
        previous_score_id = default_cache_id(self.config)
        self.config = replace(self.config, max_length=512)
        self.assertNotEqual(default_cache_id(self.config), previous_score_id)
        with self.assertRaises(ValueError):
            self.run_score(cache_id=previous_score_id, resume=True)
        self.assertEqual((original_leaf / "config.json").read_bytes(), original_config)
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 1)

    def test_centered_and_legacy_recipes_coexist_without_reusing_each_others_scores(self):
        self.reference = "ACD"
        self.mutants = ("WT", "A1V", "C2D", "A1V:C2D")
        self.sequences = ("ACD", "VCD", "ADD", "VDD")
        self.write_assay()
        legacy = replace(self.config, scoring_window="full")
        legacy_id = default_cache_id(legacy)
        centered_id = default_cache_id(self.config)
        self.assertEqual(centered_id, legacy_id + "-centered1022")
        legacy_result = score(legacy, ["tiny"], adapter_factory=FakeESM2ScoreAdapter)[0]
        legacy_leaf = Path(legacy_result["path"])
        before = (legacy_leaf / "config.json").read_bytes()
        self.run_score()
        self.assertEqual(self.score_rows(legacy_leaf), self.score_rows())
        with self.assertRaises(ValueError):
            self.run_score(cache_id=legacy_id, resume=True)
        with self.assertRaises(ValueError):
            score(legacy, ["tiny"], cache_id=centered_id, resume=True, adapter_factory=FakeESM2ScoreAdapter)
        self.assertEqual((legacy_leaf / "config.json").read_bytes(), before)

    def test_window_budget_and_legacy_length_limit_fail_before_loading_model(self):
        self.config = replace(self.config, max_tokens_per_batch=1023)
        with self.assertRaisesRegex(ValueError, "max_tokens_per_batch"):
            self.run_score()
        self.config = replace(self.config, max_tokens_per_batch=1024, scoring_window="full")
        with self.assertRaisesRegex(ValueError, "max_length"):
            self.run_score()
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 0)
        self.config = replace(self.config, scoring_window="centered")
        self.assertEqual(self.run_score()[0]["status"], "completed")
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 1)

    def test_short_protein_requires_only_its_actual_tokens(self):
        self.reference = "ACD"
        self.mutants = ("WT", "A1V", "C2D", "A1V:C2D")
        self.sequences = ("ACD", "VCD", "ADD", "VDD")
        self.write_assay()
        self.config = replace(self.config, max_tokens_per_batch=5)
        self.assertEqual(self.run_score()[0]["status"], "completed")
        self.assertEqual([row["variant_id"] for row in self.score_rows()], list(self.ids))

    def test_continues_to_long_assay_after_unavailable_assay(self):
        reports = score(self.config, ["missing", "tiny"], continue_on_error=True, adapter_factory=self.factory)
        self.assertEqual([report["status"] for report in reports], ["failed", "completed"])
        self.assertEqual(self.adapters[-1].load_count, 1)

    def test_scoring_window_is_explicit_and_does_not_change_embedding_metadata(self):
        self.assertEqual(ExtractionConfig().scoring_window, "full")
        for value in (None, True, 1, "prefix", "sliding", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ExtractionConfig(scoring_window=value)
        recipes = []
        for mode in ("full", "centered"):
            adapter = ESM2Adapter(replace(self.config, scoring_window=mode))
            adapter._manifest = {"files_sha256": {"model.safetensors": "a" * 64}}
            with patch.object(adapter, "_checkpoint"), patch("feature_generation.adapters.esm2.version", return_value="test"):
                recipes.append(adapter.metadata())
        self.assertEqual(recipes[0], recipes[1])
        self.assertNotIn("scoring_window", recipes[0])


if __name__ == "__main__":
    unittest.main()
