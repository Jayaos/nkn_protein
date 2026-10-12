"""Full-sequence pooled caches preserve centered sites and scoring identities."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from data_loading import load_feature_bundle
from feature_generation.config import ExtractionConfig
from feature_generation.runner import extract
from feature_generation.score_runner import score
from feature_generation.storage import validate_sequence_cache
from tests.test_esm2_centered_pipeline import FakeCenteredAdapter
from tests.test_esm2_centered_score_pipeline import FakeCenteredScoreAdapter
from tests.test_esm2_pooling import PrefixAdapter
from tests.test_feature_generation import residue_values, write_csv


def nonoverlap_mean(sequence, width):
    """Small independent oracle: concatenate window-local vectors, then pool."""
    return torch.cat([
        residue_values(sequence[start:start + width])
        for start in range(0, len(sequence), width)
    ]).mean(dim=0)


class FakeNonoverlapAdapter(PrefixAdapter):
    """Use the existing centered-site oracle and a separate full-global oracle."""

    def metadata(self):
        metadata = {
            **super().metadata(),
            "global_window": "non_overlapping",
            "global_window_aggregation": "length_weighted_mean",
        }
        if self.config.site_embeddings == "centered":
            metadata.update(
                site_embeddings="centered", cache_layout="mutation_sites_v1",
                site_context="mutated_sequence", window_policy="centered_clamped",
                window_indexing="zero_based_half_open",
            )
        return metadata

    def iter_site_embeddings(self, sequences, mutation_positions):
        for index, _, sites in FakeCenteredAdapter.iter_site_embeddings(self, sequences, mutation_positions):
            yield index, nonoverlap_mean(sequences[index], self.config.max_length), sites

    def iter_embeddings(self, sequences):
        for index in reversed(range(len(sequences))):
            yield [index], nonoverlap_mean(sequences[index], self.config.max_length).unsqueeze(0)


class NonoverlapPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.index = self.raw / "DMS_substitutions.csv"
        self.reference = "ACDEFGHIK"
        self.ids = ("wt", "tail", "head", "both", "middle")
        self.mutants = ("WT", "K9V", "A1V", "A1V:K9V", "F5W")
        self.sequences = ("ACDEFGHIK", "ACDEFGHIV", "VCDEFGHIK", "VCDEFGHIV", "ACDEWGHIK")
        self.write_assay()
        self.config = ExtractionConfig(
            model_id="facebook/esm2_t6_8M_UR50D", model_dir=self.root / "models",
            output_root=self.root / "features", assay_index=self.index, device="cpu",
            pooling="mean", truncation="none", global_window="non_overlapping",
            site_embeddings="centered", scoring_window="centered", max_length=4, max_tokens_per_batch=6,
        )
        self.adapters = []

    @property
    def assay_root(self):
        return self.config.output_root / "tiny"

    @property
    def leaf(self):
        return self.assay_root / "sequence" / "esm2" / self.config.cache_id

    def write_assay(self):
        write_csv(self.index, ["DMS_id", "DMS_filename", "target_seq", "seq_len"], [
            ["tiny", "tiny.csv", self.reference, len(self.reference)],
        ])
        write_csv(self.raw / "tiny.csv", ["variant_id", "mutant", "mutated_sequence", "DMS_score"], [
            [variant_id, mutant, sequence, index / 2]
            for index, (variant_id, mutant, sequence) in enumerate(zip(self.ids, self.mutants, self.sequences))
        ])

    def factory(self, config):
        adapter = FakeNonoverlapAdapter(config)
        self.adapters.append(adapter)
        return adapter

    def run_extraction(self, **kwargs):
        return extract(self.config, ["tiny"], adapter_factory=self.factory, **kwargs)

    def load_bundle(self, source, config=None, branches=None):
        config = self.config if config is None else config
        return load_feature_bundle(
            self.assay_root, feature_selection={"sequence": f"esm2/{config.cache_id}"},
            enabled_branches=branches or ["sequence_global", "sequence_site"], score_sources=[source],
        )

    def test_full_global_average_weights_partial_chunk_and_responds_to_tail(self):
        result = self.run_extraction()[0]
        self.assertEqual(result["shape"], [5, 2])
        self.assertEqual(result["site_shape"], [5, 2])
        payload = torch.load(self.leaf / "features.pt", weights_only=True)
        expected = torch.tensor([
            [sum(map(ord, sequence)) / 9, 21 / 9] for sequence in self.sequences
        ], dtype=torch.float32)
        torch.testing.assert_close(payload["embeddings"], expected)
        self.assertNotAlmostEqual(float(payload["embeddings"][0, 1]), (2.5 + 2.5 + 1) / 3)
        self.assertFalse(torch.equal(payload["embeddings"][0], payload["embeddings"][1]))
        self.assertFalse(torch.equal(payload["embeddings"][2], payload["embeddings"][3]))
        self.assertEqual(payload["represented_length"], 9)
        self.assertEqual(payload["max_length"], 4)
        self.assertEqual(payload["global_window"], "non_overlapping")
        self.assertEqual(payload["global_window_aggregation"], "length_weighted_mean")
        self.assertEqual(payload["truncation"], "none")
        self.assertEqual(payload["site_window_starts"].tolist(), [5, 0, 0, 5, 2])
        self.assertEqual(payload["site_window_ends"].tolist(), [9, 4, 4, 9, 6])
        self.assertEqual(validate_sequence_cache(self.assay_root)["reference_length"], 9)
        source = score(self.config, ["tiny"], adapter_factory=FakeCenteredScoreAdapter)[0]["score_source"]
        batch = self.load_bundle(source)
        self.assertEqual(batch.variant_ids, self.ids)
        torch.testing.assert_close(batch.global_features, expected.double())
        self.assertEqual(batch.mutation_positions.tolist(), [9, 1, 1, 9, 5])

    def test_old_prefix_cache_coexists_with_identical_sites_and_scores(self):
        prefix = replace(self.config, cache_id=None, global_window="prefix", truncation="prefix")
        self.assertIn("-mean-first4-centered-sites", prefix.cache_id)
        self.assertIn("-mean-all-nonoverlap4-centered-sites", self.config.cache_id)
        extract(prefix, ["tiny"], adapter_factory=FakeCenteredAdapter)
        old_default = json.loads((self.assay_root / "assay.json").read_text())["features"]
        old_score = score(prefix, ["tiny"], adapter_factory=FakeCenteredScoreAdapter)[0]
        old_score_bytes = (Path(old_score["path"]) / "mean_score.csv").read_bytes()
        self.run_extraction()
        current_score = score(self.config, ["tiny"], resume=True, adapter_factory=FakeCenteredScoreAdapter)[0]
        self.assertEqual(current_score["status"], "skipped")
        self.assertEqual(current_score["score_source"], old_score["score_source"])
        self.assertEqual((Path(current_score["path"]) / "mean_score.csv").read_bytes(), old_score_bytes)
        self.assertEqual(json.loads((self.assay_root / "assay.json").read_text())["features"], old_default)
        old = self.load_bundle(old_score["score_source"], prefix)
        new = self.load_bundle(current_score["score_source"])
        torch.testing.assert_close(old.local_features["sequence_site"], new.local_features["sequence_site"])
        torch.testing.assert_close(old.scores, new.scores)
        self.assertFalse(torch.equal(old.global_features, new.global_features))
        self.assertNotEqual(old.feature_signature, new.feature_signature)
        with self.assertRaisesRegex(ValueError, "new cache ID"):
            extract(replace(self.config, cache_id=prefix.cache_id), ["tiny"],
                    resume=True, adapter_factory=FakeNonoverlapAdapter)
        self.assertEqual(validate_sequence_cache(self.assay_root, prefix.cache_id)["cache_id"], prefix.cache_id)

    def test_long_sequence_global_includes_last_residue_without_dense_cache(self):
        self.reference = "A" * 2500
        self.ids, self.mutants = ("wt", "tail"), ("WT", "A2500V")
        self.sequences = (self.reference, "A" * 2499 + "V")
        self.write_assay()
        self.config = replace(self.config, cache_id=None, max_length=1022, max_tokens_per_batch=1024)
        self.run_extraction()
        payload = torch.load(self.leaf / "features.pt", weights_only=True)
        expected_position_mean = (1022 * 1023 + 456 * 457 / 2) / 2500
        expected = torch.tensor([
            [65., expected_position_mean], [65. + 21 / 2500, expected_position_mean],
        ], dtype=torch.float32)
        torch.testing.assert_close(payload["embeddings"], expected)
        self.assertFalse(torch.equal(payload["embeddings"][0], payload["embeddings"][1]))
        self.assertEqual(payload["represented_length"], 2500)
        self.assertEqual(payload["site_embeddings"].shape, (1, 2))
        self.assertEqual(payload["site_window_starts"].tolist(), [1478])
        self.assertEqual(payload["site_window_ends"].tolist(), [2500])
        self.assertFalse(any(isinstance(value, torch.Tensor) and value.ndim == 3 for value in payload.values()))
        validate_sequence_cache(self.assay_root)

    def test_compact_memory_budget_is_independent_of_global_window_coverage(self):
        output_bytes = (5 + 5) * 2 * 4 + 5 * 4 * 8
        self.config = replace(self.config, max_output_gib=output_bytes / 1024**3)
        events = []
        self.run_extraction(progress=events.append)
        self.assertEqual(next(event["output_bytes"] for event in events if event["status"] == "extracting"), output_bytes)
        self.config = replace(self.config, output_root=self.root / "too_small", max_output_gib=(output_bytes - 1) / 1024**3)
        with self.assertRaisesRegex(ValueError, "Output requires"):
            self.run_extraction()
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 1)

    def test_resume_uses_new_recipe_and_rejects_changed_window_width(self):
        self.run_extraction()
        with patch.object(FakeNonoverlapAdapter, "iter_site_embeddings", side_effect=AssertionError("must reuse cache")):
            self.assertEqual(self.run_extraction(resume=True)[0]["status"], "skipped")
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 1)
        self.config = replace(self.config, max_length=3)
        with self.assertRaisesRegex(ValueError, "new cache ID"):
            self.run_extraction(resume=True)
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 1)

    def test_global_only_full_sequence_cache_loads_without_site_payload(self):
        self.config = replace(self.config, cache_id=None, site_embeddings="none")
        result = self.run_extraction()[0]
        self.assertEqual(result["shape"], [5, 2])
        self.assertNotIn("site_shape", result)
        payload = torch.load(self.leaf / "features.pt", weights_only=True)
        self.assertNotIn("site_embeddings", payload)
        self.assertEqual(payload["represented_length"], 9)
        validate_sequence_cache(self.assay_root)
        source = score(self.config, ["tiny"], adapter_factory=FakeCenteredScoreAdapter)[0]["score_source"]
        batch = self.load_bundle(source, branches=["sequence_global"])
        expected = torch.stack([nonoverlap_mean(sequence, 4) for sequence in self.sequences])
        torch.testing.assert_close(batch.global_features, expected.double())
        self.assertEqual(batch.local_features, {})


if __name__ == "__main__":
    unittest.main()
