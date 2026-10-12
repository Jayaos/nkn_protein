"""Prefix pooling preserves assay identity while caching one vector per variant."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from data_loading import load_feature_bundle
from feature_generation.adapters.esm2 import ESM2Adapter
from feature_generation.adapters.esm2_scoring import ESM2ScoreAdapter
from feature_generation.config import ExtractionConfig
from feature_generation.runner import extract
from feature_generation.score_runner import score
from feature_generation.storage import validate_sequence_cache
from tests.test_esm2_score_pipeline import FakeESM2ScoreAdapter
from tests.test_feature_generation import FakeAdapter, FakeModel, FakeTokenizer, residue_values, write_csv


class PrefixAdapter(FakeAdapter):
    """Small explicit oracle, with deliberately reordered variant batches."""

    def metadata(self):
        return {
            **super().metadata(), "pooling": self.config.pooling,
            "truncation": self.config.truncation, "max_length": self.config.max_length,
            "includes_special_tokens": False,
        }

    def iter_embeddings(self, sequences):
        for indices, _ in super().iter_embeddings(sequences):
            yield indices, torch.stack([
                residue_values(sequences[index][:self.config.max_length]).mean(dim=0)
                for index in indices
            ])


class PrefixAdapterTests(unittest.TestCase):
    def adapter(self, **overrides):
        values = dict(
            model_id="facebook/esm2_t6_8M_UR50D", device="cpu",
            pooling="mean", truncation="prefix", max_length=3, max_tokens_per_batch=12,
        )
        values.update(overrides)
        adapter = ESM2Adapter(ExtractionConfig(**values))
        adapter._model, adapter._tokenizer = FakeModel(), FakeTokenizer()
        adapter._device = torch.device("cpu")
        return adapter

    def test_only_prefix_residues_are_averaged_excluding_special_tokens_and_padding(self):
        sequences = ("ACDEG", "ACDVV", "VCDEG")
        for layer in (0, 3, 6):
            with self.subTest(layer=layer):
                adapter = self.adapter(layer=layer)
                batches = list(adapter.iter_embeddings(sequences))
                self.assertEqual([indices for indices, _ in batches], [[0, 1], [2]])
                values = torch.cat([batch for _, batch in batches])
                expected = torch.tensor([
                    [sum(ord(aa) * 100 + pos + 1 + layer * 10000
                         for pos, aa in enumerate(sequence[:3])) / 3] * 320
                    for sequence in sequences
                ], dtype=torch.float32)
                torch.testing.assert_close(values, expected)
                torch.testing.assert_close(values[0], values[1])
                self.assertFalse(torch.equal(values[0], values[2]))
                self.assertEqual(values.shape, (3, 320))
                self.assertEqual(values.dtype, torch.float32)
                self.assertEqual(values.device.type, "cpu")
                self.assertFalse(values.requires_grad)
                self.assertTrue(all(not call["grad_enabled"] for call in adapter._model.calls))

    def test_default_boundary_keeps_residue_1022_and_discards_residue_1023(self):
        adapter = self.adapter(max_length=1022, max_tokens_per_batch=1025)
        prefix = "A" * 1021 + "C"
        sequences = (prefix + "D", prefix + "V", "A" * 1021 + "D" + "D")
        actual = torch.cat([values for _, values in adapter.iter_embeddings(sequences)])
        expected = sum(ord(aa) * 100 + pos + 1 + 60000 for pos, aa in enumerate(prefix)) / 1022
        torch.testing.assert_close(actual[0], torch.full((320,), expected, dtype=torch.float32))
        torch.testing.assert_close(actual[0], actual[1])
        self.assertFalse(torch.equal(actual[0], actual[2]))

    def test_short_sequences_use_all_residues(self):
        adapter = self.adapter()
        values = list(adapter.iter_embeddings(["AC"]))[0][1]
        expected = ((ord("A") * 100 + 1) + (ord("C") * 100 + 2)) / 2 + 60000
        torch.testing.assert_close(values, torch.full((1, 320), expected, dtype=torch.float32))

    def test_invalid_full_sequences_are_rejected_even_when_prefix_is_valid(self):
        for sequences in (["ACDX"], ["ACDE", "ACDEF"], []):
            with self.subTest(sequences=sequences):
                adapter = self.adapter()
                with patch.object(adapter, "load", side_effect=AssertionError("must validate before loading")):
                    with self.assertRaises(ValueError):
                        list(adapter.iter_embeddings(sequences))

    def test_recipe_has_distinct_default_cache_id_and_rejects_unsupported_combinations(self):
        legacy = ExtractionConfig()
        pooled = ExtractionConfig(pooling="mean", truncation="prefix")
        self.assertEqual(pooled.cache_id, legacy.cache_id + "-mean-first1022")
        for overrides in (
            {"pooling": "mean"}, {"truncation": "prefix"},
            {"pooling": "max", "truncation": "prefix"},
            {"pooling": "mean", "truncation": "prefix", "max_length": 1023},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                ExtractionConfig(**overrides)

    def test_embedding_policy_does_not_change_score_recipe_or_truncate_scores(self):
        legacy = ExtractionConfig(max_length=3)
        pooled = replace(legacy, cache_id="pooled", pooling="mean", truncation="prefix")
        recipes = []
        for config in (legacy, pooled):
            adapter = ESM2ScoreAdapter(config)
            adapter._manifest = {"files_sha256": {"model.safetensors": "a" * 64}}
            with patch.object(adapter, "_checkpoint"), patch("feature_generation.adapters.esm2.version", return_value="test"):
                recipes.append(adapter.score_metadata())
            with patch.object(adapter, "load", side_effect=AssertionError("must reject full reference before loading")):
                with self.assertRaises(ValueError):
                    list(adapter.iter_scores("ACDE", ["ACDV"]))
        self.assertEqual(recipes[0], recipes[1])
        self.assertEqual(recipes[1]["truncation"], "reject")
        self.assertNotIn("pooling", recipes[1])


class PrefixCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.index = self.raw / "DMS_substitutions.csv"
        self.reference = "ACDEG"
        self.ids = ("wt", "tail", "head", "both")
        self.sequences = ("ACDEG", "ACDEV", "VCDEG", "VCDEV")
        write_csv(self.index, ["DMS_id", "DMS_filename", "target_seq", "seq_len"], [
            ["tiny", "tiny.csv", self.reference, len(self.reference)],
        ])
        write_csv(self.raw / "tiny.csv", ["variant_id", "mutant", "mutated_sequence", "DMS_score"], [
            [variant_id, mutation, sequence, index / 2]
            for index, (variant_id, mutation, sequence) in enumerate(zip(
                self.ids, ("WT", "G5V", "A1V", "A1V:G5V"), self.sequences,
            ))
        ])
        self.config = ExtractionConfig(
            model_id="facebook/esm2_t6_8M_UR50D", cache_id="pooled",
            model_dir=self.root / "models", output_root=self.root / "features", assay_index=self.index,
            device="cpu", max_tokens_per_batch=12, max_length=3,
            pooling="mean", truncation="prefix",
        )
        self.assay_root = self.config.output_root / "tiny"
        self.leaf = self.assay_root / "sequence" / "esm2" / self.config.cache_id
        self.adapters = []

    def factory(self, config):
        adapter = PrefixAdapter(config)
        self.adapters.append(adapter)
        return adapter

    def run_extraction(self, **kwargs):
        return extract(self.config, ["tiny"], adapter_factory=self.factory, **kwargs)

    def write_score_source(self):
        # Scores independently cover complete sequences, including the tail mutation.
        score_config = replace(self.config, max_length=5, pooling="none", truncation="reject")
        return score(score_config, ["tiny"], adapter_factory=FakeESM2ScoreAdapter)[0]["score_source"]

    def load_bundle(self, source, **kwargs):
        options = dict(enabled_branches=["sequence_global"], score_sources=[source])
        options.update(kwargs)
        return load_feature_bundle(self.assay_root, **options)

    def reseal_file(self, name):
        """Exercise semantic validation rather than only the checksum guard."""
        path = self.leaf / "completion.json"
        completion = json.loads(path.read_text(encoding="utf-8"))
        completion["files_sha256"][name] = hashlib.sha256((self.leaf / name).read_bytes()).hexdigest()
        path.write_text(json.dumps(completion), encoding="utf-8")

    def test_cache_and_global_loader_preserve_full_identity_and_tail_mutations(self):
        result = self.run_extraction()[0]
        self.assertEqual(result["shape"], [4, 2])
        payload = torch.load(self.leaf / "features.pt", weights_only=True)
        expected = torch.stack([residue_values(sequence[:3]).mean(dim=0) for sequence in self.sequences])
        torch.testing.assert_close(payload["embeddings"], expected)
        self.assertEqual(payload["reference_sequence"], self.reference)
        self.assertEqual(payload["variant_ids"], list(self.ids))
        self.assertEqual(payload["sequence_hashes"], [hashlib.sha256(seq.encode()).hexdigest() for seq in self.sequences])
        torch.testing.assert_close(payload["positions"], torch.arange(1, 6))
        self.assertEqual({key: payload[key] for key in ("pooling", "truncation", "max_length", "represented_length")}, {
            "pooling": "mean", "truncation": "prefix", "max_length": 3, "represented_length": 3,
        })
        report = validate_sequence_cache(self.assay_root)
        self.assertEqual((report["reference_length"], report["embedding_dim"]), (5, 2))
        source = self.write_score_source()
        batch = self.load_bundle(source)
        self.assertEqual(batch.variant_ids, self.ids)
        self.assertEqual(batch.sequences, self.sequences)
        self.assertEqual(batch.reference_sequence, self.reference)
        self.assertEqual(batch.mutation_positions.tolist(), [5, 1, 1, 5])
        torch.testing.assert_close(batch.global_features, expected.double())
        self.assertEqual(batch.local_features, {})
        self.assertNotEqual(float(batch.scores[0, 0]), float(batch.scores[1, 0]))
        with self.assertRaisesRegex(ValueError, "sequence_site"):
            self.load_bundle(source, enabled_branches=["sequence_global", "sequence_site"])
        with self.assertRaises(ValueError):
            self.load_bundle(source, pooling="max")

    def test_output_budget_counts_pooled_vectors_and_still_guards_allocation(self):
        events = []
        # Four two-dimensional float32 vectors require 32 bytes, rather than 160.
        self.config = replace(self.config, max_output_gib=48 / 1024**3)
        self.run_extraction(progress=events.append)
        self.assertEqual(next(event["output_bytes"] for event in events if event["status"] == "extracting"), 32)
        self.config = replace(self.config, output_root=self.root / "too_small", max_output_gib=31 / 1024**3)
        with self.assertRaisesRegex(ValueError, "Output requires"):
            self.run_extraction()
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 1)

    def test_short_assay_records_its_actual_represented_length(self):
        self.config = replace(self.config, max_length=1022)
        self.run_extraction()
        payload = torch.load(self.leaf / "features.pt", weights_only=True)
        self.assertEqual(payload["represented_length"], len(self.reference))
        self.assertEqual(payload["max_length"], 1022)
        torch.testing.assert_close(payload["embeddings"], torch.stack([
            residue_values(sequence).mean(dim=0) for sequence in self.sequences
        ]))
        validate_sequence_cache(self.assay_root)

    def test_legacy_and_pooled_leaves_coexist_and_resume_without_replacing_defaults(self):
        legacy = replace(self.config, cache_id="legacy", max_length=5, pooling="none", truncation="reject")
        extract(legacy, ["tiny"], adapter_factory=FakeAdapter)
        original_default = json.loads((self.assay_root / "assay.json").read_text(encoding="utf-8"))["features"]
        self.run_extraction()
        self.assertEqual(self.run_extraction(resume=True)[0]["status"], "skipped")
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 1)
        self.assertEqual(json.loads((self.assay_root / "assay.json").read_text(encoding="utf-8"))["features"], original_default)
        self.assertEqual(validate_sequence_cache(self.assay_root)["cache_id"], "legacy")
        self.assertEqual(validate_sequence_cache(self.assay_root, "pooled")["cache_id"], "pooled")
        source = self.write_score_source()
        legacy_batch = self.load_bundle(source)
        pooled_batch = self.load_bundle(source, feature_selection={"sequence": "esm2/pooled"})
        self.assertFalse(torch.equal(legacy_batch.global_features, pooled_batch.global_features))
        self.assertNotEqual(legacy_batch.feature_signature, pooled_batch.feature_signature)
        self.config = replace(self.config, max_length=4)
        with self.assertRaisesRegex(ValueError, "new cache ID"):
            self.run_extraction(resume=True)

    def test_malformed_serialized_recipe_is_rejected_by_storage_and_loader(self):
        self.run_extraction()
        source = self.write_score_source()
        path = self.leaf / "config.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        for changes in (
            {"max_length": 3.0}, {"max_length": True}, {"pooling": True},
            {"truncation": True}, {"includes_special_tokens": 0}, {"hidden_size": 2.0},
        ):
            with self.subTest(changes=changes):
                path.write_text(json.dumps({**original, **changes}), encoding="utf-8")
                self.reseal_file("config.json")
                for check in (lambda: validate_sequence_cache(self.assay_root), lambda: self.load_bundle(source)):
                    with self.assertRaises(ValueError):
                        check()

    def test_undeclared_pooled_tensor_and_mismatched_prefix_metadata_are_rejected(self):
        self.run_extraction()
        source = self.write_score_source()
        config_path = self.leaf / "config.json"
        original_config = config_path.read_bytes()
        config = json.loads(original_config)
        config.update(pooling="none", truncation="reject")
        config_path.write_text(json.dumps(config), encoding="utf-8")
        self.reseal_file("config.json")
        for check in (lambda: validate_sequence_cache(self.assay_root), lambda: self.load_bundle(source)):
            with self.assertRaises(ValueError):
                check()
        config_path.write_bytes(original_config)
        self.reseal_file("config.json")
        path = self.leaf / "features.pt"
        original_payload = torch.load(path, weights_only=True)
        for changes in (
            {"represented_length": 2}, {"represented_length": 3.0}, {"represented_length": True},
            {"truncation": "reject"}, {"max_length": 4}, {"max_length": 3.0}, {"pooling": True},
        ):
            with self.subTest(changes=changes):
                torch.save({**original_payload, **changes}, path)
                self.reseal_file("features.pt")
                for check in (lambda: validate_sequence_cache(self.assay_root), lambda: self.load_bundle(source)):
                    with self.assertRaises(ValueError):
                        check()


if __name__ == "__main__":
    unittest.main()
