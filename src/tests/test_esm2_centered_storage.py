"""Compact ESM2 cache identity, mutation alignment, and malformed-cache checks."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from data_loading import load_feature_bundle
from data_loading.sequence_windows import compact_sequence_recipe, centered_window_bounds
from feature_generation.config import ExtractionConfig
from feature_generation.inputs import prepare_assay
from feature_generation.score_runner import score
from feature_generation.storage import make_leaf_config, validate_sequence_cache, write_sequence_cache
from tests.test_esm2_score_pipeline import FakeESM2ScoreAdapter
from tests.test_feature_generation import write_csv


class CenteredStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.index = self.raw / "DMS_substitutions.csv"
        write_csv(self.index, ["DMS_id", "DMS_filename", "target_seq", "seq_len"], [["tiny", "tiny.csv", "ACDEG", 5]])
        write_csv(self.raw / "tiny.csv", ["variant_id", "mutant", "mutated_sequence", "DMS_score"], [
            ["wt", "WT", "ACDEG", 0], ["tail", "G5V", "ACDEV", 1],
            ["head", "A1V", "VCDEG", 2], ["both", "G5V:A1V", "VCDEV", 3],
        ])
        self.inputs = prepare_assay(self.index, "tiny")
        self.assay_root = self.root / "features" / "tiny"
        self.config = make_leaf_config("centered", {
            "model_family": "esm2", "hidden_size": 2,
            "pooling": "mean", "truncation": "prefix", "max_length": 3,
            "includes_special_tokens": False, "site_embeddings": "centered",
            "cache_layout": "mutation_sites_v1", "site_context": "mutated_sequence",
            "window_policy": "centered_clamped", "window_indexing": "zero_based_half_open",
        })
        self.globals = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        self.sites = {
            "site_embeddings": torch.tensor([[15., 25.], [31., 41.], [55., 65.], [71., 81.]]),
            "site_variant_indices": torch.tensor([1, 2, 3, 3]),
            "site_positions": torch.tensor([5, 1, 5, 1]),
            "site_window_starts": torch.tensor([2, 0, 2, 0]),
            "site_window_ends": torch.tensor([5, 3, 5, 3]),
        }
        self.leaf = write_sequence_cache(
            self.assay_root, "centered", self.inputs, self.config, self.globals, {}, site_data=self.sites,
        )
        score_config = ExtractionConfig(
            model_id="facebook/esm2_t6_8M_UR50D", model_dir=self.root / "models",
            output_root=self.root / "features", assay_index=self.index, device="cpu",
            max_length=5, max_tokens_per_batch=8,
        )
        self.score_source = score(score_config, ["tiny"], adapter_factory=FakeESM2ScoreAdapter)[0]["score_source"]

    def load_bundle(self, branches=("sequence_global", "sequence_site")):
        return load_feature_bundle(self.assay_root, enabled_branches=branches, score_sources=[self.score_source])

    def reseal(self, name):
        completion_path = self.leaf / "completion.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["files_sha256"][name] = hashlib.sha256((self.leaf / name).read_bytes()).hexdigest()
        completion_path.write_text(json.dumps(completion), encoding="utf-8")

    def assert_rejected(self):
        for check in (lambda: validate_sequence_cache(self.assay_root), self.load_bundle):
            with self.assertRaises(ValueError):
                check()

    def test_cached_globals_and_sites_preserve_tail_mutations_annotation_order_and_wt(self):
        validate_sequence_cache(self.assay_root)
        batch = self.load_bundle()
        self.assertEqual(batch.variant_ids, ("wt", "tail", "head", "both"))
        self.assertEqual(batch.owners.tolist(), [1, 2, 3, 3])
        self.assertEqual(batch.mutation_positions.tolist(), [5, 1, 5, 1])
        torch.testing.assert_close(batch.global_features, self.globals.double())
        torch.testing.assert_close(batch.local_features["sequence_site"], self.sites["site_embeddings"].double())
        only_site = self.load_bundle(("sequence_site",))
        self.assertIsNone(only_site.global_features)
        torch.testing.assert_close(only_site.local_features["sequence_site"], batch.local_features["sequence_site"])

    def test_reordered_cached_variants_remap_global_and_site_features_consistently(self):
        path = self.leaf / "features.pt"
        payload = torch.load(path, weights_only=True)
        permutation, occurrences = [2, 0, 3, 1], [1, 2, 3, 0]
        payload["embeddings"] = payload["embeddings"][permutation]
        for name in ("variant_ids", "sequence_hashes"):
            payload[name] = [payload[name][index] for index in permutation]
        for name in self.sites:
            payload[name] = payload[name][occurrences]
        payload["site_variant_indices"] = torch.tensor([0, 2, 2, 3])
        torch.save(payload, path)
        self.reseal("features.pt")
        validate_sequence_cache(self.assay_root)
        batch = self.load_bundle()
        torch.testing.assert_close(batch.global_features, self.globals.double())
        torch.testing.assert_close(batch.local_features["sequence_site"], self.sites["site_embeddings"].double())
        self.assertEqual(batch.mutation_positions.tolist(), [5, 1, 5, 1])

    def test_wt_only_payload_accepts_empty_site_tensors(self):
        inputs = prepare_assay(self.index, "tiny", limit=1)
        root = self.root / "wt_only"
        sites = {key: value[:0] for key, value in self.sites.items()}
        leaf = write_sequence_cache(root, "centered", inputs, self.config, self.globals[:1], {}, site_data=sites)
        payload = torch.load(leaf / "features.pt", weights_only=True)
        self.assertEqual(payload["site_embeddings"].shape, (0, 2))
        self.assertEqual(payload["site_positions"].numel(), 0)
        validate_sequence_cache(root)

    def test_payload_tampering_is_rejected_even_with_matching_file_checksums(self):
        path = self.leaf / "features.pt"
        original = torch.load(path, weights_only=True)
        changes = [
            {"site_positions": torch.tensor([5, 1, 1, 5])},
            {"site_positions": torch.tensor([5, 1, 5, 5])},
            {"site_variant_indices": torch.tensor([0, 2, 3, 3])},
            {"site_variant_indices": torch.tensor([1, 2, 3, 3], dtype=torch.int32)},
            {"site_window_starts": torch.tensor([1, 0, 2, 0])},
            {"site_window_ends": torch.tensor([4, 3, 5, 3])},
            {"site_embeddings": self.sites["site_embeddings"].double()},
            {"site_embeddings": self.sites["site_embeddings"][:3]},
            {"site_embeddings": torch.full((4, 2), float("nan"))},
            {"embeddings": torch.full((4, 2), float("inf"))},
            {"represented_length": 3.0}, {"window_policy": "other"},
            {"sequence_hashes": ["invalid"] * 4},
            {"variant_ids": ["wt", "tail", "head", "head"]},
            {"variant_ids": None}, {"variant_ids": [["wt"], "tail", "head", "both"]},
            {"input_signature": "incorrect"},
        ]
        for change in changes:
            with self.subTest(change=list(change)):
                torch.save({**original, **change}, path)
                self.reseal("features.pt")
                self.assert_rejected()
        for key in self.sites:
            with self.subTest(missing=key):
                payload = dict(original)
                del payload[key]
                torch.save(payload, path)
                self.reseal("features.pt")
                self.assert_rejected()

    def test_recipe_partial_declarations_cannot_fall_back_to_pooled_or_legacy(self):
        path = self.leaf / "config.json"
        original = dict(self.config)
        changes = [
            {"site_embeddings": "none"}, {"site_context": "wild_type"},
            {"window_indexing": "one_based"}, {"max_length": True},
            {"max_length": 3.0}, {"max_length": 1023},
            {"includes_special_tokens": 0}, {"hidden_size": 2.0},
            {"cache_layout": "unknown"}, {"pooling": "none"},
        ]
        for change in changes:
            with self.subTest(change=change):
                path.write_text(json.dumps({**original, **change}), encoding="utf-8")
                self.reseal("config.json")
                self.assert_rejected()
        for key in ("site_embeddings", "cache_layout", "site_context", "window_policy", "window_indexing"):
            with self.subTest(missing=key):
                config = dict(original)
                del config[key]
                path.write_text(json.dumps(config), encoding="utf-8")
                self.reseal("config.json")
                self.assert_rejected()
        pooled_config = {key: value for key, value in original.items() if key not in {
            "site_embeddings", "cache_layout", "site_context", "window_policy", "window_indexing",
        }}
        path.write_text(json.dumps(pooled_config), encoding="utf-8")
        self.reseal("config.json")
        self.assert_rejected()

    def test_site_data_requires_exactly_the_compact_contract(self):
        for sites in (None, {**self.sites, "unexpected": torch.ones(1)}, {**self.sites, "site_positions": self.sites["site_positions"].double()}):
            with self.subTest(sites=None if sites is None else list(sites)), self.assertRaises(ValueError):
                write_sequence_cache(self.root / "bad", "centered", self.inputs, self.config, self.globals, {}, site_data=sites)
        legacy = {key: value for key, value in self.config.items() if key not in {
            "site_embeddings", "cache_layout", "site_context", "window_policy", "window_indexing",
        }}
        with self.assertRaisesRegex(ValueError, "site_data"):
            write_sequence_cache(self.root / "bad", "centered", self.inputs, legacy, self.globals, {}, site_data=self.sites)


class CenteredWindowContractTests(unittest.TestCase):
    def test_window_coordinates_cover_tail_and_clamp_both_ends(self):
        self.assertEqual(centered_window_bounds(1000, 3000, 1022), (488, 1510))
        self.assertEqual(centered_window_bounds(2300, 3000, 1022), (1788, 2810))
        self.assertEqual(centered_window_bounds(1, 3000, 1022), (0, 1022))
        self.assertEqual(centered_window_bounds(3000, 3000, 1022), (1978, 3000))
        self.assertEqual(centered_window_bounds(10, 10, 1022), (0, 10))
        for position in (0, 3001, True, 1.0):
            with self.subTest(position=position), self.assertRaises(ValueError):
                centered_window_bounds(position, 3000, 1022)

    def test_legacy_config_has_no_compact_recipe(self):
        self.assertIsNone(compact_sequence_recipe({"model_family": "esm2"}))
        self.assertIsNone(compact_sequence_recipe({"site_embeddings": "none"}))


if __name__ == "__main__":
    unittest.main()
