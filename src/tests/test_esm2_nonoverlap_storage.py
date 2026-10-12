"""Full-sequence global means retain distinct recipes and unchanged site caches."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from data_loading import load_feature_bundle
from data_loading.features import parse_substitutions
from data_loading.sequence_windows import centered_window_bounds
from feature_generation.config import ExtractionConfig
from feature_generation.inputs import prepare_assay
from feature_generation.score_runner import score
from feature_generation.storage import check_existing, make_leaf_config, validate_sequence_cache, write_sequence_cache
from tests.test_esm2_score_pipeline import FakeESM2ScoreAdapter
from tests.test_feature_generation import residue_values, write_csv


class NonoverlapStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        index = self.root / "DMS_substitutions.csv"
        write_csv(index, ["DMS_id", "DMS_filename", "target_seq", "seq_len"], [["tiny", "tiny.csv", "ACDEG", 5]])
        write_csv(self.root / "tiny.csv", ["variant_id", "mutant", "mutated_sequence", "DMS_score"], [
            ["wt", "WT", "ACDEG", 0], ["tail", "G5V", "ACDEV", 1],
            ["head", "A1V", "VCDEG", 2], ["both", "G5V:A1V", "VCDEV", 3],
        ])
        self.inputs = prepare_assay(index, "tiny")
        self.assay_root = self.root / "features" / "tiny"
        self.config, self.leaf, self.globals, self.sites = self.write_cache("full", compact=True)
        score_config = ExtractionConfig(
            model_id="facebook/esm2_t6_8M_UR50D", model_dir=self.root / "models",
            output_root=self.root / "features", assay_index=index,
            device="cpu", max_length=5, max_tokens_per_batch=8,
        )
        self.score_source = score(score_config, ["tiny"], adapter_factory=FakeESM2ScoreAdapter)[0]["score_source"]

    def write_cache(self, cache_id, *, compact, full=True, max_length=3):
        metadata = {
            "model_family": "esm2", "hidden_size": 2, "pooling": "mean",
            "truncation": "none" if full else "prefix", "max_length": max_length,
            "includes_special_tokens": False,
        }
        if full:
            metadata.update(global_window="non_overlapping", global_window_aggregation="length_weighted_mean")
        if compact:
            metadata.update(
                site_embeddings="centered", cache_layout="mutation_sites_v1",
                site_context="mutated_sequence", window_policy="centered_clamped",
                window_indexing="zero_based_half_open",
            )
        config = make_leaf_config(cache_id, metadata)
        globals_ = torch.stack([
            torch.cat([residue_values(sequence[start:start + max_length])
                       for start in range(0, len(sequence), max_length)]).mean(dim=0)
            if full else residue_values(sequence[:max_length]).mean(dim=0)
            for sequence in self.inputs.sequences
        ])
        site_data = None
        if compact:
            owners, positions, starts, ends, values = [], [], [], [], []
            for index, row in enumerate(self.inputs.rows):
                for position in parse_substitutions(row["mutant"], row["mutated_sequence"], self.inputs.reference_sequence):
                    start, end = centered_window_bounds(position, self.inputs.length, max_length)
                    owners.append(index)
                    positions.append(position)
                    starts.append(start)
                    ends.append(end)
                    values.append(residue_values(row["mutated_sequence"][start:end])[position - 1 - start])
            site_data = {
                "site_embeddings": torch.stack(values), "site_variant_indices": torch.tensor(owners),
                "site_positions": torch.tensor(positions), "site_window_starts": torch.tensor(starts),
                "site_window_ends": torch.tensor(ends),
            }
        leaf = write_sequence_cache(self.assay_root, cache_id, self.inputs, config, globals_, {}, site_data=site_data)
        return config, leaf, globals_, site_data

    def load_bundle(self, cache_id="full", *, compact=True):
        return load_feature_bundle(
            self.assay_root, feature_selection={"sequence": f"esm2/{cache_id}"},
            enabled_branches=["sequence_global", "sequence_site"] if compact else ["sequence_global"],
            score_sources=[self.score_source],
        )

    def reseal(self, name, leaf=None):
        leaf = leaf or self.leaf
        path = leaf / "completion.json"
        completion = json.loads(path.read_text(encoding="utf-8"))
        completion["files_sha256"][name] = hashlib.sha256((leaf / name).read_bytes()).hexdigest()
        path.write_text(json.dumps(completion), encoding="utf-8")

    def assert_rejected(self, cache_id="full", *, compact=True):
        for action in (
            lambda: validate_sequence_cache(self.assay_root, cache_id, "esm2"),
            lambda: self.load_bundle(cache_id, compact=compact),
        ):
            with self.assertRaises(ValueError):
                action()

    def test_compact_loader_returns_saved_full_global_and_unchanged_centered_sites(self):
        payload = torch.load(self.leaf / "features.pt", weights_only=True)
        self.assertEqual(payload["represented_length"], self.inputs.length)
        self.assertEqual(payload["max_length"], 3)
        self.assertEqual(payload["truncation"], "none")
        self.assertEqual(payload["global_window"], "non_overlapping")
        self.assertEqual(payload["global_window_aggregation"], "length_weighted_mean")
        self.assertEqual(payload["reference_sequence"], self.inputs.reference_sequence)
        self.assertEqual(payload["sequence_hashes"], list(self.inputs.sequence_hashes))
        torch.testing.assert_close(payload["positions"], torch.arange(1, 6))
        self.assertEqual(validate_sequence_cache(self.assay_root, "full")["reference_length"], 5)
        batch = self.load_bundle()
        torch.testing.assert_close(batch.global_features, self.globals.double())
        torch.testing.assert_close(batch.local_features["sequence_site"], self.sites["site_embeddings"].double())
        self.assertEqual(batch.mutation_positions.tolist(), [5, 1, 5, 1])
        self.assertNotEqual(float(batch.global_features[0, 0]), float(batch.global_features[1, 0]))

    def test_global_only_full_cache_loads_without_site_data(self):
        _, leaf, values, _ = self.write_cache("global_only", compact=False)
        payload = torch.load(leaf / "features.pt", weights_only=True)
        self.assertNotIn("site_embeddings", payload)
        self.assertEqual(payload["represented_length"], 5)
        validate_sequence_cache(self.assay_root, "global_only")
        batch = self.load_bundle("global_only", compact=False)
        torch.testing.assert_close(batch.global_features, values.double())
        self.assertEqual(batch.local_features, {})
        with self.assertRaisesRegex(ValueError, "sequence_site"):
            self.load_bundle("global_only")

    def test_prefix_and_full_recipes_coexist_and_cannot_resume_under_each_others_identity(self):
        prefix_config, prefix_leaf, prefix_globals, prefix_sites = self.write_cache("prefix", compact=True, full=False)
        prefix_bytes = {path.name: path.read_bytes() for path in prefix_leaf.iterdir()}
        prefix_batch, full_batch = self.load_bundle("prefix"), self.load_bundle("full")
        self.assertFalse(torch.equal(prefix_globals, self.globals))
        self.assertNotEqual(prefix_batch.feature_signature, full_batch.feature_signature)
        for key in self.sites:
            torch.testing.assert_close(prefix_sites[key], self.sites[key])
        for cache_id, config in (("prefix", prefix_config), ("full", self.config)):
            self.assertTrue(check_existing(self.assay_root, cache_id, self.inputs, config, resume=True))
        with self.assertRaisesRegex(ValueError, "new cache ID"):
            check_existing(self.assay_root, "full", self.inputs, {**prefix_config, "checkpoint": "full"}, resume=True)
        self.assertEqual({path.name: path.read_bytes() for path in prefix_leaf.iterdir()}, prefix_bytes)
        self.assertNotIn("global_window", prefix_config)
        self.assertNotIn("global_window_aggregation", prefix_config)

    def test_full_config_rejects_missing_mixed_or_invalid_global_window_declarations(self):
        path = self.leaf / "config.json"
        for changes in (
            {"global_window": "prefix"}, {"global_window": "overlapping"},
            {"global_window_aggregation": "mean_of_window_means"},
            {"truncation": "prefix"}, {"truncation": "reject"}, {"pooling": "none"},
            {"max_length": 3.0}, {"max_length": True}, {"includes_special_tokens": 0},
        ):
            with self.subTest(changes=changes):
                path.write_text(json.dumps({**self.config, **changes}), encoding="utf-8")
                self.reseal("config.json")
                self.assert_rejected()
        for key in ("global_window", "global_window_aggregation"):
            with self.subTest(missing=key):
                changed = dict(self.config)
                del changed[key]
                path.write_text(json.dumps(changed), encoding="utf-8")
                self.reseal("config.json")
                self.assert_rejected()

    def test_full_payload_rejects_truncated_length_and_policy_mismatch(self):
        path = self.leaf / "features.pt"
        original = torch.load(path, weights_only=True)
        for changes in (
            {"represented_length": 3}, {"represented_length": 5.0}, {"represented_length": True},
            {"global_window": "prefix"}, {"global_window_aggregation": "mean"}, {"truncation": "prefix"},
        ):
            with self.subTest(changes=changes):
                torch.save({**original, **changes}, path)
                self.reseal("features.pt")
                self.assert_rejected()
        for key in ("global_window", "global_window_aggregation"):
            with self.subTest(missing=key):
                payload = dict(original)
                del payload[key]
                torch.save(payload, path)
                self.reseal("features.pt")
                self.assert_rejected()

    def test_short_prefix_payload_cannot_silently_claim_full_window_policy(self):
        # Both recipes represent all five residues here, so checking only the
        # represented length cannot catch an undeclared policy change.
        for compact in (False, True):
            cache_id = f"short_prefix_{compact}"
            _, leaf, _, _ = self.write_cache(cache_id, compact=compact, full=False, max_length=10)
            path = leaf / "features.pt"
            original = torch.load(path, weights_only=True)
            self.assertEqual(original["represented_length"], 5)
            for extras in (
                {"global_window": "non_overlapping"},
                {"global_window_aggregation": "length_weighted_mean"},
                {"global_window": "non_overlapping", "global_window_aggregation": "length_weighted_mean"},
            ):
                with self.subTest(compact=compact, extras=extras):
                    torch.save({**original, **extras}, path)
                    self.reseal("features.pt", leaf)
                    self.assert_rejected(cache_id, compact=compact)


if __name__ == "__main__":
    unittest.main()
