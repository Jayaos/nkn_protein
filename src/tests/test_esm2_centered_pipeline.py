"""Compact centered-window outputs retain variant identity and NKN inputs."""

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import torch

from data_loading import load_feature_bundle
from feature_generation.config import ExtractionConfig
from feature_generation.runner import extract
from feature_generation.score_runner import score
from feature_generation.storage import validate_sequence_cache, write_sequence_cache
from tests.test_esm2_pooling import PrefixAdapter
from tests.test_esm2_score_pipeline import FakeESM2ScoreAdapter
from tests.test_feature_generation import residue_values, write_csv


class FakeCenteredAdapter(PrefixAdapter):
    """Independent window oracle yielding variants in reversed cache order."""

    def metadata(self):
        return {
            **super().metadata(),
            "site_embeddings": "centered",
            "cache_layout": "mutation_sites_v1",
            "site_context": "mutated_sequence",
            "window_policy": "centered_clamped",
            "window_indexing": "zero_based_half_open",
        }

    def iter_site_embeddings(self, sequences, mutation_positions):
        for index in reversed(range(len(sequences))):
            sequence = sequences[index]
            length = min(len(sequence), self.config.max_length)
            global_vector = residue_values(sequence[:length]).mean(dim=0)
            site_values = []
            for position in mutation_positions[index]:
                start = max(0, min(position - 1 - length // 2, len(sequence) - length))
                site_values.append(residue_values(sequence[start:start + length])[position - 1 - start])
            sites = torch.stack(site_values) if site_values else torch.empty((0, self.hidden_size))
            yield index, global_vector, sites


class CenteredPipelineTests(unittest.TestCase):
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
            model_id="facebook/esm2_t6_8M_UR50D", cache_id="compact-centered",
            model_dir=self.root / "models", output_root=self.root / "features", assay_index=self.index,
            device="cpu", max_tokens_per_batch=12, max_length=4,
            pooling="mean", truncation="prefix", site_embeddings="centered",
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
        adapter = FakeCenteredAdapter(config)
        self.adapters.append(adapter)
        return adapter

    def run_extraction(self, **kwargs):
        return extract(self.config, ["tiny"], adapter_factory=self.factory, **kwargs)

    def load_bundle(self):
        # This embedding fixture intentionally uses an independent legacy
        # full-reference score recipe for its nine-residue reference.
        scoring_config = replace(self.config, site_embeddings="none", pooling="none", truncation="reject",
                                 max_length=len(self.reference), max_tokens_per_batch=len(self.reference) + 2)
        source = score(scoring_config, ["tiny"], resume=True,
                       adapter_factory=FakeESM2ScoreAdapter)[0]["score_source"]
        return load_feature_bundle(
            self.assay_root, feature_selection={"sequence": f"esm2/{self.config.cache_id}"},
            enabled_branches=["sequence_global", "sequence_site"], score_sources=[source],
        )

    def test_prefix_global_and_centered_sites_round_trip_without_dense_residue_cache(self):
        result = self.run_extraction()[0]
        self.assertEqual(result["shape"], [5, 2])
        self.assertEqual(result["site_shape"], [5, 2])
        payload = torch.load(self.leaf / "features.pt", weights_only=True)
        globals_expected = torch.stack([residue_values(sequence[:4]).mean(dim=0) for sequence in self.sequences])
        sites_expected = torch.tensor([
            [ord("V"), 4], [ord("V"), 1], [ord("V"), 1], [ord("V"), 4], [ord("W"), 3],
        ], dtype=torch.float32)
        torch.testing.assert_close(payload["embeddings"], globals_expected)
        torch.testing.assert_close(payload["site_embeddings"], sites_expected)
        self.assertEqual(payload["site_variant_indices"].tolist(), [1, 2, 3, 3, 4])
        self.assertEqual(payload["site_positions"].tolist(), [9, 1, 1, 9, 5])
        self.assertEqual(payload["site_window_starts"].tolist(), [5, 0, 0, 5, 2])
        self.assertEqual(payload["site_window_ends"].tolist(), [9, 4, 4, 9, 6])
        self.assertEqual(payload["variant_ids"], list(self.ids))
        self.assertEqual(payload["reference_sequence"], self.reference)
        self.assertEqual(payload["sequence_hashes"], [hashlib.sha256(seq.encode()).hexdigest() for seq in self.sequences])
        self.assertFalse(any(isinstance(value, torch.Tensor) and value.ndim == 3 for value in payload.values()))
        report = validate_sequence_cache(self.assay_root)
        self.assertEqual((report["reference_length"], report["embedding_dim"]), (9, 2))
        batch = self.load_bundle()
        self.assertEqual(batch.variant_ids, self.ids)
        self.assertEqual(batch.sequences, self.sequences)
        self.assertEqual(batch.owners.tolist(), [1, 2, 3, 3, 4])
        self.assertEqual(batch.mutation_positions.tolist(), [9, 1, 1, 9, 5])
        torch.testing.assert_close(batch.global_features, globals_expected.double())
        torch.testing.assert_close(batch.local_features["sequence_site"], sites_expected.double())
        torch.testing.assert_close(batch.global_features[0], batch.global_features[1])
        torch.testing.assert_close(batch.global_features[2], batch.global_features[3])

    def test_three_thousand_residues_keep_both_distant_sites_and_prefix_mean(self):
        self.reference = "A" * 3000
        self.ids = ("wt", "first", "second", "double")
        self.mutants = ("WT", "A1000V", "A2300K", "A1000V:A2300K")
        sequences = []
        for positions in ((), ((1000, "V"),), ((2300, "K"),), ((1000, "V"), (2300, "K"))):
            sequence = list(self.reference)
            for position, amino_acid in positions:
                sequence[position - 1] = amino_acid
            sequences.append("".join(sequence))
        self.sequences = tuple(sequences)
        self.write_assay()
        self.config = replace(self.config, max_length=1022, max_tokens_per_batch=1024)
        self.run_extraction()
        payload = torch.load(self.leaf / "features.pt", weights_only=True)
        self.assertEqual(payload["site_positions"].tolist(), [1000, 2300, 1000, 2300])
        self.assertEqual(payload["site_window_starts"].tolist(), [488, 1788, 488, 1788])
        self.assertEqual(payload["site_window_ends"].tolist(), [1510, 2810, 1510, 2810])
        torch.testing.assert_close(payload["site_embeddings"], torch.tensor([
            [ord("V"), 512], [ord("K"), 512], [ord("V"), 512], [ord("K"), 512],
        ], dtype=torch.float32))
        torch.testing.assert_close(payload["embeddings"][0], payload["embeddings"][2])
        torch.testing.assert_close(payload["embeddings"][1], payload["embeddings"][3])
        self.assertFalse(torch.equal(payload["embeddings"][0], payload["embeddings"][1]))
        self.assertEqual(validate_sequence_cache(self.assay_root)["reference_length"], 3000)

    def test_guard_counts_global_site_and_index_storage_before_loading_weights(self):
        # Five global + five site vectors, plus four int64 arrays for each site.
        output_bytes = (5 + 5) * 2 * 4 + 5 * 4 * 8
        self.config = replace(self.config, max_output_gib=output_bytes / 1024**3)
        events = []
        self.run_extraction(progress=events.append)
        self.assertEqual(next(event["output_bytes"] for event in events if event["status"] == "extracting"), output_bytes)
        self.config = replace(self.config, output_root=self.root / "too_small", max_output_gib=(output_bytes - 1) / 1024**3)
        with self.assertRaisesRegex(ValueError, "Output requires"):
            self.run_extraction()
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 1)

    def test_completed_resume_reuses_cache_and_changed_recipe_cannot_resume(self):
        self.run_extraction()
        with patch.object(FakeCenteredAdapter, "iter_site_embeddings", side_effect=AssertionError("must reuse cache")):
            self.assertEqual(self.run_extraction(resume=True)[0]["status"], "skipped")
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 1)
        self.config = replace(self.config, max_length=5)
        with self.assertRaisesRegex(ValueError, "new cache ID"):
            self.run_extraction(resume=True)
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 1)

    def test_wild_type_only_has_empty_site_tensor_and_loads_both_branches(self):
        self.ids, self.mutants, self.sequences = ("wt",), ("WT",), (self.reference,)
        self.write_assay()
        self.run_extraction()
        payload = torch.load(self.leaf / "features.pt", weights_only=True)
        self.assertEqual(payload["site_embeddings"].shape, (0, 2))
        self.assertEqual(payload["site_positions"].shape, (0,))
        batch = self.load_bundle()
        self.assertEqual(batch.local_features["sequence_site"].shape, (0, 2))
        self.assertEqual(batch.owners.numel(), 0)
        self.assertEqual(batch.global_features.shape, (1, 2))

    def test_invalid_adapter_outputs_never_publish_complete_cache(self):
        original = FakeCenteredAdapter.iter_site_embeddings
        for problem in ("duplicate", "missing", "index", "global_shape", "site_shape", "nonfinite", "dtype"):
            with self.subTest(problem=problem):
                self.config = replace(self.config, output_root=self.root / problem)

                def invalid(adapter, sequences, mutation_positions):
                    outputs = list(original(adapter, sequences, mutation_positions))
                    index, global_vector, sites = outputs[0]
                    if problem == "duplicate":
                        outputs.append(outputs[0])
                    elif problem == "missing":
                        outputs.pop()
                    elif problem == "index":
                        outputs[0] = (len(sequences), global_vector, sites)
                    elif problem == "global_shape":
                        outputs[0] = (index, global_vector[:1], sites)
                    elif problem == "site_shape":
                        outputs[0] = (index, global_vector, sites[:0])
                    elif problem == "nonfinite":
                        global_vector[0] = float("nan")
                    else:
                        outputs[0] = (index, global_vector.double(), sites)
                    yield from outputs

                with patch.object(FakeCenteredAdapter, "iter_site_embeddings", invalid):
                    with self.assertRaises(ValueError):
                        self.run_extraction()
                self.assertFalse((self.leaf / "completion.json").exists())

    def test_interrupted_assay_retries_and_continue_on_error_reports_failure(self):
        original = FakeCenteredAdapter.iter_site_embeddings

        def interrupted(adapter, sequences, mutation_positions):
            yield next(original(adapter, sequences, mutation_positions))
            raise RuntimeError("simulated interruption")

        with patch.object(FakeCenteredAdapter, "iter_site_embeddings", interrupted):
            reports = self.run_extraction(continue_on_error=True)
        self.assertEqual(reports[0]["status"], "failed")
        self.assertIn("simulated interruption", reports[0]["error"])
        self.assertFalse((self.leaf / "completion.json").exists())
        self.assertEqual(self.run_extraction(resume=True)[0]["status"], "completed")
        validate_sequence_cache(self.assay_root)

    def test_previous_assay_site_tensor_is_released_before_next_inference(self):
        write_csv(self.index, ["DMS_id", "DMS_filename", "target_seq", "seq_len"], [
            ["tiny", "tiny.csv", self.reference, len(self.reference)],
            ["second", "second.csv", self.reference, len(self.reference)],
        ])
        (self.raw / "second.csv").write_bytes((self.raw / "tiny.csv").read_bytes())
        previous_tensors = []
        original = FakeCenteredAdapter.iter_site_embeddings

        def recording_writer(*args, **kwargs):
            previous_tensors.append(weakref.ref(kwargs["site_data"]["site_embeddings"]))
            return write_sequence_cache(*args, **kwargs)

        def checking_iterator(adapter, sequences, mutation_positions):
            if previous_tensors:
                self.assertIsNone(previous_tensors[-1](), "Previous assay site cache is still retained")
            yield from original(adapter, sequences, mutation_positions)

        with patch("feature_generation.runner.write_sequence_cache", recording_writer), \
                patch.object(FakeCenteredAdapter, "iter_site_embeddings", checking_iterator):
            reports = extract(self.config, ["tiny", "second"], adapter_factory=self.factory)
        self.assertEqual([report["status"] for report in reports], ["completed", "completed"])
        self.assertEqual(len(previous_tensors), 2)
        self.assertTrue(all(reference() is None for reference in previous_tensors))


if __name__ == "__main__":
    unittest.main()
