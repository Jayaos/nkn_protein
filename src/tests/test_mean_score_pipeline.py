"""Offline sequence-score generation, immutable caching, and GP mean integration."""

from __future__ import annotations

import contextlib
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from data_loading import load_feature_bundle
from feature_generation.cli import main
from feature_generation.config import ExtractionConfig
from feature_generation.inverse_config import InverseFoldingConfig
from feature_generation.score_runner import score
from feature_generation.score_storage import validate_score_cache
from tests.test_data import make_bundle, write_csv
from tests.test_feature_generation import FakeAdapter as FakeSequenceAdapter
from tests.test_inverse_pipeline import FakeAdapter as FakeInverseAdapter
from tests.test_inverse_pipeline import structure_metadata, refresh_structure_signature


def sequence_score(sequence):
    # Context-sensitive dummy output: a mutation changes its neighbors' terms.
    # This is deliberately not a sum over the mutated positions or WT-centered.
    return -(sum(ord(aa) for aa in sequence) + ord(sequence[0]) * ord(sequence[-1]) / 100) / 100


class FakeScoreAdapter:
    def __init__(self, config):
        self.config = config
        self.load_count = 0
        self.score_calls = []

    def score_metadata(self, *, num_orders=1):
        return {
            "model_id": self.config.model_id,
            "revision": self.config.revision,
            "checkpoint_sha256": "c" * 64,
            "scoring_recipe_version": 1,
            "score_definition": "negative_mean_sequence_nll",
            "conditioning_mode": "sequence_autoregressive",
            "inference_method": "forward",
            "num_orders": num_orders,
            "seed": self.config.seed,
            "decoding_order": "shared_random_orders",
            "normalization": "native_21_amino_acid_tokens",
            "alphabet": "ACDEFGHIKLMNPQRSTVWYX",
            "output_dtype": "float32",
            "score_direction": "higher_is_more_likely",
        }

    def load(self):
        self.load_count += 1

    def iter_scores(self, structure, sequences, *, batch_size=8, num_orders=1):
        if torch.is_grad_enabled():
            raise AssertionError("Scoring must disable gradients")
        if not all(isinstance(sequence, str) for sequence in sequences):
            raise AssertionError("The model must receive sequences, not labeled assay rows")
        self.score_calls.append((tuple(sequences), batch_size, num_orders))
        for stop in range(len(sequences), 0, -batch_size):
            indices = list(range(max(0, stop - batch_size), stop))[::-1]
            yield indices, torch.tensor([sequence_score(sequences[index]) for index in indices], dtype=torch.float32)


class MeanScorePipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.index = self.raw / "DMS_substitutions.csv"
        self.ids = ("wt", "first", "second", "double")
        self.mutants = ("WT", "A1V", "C2D", "A1V:C2D")
        self.sequences = ("ACD", "VCD", "ADD", "VDD")
        write_csv(self.index, ["DMS_id", "DMS_filename", "target_seq", "seq_len"], [["tiny", "tiny.csv", "ACD", 3]])
        self.write_raw_labels([0, 1, 2, 3])
        self.config = InverseFoldingConfig(
            model_family="proteinmpnn", cache_id="tiny_distributions",
            output_root=self.root / "features", assay_index=self.index,
            structure_manifest_dir=self.root / "manifests",
            model_dir=self.root / "models", max_length=10, seed=0, device="cpu",
        )
        self.metadata = structure_metadata()
        self.adapters = []
        self.structure_loader = patch("feature_generation.score_runner.load_structure", side_effect=self.load_structure)
        self.mock_structure = self.structure_loader.start()
        self.addCleanup(self.structure_loader.stop)

    def write_raw_labels(self, labels):
        write_csv(self.raw / "tiny.csv", ["variant_id", "mutant", "mutated_sequence", "DMS_score"], [
            [variant_id, mutant, sequence, label]
            for variant_id, mutant, sequence, label in zip(self.ids, self.mutants, self.sequences, labels)
        ])

    def load_structure(self, manifest_path, reference_sequence, require_oxygen=True):
        metadata = copy.deepcopy(self.metadata)
        return SimpleNamespace(
            metadata=metadata, reference_sequence=reference_sequence,
            positions=torch.arange(1, 4), coords=torch.zeros(3, 4, 3),
            structure_sha256=metadata["structure_sha256"], input_signature=metadata["input_signature"],
        )

    def factory(self, config):
        adapter = FakeScoreAdapter(config)
        self.adapters.append(adapter)
        return adapter

    @property
    def assay_root(self):
        return self.config.output_root / "tiny"

    def leaf(self, cache_id=None, num_orders=1):
        cache_id = cache_id or (
            f"{self.config.structure_source}--{self.config.model_id.split('/')[-1]}-sequence_log_likelihood-"
            f"orders{num_orders}-seed{self.config.seed}-{self.config.revision[:12]}"
        )
        return self.assay_root / "scores" / "proteinmpnn" / cache_id

    def run_score(self, **kwargs):
        return score(self.config, ["tiny"], adapter_factory=self.factory, **kwargs)

    def rows(self, leaf=None):
        from data_loading.features import read_csv
        return read_csv((leaf or self.leaf()) / "mean_score.csv")

    def expected_scores(self):
        return torch.tensor([sequence_score(sequence) for sequence in self.sequences], dtype=torch.float32)

    def validate(self, leaf=None):
        return validate_score_cache(self.assay_root, model_family="proteinmpnn", cache_id=(leaf or self.leaf()).name)

    def prepare_representations(self):
        from feature_generation.inverse_runner import extract as extract_inverse
        from feature_generation.runner import extract as extract_sequence
        with patch("feature_generation.inverse_runner.load_structure", side_effect=self.load_structure):
            extract_inverse(self.config, ["tiny"], adapter_factory=FakeInverseAdapter)
        sequence_config = ExtractionConfig(
            output_root=self.config.output_root, assay_index=self.index,
            model_dir=self.root / "sequence_models", device="cpu", cache_id="tiny_embeddings",
        )
        extract_sequence(sequence_config, ["tiny"], adapter_factory=FakeSequenceAdapter)

    def test_reordered_batches_store_one_raw_score_per_variant_without_labels(self):
        results = self.run_score(batch_size=2, num_orders=3)
        leaf = self.leaf(num_orders=3)
        self.assertEqual(results[0]["status"], "completed")
        self.assertEqual(results[0]["num_variants"], 4)
        self.assertEqual([row["variant_id"] for row in self.rows(leaf)], list(self.ids))
        values = torch.tensor([float(row["score"]) for row in self.rows(leaf)], dtype=torch.float32)
        torch.testing.assert_close(values, self.expected_scores())
        self.assertLess(float(values[0]), 0)  # WT retains its full-sequence likelihood.
        self.assertEqual(self.adapters[0].score_calls, [(self.sequences, 2, 3)])
        self.assertEqual(self.adapters[0].load_count, 1)
        self.assertTrue(self.mock_structure.call_args.kwargs["require_oxygen"])
        self.assertFalse((leaf / "zero_shot.csv").exists())
        config = json.loads((leaf / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["feature_type"], "scores")
        self.assertEqual(config["mean_score"], {"file": "mean_score.csv", "id_column": "variant_id", "score_column": "score"})
        provenance = json.loads((leaf / "provenance.json").read_text(encoding="utf-8"))
        self.assertIs(provenance["uses_dms_labels"], False)
        self.assertEqual(self.validate(leaf)["num_variants"], 4)

    def test_changing_experimental_labels_does_not_change_scores(self):
        self.run_score()
        original = (self.leaf() / "mean_score.csv").read_bytes()
        self.write_raw_labels([900, -400, 70, 23])
        self.config = replace(self.config, output_root=self.root / "other_features")
        self.run_score()
        self.assertEqual((self.leaf() / "mean_score.csv").read_bytes(), original)

    def test_resume_validates_and_skips_inference_without_touching_completed_files(self):
        self.run_score()
        original = {path.name: path.read_bytes() for path in self.leaf().iterdir() if path.is_file()}
        results = self.run_score(resume=True, batch_size=1)
        self.assertEqual(results[0]["status"], "skipped")
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 1)
        self.assertEqual(sum(len(adapter.score_calls) for adapter in self.adapters), 1)
        self.assertEqual({path.name: path.read_bytes() for path in self.leaf().iterdir() if path.is_file()}, original)
        with self.assertRaises(FileExistsError):
            self.run_score()

    def test_recipe_and_structure_changes_cannot_reuse_explicit_cache_id(self):
        self.run_score(cache_id="fixed_recipe", num_orders=2)
        with self.assertRaises(ValueError):
            self.run_score(cache_id="fixed_recipe", num_orders=3, resume=True)
        self.config = replace(self.config, seed=1)
        with self.assertRaises(ValueError):
            self.run_score(cache_id="fixed_recipe", num_orders=2, resume=True)
        self.config = replace(self.config, seed=0)
        self.metadata["structure_sha256"] = "f" * 64
        refresh_structure_signature(self.metadata)
        with self.assertRaises(ValueError):
            self.run_score(cache_id="fixed_recipe", num_orders=2, resume=True)
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 1)

    def test_variant_coverage_is_immutable_across_scoring_and_extraction(self):
        self.prepare_representations()
        original = (self.assay_root / "variants.csv").read_bytes()
        with self.assertRaises(ValueError):
            self.run_score(limit=2)
        self.assertEqual((self.assay_root / "variants.csv").read_bytes(), original)
        self.assertFalse(self.leaf().exists())
        self.run_score()
        self.assertEqual((self.assay_root / "variants.csv").read_bytes(), original)

    def test_mean_score_source_loads_with_two_kernel_branches_and_preserves_feature_caches(self):
        self.prepare_representations()
        features_before = {
            path.relative_to(self.assay_root): path.read_bytes()
            for family in ("sequence", "inverse_folding")
            for path in (self.assay_root / family).rglob("*") if path.is_file()
        }
        defaults_before = json.loads((self.assay_root / "assay.json").read_text(encoding="utf-8"))["features"]
        self.run_score()
        source = self.leaf().relative_to(self.assay_root).as_posix()
        batch = load_feature_bundle(
            self.assay_root, enabled_branches=["sequence_global", "inverse_folding"],
            score_sources=[source],
        )
        self.assertEqual(batch.variant_ids, self.ids)
        self.assertEqual(batch.score_names, (source,))
        self.assertEqual(batch.scores.shape, (4, 1))
        self.assertEqual(batch.global_features.shape, (4, 2))
        self.assertEqual(tuple(batch.local_features), ("inverse_folding",))
        torch.testing.assert_close(batch.scores[:, 0], self.expected_scores().double())
        torch.testing.assert_close(batch.targets, torch.arange(4, dtype=torch.float64))
        self.assertIn(f"score:{source}.config_sha256", batch.feature_signature)
        self.assertEqual(json.loads((self.assay_root / "assay.json").read_text(encoding="utf-8"))["features"], defaults_before)
        for relative, content in features_before.items():
            self.assertEqual((self.assay_root / relative).read_bytes(), content)

    def test_corrupt_or_incomplete_score_cache_is_rejected_by_validator_and_gp_loader(self):
        self.prepare_representations()
        self.run_score()
        source = self.leaf().relative_to(self.assay_root).as_posix()
        score_path = self.leaf() / "mean_score.csv"
        original = score_path.read_bytes()
        score_path.write_bytes(original.replace(b"wt,", b"unknown,"))
        for action in (
            self.validate,
            lambda: load_feature_bundle(self.assay_root, enabled_branches=["sequence_global"], score_sources=[source]),
        ):
            with self.assertRaises(ValueError):
                action()
        self.assertEqual(self.run_score(resume=True)[0]["status"], "completed")
        self.validate()
        (self.leaf() / "completion.json").unlink()
        for action in (
            self.validate,
            lambda: load_feature_bundle(self.assay_root, enabled_branches=["sequence_global"], score_sources=[source]),
        ):
            with self.assertRaises((ValueError, FileNotFoundError)):
                action()
        self.assertEqual(self.run_score(resume=True, cleanup_and_save=True)[0]["status"], "completed")
        self.validate()

    def test_invalid_model_batches_never_publish_scores(self):
        cases = {
            "nan": [([0, 1, 2, 3], torch.tensor([1.0, 2.0, float("nan"), 4.0]))],
            "duplicate": [([0, 0, 2, 3], torch.ones(4))],
            "missing": [([0, 1, 2], torch.ones(3))],
            "outside": [([0, 1, 2, 4], torch.ones(4))],
            "shape": [([0, 1, 2, 3], torch.ones(4, 1))],
        }
        for name, batches in cases.items():
            with self.subTest(case=name):
                self.config = replace(self.config, output_root=self.root / name)
                with patch.object(FakeScoreAdapter, "iter_scores", return_value=iter(batches)):
                    with self.assertRaises(ValueError):
                        self.run_score()
                self.assertFalse(self.leaf().exists())

    def test_invalid_options_and_missing_structure_fail_before_model_load(self):
        for kwargs in ({"batch_size": 0}, {"num_orders": 0}, {"batch_size": True}, {"num_orders": True}, {"cache_id": "../escape"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_score(**kwargs)
        self.config = replace(self.config, max_length=2)
        with self.assertRaisesRegex(ValueError, "max_length"):
            self.run_score()
        self.mock_structure.assert_not_called()
        self.config = replace(self.config, max_length=10)
        self.mock_structure.side_effect = FileNotFoundError("no structure manifest")
        with self.assertRaisesRegex(FileNotFoundError, "no structure manifest"):
            self.run_score()
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 0)

    def test_continue_on_error_and_separate_subset_output(self):
        results = score(self.config, ["missing", "tiny"], continue_on_error=True, adapter_factory=self.factory)
        self.assertEqual([result["status"] for result in results], ["failed", "completed"])
        self.config = replace(self.config, output_root=self.root / "singles")
        results = self.run_score(singles_only=True)
        self.assertEqual(results[0]["num_variants"], 2)
        self.assertEqual([row["variant_id"] for row in self.rows()], ["first", "second"])
        self.assertEqual(self.adapters[-1].score_calls[0][0], ("VCD", "ADD"))


class MeanScoreDescriptorTests(unittest.TestCase):
    def test_new_descriptor_and_legacy_descriptor_remain_compatible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_bundle(root)
            original = load_feature_bundle(root)
            leaf = root / "sequence/synthetic/v1"
            path = leaf / "config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            config["mean_score"] = config.pop("zero_shot")
            config["mean_score"]["file"] = "mean_score.csv"
            (leaf / "zero_shot.csv").rename(leaf / "mean_score.csv")
            path.write_text(json.dumps(config), encoding="utf-8")
            actual = load_feature_bundle(root)
            torch.testing.assert_close(actual.scores, original.scores)
            config["zero_shot"] = {**config["mean_score"], "score_column": "other"}
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_feature_bundle(root)


class MeanScoreCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "proteinmpnn.json"
        self.config.write_text(json.dumps({"model_family": "proteinmpnn"}), encoding="utf-8")

    def invoke(self, arguments):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            status = main([str(argument) for argument in arguments])
        return status, output.getvalue(), errors.getvalue()

    def test_score_routes_configuration_and_controls_to_sequence_scorer(self):
        with patch("feature_generation.score_runner.score", return_value=[{"assay_id": "tiny", "status": "completed"}]) as scorer, patch(
            "feature_generation.inverse_runner.extract"
        ) as extractor:
            status, output, errors = self.invoke([
                "score", "--config", self.config, "--assay", "tiny", "--assay", "other",
                "--assay-index", self.root / "index.csv", "--output-root", self.root / "outputs",
                "--structure-manifest-dir", self.root / "manifests", "--device", "cpu",
                "--batch-size", "2", "--num-orders", "3", "--cache-id", "scores_v1",
                "--limit", "4", "--singles-only", "--resume", "--continue-on-error", "--quiet",
            ])
        self.assertEqual(status, 0, errors)
        extractor.assert_not_called()
        config, ids = scorer.call_args.args
        self.assertIsInstance(config, InverseFoldingConfig)
        self.assertEqual(ids, ["tiny", "other"])
        self.assertEqual(config.output_root, (self.root / "outputs").resolve())
        self.assertEqual(config.assay_index, (self.root / "index.csv").resolve())
        self.assertEqual(config.structure_manifest_dir, (self.root / "manifests").resolve())
        self.assertEqual(config.device, "cpu")
        self.assertEqual(scorer.call_args.kwargs, {
            "batch_size": 2, "num_orders": 3, "cache_id": "scores_v1", "limit": 4,
            "singles_only": True, "resume": True, "cleanup_and_save": False,
            "continue_on_error": True, "progress": None,
        })
        self.assertEqual(json.loads(output)["results"][0]["status"], "completed")

    def test_score_all_assays_and_failed_report_return_nonzero(self):
        records = [SimpleNamespace(assay_id="tiny"), SimpleNamespace(assay_id="missing")]
        with patch("data_loading.AssayCatalog", return_value=records), patch("feature_generation.score_runner.score", return_value=[
            {"assay_id": "tiny", "status": "completed"}, {"assay_id": "missing", "status": "failed"},
        ]) as scorer:
            status, output, errors = self.invoke(["score", "--config", self.config, "--all-assays", "--continue-on-error"])
        self.assertEqual(status, 1, errors)
        self.assertEqual(scorer.call_args.args[1], ["tiny", "missing"])
        self.assertTrue(callable(scorer.call_args.kwargs["progress"]))
        self.assertEqual(json.loads(output)["results"][1]["status"], "failed")

    def test_score_validation_selects_scores_without_distributions(self):
        with patch("feature_generation.score_storage.validate_score_cache", return_value={"num_variants": 4}) as validator, patch(
            "feature_generation.inverse_storage.validate_inverse_cache"
        ) as inverse:
            status, output, errors = self.invoke([
                "validate", "--features", self.root, "--feature-type", "scores",
                "--model-family", "proteinmpnn", "--cache-id", "scores_v1",
            ])
        self.assertEqual(status, 0, errors)
        inverse.assert_not_called()
        validator.assert_called_once_with(self.root, model_family="proteinmpnn", cache_id="scores_v1")
        self.assertEqual(json.loads(output), {"num_variants": 4})

    def test_invalid_score_model_configs_are_rejected_before_inference(self):
        for family in ("unknown_model", "prosst"):
            with self.subTest(family=family):
                self.config.write_text(json.dumps({"model_family": family}), encoding="utf-8")
                with patch("feature_generation.adapters.proteinmpnn.ProteinMPNNAdapter") as adapter:
                    status, output, errors = self.invoke(["score", "--config", self.config, "--assay", "tiny"])
                self.assertEqual(status, 1)
                self.assertEqual(output, "")
                self.assertIn("error:", errors)
                adapter.assert_not_called()

    def test_esm_if1_score_routes_and_rejects_explicit_random_orders(self):
        self.config.write_text(json.dumps({"model_family": "esm_if1"}), encoding="utf-8")
        with patch("feature_generation.score_runner.score", return_value=[]) as scorer:
            status, _, errors = self.invoke(["score", "--config", self.config, "--assay", "tiny"])
            self.assertEqual(status, 0, errors)
            self.assertEqual(scorer.call_args.args[0].model_family, "esm_if1")
            self.assertIsNone(scorer.call_args.kwargs["batch_size"])
        with patch("feature_generation.score_runner.score") as scorer:
            status, _, errors = self.invoke(["score", "--config", self.config, "--assay", "tiny", "--num-orders", "1"])
            self.assertEqual(status, 1)
            self.assertIn("only to ProteinMPNN", errors)
            scorer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
