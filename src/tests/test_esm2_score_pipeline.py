"""ESM2 masked-marginal score caches and independent GP mean integration."""

from __future__ import annotations

import contextlib
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
from gp.training import FittedGP, fit_gp
from tests.test_data import write_csv
from tests.test_feature_generation import FakeAdapter as FakeSequenceAdapter
from tests.test_inverse_pipeline import FakeAdapter as FakeInverseAdapter, structure_metadata
from tests.test_mean_score_pipeline import FakeScoreAdapter as FakeProteinMPNNScoreAdapter, sequence_score
from tests.test_pipeline import small_config


def masked_score(reference, sequence):
    # A transparent additive WT-relative oracle for exercising the pipeline.
    return sum((index + 1) * (ord(mutant) - ord(wild_type)) / 100
               for index, (wild_type, mutant) in enumerate(zip(reference, sequence)))


class FakeESM2ScoreAdapter:
    def __init__(self, config):
        self.config = config
        self.load_count = 0
        self.score_calls = []

    def score_metadata(self):
        return {
            "model_family": "esm2",
            "model_id": self.config.model_id,
            "revision": self.config.revision,
            "checkpoint_sha256": "c" * 64,
            "scoring_recipe_version": 1,
            "score_definition": "sum_masked_marginal_log_odds",
            "conditioning_mode": "masked_wild_type",
            "inference_method": "masked_marginals",
            "normalization": "model_vocabulary",
            "inference_dtype": self.config.inference_dtype,
            "output_dtype": "float32",
            "score_direction": "higher_is_more_likely",
        }

    def load(self):
        self.load_count += 1

    def iter_scores(self, reference_sequence, sequences, *, batch_size=8):
        if torch.is_grad_enabled():
            raise AssertionError("Scoring must disable gradients")
        if not isinstance(reference_sequence, str) or not all(isinstance(item, str) for item in sequences):
            raise AssertionError("Scoring must receive sequences without assay labels")
        self.score_calls.append((reference_sequence, tuple(sequences), batch_size))
        for stop in range(len(sequences), 0, -batch_size):
            indices = list(range(max(0, stop - batch_size), stop))[::-1]
            yield indices, torch.tensor(
                [masked_score(reference_sequence, sequences[index]) for index in indices], dtype=torch.float32,
            )


class ESM2ScorePipelineTests(unittest.TestCase):
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
        self.write_labels([0, 1, 2, 3])
        self.config = ExtractionConfig(
            model_id="facebook/esm2_t6_8M_UR50D", cache_id="tiny_embeddings",
            output_root=self.root / "features", assay_index=self.index,
            model_dir=self.root / "models", max_length=10, device="cpu",
            max_tokens_per_batch=20,
        )
        self.adapters = []
        self.structure_patch = patch(
            "feature_generation.score_runner.load_structure", side_effect=AssertionError("ESM2 must not need structures"),
        )
        self.mock_structure = self.structure_patch.start()
        self.addCleanup(self.structure_patch.stop)

    @property
    def assay_root(self):
        return self.config.output_root / "tiny"

    def leaf(self, cache_id=None):
        cache_id = cache_id or (
            f"{self.config.model_id.split('/')[-1]}-masked_marginals-"
            f"{self.config.inference_dtype}-{self.config.revision[:12]}"
        )
        return self.assay_root / "scores" / "esm2" / cache_id

    def write_labels(self, labels):
        write_csv(self.raw / "tiny.csv", ["variant_id", "mutant", "mutated_sequence", "DMS_score"], [
            [variant_id, mutant, sequence, label]
            for variant_id, mutant, sequence, label in zip(self.ids, self.mutants, self.sequences, labels)
        ])

    def factory(self, config):
        adapter = FakeESM2ScoreAdapter(config)
        self.adapters.append(adapter)
        return adapter

    def run_score(self, **kwargs):
        return score(self.config, ["tiny"], adapter_factory=self.factory, **kwargs)

    def rows(self):
        from data_loading.features import read_csv
        return read_csv(self.leaf() / "mean_score.csv")

    def expected_scores(self):
        return torch.tensor([masked_score("ACD", sequence) for sequence in self.sequences], dtype=torch.float32)

    def validate(self):
        return validate_score_cache(self.assay_root, model_family="esm2", cache_id=self.leaf().name)

    def prepare_sequence_representations(self):
        from feature_generation.runner import extract
        extract(self.config, ["tiny"], adapter_factory=FakeSequenceAdapter)

    def proteinmpnn_config(self):
        return InverseFoldingConfig(
            model_family="proteinmpnn", cache_id="tiny_distributions",
            output_root=self.config.output_root, assay_index=self.index,
            structure_manifest_dir=self.root / "manifests", model_dir=self.root / "proteinmpnn_models",
            max_length=10, device="cpu",
        )

    def fake_structure(self, *_args, **_kwargs):
        metadata = structure_metadata()
        return SimpleNamespace(
            metadata=metadata, reference_sequence="ACD", positions=torch.arange(1, 4),
            coords=torch.zeros(3, 4, 3), structure_sha256=metadata["structure_sha256"],
            input_signature=metadata["input_signature"],
        )

    def test_standalone_scoring_aligns_variants_and_writes_mean_score_without_structures(self):
        result = self.run_score(batch_size=2)[0]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["model_family"], "esm2")
        self.assertEqual(result["num_variants"], 4)
        self.assertEqual(result["score_source"], self.leaf().relative_to(self.assay_root).as_posix())
        self.assertEqual([row["variant_id"] for row in self.rows()], list(self.ids))
        values = torch.tensor([float(row["score"]) for row in self.rows()], dtype=torch.float32)
        torch.testing.assert_close(values, self.expected_scores())
        self.assertEqual(float(values[0]), 0)
        self.assertAlmostEqual(float(values[3]), float(values[1] + values[2]))
        self.assertEqual(self.adapters[0].score_calls, [("ACD", self.sequences, 2)])
        self.mock_structure.assert_not_called()
        self.assertFalse((self.assay_root / "sequence").exists())
        self.assertFalse((self.assay_root / "inverse_folding").exists())
        self.assertFalse((self.root / "manifests").exists())
        config = json.loads((self.leaf() / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["mean_score"], {"file": "mean_score.csv", "id_column": "variant_id", "score_column": "score"})
        self.assertEqual(config["score_definition"], "sum_masked_marginal_log_odds")
        self.assertIsNone(config.get("structure"))
        provenance = json.loads((self.leaf() / "provenance.json").read_text(encoding="utf-8"))
        self.assertIs(provenance["uses_dms_labels"], False)
        self.assertIsNone(provenance.get("structure_manifest"))
        self.assertEqual(self.validate()["num_variants"], 4)
        self.assertFalse((self.leaf() / "zero_shot.csv").exists())

    def test_score_generation_does_not_depend_on_experimental_labels(self):
        self.run_score()
        first = (self.leaf() / "mean_score.csv").read_bytes()
        self.write_labels([-400, 900, -30, 40])
        self.config = replace(self.config, output_root=self.root / "different_labels")
        self.run_score()
        self.assertEqual((self.leaf() / "mean_score.csv").read_bytes(), first)

    def test_resume_skips_inference_and_reuses_cache_across_batch_settings(self):
        self.run_score(batch_size=2)
        first = {path.name: path.read_bytes() for path in self.leaf().iterdir() if path.is_file()}
        self.config = replace(self.config, max_tokens_per_batch=30)
        self.assertEqual(self.run_score(batch_size=3, resume=True)[0]["status"], "skipped")
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 1)
        self.assertEqual(sum(len(adapter.score_calls) for adapter in self.adapters), 1)
        self.assertEqual({path.name: path.read_bytes() for path in self.leaf().iterdir() if path.is_file()}, first)
        with self.assertRaises(FileExistsError):
            self.run_score()

    def test_inference_recipe_and_checkpoint_changes_require_a_new_cache(self):
        original = self.config
        self.run_score(cache_id="fixed")
        for changed in (replace(original, inference_dtype="bfloat16"), replace(original, revision="a" * 40)):
            with self.subTest(config=changed):
                self.config = changed
                with self.assertRaises(ValueError):
                    self.run_score(cache_id="fixed", resume=True)
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 1)

    def test_shared_variant_coverage_and_representation_files_are_immutable(self):
        self.prepare_sequence_representations()
        first = {
            path.relative_to(self.assay_root): path.read_bytes()
            for path in (self.assay_root / "sequence").rglob("*") if path.is_file()
        }
        variant_bytes = (self.assay_root / "variants.csv").read_bytes()
        defaults = json.loads((self.assay_root / "assay.json").read_text(encoding="utf-8"))["features"]
        with self.assertRaises(ValueError):
            self.run_score(limit=2)
        self.assertFalse(self.leaf().exists())
        self.run_score()
        self.assertEqual((self.assay_root / "variants.csv").read_bytes(), variant_bytes)
        self.assertEqual(json.loads((self.assay_root / "assay.json").read_text(encoding="utf-8"))["features"], defaults)
        for path, contents in first.items():
            self.assertEqual((self.assay_root / path).read_bytes(), contents)

    def test_esm_mean_source_alone_and_with_proteinmpnn_support_partial_branch_fit(self):
        self.prepare_sequence_representations()
        self.run_score()
        esm_source = self.leaf().relative_to(self.assay_root).as_posix()
        batch = load_feature_bundle(self.assay_root, enabled_branches=["sequence_global"], score_sources=[esm_source])
        self.assertEqual(batch.score_names, (esm_source,))
        self.assertEqual(batch.local_features, {})
        torch.testing.assert_close(batch.scores[:, 0], self.expected_scores().double())
        # Adding a second score family must not reorder or replace the first column.
        with patch("feature_generation.score_runner.load_structure", side_effect=self.fake_structure):
            protein_result = score(self.proteinmpnn_config(), ["tiny"], adapter_factory=FakeProteinMPNNScoreAdapter)[0]
        protein_source = protein_result["score_source"]
        from feature_generation.inverse_runner import extract
        with patch("feature_generation.inverse_runner.load_structure", side_effect=self.fake_structure):
            extract(self.proteinmpnn_config(), ["tiny"], adapter_factory=FakeInverseAdapter)
        batch = load_feature_bundle(
            self.assay_root, enabled_branches=["sequence_global", "inverse_folding"],
            score_sources=[protein_source, esm_source],
        )
        self.assertEqual(batch.variant_ids, self.ids)
        self.assertEqual(batch.score_names, (protein_source, esm_source))
        torch.testing.assert_close(batch.scores[:, 0], torch.tensor([sequence_score(s) for s in self.sequences]).double())
        torch.testing.assert_close(batch.scores[:, 1], self.expected_scores().double())
        self.assertIn(f"score:{esm_source}.config_sha256", batch.feature_signature)
        config = small_config()
        config = replace(config, model=replace(config.model, enabled_branches=("inverse_folding", "sequence_global")))
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)
        fitted = fit_gp(batch.select([0, 1, 2]), config.model, config.training)
        test = replace(batch.select([3]), targets=None)
        before = fitted.predict(test)
        self.assertTrue(torch.isfinite(before.mean).all())
        checkpoint = self.root / "gp.pt"
        fitted.save(checkpoint)
        restored = FittedGP.load(checkpoint)
        self.assertEqual(restored.train.score_names, (protein_source, esm_source))
        after = restored.predict(test)
        torch.testing.assert_close(after.mean, before.mean)
        torch.testing.assert_close(after.observation_covariance, before.observation_covariance)

    def test_corrupt_and_incomplete_scores_are_rejected_by_resume_and_gp_loader(self):
        self.prepare_sequence_representations()
        self.run_score()
        source = self.leaf().relative_to(self.assay_root).as_posix()
        path = self.leaf() / "mean_score.csv"
        original = path.read_bytes()
        path.write_bytes(original.replace(b"wt,", b"unknown,"))
        actions = (
            self.validate,
            lambda: self.run_score(resume=True),
            lambda: load_feature_bundle(self.assay_root, enabled_branches=["sequence_global"], score_sources=[source]),
        )
        for action in actions:
            with self.assertRaises(ValueError):
                action()
        path.write_bytes(original)
        (self.leaf() / "completion.json").unlink()
        for action in actions:
            with self.assertRaises((ValueError, FileNotFoundError)):
                action()

    def test_missing_or_nonfinite_score_batches_never_publish_cache(self):
        cases = {
            "missing": [([0, 1, 2], torch.zeros(3))],
            "nonfinite": [([0, 1, 2, 3], torch.tensor([0., 1., float("nan"), 2.]))],
        }
        for name, batches in cases.items():
            self.config = replace(self.config, output_root=self.root / name)
            with self.subTest(case=name), patch.object(FakeESM2ScoreAdapter, "iter_scores", return_value=iter(batches)):
                with self.assertRaises(ValueError):
                    self.run_score()
            self.assertFalse(self.leaf().exists())

    def test_invalid_controls_fail_before_model_load(self):
        for arguments in ({"num_orders": 2}, {"num_orders": 0}, {"batch_size": 0}, {"batch_size": True}, {"cache_id": "../escape"}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.run_score(**arguments)
        self.config = replace(self.config, max_length=2)
        with self.assertRaisesRegex(ValueError, "max_length"):
            self.run_score()
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 0)
        self.mock_structure.assert_not_called()

    def test_continue_on_error_and_explicit_subset_use_separate_caches(self):
        results = score(self.config, ["missing", "tiny"], continue_on_error=True, adapter_factory=self.factory)
        self.assertEqual([result["status"] for result in results], ["failed", "completed"])
        self.config = replace(self.config, output_root=self.root / "single_mutants")
        self.assertEqual(self.run_score(singles_only=True)[0]["num_variants"], 2)
        self.assertEqual([row["variant_id"] for row in self.rows()], ["first", "second"])
        self.assertEqual(self.adapters[-1].score_calls, [("ACD", ("VCD", "ADD"), 8)])


class ESM2MeanScoreCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "esm2.json"
        self.config.write_text(json.dumps({"model_id": "facebook/esm2_t6_8M_UR50D"}), encoding="utf-8")

    def invoke(self, arguments):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            status = main([str(argument) for argument in arguments])
        return status, output.getvalue(), errors.getvalue()

    def test_sequence_score_routes_token_budget_batch_cap_and_cache_selection(self):
        with patch("feature_generation.score_runner.score", return_value=[{"assay_id": "tiny", "status": "completed"}]) as scorer:
            status, output, errors = self.invoke([
                "score", "--config", self.config, "--assay", "tiny", "--assay", "other",
                "--assay-index", self.root / "index.csv", "--output-root", self.root / "outputs",
                "--device", "cpu", "--max-tokens-per-batch", "256", "--batch-size", "4",
                "--cache-id", "masked_v1", "--limit", "4", "--singles-only", "--resume",
                "--continue-on-error", "--quiet",
            ])
        self.assertEqual(status, 0, errors)
        config, ids = scorer.call_args.args
        self.assertIsInstance(config, ExtractionConfig)
        self.assertEqual(ids, ["tiny", "other"])
        self.assertEqual(config.max_tokens_per_batch, 256)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.output_root, (self.root / "outputs").resolve())
        self.assertEqual(config.assay_index, (self.root / "index.csv").resolve())
        self.assertEqual(scorer.call_args.kwargs["batch_size"], 4)
        self.assertEqual(scorer.call_args.kwargs["cache_id"], "masked_v1")
        self.assertEqual(scorer.call_args.kwargs["limit"], 4)
        self.assertTrue(scorer.call_args.kwargs["singles_only"])
        self.assertTrue(scorer.call_args.kwargs["resume"])
        self.assertTrue(scorer.call_args.kwargs["continue_on_error"])
        self.assertIsNone(scorer.call_args.kwargs["progress"])
        self.assertEqual(json.loads(output)["results"][0]["status"], "completed")

    def test_esm_rejects_structure_and_any_explicit_decoding_order_flag(self):
        for options in (["--structure-manifest-dir", self.root / "manifests"], ["--num-orders", "1"], ["--num-orders", "2"]):
            with self.subTest(options=options), patch("feature_generation.score_runner.score") as scorer:
                status, output, errors = self.invoke(["score", "--config", self.config, "--assay", "tiny", *options])
            self.assertEqual(status, 1)
            self.assertEqual(output, "")
            self.assertIn("error:", errors)
            self.assertIn(str(options[0]), errors)
            scorer.assert_not_called()

    def test_esm_score_validation_does_not_load_representations(self):
        with patch("feature_generation.score_storage.validate_score_cache", return_value={"num_variants": 4}) as validator, patch(
            "feature_generation.storage.validate_sequence_cache",
        ) as representation:
            status, output, errors = self.invoke([
                "validate", "--features", self.root, "--feature-type", "scores", "--model-family", "esm2", "--cache-id", "masked_v1",
            ])
        self.assertEqual(status, 0, errors)
        validator.assert_called_once_with(self.root, model_family="esm2", cache_id="masked_v1")
        representation.assert_not_called()
        self.assertEqual(json.loads(output), {"num_variants": 4})


if __name__ == "__main__":
    unittest.main()
