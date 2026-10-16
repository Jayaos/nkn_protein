"""Assay restrictions through real caches, kernel math, CLI, and Slurm preflight."""

import contextlib
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from assay_policies import PROTEINMPNN_SCORE_SKIP_REASONS
from baselines.kermut.cli import main
from baselines.kermut.config import KermutConfig
from baselines.kermut.data import load_feature_bundle
from baselines.kermut.kernel import KermutKernel
from baselines.kermut.runner import run_cv, run_extrapolation, run_predict
from baselines.kermut.training import FittedKermut, fit_gp
from feature_generation.config import ExtractionConfig
from feature_generation.inputs import prepare_assay
from feature_generation.inverse_config import InverseFoldingConfig
from feature_generation.inverse_runner import extract as extract_inverse
from feature_generation.runner import extract as extract_sequence
from feature_generation.score_storage import make_score_config, write_score_cache
from feature_generation.structure_fragments import load_inverse_structure, prepare_structure_fragments
from feature_generation.structure_inputs import prepare_structure
from feature_generation.structure_mappings import CAS9_ASSAY, DOMAIN_RANGES, P53_ASSAYS
from tests.test_feature_generation import FakeAdapter as SequenceAdapter
from tests.test_inverse_fragments import FragmentAdapter
from tests.test_structure_inputs import pdb_text
from fixtures import scalar_kernel, small_config, synthetic_batch, write_csv


BRCA2 = "BRCA2_HUMAN_Erwood_2022_HEK293T"
REPOSITORY = Path(__file__).resolve().parents[3]


class PooledSequenceAdapter(SequenceAdapter):
    def metadata(self):
        return dict(super().metadata(), pooling="mean", truncation="none", includes_special_tokens=False,
                    max_length=self.config.max_length, global_window="non_overlapping",
                    global_window_aggregation="length_weighted_mean")

    def iter_embeddings(self, sequences):
        for indices, values in super().iter_embeddings(sequences):
            yield indices, values.mean(dim=1)


class AssayExceptionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def make_assay(self, assay):
        root = self.root / assay
        root.mkdir()
        if assay in DOMAIN_RANGES:
            length, start, end, filename = DOMAIN_RANGES[assay]
            reference = "A" * length
        else:
            reference, start, end, filename = "ACDVA", 1, 5, "BRCA2_HUMAN.pdb"
        mutations = [[(position, aa)] for position in range(start, start + 5) for aa in "WY"]
        mutations += [[(first, "W"), (second, "Y")] for first in range(start, start + 5)
                      for second in range(first + 1, start + 5)]
        rows = []
        for index, changes in enumerate(mutations):
            sequence = list(reference)
            mutant = ":".join(f"{reference[p - 1]}{p}{aa}" for p, aa in changes)
            for position, aa in changes:
                sequence[position - 1] = aa
            rows.append([mutant, "".join(sequence), index / 10 + (index % 3) * 0.1])
        catalog = root / "index.csv"
        write_csv(catalog, ["DMS_id", "DMS_filename", "target_seq", "pdb_file"], [[assay, "variants.csv", reference, filename]])
        write_csv(root / "variants.csv", ["mutant", "mutated_sequence", "DMS_score"], rows)
        manifests = root / "manifests"
        if assay == BRCA2:
            files = [root / "first.pdb", root / "second.pdb"]
            files[0].write_text(pdb_text(reference[:2]))
            files[1].write_text(pdb_text(reference[2:]))
            prepare_structure_fragments(catalog, assay, files, [(1, 2), (3, 5)], output_dir=manifests)
        else:
            pdb = root / filename
            pdb.write_text(pdb_text(reference[start - 1:end]))
            prepare_structure(catalog, assay, pdb, output_dir=manifests)
        inverse = InverseFoldingConfig(assay_index=catalog, output_root=root / "features", model_dir=root / "models",
                                       structure_manifest_dir=manifests, device="cpu", conditioning_mode="sequence_conditional")
        sequence = ExtractionConfig(assay_index=catalog, output_root=inverse.output_root, model_dir=root / "sequence_model",
                                    device="cpu", pooling="mean", global_window="non_overlapping", truncation="none")
        extract_inverse(inverse, [assay], adapter_factory=FragmentAdapter)
        extract_sequence(sequence, [assay], adapter_factory=PooledSequenceAdapter)
        features = inverse.output_root / assay
        inputs = prepare_assay(catalog, assay)
        score_config = make_score_config("test", "esm2", {
            "scoring_recipe_version": 1, "score_definition": "sum_masked_marginal_log_odds",
            "conditioning_mode": "masked_wild_type", "inference_method": "masked_marginals",
            "normalization": "model_vocabulary", "output_dtype": "float32",
        })
        write_score_cache(features, "esm2", "test", inputs, score_config, torch.arange(20, dtype=torch.float32) / 20,
                          {"uses_dms_labels": False})
        config = replace(small_config(n_steps=1), feature_selection={
            "sequence": f"esm2/{sequence.cache_id}", "inverse_folding": f"proteinmpnn/{inverse.cache_id}"},
            score_sources=("scores/esm2/test", "scores/proteinmpnn/absent"), structure_manifest_dir=str(manifests))
        return features, config

    def test_policies_preserve_original_config_and_unaffected_assays(self):
        config = KermutConfig.load(REPOSITORY / "configs/kermut_esm2_proteinmpnn.json")
        for assay in PROTEINMPNN_SCORE_SKIP_REASONS:
            resolved = config.for_assay(assay)
            self.assertEqual(resolved.score_sources, config.score_sources[:1])
            self.assertEqual(resolved.model.use_structure_distance, assay != BRCA2)
            self.assertIs(resolved.for_assay(assay), resolved)
            self.assertEqual(KermutConfig.from_dict(resolved.to_dict()), resolved)
            with self.assertRaisesRegex(ValueError, "configure an ESM2"):
                replace(config, score_sources=config.score_sources[1:]).for_assay(assay)
        self.assertEqual(len(config.score_sources), 2)
        self.assertTrue(config.model.use_structure_distance)
        for assay in [CAS9_ASSAY, *P53_ASSAYS, "another_assay"]:
            self.assertIs(config.for_assay(assay), config)
        with self.assertRaisesRegex(ValueError, "boolean"):
            replace(config.model, use_structure_distance=1)

    def test_domain_caches_load_without_proteinmpnn_scores_and_keep_correct_coordinates(self):
        for assay in DOMAIN_RANGES:
            with self.subTest(assay=assay):
                features, config = self.make_assay(assay)
                batch = load_feature_bundle(features, config)
                self.assertEqual(batch.score_names, ("scores/esm2/test",))
                structure = load_inverse_structure(Path(config.structure_manifest_dir) / f"{assay}.json", batch.reference_sequence)
                indices = batch.mutation_positions - structure.positions[0]
                torch.testing.assert_close(batch.local_features["structure"], structure.coords[indices, 1].double())
                fitted = fit_gp(batch.select(range(8)), config.model, config.training)
                self.assertTrue(fitted.model_config.use_structure_distance)
                self.assertTrue(torch.isfinite(fitted.predict(batch.select(range(10, 20))).mean).all())

    def test_brca2_kernel_matches_independent_equation_without_spatial_factor(self):
        original = synthetic_batch()
        batch = replace(original, assay_id=BRCA2, local_features={"inverse_folding": original.local_features["inverse_folding"]})
        config = small_config()
        kernel = KermutKernel(batch, config.model)
        # Zero distances make the spatial factor exactly one in the independent scalar formula.
        expected_batch = replace(original, local_features={**original.local_features, "structure": torch.zeros_like(original.local_features["structure"])})
        torch.testing.assert_close(kernel(batch), scalar_kernel(expected_batch, expected_batch, config.model), rtol=1e-10, atol=1e-10)
        self.assertIsNone(kernel._train_distances.spatial)
        self.assertFalse(kernel.raw_d_lengthscale.requires_grad)
        kernel(batch).sum().backward()
        self.assertIsNone(kernel.raw_d_lengthscale.grad)
        self.assertIsNotNone(kernel.raw_h_lengthscale.grad)
        self.assertEqual(kernel.d_lengthscale.item(), 0)

    def test_brca2_cli_validation_fit_cv_and_saved_checkpoint_prediction(self):
        features, config = self.make_assay(BRCA2)
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(config.to_dict()))
        batch = load_feature_bundle(features, config)
        self.assertNotIn("structure", batch.local_features)
        self.assertEqual(batch.num_scores, 1)
        folds = self.root / "folds.csv"
        write_csv(folds, ["mutant", "fold_random_5"], [[variant, (i % 10) // 2] for i, variant in enumerate(batch.variant_ids)])
        for command, extra in (
            ("validate", []), ("fit", ["--setting", "single_to_double", "--sample-ratio", "1"]),
            ("cv", ["--setting", "singles_random", "--folds", str(folds)]),
            ("predict", ["--checkpoint", str(self.root / "fit/checkpoint.pt")]),
        ):
            with self.subTest(command=command):
                arguments = [command, "--features", str(features), "--output", str(self.root / command), "--quiet", *extra]
                if command != "predict":
                    arguments += ["--config", str(config_path)]
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    status = main(arguments)
                self.assertEqual(status, 0, stderr.getvalue())
                resolved = KermutConfig.load(self.root / command / "resolved_config.json")
                self.assertFalse(resolved.model.use_structure_distance)
                self.assertEqual(resolved.score_sources, ("scores/esm2/test",))
        fitted = FittedKermut.load(self.root / "fit/checkpoint.pt")
        self.assertFalse(fitted.model_config.use_structure_distance)
        self.assertNotIn("structure", fitted.train.local_features)
        self.assertTrue(torch.isfinite(fitted.predict(replace(batch, targets=None)).mean).all())
        self.assertFalse(fitted.fit_summary()["use_structure_distance"])

    def test_programmatic_runners_save_effective_brca2_settings(self):
        features, config = self.make_assay(BRCA2)
        batch = load_feature_bundle(features, config)
        run_extrapolation(batch, config, self.root / "fit", sample_ratio=1)
        run_cv(batch.select(range(10)), torch.arange(10) // 2, config, self.root / "cv")
        run_predict(batch, self.root / "fit/checkpoint.pt", config, self.root / "predict")
        for directory in ("fit", "cv", "cv/fold_0", "predict"):
            saved = KermutConfig.load(self.root / directory / "resolved_config.json")
            self.assertFalse(saved.model.use_structure_distance)
            self.assertEqual(saved.score_sources, ("scores/esm2/test",))

    def test_slurm_resolves_exceptions_before_saving_runtime_config(self):
        source = (REPOSITORY / "sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn.sbatch").read_text()
        runtime_source = source.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        for assay in PROTEINMPNN_SCORE_SKIP_REASONS:
            features, config = self.make_assay(assay)
            config_path = features.parent / "config.json"
            config_path.write_text(json.dumps(config.to_dict()))
            runtime_path = features.parent / "runtime.json"
            arguments = ["-", str(config_path), str(runtime_path), str(features), assay,
                         "false", "4", "1", "1", "singles_random"]
            log = io.StringIO()
            with patch.object(sys, "argv", arguments), patch.dict(os.environ, {"OMP_NUM_THREADS": "4"}), contextlib.redirect_stdout(log):
                exec(compile(runtime_source, "<kermut-sbatch-runtime>", "exec"), {})
            runtime = KermutConfig.load(runtime_path)
            self.assertEqual(runtime.score_sources, ("scores/esm2/test",))
            self.assertEqual(runtime.model.use_structure_distance, assay != BRCA2)
            self.assertIn("ProteinMPNN mean scores disabled", log.getvalue())
            self.assertEqual(load_feature_bundle(features, runtime).num_scores, 1)

    def test_parallel_brca2_fold_checkpoints_preserve_exceptions(self):
        features, config = self.make_assay(BRCA2)
        batch = load_feature_bundle(features, config).select(range(10))
        run_cv(batch, torch.arange(10) // 2, config, self.root / "parallel", fold_workers=2)
        for fold in range(5):
            fitted = FittedKermut.load(self.root / "parallel" / f"fold_{fold}" / "checkpoint.pt")
            self.assertFalse(fitted.model_config.use_structure_distance)
            self.assertEqual(fitted.train.score_names, ("scores/esm2/test",))
            self.assertNotIn("structure", fitted.train.local_features)

    def test_legacy_checkpoint_defaults_to_distance_enabled_and_rejects_conflicting_config(self):
        batch = synthetic_batch().select(range(8))
        config = small_config(n_steps=1)
        fitted = fit_gp(batch, config.model, config.training)
        expected = fitted.predict(batch).mean
        path = self.root / "legacy.pt"
        fitted.save(path)
        state = torch.load(path, weights_only=True)
        state["model_config"].pop("use_structure_distance")
        torch.save(state, path)
        restored = FittedKermut.load(path)
        self.assertTrue(restored.model_config.use_structure_distance)
        torch.testing.assert_close(restored.predict(batch).mean, expected, rtol=0, atol=0)
        conflicting = replace(config, model=replace(config.model, use_structure_distance=False))
        with self.assertRaisesRegex(ValueError, "structure-distance setting"):
            run_predict(batch, path, conflicting, self.root / "predict")
        self.assertFalse((self.root / "predict").exists())


if __name__ == "__main__":
    unittest.main()
