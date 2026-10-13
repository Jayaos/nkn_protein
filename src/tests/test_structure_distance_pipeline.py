"""Physical coordinates survive the experiment, checkpoint and worker paths."""

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from data_loading import load_feature_bundle
from experiments import run_cv
from experiments.cli import main
from feature_generation.structure_inputs import prepare_structure
from gp.training import FittedGP, fit_gp
from tests.test_data import make_bundle, write_csv
from tests.test_pipeline import read_predictions, small_config, synthetic_batch
from tests.test_structure_inputs import pdb_text


class StructureDistancePipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        self.features = self.root / "features"
        self.features.mkdir()
        self.ids, self.sequences, _ = make_bundle(self.features)
        self.index = self.root / "index.csv"
        write_csv(self.index, ["DMS_id", "DMS_filename", "target_seq", "seq_len"],
                  [["tiny", "tiny.csv", "ACD", 3]])
        write_csv(self.root / "tiny.csv", ["mutant", "mutated_sequence", "DMS_score"],
                  [["WT", "ACD", 0]])
        self.pdb = self.root / "tiny.pdb"
        self.pdb.write_text(pdb_text(), encoding="utf-8")
        self.manifests = self.root / "manifests"
        prepare_structure(self.index, "tiny", self.pdb, output_dir=self.manifests)

    def config(self, branches):
        config = small_config()
        return replace(
            config,
            model=replace(config.model, enabled_branches=branches),
            score_sources=["sequence/synthetic/v1"],
            structure_manifest_dir=str(self.manifests),
        )

    def load_batch(self, config):
        return load_feature_bundle(
            self.features, enabled_branches=config.model.enabled_branches,
            score_sources=config.score_sources,
            structure_manifest_dir=config.structure_manifest_dir,
        )

    def invoke(self, arguments):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            status = main([str(argument) for argument in arguments])
        return status, output.getvalue(), errors.getvalue()

    def test_cli_mixed_branches_fit_and_label_free_predict_recover_manifest_config(self):
        config = self.config(("inverse_folding", "sequence_site", "sequence_global", "structure_distance"))
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
        for command in ("validate", "fit"):
            extra = ["--setting", "single_to_double", "--sample-ratio", "1"] if command == "fit" else []
            status, output, errors = self.invoke([
                command, "--features", self.features, "--config", config_path,
                "--output", self.root / command, "--quiet", *extra,
            ])
            self.assertEqual(status, 0, errors)
            summary = json.loads(output)
            if command == "validate":
                self.assertTrue(summary["valid"])
                self.assertEqual(summary["enabled_branches"], list(config.model.enabled_branches))
                self.assertEqual(summary["local_feature_dimensions"]["structure_distance"], 3)

        checkpoint = self.root / "fit/checkpoint.pt"
        fitted = FittedGP.load(checkpoint)
        self.assertEqual(fitted.model_config.enabled_branches, config.model.enabled_branches)
        self.assertEqual(fitted.train.local_features["structure_distance"].shape, (2, 3))
        saved_config = json.loads((checkpoint.parent / "resolved_config.json").read_text())
        self.assertEqual(saved_config["structure_manifest_dir"], str(self.manifests))

        write_csv(self.features / "variants.csv", ["variant_id", "mutant", "mutated_sequence"],
                  [[variant, variant, sequence] for variant, sequence in zip(self.ids, self.sequences)])
        # Omitting --config must recover both the branch and custom manifest directory.
        status, output, errors = self.invoke([
            "predict", "--features", self.features, "--checkpoint", checkpoint,
            "--output", self.root / "prediction", "--quiet",
        ])
        self.assertEqual(status, 0, errors)
        self.assertEqual(json.loads(output)["n_variants"], 4)
        rows = read_predictions(self.root / "prediction/predictions.csv")
        self.assertEqual([row["variant_id"] for row in rows], self.ids)
        self.assertTrue(all(row["target"] == "" for row in rows))
        expected = fitted.predict(self.load_batch(config))
        actual = torch.tensor([float(row["mean"]) for row in rows], dtype=torch.float64)
        torch.testing.assert_close(actual, expected.mean, rtol=1e-12, atol=1e-12)
        actual_variances = torch.tensor([float(row["observation_variance"]) for row in rows], dtype=torch.float64)
        torch.testing.assert_close(actual_variances, expected.observation_covariance.diagonal(), rtol=1e-12, atol=1e-12)

    def test_structure_distance_only_checkpoint_round_trip_and_changed_structure_rejection(self):
        config = self.config(("structure_distance",))
        batch = self.load_batch(config)
        fitted = fit_gp(batch.select([1, 2]), config.model, config.training)
        test = replace(batch.select([0, 3]), targets=None)
        expected = fitted.predict(test)
        path = self.root / "structure_distance_only.pt"
        fitted.save(path)
        restored = FittedGP.load(path)
        self.assertIsNone(restored.train.global_features)
        self.assertEqual(set(restored.train.local_features), {"structure_distance"})
        self.assertEqual(restored.train.feature_signature, batch.feature_signature)
        actual = restored.predict(test)
        torch.testing.assert_close(actual.mean, expected.mean, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual.observation_covariance, expected.observation_covariance, rtol=1e-12, atol=1e-12)
        self.assertTrue(torch.isfinite(actual.mean).all())

        alternate_pdb = self.root / "alternate.pdb"
        # Move site 2's CA from x=8.3 to x=10.3 while retaining its residue identity.
        original = pdb_text()
        moved = original.replace("   8.300", "  10.300")
        self.assertNotEqual(moved, original)
        alternate_pdb.write_text(moved, encoding="utf-8")
        alternate_manifests = self.root / "alternate_manifests"
        prepare_structure(self.index, "tiny", alternate_pdb, output_dir=alternate_manifests)
        changed = self.load_batch(replace(config, structure_manifest_dir=str(alternate_manifests)))
        self.assertFalse(torch.equal(changed.local_features["structure_distance"], batch.local_features["structure_distance"]))
        self.assertNotEqual(changed.feature_signature, batch.feature_signature)
        with self.assertRaisesRegex(ValueError, "feature sources or preprocessing"):
            restored.predict(replace(changed, targets=None))

    def test_structure_distance_branch_spawned_cv_matches_serial(self):
        original = synthetic_batch()
        batch = replace(original, local_features={"structure_distance": original.local_features["structure"]})
        config = self.config(("sequence_global", "structure_distance"))
        folds = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        serial, parallel = self.root / "serial", self.root / "parallel"
        expected = run_cv(batch, folds, config, serial)
        actual = run_cv(batch, folds, config, parallel, fold_workers=2)
        self.assertEqual(actual, expected)
        self.assertEqual(read_predictions(serial / "predictions.csv"), read_predictions(parallel / "predictions.csv"))
        for fold in range(3):
            restored = FittedGP.load(parallel / f"fold_{fold}/checkpoint.pt")
            self.assertEqual(set(restored.train.local_features), {"structure_distance"})
            expected_train = batch.select(folds != fold)
            torch.testing.assert_close(restored.train.local_features["structure_distance"],
                                       expected_train.local_features["structure_distance"])


if __name__ == "__main__":
    unittest.main()
