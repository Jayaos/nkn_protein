"""NKN uses conditional caches through fitting, worker processes, and reloads."""

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import torch

from data_loading import load_feature_bundle
from experiments import run_cv
from experiments.cli import main
from gp.training import FittedGP
from tests.test_data import make_bundle
from tests.test_pipeline import read_predictions, small_config, synthetic_batch


RECIPE = {
    "model_family": "proteinmpnn", "conditioning_mode": "sequence_conditional",
    "inference_method": "conditional_probs", "normalization": "native_21_amino_acid_tokens",
    "noncanonical_tokens": "excluded_without_renormalization",
    "order_aggregation": "geometric_mean_of_probabilities", "num_orders": 10, "seed": 37,
}


class ConditionalNknPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        defaults = small_config()
        self.config = replace(
            defaults, model=replace(defaults.model, enabled_branches=("inverse_folding", "sequence_global")),
            inverse_folding_policy="sequence_conditional",
        )

    def invoke(self, *args):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            status = main([str(value) for value in args])
        self.assertEqual(status, 0, errors.getvalue())
        return json.loads(output.getvalue())

    def test_cli_preserves_raw_values_and_reload_recovers_policy_without_config_argument(self):
        features = self.root / "features"
        features.mkdir()
        make_bundle(features)
        source = features / "inverse_folding/synthetic/v1"
        leaf = features / "inverse_folding/proteinmpnn/conditional"
        leaf.mkdir(parents=True)
        payload = torch.load(source / "features.pt", weights_only=True)
        payload["probabilities"] *= torch.tensor([0.80, 0.90, 0.95])[:, None]
        payload["log_probs"] = payload["probabilities"].log()
        torch.save(payload, leaf / "features.pt")
        metadata = json.loads((source / "config.json").read_text())
        metadata.update(RECIPE, checkpoint="conditional")
        (leaf / "config.json").write_text(json.dumps(metadata), encoding="utf-8")
        shutil.copyfile(source / "zero_shot.csv", leaf / "zero_shot.csv")
        sequence_path = features / "sequence/synthetic/v1/features.pt"
        sequence = torch.load(sequence_path, weights_only=True)
        # Exercise the actual 650M representation dimension without model downloads.
        sequence["embeddings"] = sequence["embeddings"].repeat(1, 1, 640)
        torch.save(sequence, sequence_path)
        config = replace(
            self.config, feature_selection={"sequence": "synthetic/v1", "inverse_folding": "proteinmpnn/conditional"},
            score_sources=["sequence/synthetic/v1", "inverse_folding/proteinmpnn/conditional"],
        )
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
        result = self.invoke("validate", "--features", features, "--config", config_path,
                             "--output", self.root / "validate")
        self.assertEqual(result["global_feature_dimension"], 1280)
        self.assertEqual(result["n_score_sources"], 2)
        self.invoke("fit", "--features", features, "--config", config_path,
                    "--setting", "single_to_double", "--sample-ratio", "1",
                    "--output", self.root / "fit", "--quiet")
        checkpoint = self.root / "fit/checkpoint.pt"
        fitted = FittedGP.load(checkpoint)
        expected_train = payload["probabilities"][fitted.train.mutation_positions - 1].double()
        torch.testing.assert_close(fitted.train.local_features["inverse_folding"], expected_train, rtol=0, atol=0)
        self.assertTrue((expected_train.sum(-1) < 1).all())
        self.assertEqual(json.loads(fitted.train.feature_signature["inverse_folding.probability_recipe"]), RECIPE)
        saved = json.loads((self.root / "fit/resolved_config.json").read_text())
        self.assertEqual(saved["inverse_folding_policy"], "sequence_conditional")
        batch = load_feature_bundle(
            features, feature_selection=config.feature_selection, score_sources=config.score_sources,
            enabled_branches=config.model.enabled_branches, inverse_folding_policy=config.inverse_folding_policy,
        )
        expected = fitted.predict(batch)
        self.invoke("predict", "--features", features, "--checkpoint", checkpoint,
                    "--output", self.root / "predict", "--quiet")
        rows = read_predictions(self.root / "predict/predictions.csv")
        actual = torch.tensor([float(row["mean"]) for row in rows], dtype=torch.float64)
        torch.testing.assert_close(actual, expected.mean, rtol=1e-12, atol=1e-12)
        changed_signature = dict(batch.feature_signature)
        changed_signature.pop("inverse_folding.probability_recipe")
        with self.assertRaisesRegex(ValueError, "feature sources"):
            fitted.predict(replace(batch, feature_signature=changed_signature))

    def test_conditional_cv_matches_serial_after_worker_and_checkpoint_round_trips(self):
        batch = synthetic_batch()
        original = batch.local_features["inverse_folding"]
        raw = original * (0.8 + 0.02 * batch.mutation_positions[:, None])
        batch = replace(
            batch, local_features={"inverse_folding": raw},
            feature_signature={**batch.feature_signature,
                               "inverse_folding.probability_recipe": json.dumps(RECIPE, sort_keys=True)},
        )
        folds = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        serial, parallel = self.root / "serial", self.root / "parallel"
        expected = run_cv(batch, folds, self.config, serial)
        actual = run_cv(batch, folds, self.config, parallel, fold_workers=2)
        self.assertEqual(actual, expected)
        self.assertEqual(read_predictions(serial / "predictions.csv"), read_predictions(parallel / "predictions.csv"))
        for fold in range(3):
            restored = FittedGP.load(parallel / f"fold_{fold}/checkpoint.pt")
            train = batch.select(folds != fold)
            torch.testing.assert_close(restored.train.local_features["inverse_folding"],
                                       train.local_features["inverse_folding"], rtol=0, atol=0)
            self.assertTrue(torch.isfinite(restored.predict(batch.select(folds == fold)).mean).all())


if __name__ == "__main__":
    unittest.main()
