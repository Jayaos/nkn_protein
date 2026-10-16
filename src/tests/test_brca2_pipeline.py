"""The fragmented BRCA2 assay uses truthful configs without distance kernels."""

import contextlib
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from config import ExperimentConfig
from data_loading import load_feature_bundle
from experiments import run_cv, run_extrapolation, run_predict
from experiments.cli import main
from gp.training import FittedGP
from tests.test_data import make_bundle, write_csv
from tests.test_pipeline import small_config, synthetic_batch


ASSAY = "BRCA2_HUMAN_Erwood_2022_HEK293T"
REPOSITORY = Path(__file__).resolve().parents[2]


class Brca2PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        self.features = self.root / "features"
        self.features.mkdir()
        self.ids, self.sequences, _ = make_bundle(self.features)
        metadata_path = self.features / "assay.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["assay_id"] = ASSAY
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        for path in self.features.rglob("features.pt"):
            payload = torch.load(path, weights_only=True)
            payload["assay_id"] = ASSAY
            torch.save(payload, path)
        config = small_config()
        self.config = replace(
            config,
            model=replace(config.model, enabled_branches=(
                "inverse_folding", "sequence_site", "sequence_global", "structure_distance",
            )),
            score_sources=["sequence/synthetic/v1"],
            structure_manifest_dir=str(self.root / "no_structure_manifests"),
        )
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config.to_dict()), encoding="utf-8")

    def invoke(self, arguments):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            status = main([str(argument) for argument in arguments])
        return status, output.getvalue(), errors.getvalue()

    def test_assay_resolution_preserves_other_branches_and_original_config(self):
        resolved = self.config.for_assay(ASSAY)
        self.assertEqual(resolved.model.enabled_branches, ("inverse_folding", "sequence_site", "sequence_global"))
        self.assertIn("structure_distance", self.config.model.enabled_branches)
        self.assertEqual(resolved.training, self.config.training)
        self.assertEqual(resolved.feature_selection, self.config.feature_selection)
        self.assertEqual(resolved.model.branches, self.config.model.branches)
        self.assertIs(resolved.for_assay(ASSAY), resolved)
        for assay in ("A0A1I9GEU1_NEIME_Kennouche_2019", "BRCA2_HUMAN_another_assay", None):
            self.assertIs(self.config.for_assay(assay), self.config)
        restored = ExperimentConfig.from_dict(resolved.to_dict())
        self.assertEqual(restored.model.enabled_branches, resolved.model.enabled_branches)

    def test_cli_validate_fit_and_predict_without_structure_records_effective_branches(self):
        branches = list(self.config.for_assay(ASSAY).model.enabled_branches)
        for command in ("validate", "fit"):
            extra = ["--setting", "single_to_double", "--sample-ratio", "1"] if command == "fit" else []
            status, output, errors = self.invoke([
                command, "--features", self.features, "--config", self.config_path,
                "--output", self.root / command, "--quiet", *extra,
            ])
            self.assertEqual(status, 0, errors)
            self.assertIn("structure_distance disabled", errors)
            resolved = json.loads((self.root / command / "resolved_config.json").read_text())
            self.assertEqual(resolved["model"]["enabled_branches"], branches)
            if command == "validate":
                self.assertEqual(json.loads(output)["enabled_branches"], branches)
                self.assertNotIn("structure_distance", json.loads(output)["local_feature_dimensions"])
        fitted = FittedGP.load(self.root / "fit/checkpoint.pt")
        self.assertEqual(list(fitted.model_config.enabled_branches), branches)
        self.assertNotIn("structure_distance", fitted.train.local_features)
        self.assertFalse(any(key.startswith("structure_distance.") for key in fitted.train.feature_signature))
        write_csv(self.features / "variants.csv", ["variant_id", "mutant", "mutated_sequence"],
                  [[variant, variant, sequence] for variant, sequence in zip(self.ids, self.sequences)])
        for explicit_config in (False, True):
            extra = ["--config", self.config_path] if explicit_config else []
            destination = self.root / f"predict_{explicit_config}"
            status, output, errors = self.invoke([
                "predict", "--features", self.features, "--checkpoint", self.root / "fit/checkpoint.pt",
                "--output", destination, "--quiet", *extra,
            ])
            self.assertEqual(status, 0, errors)
            self.assertEqual(json.loads(output)["n_variants"], 4)
            saved = json.loads((destination / "resolved_config.json").read_text())
            self.assertEqual(saved["model"]["enabled_branches"], branches)

    def test_two_score_preset_resolves_to_esm2_without_changing_other_assays(self):
        config = ExperimentConfig.load(REPOSITORY / "configs/esm2_proteinmpnn_nkn_two_scores_additive_four_branches_noise_initial_0p5.json")
        resolved = config.for_assay(ASSAY)
        self.assertEqual(resolved.score_sources, [config.score_sources[0]])
        self.assertEqual(len(config.score_sources), 2)
        self.assertEqual(resolved.training.noise_initial, 0.5)
        self.assertIn("inverse_folding", resolved.model.enabled_branches)
        self.assertIs(resolved.for_assay(ASSAY), resolved)
        self.assertIs(config.for_assay("A0A1I9GEU1_NEIME_Kennouche_2019"), config)
        with self.assertRaisesRegex(ValueError, "configure an ESM2 mean score source"):
            replace(config, score_sources=[config.score_sources[1]]).for_assay(ASSAY)

    def test_cli_drops_unsupported_score_even_without_distance_branch(self):
        config = self.config.for_assay(ASSAY)
        config = replace(config, score_sources=[*config.score_sources, "scores/proteinmpnn/absent"])
        self.config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
        destination = self.root / "validate_scores"
        status, _, errors = self.invoke([
            "validate", "--features", self.features, "--config", self.config_path,
            "--output", destination, "--quiet",
        ])
        self.assertEqual(status, 0, errors)
        self.assertIn("ProteinMPNN mean scores disabled", errors)
        resolved = ExperimentConfig.load(destination / "resolved_config.json")
        self.assertEqual(resolved.score_sources, self.config.score_sources)

    def test_distance_only_configuration_fails_before_feature_loading_or_output(self):
        config = replace(self.config, model=replace(self.config.model, enabled_branches=("structure_distance",)))
        self.config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
        with patch("experiments.cli.load_feature_bundle") as loader:
            status, _, errors = self.invoke([
                "validate", "--features", self.features, "--config", self.config_path,
                "--output", self.root / "rejected", "--quiet",
            ])
        self.assertEqual(status, 2)
        self.assertIn("enable at least one other kernel branch", errors)
        loader.assert_not_called()
        self.assertFalse((self.root / "rejected").exists())

    def test_programmatic_cv_fit_and_predict_resolve_configuration(self):
        original = synthetic_batch()
        batch = replace(original, assay_id=ASSAY, local_features={
            name: values for name, values in original.local_features.items() if name != "structure"
        })
        branches = self.config.for_assay(ASSAY).model.enabled_branches
        run_cv(batch, torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]), self.config, self.root / "cv")
        run_extrapolation(batch, self.config, self.root / "fit", sample_ratio=1)
        run_predict(replace(batch, targets=None), self.root / "fit/checkpoint.pt", self.config, self.root / "predict")
        for folder in ("cv", "cv/fold_0", "fit", "predict"):
            resolved = ExperimentConfig.load(self.root / folder / "resolved_config.json")
            self.assertEqual(resolved.model.enabled_branches, branches)
        for folder in ("cv/fold_0", "fit"):
            fitted = FittedGP.load(self.root / folder / "checkpoint.pt")
            self.assertEqual(fitted.model_config.enabled_branches, branches)
            self.assertNotIn("structure_distance", fitted.train.local_features)

    def test_programmatic_entries_reject_stale_distance_features_before_output(self):
        original = synthetic_batch()
        batch = replace(original, assay_id=ASSAY, local_features={
            **original.local_features, "structure_distance": original.local_features["structure"],
        })
        for name, action in (
            ("cv", lambda output: run_cv(batch, torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]), self.config, output)),
            ("fit", lambda output: run_extrapolation(batch, self.config, output)),
            ("predict", lambda output: run_predict(batch, self.root / "absent.pt", self.config, output)),
        ):
            with self.subTest(entry=name), self.assertRaisesRegex(ValueError, "reload features using"):
                action(self.root / name)
            self.assertFalse((self.root / name).exists())

    def test_sbatch_resolves_assay_before_writing_runtime_config_and_preflight(self):
        script = (REPOSITORY / "sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch").read_text()
        runtime_source = script.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        metadata_path = self.features / "assay.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["synthetic"] = False
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        # The runtime block requires the published five-fold configuration.
        config = replace(self.config, evaluation=replace(self.config.evaluation, n_folds=5),
                         score_sources=[*self.config.score_sources, "scores/proteinmpnn/absent"])
        self.config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
        runtime_path = self.root / "runtime_config.json"
        arguments = ["-", str(self.config_path), str(runtime_path), str(self.features), ASSAY,
                     "false", "2", "1", "1", "0", "training_nll", "100", "100", "1000",
                     "none", "0.1", "singles_random"]
        output = io.StringIO()
        with patch.object(sys, "argv", arguments), patch.dict(os.environ, {"OMP_NUM_THREADS": "2"}), contextlib.redirect_stdout(output):
            exec(compile(runtime_source, "<sbatch-runtime-config>", "exec"), {})
        runtime = ExperimentConfig.load(runtime_path)
        self.assertEqual(runtime.model.enabled_branches, config.for_assay(ASSAY).model.enabled_branches)
        self.assertEqual(runtime.num_threads, 2)
        self.assertIn("structure_distance disabled", output.getvalue())
        self.assertIn("ProteinMPNN mean scores disabled", output.getvalue())
        self.assertEqual(runtime.score_sources, self.config.score_sources)
        # Keep the small cache's genuine provenance for the scientific preflight.
        metadata["synthetic"] = True
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        batch = load_feature_bundle(
            self.features, enabled_branches=runtime.model.enabled_branches,
            score_sources=runtime.score_sources, structure_manifest_dir=runtime.structure_manifest_dir,
        )
        self.assertNotIn("structure_distance", batch.local_features)

    def test_preparation_verification_imports_the_real_fragment_loader(self):
        script = (REPOSITORY / "sbatch_run/run_prepare_rep/prepare_proteinmpnn.sbatch").read_text()
        source = next(block.split("\nPY\n", 1)[0] for block in script.split("<<'PY'\n")[1:]
                      if "catalog = AssayCatalog" in block.split("\nPY\n", 1)[0])
        index = self.root / "index.csv"
        write_csv(index, ["DMS_id", "DMS_filename", "target_seq", "seq_len"], [[ASSAY, "tiny.csv", "ACD", 3]])
        write_csv(self.root / "tiny.csv", ["mutant", "mutated_sequence", "DMS_score"], [["A1V", "VCD", 1]])
        manifests = self.root / "manifests"
        manifests.mkdir()
        manifest = manifests / f"{ASSAY}.json"
        manifest.write_text("{}", encoding="utf-8")
        config = SimpleNamespace(assay_index=index, structure_manifest_dir=manifests)
        # Patching the actual modules also verifies that the imports name real APIs.
        with patch("feature_generation.inverse_config.InverseFoldingConfig.load", return_value=config), \
                patch("feature_generation.structure_fragments.load_inverse_structure") as loader, \
                patch("feature_generation.structure_inputs.download_proteingym_structure") as download, \
                patch.object(sys, "argv", ["-", str(self.config_path), ASSAY]), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as ended:
            exec(compile(source, "<preparation-structure-verification>", "exec"), {})
        self.assertEqual(ended.exception.code, False)
        loader.assert_called_once_with(manifest, "ACD")
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
