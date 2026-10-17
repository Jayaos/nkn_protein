"""Check the real-assay GPU launch contract without fitting a large model."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import csv
from importlib.util import find_spec
import io
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from data_loading import FeatureBatch
from experiments import large_assay_gpu as gpu


ROOT = Path(__file__).resolve().parents[2]
FOLD_ROOT = ROOT / "data/DMS_ProteinGym_substitutions/cv_folds_singles_substitutions"
FOLDS = FOLD_ROOT / f"{gpu.ASSAY}.csv"


def human_single_batch(assay):
    raw = ROOT / "data/DMS_ProteinGym_substitutions/raw_substitutions" / f"{assay}.csv"
    with raw.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if ":" not in row["mutant"]]
    ids = tuple(row.get("variant_id") or row["mutant"] for row in rows)
    positions = [int(re.fullmatch(r"[A-Z](\d+)[A-Z]", row["mutant"]).group(1)) for row in rows]
    n = len(rows)
    # Only IDs/positions come from these human assays; numerical values are
    # dummy tensors. Fitting is mocked in these orchestration tests.
    return FeatureBatch(
        assay_id=assay, variant_ids=ids,
        targets=torch.arange(n, dtype=torch.float64),
        scores=torch.ones(n, 2, dtype=torch.float64),
        global_features=torch.ones(n, 1, dtype=torch.float64),
        local_features={}, owners=torch.arange(n),
        mutation_positions=torch.tensor(positions),
    )


class LargeAssayGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.batch = human_single_batch(gpu.ASSAY)
        cls.ids = cls.batch.variant_ids

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.features = self.root / "features"
        self.features.mkdir()
        self.metadata = self.features / "assay.json"
        self.metadata.write_text(json.dumps({"assay_id": gpu.ASSAY, "synthetic": False}))
        self.output = self.root / "output"

    def invoke(self, *extra, cuda=True, assay=gpu.ASSAY, batch=None, use_default_paths=False):
        args = [
            "--assay", assay,
            "--output", str(self.output), "--num-threads", "1",
            "--max-steps", "2", "--min-steps", "0",
        ]
        if not use_default_paths:
            args.extend(["--features", str(self.features), "--folds", str(FOLD_ROOT / f"{assay}.csv")])
        args.extend(extra)
        stream = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(stream))
            stack.enter_context(redirect_stderr(stream))
            stack.enter_context(patch.object(gpu.torch.cuda, "is_available", return_value=cuda))
            for name in ("synchronize", "reset_peak_memory_stats"):
                stack.enter_context(patch.object(gpu.torch.cuda, name))
            for name in ("max_memory_allocated", "max_memory_reserved"):
                stack.enter_context(patch.object(gpu.torch.cuda, name, return_value=1024))
            stack.enter_context(patch.object(gpu.torch.cuda, "get_device_name", return_value="test GPU"))
            stack.enter_context(patch.object(gpu.torch.cuda, "get_device_properties",
                                            return_value=SimpleNamespace(total_memory=80 * 2**30)))
            loader = stack.enter_context(patch.object(gpu, "load_feature_bundle",
                                                     return_value=self.batch if batch is None else batch))
            coverage = stack.enter_context(patch.object(gpu, "validate_assay_coverage"))
            fit = stack.enter_context(patch.object(gpu, "run_cv"))
            status = gpu.main(args)
        return status, stream.getvalue(), loader, coverage, fit

    def test_full_human_assay_and_published_folds_reach_serial_cuda_runner(self):
        self.assertEqual(len(self.ids), 10094)
        self.assertEqual(len(set(self.ids)), 10094)
        with (ROOT / "data/DMS_substitutions.csv").open(newline="", encoding="utf-8") as handle:
            record = next(row for row in csv.DictReader(handle) if row["DMS_id"] == gpu.ASSAY)
        self.assertEqual(record["source_organism"], "Homo sapiens")
        self.assertEqual(int(record["DMS_number_single_mutants"]), len(self.ids))
        status, log, loader, coverage, fit = self.invoke()
        self.assertEqual(status, 0, log)
        self.assertEqual(loader.call_args.kwargs["device"], "cpu")
        self.assertTrue(coverage.call_args.kwargs["singles_only"])
        self.assertEqual(fit.call_count, 3)
        for call, setting in zip(fit.call_args_list, gpu.SETTINGS):
            batch, folds, config, output = call.args
            self.assertEqual(batch.num_variants, 10094)
            self.assertEqual(set(folds.tolist()), set(range(5)))
            self.assertEqual(config.device, "cuda")
            self.assertEqual(config.training.max_steps, 2)
            self.assertEqual(config.training.inference_backend, "cholesky")
            self.assertIsNone(config.training.max_train_variants)
            self.assertEqual(call.kwargs["setting"], setting)
            self.assertEqual(call.kwargs["fold_workers"], 1)
            self.assertEqual(call.kwargs["split_source"], "provided")
        self.assertEqual((self.output / "folds/singles.csv").read_bytes(), FOLDS.read_bytes())
        manifest = json.loads((self.output / "fold_manifest.json").read_text())
        for counts in manifest["held_out_counts"].values():
            self.assertEqual(sum(counts), 10094)
        self.assertEqual(len(json.loads((self.output / "benchmark.json").read_text())["settings"]), 3)
        self.assertNotIn("preflight", log.lower())
        for filename in ("device_parity.json", "backend_parity.json"):
            self.assertFalse((self.output / filename).exists())

    def test_selected_smaller_human_assays_use_their_published_folds(self):
        for assay, expected_count in (
            ("A4_HUMAN_Seuma_2022", 796),
            ("BRCA2_HUMAN_Erwood_2022_HEK293T", 265),
            ("FKBP3_HUMAN_Tsuboyama_2023_2KFV", 1237),
        ):
            with self.subTest(assay=assay):
                self.metadata.write_text(json.dumps({"assay_id": assay, "synthetic": False}))
                self.output = self.root / assay
                batch = human_single_batch(assay)
                self.assertEqual(batch.num_variants, expected_count)
                status, log, loader, coverage, fit = self.invoke(assay=assay, batch=batch)
                self.assertEqual(status, 0, log)
                self.assertEqual(coverage.call_args.args[0].assay_id, assay)
                self.assertEqual(fit.call_count, 3)
                source = FOLD_ROOT / f"{assay}.csv"
                with source.open(newline="", encoding="utf-8") as handle:
                    published = {row["mutant"]: row for row in csv.DictReader(handle)}
                for call, setting in zip(fit.call_args_list, gpu.SETTINGS):
                    actual_batch, folds, config, _ = call.args
                    column = f"fold_{setting.removeprefix('singles_')}_5"
                    expected_folds = torch.tensor([int(published[variant][column]) for variant in batch.variant_ids])
                    self.assertTrue(torch.equal(folds, expected_folds))
                    self.assertEqual(actual_batch.assay_id, assay)
                    self.assertEqual(config.device, "cuda")
                    self.assertEqual(call.kwargs["split_source"], "provided")
                    self.assertEqual(call.kwargs["fold_workers"], 1)
                self.assertEqual((self.output / "folds/singles.csv").read_bytes(), source.read_bytes())
                for filename in ("execution_plan.json", "fold_manifest.json", "benchmark.json"):
                    self.assertEqual(json.loads((self.output / filename).read_text())["assay_id"], assay)
                if assay == "BRCA2_HUMAN_Erwood_2022_HEK293T":
                    # Apply fragment restrictions before requesting prepared assets.
                    kwargs = loader.call_args.kwargs
                    self.assertNotIn("structure_distance", kwargs["enabled_branches"])
                    self.assertIn("inverse_folding", kwargs["enabled_branches"])
                    self.assertTrue(kwargs["score_sources"])
                    self.assertTrue(all(source.startswith("scores/esm2/") for source in kwargs["score_sources"]))
                    saved = json.loads((self.output / "config.json").read_text())
                    self.assertEqual(saved["score_sources"], kwargs["score_sources"])
                    self.assertNotIn("structure_distance", saved["model"]["enabled_branches"])

    def test_default_asset_and_fold_paths_follow_selected_assay(self):
        assay = "BRCA2_HUMAN_Erwood_2022_HEK293T"
        repository = self.root / "repository"
        features = repository / "data/features/pretrained/singles" / assay
        features.mkdir(parents=True)
        (features / "assay.json").write_text(json.dumps({"assay_id": assay, "synthetic": False}))
        folds = repository / "data/DMS_ProteinGym_substitutions/cv_folds_singles_substitutions" / f"{assay}.csv"
        folds.parent.mkdir(parents=True)
        folds.write_bytes((FOLD_ROOT / f"{assay}.csv").read_bytes())
        with patch.object(gpu, "ROOT", repository):
            status, log, loader, coverage, fit = self.invoke(
                "singles_random", assay=assay, batch=human_single_batch(assay), use_default_paths=True,
            )
        self.assertEqual(status, 0, log)
        self.assertEqual(loader.call_args.args[0], features)
        self.assertEqual(coverage.call_args.args[1], repository
                         / "data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv")
        self.assertEqual(fit.call_count, 1)
        manifest = json.loads((self.output / "fold_manifest.json").read_text())
        self.assertEqual(Path(manifest["source"]), folds.resolve())

    def test_unsupported_assay_is_rejected_before_loading(self):
        stream = io.StringIO()
        with redirect_stderr(stream), patch.object(gpu, "load_feature_bundle") as loader:
            with self.assertRaises(SystemExit) as error:
                gpu.main(["--assay", "unknown_assay", "--output", str(self.output)])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("invalid choice", stream.getvalue())
        loader.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_loaded_batch_identity_must_match_selected_assay(self):
        assay = "BRCA2_HUMAN_Erwood_2022_HEK293T"
        self.metadata.write_text(json.dumps({"assay_id": assay, "synthetic": False}))
        status, log, loader, coverage, fit = self.invoke(assay=assay)
        self.assertEqual(status, 2)
        self.assertIn(f"Loaded feature batch does not identify {assay}", log)
        loader.assert_called_once()
        coverage.assert_not_called()
        fit.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_missing_prepared_metadata_stops_before_loading_or_fitting(self):
        self.metadata.unlink()
        status, log, loader, coverage, fit = self.invoke()
        self.assertEqual(status, 2)
        self.assertIn("assay.json", log)
        loader.assert_not_called()
        coverage.assert_not_called()
        fit.assert_not_called()
        self.assertFalse(self.output.exists())

    @unittest.skipUnless(find_spec("linear_operator"), "optional linear_operator dependency required")
    def test_bbmm_controls_reach_all_settings_and_saved_config(self):
        status, log, _, _, fit = self.invoke(
            "--inference-backend", "bbmm", "--bbmm-cg-tolerance", "0.0001",
            "--bbmm-max-cg-iterations", "200", "--bbmm-num-trace-samples", "16",
            "--bbmm-max-lanczos-iterations", "40", "--bbmm-preconditioner-size", "10",
            "--bbmm-probe-seed", "37",
        )
        self.assertEqual(status, 0, log)
        self.assertEqual(fit.call_count, 3)
        expected = {
            "inference_backend": "bbmm", "bbmm_cg_tolerance": 0.0001,
            "bbmm_max_cg_iterations": 200, "bbmm_num_trace_samples": 16,
            "bbmm_max_lanczos_iterations": 40, "bbmm_preconditioner_size": 10,
            "bbmm_probe_seed": 37,
        }
        for call in fit.call_args_list:
            training = call.args[2].training
            for name, value in expected.items():
                self.assertEqual(getattr(training, name), value)
            self.assertEqual(call.args[0].num_variants, 10094)
            self.assertEqual(call.kwargs["fold_workers"], 1)
        saved = json.loads((self.output / "config.json").read_text())["training"]
        for name, value in expected.items():
            self.assertEqual(saved[name], value)
        self.assertIn("bbmm", log.lower())
        self.assertNotIn("preflight", log.lower())
        for filename in ("device_parity.json", "backend_parity.json"):
            self.assertFalse((self.output / filename).exists())
        for filename in ("execution_plan.json", "benchmark.json"):
            metadata = json.loads((self.output / filename).read_text())
            self.assertEqual(metadata["inference_backend"], "bbmm")
            self.assertEqual(metadata["inference"]["bbmm_probe_seed"], 37)

    def test_invalid_later_split_stops_before_any_fit(self):
        broken = self.root / "broken_folds.csv"
        broken.write_text(FOLDS.read_text().replace("fold_contiguous_5", "missing_contiguous"))
        status, log, _, _, fit = self.invoke("--folds", str(broken))
        self.assertEqual(status, 2)
        self.assertIn("fold_contiguous_5", log)
        fit.assert_not_called()
        self.assertTrue((self.output / "failure.json").is_file())

    def test_identity_and_no_cuda_fail_before_loading(self):
        self.metadata.write_text(json.dumps({"assay_id": "wrong", "synthetic": False}))
        status, log, loader, _, fit = self.invoke()
        self.assertEqual(status, 2)
        self.assertIn("Expected real prepared assets", log)
        loader.assert_not_called()
        fit.assert_not_called()
        self.assertFalse(self.output.exists())
        status, log, loader, _, fit = self.invoke(cuda=False)
        self.assertEqual(status, 2)
        self.assertIn("CUDA is unavailable", log)
        loader.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_existing_results_are_preserved(self):
        self.output.mkdir()
        marker = self.output / "marker"
        marker.write_text("keep")
        status, log, loader, _, fit = self.invoke()
        self.assertEqual(status, 2)
        self.assertIn("already exists", log)
        self.assertEqual(marker.read_text(), "keep")
        loader.assert_not_called()
        fit.assert_not_called()



if __name__ == "__main__":
    unittest.main()
