"""Independent feature structures retain identity checks without model runtimes."""

import ast
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from config import ExperimentConfig
from structure_provenance import validate_feature_structure


ROOT = Path(__file__).resolve().parents[2]


class StructureProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.geometry = {"assay_id": "tiny", "input_signature": "a" * 64, "structure_sha256": "b" * 64}
        self.independent = {"assay_id": "tiny", "input_signature": "c" * 64, "structure_sha256": "d" * 64}

    @staticmethod
    def records(structure):
        return [{"structure_input_signature": structure["input_signature"],
                 "structure_sha256": structure["structure_sha256"]} for _ in range(4)]

    def check(self, structure, records=None, consistency="shared"):
        validate_feature_structure("sequence", "tiny", self.geometry, structure,
                                   self.records(structure) if records is None else records, consistency)

    def test_shared_default_accepts_matching_and_rejects_independent_structure(self):
        self.check(self.geometry)
        with self.assertRaisesRegex(ValueError, "Selected structure differs"):
            self.check(self.independent)

    def test_per_cache_accepts_independent_feature_structure(self):
        self.check(self.independent, consistency="per_cache")
        self.check(self.geometry, consistency="per_cache")

    def test_per_cache_rejects_corrupt_config_tensor_and_sidecar_provenance(self):
        for index in range(4):
            for key in ("structure_input_signature", "structure_sha256"):
                records = self.records(self.independent)
                records[index][key] = "wrong"
                with self.subTest(record=index, key=key), self.assertRaisesRegex(ValueError, "provenance disagrees"):
                    self.check(self.independent, records, "per_cache")

    def test_per_cache_does_not_accept_geometry_identity_for_a_different_feature(self):
        with self.assertRaisesRegex(ValueError, "provenance disagrees"):
            self.check(self.independent, self.records(self.geometry), "per_cache")

    def test_wrong_assay_and_invalid_provenance_are_rejected(self):
        for consistency in ("shared", "per_cache"):
            with self.subTest(consistency=consistency):
                with self.assertRaisesRegex(ValueError, "different assay"):
                    self.check({**self.geometry, "assay_id": "other"}, consistency=consistency)
                with self.assertRaisesRegex(ValueError, "must be an object"):
                    self.check(self.geometry, [None], consistency)

    def test_undeclared_structure_still_checks_geometry_provenance(self):
        validate_feature_structure("sequence", "tiny", self.geometry, None,
                                   self.records(self.geometry), "per_cache")
        with self.assertRaisesRegex(ValueError, "provenance disagrees"):
            validate_feature_structure("sequence", "tiny", self.geometry, None,
                                       self.records(self.independent), "per_cache")

    def test_config_default_validation_and_round_trip(self):
        self.assertEqual(ExperimentConfig().structure_consistency, "shared")
        configured = replace(ExperimentConfig(), structure_consistency="per_cache")
        self.assertEqual(ExperimentConfig.from_dict(configured.to_dict()).structure_consistency, "per_cache")
        for value in (None, True, "ignore", "", {}, []):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "structure_consistency"):
                replace(configured, structure_consistency=value)

    def test_cli_forwards_consistency_to_the_feature_loader(self):
        # Run the actual CLI loading function with a captured loader, so this
        # routing regression does not require importing PyTorch or fitting a GP.
        tree = ast.parse((ROOT / "src/experiments/cli.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_load_batch")
        calls = []
        sentinel = object()

        def loader(*args, **kwargs):
            calls.append(kwargs)
            return sentinel

        namespace = {"load_feature_bundle": loader}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<cli loading>", "exec"), namespace)
        args = SimpleNamespace(full_assay=False, allow_subset=False, assay_index=None,
                               features="features", singles_only=False)
        config = ExperimentConfig.load(ROOT / "configs/proteinmpnn_prosst_nkn.json")
        self.assertIs(namespace["_load_batch"](args, config), sentinel)
        self.assertEqual(calls[0]["structure_consistency"], "per_cache")
        self.assertEqual(calls[0]["structure_manifest_dir"], "data/structures/proteingym/manifests")

    def test_local_prosst_and_proteinmpnn_metadata_can_use_independent_structures(self):
        preset = ExperimentConfig.load(ROOT / "configs/proteinmpnn_prosst_nkn.json")
        feature_root = ROOT / "data/features/pretrained/singles"
        sources = {
            family: json.loads((ROOT / "configs/extraction" / filename).read_text())["structure_source"]
            for family, filename in (("prosst", "prosst_2048_singles.json"),
                                     ("proteinmpnn", "proteinmpnn_v48_020_kermut.json"))
        }
        checked = 0
        for assay_root in feature_root.glob("*"):
            manifest = ROOT / preset.structure_manifest_dir / f"{assay_root.name}.json"
            if not manifest.is_file():
                continue
            geometry = json.loads(manifest.read_text())
            for name, selection in preset.feature_selection.items():
                leaf = assay_root / name / selection
                if not leaf.exists():
                    family, recipe = selection.split("/", 1)
                    leaf = assay_root / name / family / f"{sources[family]}--{recipe}"
                if not (leaf / "config.json").is_file():
                    continue
                config = json.loads((leaf / "config.json").read_text())
                provenance = json.loads((leaf / "provenance.json").read_text())
                original = deepcopy(config)
                validate_feature_structure(name, assay_root.name, geometry, config["structure"],
                                           [config, provenance], preset.structure_consistency)
                self.assertEqual(config, original)
                if config["structure"]["input_signature"] != geometry["input_signature"]:
                    with self.assertRaisesRegex(ValueError, "Selected structure differs"):
                        validate_feature_structure(name, assay_root.name, geometry, config["structure"],
                                                   [config, provenance])
                    checked += 1
        if not checked:
            self.skipTest("No local feature cache with a different structure is available")


if __name__ == "__main__":
    unittest.main()
