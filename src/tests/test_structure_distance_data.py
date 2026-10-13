"""Mapped CA coordinates remain independent of learned structure caches."""

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from config import DEFAULT_BRANCHES
from data_loading import load_feature_bundle
from feature_generation.structure_inputs import load_structure, prepare_structure
from tests.test_data import make_bundle, write_csv
from tests.test_structure_inputs import pdb_text


class StructureDistanceDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.features = self.root / "features"
        self.features.mkdir()
        make_bundle(self.features)
        self.index = self.root / "index.csv"
        write_csv(self.index, ["DMS_id", "DMS_filename", "target_seq"], [["tiny", "tiny.csv", "ACD"]])
        write_csv(self.root / "tiny.csv", ["mutant", "mutated_sequence", "DMS_score"], [["WT", "ACD", 0]])
        self.pdb = self.root / "tiny.pdb"
        # Mapping uses reference order, including gapped author numbering and
        # insertion codes. Distance features do not need backbone oxygen.
        self.pdb.write_text(pdb_text(residue_ids=[(10, ""), (10, "A"), (15, "")],
                                     omit={(2, "O")}), encoding="utf-8")
        self.manifests = self.root / "manifests"
        self.manifest = prepare_structure(self.index, "tiny", self.pdb, output_dir=self.manifests)

    def load(self, branches=("inverse_folding", "structure_distance"), **kwargs):
        return load_feature_bundle(self.features, enabled_branches=branches,
                                   structure_manifest_dir=self.manifests, **kwargs)

    def test_reference_mapping_packing_selection_and_units(self):
        batch = self.load()
        structure = load_structure(self.manifest, "ACD", require_oxygen=False)
        expected = structure.coords[:, 1].double()[[0, 1, 0, 1]]
        torch.testing.assert_close(batch.local_features["structure_distance"], expected)
        self.assertEqual(batch.owners.tolist(), [1, 2, 3, 3])
        self.assertEqual(batch.mutation_positions.tolist(), [1, 2, 1, 2])
        self.assertAlmostEqual(torch.dist(expected[0], expected[1]).item(), 4.0, places=5)
        selected = batch.select([3, 0, 1])
        torch.testing.assert_close(selected.local_features["structure_distance"], expected[[2, 3, 0]])
        self.assertEqual(batch.select([0]).local_features["structure_distance"].shape, (0, 3))
        self.assertEqual(batch.feature_signature["structure_distance.input_signature"], structure.input_signature)
        self.assertEqual(batch.feature_signature["structure_distance.structure_sha256"], structure.structure_sha256)
        self.assertEqual(batch.feature_signature["structure_distance.coordinates"], "reference_CA_angstrom")

    def test_geometry_uses_explicit_scores_without_any_representation_tensor(self):
        for name in ("sequence", "structure", "inverse_folding"):
            (self.features / name / "synthetic/v1/features.pt").unlink()
        metadata_path = self.features / "assay.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["features"] = {}
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        batch = self.load(("structure_distance",), score_sources=["sequence/synthetic/v1"])
        self.assertEqual(tuple(batch.local_features), ("structure_distance",))
        self.assertIsNone(batch.global_features)
        self.assertEqual(batch.score_names, ("sequence/synthetic/v1",))
        with self.assertRaisesRegex(ValueError, "mean score"):
            self.load(("structure_distance",))

    def test_enabled_geometry_adds_no_mean_source_and_default_stays_legacy(self):
        legacy = load_feature_bundle(self.features, structure_manifest_dir=self.root / "absent")
        explicit = self.load(DEFAULT_BRANCHES)
        self.assertEqual(legacy.feature_signature, explicit.feature_signature)
        self.assertNotIn("enabled_branches", legacy.feature_signature)
        augmented = self.load((*DEFAULT_BRANCHES, "structure_distance"))
        self.assertEqual(augmented.score_names, legacy.score_names)
        torch.testing.assert_close(augmented.scores, legacy.scores)
        self.assertEqual(set(augmented.local_features), {*legacy.local_features, "structure_distance"})

    def test_relative_manifest_directory_resolves_from_project_root(self):
        import data_loading.features as feature_module

        # Patch the module path to make the temporary directory the project
        # root, then run from an unrelated working directory.
        module_path = self.root / "src/data_loading/features.py"
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.features)
        with patch.object(feature_module, "__file__", str(module_path)):
            batch = load_feature_bundle(self.features, enabled_branches=("inverse_folding", "structure_distance"),
                                        structure_manifest_dir="manifests")
        self.assertEqual(batch.local_features["structure_distance"].shape, (4, 3))

    def test_checksum_assay_and_shape_are_checked(self):
        batch = self.load()
        with self.assertRaisesRegex(ValueError, "three-dimensional"):
            replace(batch, local_features={"structure_distance": torch.ones(4, 2, dtype=torch.float64)})
        metadata = json.loads(self.manifest.read_text(encoding="utf-8"))
        metadata["assay_id"] = "other"
        self.manifest.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different assay"):
            self.load()
        metadata["assay_id"] = "tiny"
        self.manifest.write_text(json.dumps(metadata), encoding="utf-8")
        with self.pdb.open("a", encoding="utf-8") as handle:
            handle.write("REMARK changed\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.load()

    def test_selected_learned_features_must_use_the_same_structure(self):
        original = json.loads(self.manifest.read_text(encoding="utf-8"))
        alternative = self.root / "other.pdb"
        alternative.write_text(pdb_text(), encoding="utf-8")
        alternative_manifest = prepare_structure(self.index, "tiny", alternative,
                                                 output_dir=self.root / "other_manifests")
        mismatched = json.loads(alternative_manifest.read_text(encoding="utf-8"))
        for name, branch in (("structure", "structure"), ("inverse_folding", "inverse_folding"),
                             ("sequence", "sequence_site")):
            with self.subTest(branch=name):
                path = self.features / name / "synthetic/v1/config.json"
                config = json.loads(path.read_text(encoding="utf-8"))
                config["structure"] = original
                path.write_text(json.dumps(config), encoding="utf-8")
                self.load((branch, "structure_distance"))
                config["structure"] = mismatched
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "differs from"):
                    self.load((branch, "structure_distance"))
                del config["structure"]
                path.write_text(json.dumps(config), encoding="utf-8")

    def test_tensor_and_separate_provenance_mismatches_are_rejected(self):
        for name, branch in (("structure", "structure"), ("inverse_folding", "inverse_folding"),
                             ("sequence", "sequence_global")):
            leaf = self.features / name / "synthetic/v1"
            path = leaf / "features.pt"
            store = torch.load(path, weights_only=True)
            for nested in (False, True):
                with self.subTest(branch=name, nested=nested):
                    modified = {**store, **({"provenance": {"structure_input_signature": "wrong"}}
                                          if nested else {"structure_sha256": "wrong"})}
                    torch.save(modified, path)
                    with self.assertRaisesRegex(ValueError, "provenance disagrees"):
                        self.load((branch, "structure_distance"))
            torch.save(store, path)
            provenance = leaf / "provenance.json"
            provenance.write_text(json.dumps({"structure_input_signature": "wrong"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance disagrees"):
                self.load((branch, "structure_distance"))
            provenance.unlink()


if __name__ == "__main__":
    unittest.main()
