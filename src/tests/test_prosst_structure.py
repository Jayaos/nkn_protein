"""Structure selection, token identity, integrity, and atomic publication tests."""
import copy
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import types
import sys
import unittest
from unittest.mock import Mock, patch

import torch

from feature_generation.prosst_config import ProSSTConfig
from feature_generation.prosst_structure import (
    DEPENDENCIES, canonical_pdb_bytes, import_pinned_quantizer, prepare_structure_tokens,
    token_ids_sha256, validate_structure_token_cache, validate_structure_token_metadata,
)
from feature_generation.structure_inputs import load_structure, prepare_structure
from test_structure_inputs import pdb_text


class ProSSTStructureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        index = self.root / "index.csv"
        with index.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["DMS_id", "DMS_filename", "target_seq"])
            writer.writerow(["tiny", "tiny.csv", "ACD"])
        (self.root / "tiny.csv").write_text("mutant,mutated_sequence,DMS_score\nWT,ACD,0\n", encoding="utf-8")
        self.structure_path = self.root / "input.pdb"
        # Author IDs and insertion codes must not affect token position ordering.
        self.structure_path.write_text(pdb_text(chain="Z", residue_ids=[(4, ""), (4, "A"), (17, "")]), encoding="ascii")
        manifest = prepare_structure(index, "tiny", self.structure_path, chain_id="Z", output_dir=self.root / "manifests")
        self.structure = load_structure(manifest, "ACD")
        self.config = ProSSTConfig(structure_token_dir=self.root / "tokens")
        runtime = {name: "test-runtime" for name in DEPENDENCIES}
        runtime.update({"scikit-learn": "1.2.2", "python": "3.10.0", "platform": "Windows", "machine": "AMD64"})
        self.runtime = patch("feature_generation.prosst_structure._runtime_versions", return_value=runtime)
        self.runtime.start()
        self.addCleanup(self.runtime.stop)
        self.local = patch("feature_generation.prosst_checkpoints.local_checkpoint", return_value=self.root / "model")
        self.local.start()
        self.addCleanup(self.local.stop)
        self.predict = patch("feature_generation.prosst_structure._native_quantize", return_value=torch.tensor([0, 731, 2047], dtype=torch.int64))
        self.quantizer = self.predict.start()
        self.addCleanup(self.predict.stop)

    def produce(self):
        return prepare_structure_tokens(self.config, self.structure)

    def rewrite_completion(self, path):
        completion = {"schema_version": 1, "files_sha256": {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in ("canonical.pdb", "tokens.pt", "metadata.json")}}
        (path / "completion.json").write_text(json.dumps(completion), encoding="utf-8")

    def test_canonical_pdb_contains_one_reference_chain_and_model(self):
        from Bio.PDB import PDBParser
        data = canonical_pdb_bytes(self.structure)
        path = self.root / "canonical.pdb"
        path.write_bytes(data)
        parsed = PDBParser(QUIET=True).get_structure("canonical", path)
        self.assertEqual(len(list(parsed.get_models())), 1)
        self.assertEqual([chain.id for chain in parsed.get_chains()], ["A"])
        residues = list(parsed.get_residues())
        self.assertEqual([row.id for row in residues], [(" ", 1, " "), (" ", 2, " "), (" ", 3, " ")])
        self.assertEqual([row.resname for row in residues], ["ALA", "CYS", "ASP"])
        self.assertEqual([[a.name for a in row] for row in residues], [["N", "CA", "C", "O"]] * 3)
        observed = torch.tensor([[a.coord.tolist() for a in row] for row in residues])
        torch.testing.assert_close(observed, self.structure.coords)

    def test_missing_oxygen_and_wrong_position_order_are_rejected(self):
        coords = self.structure.coords.clone()
        coords[1, 3, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "N, CA, C, O"):
            prepare_structure_tokens(self.config, replace(self.structure, coords=coords))
        with self.assertRaisesRegex(ValueError, "1..L"):
            canonical_pdb_bytes(replace(self.structure, positions=torch.tensor([1, 3, 2])))
        self.quantizer.assert_not_called()

    def test_pdb_coordinate_overflow_rejected(self):
        coords = self.structure.coords.clone()
        coords[0, 0, 0] = 10000.0
        with self.assertRaisesRegex(ValueError, "field width"):
            canonical_pdb_bytes(replace(self.structure, coords=coords))

    def test_verified_cache_reused_across_assay_labels(self):
        first = self.produce()
        metadata = dict(self.structure.metadata, assay_id="another_assay")
        second = prepare_structure_tokens(self.config, replace(self.structure, metadata=metadata))
        self.assertEqual(first.path, second.path)
        self.assertEqual(self.quantizer.call_count, 1)
        self.assertEqual(first.tokens.tolist(), [0, 731, 2047])
        validate_structure_token_metadata(first.metadata, self.structure.metadata)
        self.assertEqual(first.metadata["token_ids_sha256"], token_ids_sha256(first.tokens))
        self.assertEqual(sorted(p.name for p in first.path.iterdir()), ["canonical.pdb", "completion.json", "metadata.json", "tokens.pt"])

    def test_readiness_validates_referenced_tokens_without_quantizing(self):
        from feature_generation.readiness import validate_prosst_tokens

        tokens = self.produce()
        saved = {"structure_token_input_signature": tokens.input_signature, "structure_tokens": tokens.metadata}
        before = {path.name: path.read_bytes() for path in tokens.path.iterdir()}
        self.quantizer.side_effect = AssertionError("readiness must not quantize")
        validate_prosst_tokens(self.config, saved, self.structure)
        self.assertEqual({path.name: path.read_bytes() for path in tokens.path.iterdir()}, before)
        with self.assertRaisesRegex(ValueError, "quantizer_batch_size"):
            validate_prosst_tokens(replace(self.config, quantizer_batch_size=1), saved, self.structure)
        (tokens.path / "tokens.pt").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            validate_prosst_tokens(self.config, saved, self.structure)
        self.assertEqual(self.quantizer.call_count, 1)

    def test_coordinate_or_runtime_recipe_changes_create_new_identity(self):
        first = self.produce()
        coords = self.structure.coords.clone()
        coords[1, 2, 0] += 0.1
        second = prepare_structure_tokens(self.config, replace(self.structure, coords=coords))
        third = prepare_structure_tokens(replace(self.config, quantizer_batch_size=1), self.structure)
        self.assertEqual(len({first.input_signature, second.input_signature, third.input_signature}), 3)
        self.assertEqual(self.quantizer.call_count, 3)

    def test_corrupt_cache_is_rejected_without_overwriting(self):
        first = self.produce()
        path = first.path / "tokens.pt"
        data = path.read_bytes() + b"broken"
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.produce()
        self.assertEqual(path.read_bytes(), data)
        self.assertEqual(self.quantizer.call_count, 1)

    def test_incomplete_cache_is_not_silently_replaced(self):
        first = self.produce()
        (first.path / "completion.json").unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.produce()
        self.assertEqual(self.quantizer.call_count, 1)

    def test_failed_quantizer_does_not_publish_partial_cache(self):
        self.quantizer.side_effect = RuntimeError("simulated model failure")
        with self.assertRaisesRegex(RuntimeError, "simulated model failure"):
            self.produce()
        self.assertEqual(list(self.config.structure_token_dir.rglob("completion.json")), [])
        self.assertEqual(list(self.config.structure_token_dir.rglob(".building-*")), [])
        self.quantizer.side_effect = None
        self.produce()

    def test_bad_token_shape_dtype_and_bounds_fail_before_publication(self):
        for tokens in (torch.tensor([0, 1]), torch.tensor([0., 1., 2.]), torch.tensor([0, -1, 2]), torch.tensor([0, 2048, 2])):
            with self.subTest(tokens=tokens):
                self.quantizer.return_value = tokens
                with self.assertRaises(ValueError):
                    self.produce()
        self.assertEqual(list(self.config.structure_token_dir.rglob("completion.json")), [])

    def test_semantically_invalid_tokens_fail_even_with_updated_file_hashes(self):
        first = self.produce()
        bad = torch.tensor([0, 2048, 2047])
        torch.save(bad, first.path / "tokens.pt")
        metadata = dict(first.metadata, tokens_sha256=hashlib.sha256((first.path / "tokens.pt").read_bytes()).hexdigest(), token_ids_sha256=token_ids_sha256(bad))
        (first.path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        self.rewrite_completion(first.path)
        with self.assertRaisesRegex(ValueError, "0..2047"):
            validate_structure_token_cache(first.path)

    def test_metadata_binds_reference_structure_quantizer_and_recipe(self):
        result = self.produce()
        for field, value in (("reference_sequence", "ACV"), ("structure_input_signature", "0" * 64), ("quantizer", {}), ("num_residues", 2)):
            with self.subTest(field=field):
                changed = copy.deepcopy(result.metadata)
                changed[field] = value
                with self.assertRaises(ValueError):
                    validate_structure_token_metadata(changed, self.structure.metadata)
        changed = copy.deepcopy(result.metadata)
        changed["recipe"]["batch_size"] = 64
        with self.assertRaisesRegex(ValueError, "signature"):
            validate_structure_token_metadata(changed)

    def test_foreign_prosst_namespace_is_rejected(self):
        other = types.ModuleType("prosst")
        other.__path__ = [str(self.root / "unrelated")]
        with patch.dict(sys.modules, {"prosst": other}):
            with self.assertRaisesRegex(RuntimeError, "Another prosst"):
                import_pinned_quantizer(self.root / "source")

    def test_repeated_import_allows_only_the_recorded_pyg_generated_module(self):
        source = self.root / "source"
        namespace = types.ModuleType("prosst")
        namespace.__path__ = [str(source / "prosst")]
        generated_name = "prosst.structure.encoder.layer_GVPConv_propagate"
        generated = types.ModuleType(generated_name)
        generated.__file__ = str(self.root / "temporary_pyg_helper.py")
        imported = Mock()
        with patch.dict(sys.modules, {"prosst": namespace, generated_name: generated}):
            with patch.dict("feature_generation.prosst_structure._GENERATED_MODULES", {generated_name: generated}, clear=True):
                with patch("feature_generation.prosst_structure.importlib.import_module", return_value=imported):
                    import_pinned_quantizer(source)
                    sys.modules[generated_name] = types.ModuleType(generated_name)
                    with self.assertRaisesRegex(RuntimeError, "Another prosst"):
                        import_pinned_quantizer(source)

    def test_wrong_token_content_checksum_is_rejected(self):
        result = self.produce()
        metadata = dict(result.metadata, token_ids_sha256="0" * 64)
        (result.path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        self.rewrite_completion(result.path)
        with self.assertRaisesRegex(ValueError, "canonical checksum"):
            validate_structure_token_cache(result.path)


if __name__ == "__main__":
    unittest.main()
