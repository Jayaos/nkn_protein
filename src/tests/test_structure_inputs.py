"""Structure-to-reference mapping, integrity and offline acquisition checks."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import torch

from feature_generation.structure_inputs import (
    download_proteingym_structure, download_structure, ensure_structure_manifest,
    load_structure, prepare_structure, validate_structure_metadata,
    PROTEINGYM_STRUCTURE_SHA256, PROTEINGYM_STRUCTURE_URL,
)
from feature_generation.inputs import sha256_file


THREE = {"A": "ALA", "C": "CYS", "D": "ASP", "V": "VAL"}


def pdb_text(sequence="ACD", chain="A", residue_ids=None, omit=(), alternate=None, nan_atom=None):
    ids = residue_ids or [(index, "") for index in range(1, len(sequence) + 1)]
    lines, serial = [], 1
    for position, (aa, (number, insertion)) in enumerate(zip(sequence, ids), 1):
        for atom_index, atom in enumerate(("N", "CA", "C", "O")):
            if (position, atom) in omit:
                continue
            altloc = "A" if alternate == (position, atom) else " "
            x = float("nan") if nan_atom == (position, atom) else position * 4.0 + atom_index * 0.3
            lines.append(f"ATOM  {serial:5d} {atom:^4s}{altloc}{THREE[aa]} {chain}{number:4d}{insertion or ' ' :1s}   {x:8.3f}{1.2:8.3f}{2.4:8.3f}{1.0:6.2f}{20.0:6.2f}          {atom[0]:>2s}  ")
            serial += 1
    return "\n".join(lines) + "\nTER\n"


class StructureInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = self.root / "index.csv"
        with self.index.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["DMS_id", "DMS_filename", "target_seq", "pdb_file", "pdb_range"])
            writer.writerow(["tiny", "tiny.csv", "ACD", "tiny.pdb", "1-3"])
        (self.root / "tiny.csv").write_text("mutant,mutated_sequence,DMS_score\nWT,ACD,0\n", encoding="utf-8")
        self.pdb = self.root / "tiny.pdb"
        self.pdb.write_text(pdb_text(), encoding="utf-8")
        self.manifests = self.root / "manifests"

    def register(self, **kwargs):
        return prepare_structure(self.index, "tiny", self.pdb, output_dir=self.manifests, **kwargs)

    def test_complete_mapping_and_manifest_reuse(self):
        path = self.register()
        observed = load_structure(path, "ACD")
        self.assertEqual(observed.coords.shape, (3, 4, 3))
        self.assertEqual(observed.coords.dtype, torch.float32)
        self.assertEqual(observed.positions.tolist(), [1, 2, 3])
        self.assertEqual(observed.chain_id, "A")
        self.assertEqual(observed.metadata["structure_path"], "../tiny.pdb")
        self.assertTrue(torch.isfinite(observed.coords).all())
        self.assertEqual(self.register(), path)
        validate_structure_metadata(observed.metadata, "ACD")

    def test_author_gaps_and_insertion_codes_preserved(self):
        self.pdb.write_text(pdb_text(residue_ids=[(10, ""), (10, "A"), (15, "")]), encoding="utf-8")
        observed = load_structure(self.register(), "ACD")
        self.assertEqual([(row["residue_number"], row["insertion_code"]) for row in observed.residue_mapping], [(10, ""), (10, "A"), (15, "")])
        self.assertEqual(observed.positions.tolist(), [1, 2, 3])

    def test_missing_reference_residue_is_rejected(self):
        self.pdb.write_text(pdb_text("AD", residue_ids=[(1, ""), (3, "")]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "full reference sequence"):
            self.register()

    def test_reference_substitution_is_rejected(self):
        self.pdb.write_text(pdb_text("AVD"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "full reference sequence"):
            self.register()

    def test_extra_protein_residues_rejected(self):
        self.pdb.write_text(pdb_text("ACDA"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "full reference sequence"):
            self.register()

    def test_missing_oxygen_is_explicitly_backend_specific(self):
        self.pdb.write_text(pdb_text(omit={(2, "O")}), encoding="utf-8")
        manifest = self.register()
        observed = load_structure(manifest, "ACD", require_oxygen=False)
        self.assertTrue(torch.isfinite(observed.coords[:, :3]).all())
        self.assertTrue(torch.isnan(observed.coords[1, 3]).all())
        with self.assertRaisesRegex(ValueError, "requires an O backbone atom"):
            load_structure(manifest, "ACD", require_oxygen=True)

    def test_missing_n_ca_or_c_rejected(self):
        for atom in ("N", "CA", "C"):
            with self.subTest(atom=atom):
                self.pdb.write_text(pdb_text(omit={(2, atom)}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Missing backbone atom"):
                    self.register()

    def test_alternate_backbone_atom_is_rejected(self):
        self.pdb.write_text(pdb_text(alternate=(2, "CA")), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "alternate backbone atom"):
            self.register()

    def test_non_finite_coordinates_rejected(self):
        self.pdb.write_text(pdb_text(nan_atom=(2, "CA")), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.register()

    def test_duplicate_atom_is_rejected_by_strict_parser(self):
        text = pdb_text()
        self.pdb.write_text(text.splitlines()[0] + "\n" + text, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Cannot parse structure"):
            self.register()

    def test_multiple_chains_require_explicit_selection(self):
        self.pdb.write_text(pdb_text(chain="A") + pdb_text(chain="B"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Specify chain_id"):
            self.register()
        observed = load_structure(self.register(chain_id="B"), "ACD")
        self.assertEqual(observed.chain_id, "B")
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.register(chain_id="C")

    def test_model_selection_is_one_based_file_order(self):
        self.pdb.write_text("MODEL        7\n" + pdb_text("AVD") + "ENDMDL\nMODEL       12\n" + pdb_text() + "ENDMDL\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "full reference sequence"):
            self.register()
        observed = load_structure(self.register(model_index=2), "ACD")
        self.assertEqual(observed.model_index, 2)
        with self.assertRaisesRegex(ValueError, "structure has 2 models"):
            self.register(model_index=3)
        with self.assertRaisesRegex(ValueError, "one-based integer"):
            self.register(model_index=0)

    def test_file_corruption_is_detected(self):
        manifest = self.register()
        self.pdb.write_text(pdb_text() + "REMARK changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_structure(manifest, "ACD")

    def test_manifest_mapping_corruption_is_detected(self):
        manifest = self.register()
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        metadata["residue_mapping"][0]["residue_number"] = 100
        manifest.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "input signature"):
            load_structure(manifest, "ACD")

    def test_reference_mismatch_in_manifest_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_structure(self.register(), "AVD")

    def test_conflicting_registration_rejected(self):
        self.register()
        alternative = self.root / "other.pdb"
        alternative.write_text(pdb_text(residue_ids=[(11, ""), (12, ""), (13, "")]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different structure is already registered"):
            prepare_structure(self.index, "tiny", alternative, output_dir=self.manifests)

    def test_mmcif_uses_author_ids_and_same_coordinates(self):
        from Bio.PDB import MMCIFIO, PDBParser
        structure = PDBParser(PERMISSIVE=False, QUIET=True).get_structure("tiny", str(self.pdb))
        cif = self.root / "tiny.cif"
        writer = MMCIFIO()
        writer.set_structure(structure)
        writer.save(str(cif))
        manifest = prepare_structure(self.index, "tiny", cif, output_dir=self.manifests)
        observed = load_structure(manifest, "ACD")
        self.assertEqual(observed.chain_id, "A")
        self.assertEqual(observed.coords[0, 0].tolist(), [4.0, torch.tensor(1.2).item(), torch.tensor(2.4).item()])

    def test_download_validates_and_reuses_without_network(self):
        with patch("feature_generation.structure_inputs.urlopen", return_value=io.BytesIO(pdb_text().encode())) as fetch:
            path = download_structure(self.index, "tiny", "https://example.org/tiny.pdb", raw_dir=self.root / "raw", manifest_dir=self.manifests)
            self.assertEqual(fetch.call_count, 1)
        with patch("feature_generation.structure_inputs.urlopen", side_effect=AssertionError("network request on resume")):
            self.assertEqual(download_structure(self.index, "tiny", "https://example.org/tiny.pdb", raw_dir=self.root / "raw", manifest_dir=self.manifests), path)
        self.assertEqual(load_structure(path, "ACD").metadata["source_url"], "https://example.org/tiny.pdb")

    def test_bad_download_is_not_published(self):
        with patch("feature_generation.structure_inputs.urlopen", return_value=io.BytesIO(pdb_text("AD").encode())):
            with self.assertRaisesRegex(ValueError, "full reference sequence"):
                download_structure(self.index, "tiny", "https://example.org/tiny.pdb", raw_dir=self.root / "raw", manifest_dir=self.manifests)
        self.assertEqual(list((self.root / "raw").iterdir()), [])
        self.assertFalse(self.manifests.exists())


    def archive_downloader(self, members):
        def write_archive(url, destination, expected_hash):
            self.assertEqual(url, PROTEINGYM_STRUCTURE_URL)
            self.assertEqual(expected_hash, PROTEINGYM_STRUCTURE_SHA256)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(destination, "w") as archive:
                for member in members:
                    archive.writestr(member, pdb_text())
            return destination
        return write_archive

    def test_proteingym_extracts_only_named_file_and_reuses(self):
        with patch("feature_generation.downloads.download_verified", side_effect=self.archive_downloader(["ProteinGym_AF2/tiny.pdb", "ProteinGym_AF2/unused.pdb"])) as fetch:
            manifest = download_proteingym_structure(self.index, "tiny", raw_dir=self.root / "raw", manifest_dir=self.manifests)
            self.assertEqual(fetch.call_count, 1)
        self.assertEqual([item.name for item in (self.root / "raw").iterdir()], ["tiny.pdb"])
        observed = load_structure(manifest, "ACD")
        self.assertEqual(observed.metadata["source_archive"]["sha256"], PROTEINGYM_STRUCTURE_SHA256)
        self.assertEqual(observed.metadata["source_archive"]["member"], "ProteinGym_AF2/tiny.pdb")
        with patch("feature_generation.downloads.download_verified", side_effect=AssertionError("network on resume")):
            self.assertEqual(download_proteingym_structure(self.index, "tiny", raw_dir=self.root / "raw", manifest_dir=self.manifests), manifest)

    def test_archive_member_path_cannot_escape_raw_directory(self):
        with patch("feature_generation.downloads.download_verified", side_effect=self.archive_downloader(["../../outside/tiny.pdb"])):
            manifest = download_proteingym_structure(self.index, "tiny", raw_dir=self.root / "raw", manifest_dir=self.manifests)
        self.assertFalse((self.root / "outside").exists())
        self.assertTrue((self.root / "raw/tiny.pdb").is_file())
        self.assertEqual(load_structure(manifest, "ACD").coords.shape, (3, 4, 3))

    def test_duplicate_archive_basename_is_rejected(self):
        with patch("feature_generation.downloads.download_verified", side_effect=self.archive_downloader(["one/tiny.pdb", "two/tiny.pdb"])):
            with self.assertRaisesRegex(ValueError, "Expected one ProteinGym archive member"):
                download_proteingym_structure(self.index, "tiny", raw_dir=self.root / "raw", manifest_dir=self.manifests)
        self.assertEqual(list((self.root / "raw").iterdir()), [])

    def test_missing_archive_basename_is_rejected(self):
        with patch("feature_generation.downloads.download_verified", side_effect=self.archive_downloader(["one/other.pdb"])):
            with self.assertRaisesRegex(ValueError, "found 0"):
                download_proteingym_structure(self.index, "tiny", raw_dir=self.root / "raw", manifest_dir=self.manifests)

    def test_conflicting_raw_file_is_not_overwritten(self):
        raw = self.root / "raw"
        raw.mkdir()
        existing = raw / "tiny.pdb"
        existing.write_text("existing unrelated data", encoding="utf-8")
        with patch("feature_generation.downloads.download_verified", side_effect=self.archive_downloader(["one/tiny.pdb"])):
            with self.assertRaisesRegex(ValueError, "differs from the pinned ProteinGym archive"):
                download_proteingym_structure(self.index, "tiny", raw_dir=raw, manifest_dir=self.manifests)
        self.assertEqual(existing.read_text(encoding="utf-8"), "existing unrelated data")

    def protssn_collection(self, filename="tiny.pdb", sequence="ACD", aliases=None):
        raw = self.root / "protssn/raw"
        raw.mkdir(parents=True)
        pdb = raw / filename
        pdb.write_text(pdb_text(sequence), encoding="utf-8")
        source = {"schema_version": 1, "source_id": "protssn_proteingym",
                  "archive_url": "https://example.org/protssn.zip", "archive_sha256": None,
                  "files_sha256": {f"raw/{filename}": sha256_file(pdb)},
                  "filename_aliases": aliases or {}}
        (raw.parent / "source.json").write_text(json.dumps(source), encoding="utf-8")
        return raw, pdb

    def ensure_protssn(self, **kwargs):
        return ensure_structure_manifest(
            self.index, "tiny", structure_source="protssn_proteingym",
            raw_dir=self.root / "protssn/raw", manifest_dir=self.manifests, **kwargs,
        )

    def test_protssn_registers_raw_structure_and_reuses_manifest_offline(self):
        raw, pdb = self.protssn_collection()
        with patch("feature_generation.structure_inputs.download_proteingym_structure") as official:
            manifest = self.ensure_protssn()
            original = manifest.read_bytes()
            observed = load_structure(manifest, "ACD")
            self.assertEqual(observed.structure_sha256, sha256_file(pdb))
            self.assertEqual(observed.metadata["source_archive"]["source_id"], "protssn_proteingym")
            self.assertEqual(observed.metadata["source_url"], "https://example.org/protssn.zip")
            (raw.parent / "source.json").unlink()
            self.assertEqual(self.ensure_protssn(), manifest)
            self.assertEqual(manifest.read_bytes(), original)
            official.assert_not_called()

    def test_protssn_resolves_inventory_alias_without_renaming_source(self):
        raw, pdb = self.protssn_collection("original_name.pdb", aliases={"tiny": "original_name.pdb"})
        manifest = self.ensure_protssn()
        observed = load_structure(manifest, "ACD")
        self.assertEqual((manifest.parent / observed.metadata["structure_path"]).resolve(), pdb)
        self.assertEqual([path.name for path in raw.iterdir()], ["original_name.pdb"])

    def test_protssn_rejects_modified_raw_file_without_manifest(self):
        _, pdb = self.protssn_collection()
        pdb.write_text(pdb_text("ACV"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.ensure_protssn()
        self.assertFalse(self.manifests.exists())

    def test_protssn_rejects_reference_mismatch_despite_valid_checksum(self):
        self.protssn_collection(sequence="ACV")
        with self.assertRaisesRegex(ValueError, "full reference sequence"):
            self.ensure_protssn()
        self.assertFalse(self.manifests.exists())

    def test_protssn_missing_raw_file_never_falls_back_to_official(self):
        _, pdb = self.protssn_collection()
        pdb.unlink()
        with patch("feature_generation.structure_inputs.download_proteingym_structure") as official:
            with self.assertRaisesRegex(FileNotFoundError, "Missing ProtSSN raw structure"):
                self.ensure_protssn()
            official.assert_not_called()
        self.assertFalse(self.manifests.exists())

    def test_protssn_rejects_wrong_collection_unpinned_file_and_unsafe_alias(self):
        raw, _ = self.protssn_collection()
        source_path = raw.parent / "source.json"
        original = json.loads(source_path.read_text())
        cases = (({"source_id": "proteingym"}, "Invalid protssn_proteingym"),
                 ({"files_sha256": {}}, "No valid ProtSSN checksum"),
                 ({"filename_aliases": {"tiny": "../tiny.pdb"}}, "Invalid structure filename"))
        for changes, message in cases:
            with self.subTest(changes=changes):
                source_path.write_text(json.dumps(dict(original, **changes)), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    self.ensure_protssn()
                self.assertFalse(self.manifests.exists())

    def test_collection_preparation_preserves_custom_manifest_and_checks_integrity(self):
        manifest = self.register()
        original = manifest.read_bytes()
        for source in ("proteingym", "protssn_proteingym"):
            with self.subTest(source=source):
                self.assertEqual(ensure_structure_manifest(
                    self.index, "tiny", structure_source=source, manifest_dir=self.manifests,
                ), manifest)
                self.assertEqual(manifest.read_bytes(), original)
        self.pdb.write_text(pdb_text("ACV"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.ensure_protssn()

    def test_collection_preparation_keeps_official_archive_acquisition(self):
        with patch("feature_generation.downloads.download_verified",
                   side_effect=self.archive_downloader(["ProteinGym_AF2/tiny.pdb"])) as fetch:
            manifest = ensure_structure_manifest(
                self.index, "tiny", raw_dir=self.root / "raw", manifest_dir=self.manifests,
            )
        fetch.assert_called_once()
        self.assertEqual(load_structure(manifest, "ACD").metadata["source_archive"]["sha256"],
                         PROTEINGYM_STRUCTURE_SHA256)


if __name__ == "__main__":
    unittest.main()
