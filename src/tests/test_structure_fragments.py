"""Explicit partial structural coverage and independently verified fragments."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

from feature_generation.structure_fragments import (
    BRCA2_FRAGMENTS, StructureFragments, inverse_structure_positions, load_inverse_structure,
    prepare_structure_fragments, validate_inverse_structure_metadata,
)
from feature_generation.structure_inputs import (
    PROTEINGYM_STRUCTURE_SHA256, PROTEINGYM_STRUCTURE_URL,
    download_proteingym_structure, load_structure, prepare_structure,
)
from tests.test_data import write_csv
from tests.test_structure_inputs import pdb_text


class StructureFragmentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reference = "ACDVACD"
        self.index = self.root / "index.csv"
        write_csv(self.index, ["DMS_id", "DMS_filename", "target_seq"], [["tiny", "tiny.csv", self.reference]])
        write_csv(self.root / "tiny.csv", ["mutant", "mutated_sequence", "DMS_score"], [["WT", self.reference, 0]])
        self.files = []
        for index, sequence in enumerate(("AC", "DV", "AC")):
            path = self.root / f"fragment{index}.pdb"
            path.write_text(pdb_text(sequence, residue_ids=[(10, ""), (10, "A")]), encoding="utf-8")
            self.files.append(path)
        self.ranges = ((1, 2), (3, 4), (5, 6))
        self.manifests = self.root / "manifests"

    def register(self, **kwargs):
        return prepare_structure_fragments(self.index, "tiny", self.files, self.ranges,
                                           output_dir=self.manifests, **kwargs)

    def test_partial_reference_mapping_keeps_coordinate_frames_separate(self):
        path = self.register()
        bundle = load_inverse_structure(path, self.reference)
        self.assertIsInstance(bundle, StructureFragments)
        self.assertFalse(hasattr(bundle, "coords"))
        self.assertEqual(bundle.reference_sequence, self.reference)
        self.assertEqual(bundle.reference_ranges, self.ranges)
        self.assertEqual(bundle.positions.tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual([fragment.reference_sequence for fragment in bundle.fragments], ["AC", "DV", "AC"])
        self.assertEqual([fragment.coords.shape for fragment in bundle.fragments], [(2, 4, 3)] * 3)
        self.assertTrue(all(fragment.positions.tolist() == [1, 2] for fragment in bundle.fragments))
        mapping = bundle.metadata["residue_mapping"]
        self.assertEqual([row["fragment_index"] for row in mapping], [0, 0, 1, 1, 2, 2])
        self.assertEqual([row["local_reference_position"] for row in mapping], [1, 2, 1, 2, 1, 2])
        self.assertEqual([(row["residue_number"], row["insertion_code"]) for row in mapping], [(10, ""), (10, "A")] * 3)
        torch.testing.assert_close(inverse_structure_positions(bundle.metadata, self.reference), bundle.positions)
        self.assertEqual(self.register(), path)
        with self.assertRaisesRegex(ValueError, "shared coordinate frame"):
            load_structure(path, self.reference)

    def test_internal_uncovered_sites_remain_absent(self):
        self.files = [self.files[0], self.files[2]]
        self.ranges = ((1, 2), (5, 6))
        bundle = load_inverse_structure(self.register(), self.reference)
        self.assertEqual(bundle.positions.tolist(), [1, 2, 5, 6])
        self.assertEqual([row["reference_position"] for row in bundle.metadata["residue_mapping"]], [1, 2, 5, 6])

    def test_invalid_ranges_and_count_are_rejected(self):
        cases = ((), ((0, 2),), ((1, 2), (2, 4)), ((5, 6), (1, 2)), ((1, 99),), ((True, 2),))
        for ranges in cases:
            with self.subTest(ranges=ranges), self.assertRaisesRegex(ValueError, "range"):
                prepare_structure_fragments(self.index, "tiny", self.files[:len(ranges)], ranges,
                                            output_dir=self.manifests)
        with self.assertRaisesRegex(ValueError, "one structure file"):
            prepare_structure_fragments(self.index, "tiny", self.files[:1], self.ranges, output_dir=self.manifests)

    def test_each_fragment_must_match_the_explicit_reference_slice(self):
        with self.assertRaisesRegex(ValueError, "exactly match"):
            prepare_structure_fragments(self.index, "tiny", [self.files[0]], [(2, 3)], output_dir=self.manifests)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            prepare_structure_fragments(self.index, "tiny", [self.files[0]], [(1, 3)], output_dir=self.manifests)

    def test_required_atoms_and_optional_oxygen_are_verified_per_fragment(self):
        self.files[1].write_text(pdb_text("DV", omit={(2, "O")}), encoding="utf-8")
        path = self.register()
        bundle = load_inverse_structure(path, self.reference, require_oxygen=False)
        self.assertTrue(torch.isnan(bundle.fragments[1].coords[1, 3]).all())
        with self.assertRaisesRegex(ValueError, "requires an O"):
            load_inverse_structure(path, self.reference)
        self.files[1].write_text(pdb_text("DV", omit={(2, "CA")}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Missing backbone atom"):
            self.register()

    def test_file_checksum_and_metadata_identity_are_verified(self):
        path = self.register()
        metadata = json.loads(path.read_text(encoding="utf-8"))
        cases = (
            ("mapping", lambda data: data["residue_mapping"][0].update(reference_position=2)),
            ("policy", lambda data: data.update(stitch_policy="invent_interfragment_contacts")),
            ("SHA-256", lambda data: data.update(structure_sha256="0" * 64)),
            ("signature", lambda data: data.update(input_signature="0" * 64)),
            ("different assay", lambda data: data["fragments"][1]["structure"].update(assay_id="other")),
        )
        for message, mutate in cases:
            with self.subTest(message=message):
                changed = deepcopy(metadata)
                mutate(changed)
                with self.assertRaisesRegex(ValueError, message):
                    validate_inverse_structure_metadata(changed, self.reference)
        self.files[1].write_text(self.files[1].read_text(encoding="utf-8") + "REMARK modified\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_inverse_structure(path, self.reference)

    def test_changed_ranges_or_fragments_cannot_replace_registered_identity(self):
        self.register()
        self.files = [self.files[0], self.files[2]]
        self.ranges = ((1, 2), (5, 6))
        with self.assertRaisesRegex(ValueError, "different structure"):
            self.register()

    def test_legacy_structure_dispatch_preserves_signature(self):
        full = self.root / "full.pdb"
        full.write_text(pdb_text(self.reference), encoding="utf-8")
        path = prepare_structure(self.index, "tiny", full, output_dir=self.manifests)
        original = load_structure(path, self.reference)
        dispatched = load_inverse_structure(path, self.reference)
        self.assertEqual(dispatched.input_signature, original.input_signature)
        self.assertEqual(dispatched.metadata, original.metadata)
        torch.testing.assert_close(dispatched.coords, original.coords)
        torch.testing.assert_close(inverse_structure_positions(original.metadata, self.reference), original.positions)

    def test_brca2_acquisition_uses_pinned_named_fragments_and_preserves_uncovered_tail(self):
        assay_id = "BRCA2_HUMAN_Erwood_2022_HEK293T"
        reference = "A" * 1000 + "C" * 1085 + "D" * 747 + "V" * 586
        write_csv(self.index, ["DMS_id", "DMS_filename", "target_seq", "pdb_file", "pdb_range"],
                  [[assay_id, "tiny.csv", reference, "BRCA2_HUMAN.pdb", "1-2832"]])
        raw = self.root / "structures/raw"
        archive_path = raw.parent / "downloads/ProteinGym_v1.1_AF2_structures.zip"
        archive_path.parent.mkdir(parents=True)
        with zipfile.ZipFile(archive_path, "w") as archive:
            for filename, start, end in BRCA2_FRAGMENTS:
                archive.writestr(f"ProteinGym_AF2_structures/{filename}", pdb_text(reference[start - 1:end]))
            archive.writestr("unrelated.pdb", pdb_text())
        with patch("feature_generation.downloads.download_verified") as download:
            path = download_proteingym_structure(self.index, assay_id, raw_dir=raw, manifest_dir=self.manifests)
        download.assert_called_once_with(PROTEINGYM_STRUCTURE_URL, archive_path, PROTEINGYM_STRUCTURE_SHA256)
        bundle = load_inverse_structure(path, reference)
        self.assertEqual(bundle.reference_ranges, ((1, 1000), (1001, 2085), (2086, 2832)))
        self.assertEqual(bundle.positions.tolist(), list(range(1, 2833)))
        self.assertEqual(len(bundle.reference_sequence), 3418)
        self.assertEqual([fragment.reference_sequence for fragment in bundle.fragments],
                         [reference[:1000], reference[1000:2085], reference[2085:2832]])
        self.assertEqual(set(item.name for item in raw.iterdir()), {row[0] for row in BRCA2_FRAGMENTS})
        with patch("feature_generation.downloads.download_verified", side_effect=AssertionError("must reuse manifest")):
            self.assertEqual(download_proteingym_structure(self.index, assay_id, raw_dir=raw,
                                                          manifest_dir=self.manifests), path)


if __name__ == "__main__":
    unittest.main()
