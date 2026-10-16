"""Kermut exceptions preserve inference context and explicit reference coverage."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import torch

from config import ExperimentConfig
from data_loading import load_feature_bundle
from feature_generation.inputs import prepare_assay
from feature_generation.inverse_config import InverseFoldingConfig
from feature_generation.inverse_runner import extract
from feature_generation.inverse_storage import validate_inverse_cache
from feature_generation.score_runner import score
from feature_generation.score_storage import make_score_config, write_score_cache
from feature_generation.structure_inputs import prepare_structure, load_structure, download_proteingym_structure
from feature_generation.structure_fragments import load_inverse_structure
from feature_generation.structure_mappings import (
    CAS9_ASSAY, DOMAIN_RANGES, P53_ASSAYS, mapping_recipe, validate_mapped_metadata,
)
from tests.test_inverse_fragments import FragmentAdapter, fragment_log_probs
from tests.test_inverse_pipeline import write_csv
from tests.test_mean_score_pipeline import FakeScoreAdapter
from tests.test_structure_inputs import pdb_text, THREE


class StructureMappingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def fixture(self, assay):
        if assay in DOMAIN_RANGES:
            length, start, end, filename = DOMAIN_RANGES[assay]
            reference = "A" * length
            source = reference[start - 1:end]
        elif assay == CAS9_ASSAY:
            reference, source, filename, start, end = "A" * 1368, "A" * 1368 + "C" * 22, "CAS9_STRP1.pdb", 1, 1368
        else:
            reference = "A" * 71 + "R" + "A" * 321
            source = "A" * 71 + "P" + "A" * 321
            filename, start, end = "P53_HUMAN.pdb", 1, 393
        directory = self.root / assay
        directory.mkdir()
        index = directory / "index.csv"
        write_csv(index, ["DMS_id", "DMS_filename", "target_seq", "pdb_file"],
                  [[assay, "variants.csv", reference, filename]])
        mutants = ["WT", f"{reference[start - 1]}{start}V", f"{reference[end - 1]}{end}V"]
        rows = []
        for mutant in mutants:
            sequence = list(reference)
            if mutant != "WT":
                sequence[int(mutant[1:-1]) - 1] = mutant[-1]
            rows.append([mutant, mutant, "".join(sequence), len(rows)])
        write_csv(directory / "variants.csv", ["variant_id", "mutant", "mutated_sequence", "DMS_score"], rows)
        pdb = directory / filename
        with patch.dict(THREE, {"R": "ARG", "P": "PRO"}):
            pdb.write_text(pdb_text(source), encoding="utf-8")
        config = InverseFoldingConfig(
            assay_index=index, structure_manifest_dir=directory / "manifests", output_root=directory / "features",
            model_dir=directory / "models", device="cpu", conditioning_mode="sequence_conditional", seed=37,
        )
        manifest = prepare_structure(index, assay, pdb, output_dir=config.structure_manifest_dir)
        return reference, source, config, manifest, pdb

    def test_all_six_infer_original_sequence_then_map_outputs_and_resume(self):
        for assay in [*DOMAIN_RANGES, CAS9_ASSAY, *sorted(P53_ASSAYS)]:
            with self.subTest(assay=assay):
                reference, source, config, manifest, pdb = self.fixture(assay)
                original = pdb.read_bytes()
                structure = load_inverse_structure(manifest, reference)
                adapter = FragmentAdapter(config)
                result = extract(config, [assay], adapter_factory=lambda _: adapter)[0]
                self.assertEqual(adapter.calls, [source])
                self.assertEqual(result["covered_residues"], len(structure.positions))
                root = config.output_root / assay
                leaf = root / "inverse_folding/proteinmpnn" / config.cache_id
                payload = torch.load(leaf / "features.pt", weights_only=True)
                expected = fragment_log_probs(source)[structure.source_indices]
                torch.testing.assert_close(payload["log_probs"], expected, rtol=0, atol=0)
                self.assertEqual(payload["positions"].tolist(), structure.positions.tolist())
                self.assertEqual(payload["reference_sequence"], reference)
                validate_inverse_cache(root, "proteinmpnn", config.cache_id)
                before = {p.name: p.read_bytes() for p in leaf.iterdir() if p.is_file()}
                self.assertEqual(extract(config, [assay], adapter_factory=lambda _: adapter, resume=True)[0]["status"], "skipped")
                self.assertEqual(adapter.calls, [source])
                self.assertEqual(before, {p.name: p.read_bytes() for p in leaf.iterdir() if p.is_file()})
                self.assertEqual(pdb.read_bytes(), original)
                # ESM2 mean scores plus mapped inverse features and coordinates must load together.
                inputs = prepare_assay(config.assay_index, assay)
                score_config = make_score_config("test", "esm2", {
                    "scoring_recipe_version": 1, "score_definition": "sum_masked_marginal_log_odds",
                    "conditioning_mode": "masked_wild_type", "inference_method": "masked_marginals",
                    "normalization": "model_vocabulary", "output_dtype": "float32",
                })
                write_score_cache(root, "esm2", "test", inputs, score_config, torch.arange(3, dtype=torch.float32),
                                  {"uses_dms_labels": False})
                batch = load_feature_bundle(root, enabled_branches=["inverse_folding", "structure_distance"],
                                            inverse_folding_policy="sequence_conditional", score_sources=["scores/esm2/test"],
                                            structure_manifest_dir=config.structure_manifest_dir)
                self.assertEqual(batch.mutation_positions.tolist(), [int(structure.positions[0]), int(structure.positions[-1])])
                torch.testing.assert_close(batch.local_features["structure_distance"], structure.coords[[0, -1], 1].double())

    def test_full_reference_scores_supported_only_for_cas9_and_p53(self):
        for assay in [CAS9_ASSAY, sorted(P53_ASSAYS)[0], *DOMAIN_RANGES]:
            with self.subTest(assay=assay):
                reference, source, config, manifest, _ = self.fixture(assay)
                if assay in DOMAIN_RANGES:
                    with self.assertRaisesRegex(ValueError, "Domain-only"):
                        load_structure(manifest, reference)
                    continue
                structure = load_structure(manifest, reference)
                self.assertEqual(structure.reference_sequence, reference)
                self.assertEqual(structure.coords.shape, (len(reference), 4, 3))
                adapter = FakeScoreAdapter(config)
                self.assertEqual(score(config, [assay], adapter_factory=lambda _: adapter)[0]["status"], "completed")
                self.assertEqual(score(config, [assay], adapter_factory=lambda _: adapter, resume=True)[0]["status"], "skipped")
                self.assertEqual(len(adapter.score_calls), 1)

    def test_p53_only_documented_mismatch_and_manifest_integrity(self):
        assay = sorted(P53_ASSAYS)[0]
        reference, source, config, manifest, _ = self.fixture(assay)
        metadata = json.loads(manifest.read_text())
        self.assertEqual(metadata["source"]["residue_mapping"][71]["amino_acid"], "P")
        self.assertEqual(metadata["residue_mapping"][71]["amino_acid"], "R")
        with self.assertRaisesRegex(ValueError, "only the source"):
            mapping_recipe(assay, reference, "C" + source[1:])
        with self.assertRaisesRegex(ValueError, "No explicit"):
            mapping_recipe("another_assay", reference, source)
        changed = copy.deepcopy(metadata)
        changed["source_positions"][0] = 2
        with self.assertRaisesRegex(ValueError, "explicit Kermut recipe"):
            validate_mapped_metadata(changed, reference)
        changed = copy.deepcopy(metadata)
        changed["source"]["reference_sequence"] = reference
        with self.assertRaises(ValueError):
            validate_mapped_metadata(changed, reference)

    def test_uncovered_mutation_rejected_before_inference(self):
        assay = next(iter(DOMAIN_RANGES))
        reference, _, config, _, _ = self.fixture(assay)
        write_csv(config.assay_index.parent / "variants.csv", ["mutant", "mutated_sequence", "DMS_score"],
                  [["A1V", "V" + reference[1:], 1]])
        adapter = FragmentAdapter(config)
        with self.assertRaisesRegex(ValueError, "do not cover mutation positions"):
            extract(config, [assay], adapter_factory=lambda _: adapter)
        self.assertEqual(adapter.calls, [])

    def test_download_accepts_original_pdb_and_reuses_verified_manifest(self):
        for assay in [*DOMAIN_RANGES, CAS9_ASSAY, sorted(P53_ASSAYS)[0]]:
            with self.subTest(assay=assay):
                reference, source, config, _, pdb = self.fixture(assay)
                def downloaded(url, path, checksum):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(path, "w") as archive:
                        archive.writestr("structures/" + pdb.name, pdb.read_bytes())
                kwargs = dict(raw_dir=pdb.parent / "raw", manifest_dir=pdb.parent / "downloaded_manifests")
                with patch("feature_generation.downloads.download_verified", side_effect=downloaded) as download:
                    manifest = download_proteingym_structure(config.assay_index, assay, **kwargs)
                    self.assertEqual(load_inverse_structure(manifest, reference).source.reference_sequence, source)
                    self.assertEqual(download_proteingym_structure(config.assay_index, assay, **kwargs), manifest)
                    download.assert_called_once()
                self.assertEqual((kwargs["raw_dir"] / pdb.name).read_bytes(), pdb.read_bytes())

    def test_domain_config_retains_distance_but_uses_esm2_mean(self):
        config = ExperimentConfig(score_sources=["scores/esm2/test", "scores/proteinmpnn/test"])
        config = replace(config, model=replace(config.model, enabled_branches=("inverse_folding", "structure_distance")))
        for assay in DOMAIN_RANGES:
            resolved = config.for_assay(assay)
            self.assertEqual(resolved.model.enabled_branches, config.model.enabled_branches)
            self.assertEqual(resolved.score_sources, ["scores/esm2/test"])
        self.assertEqual(len(config.score_sources), 2)


if __name__ == "__main__":
    unittest.main()
