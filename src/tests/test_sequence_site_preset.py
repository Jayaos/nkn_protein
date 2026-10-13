"""Exercise the shipped NKN selections against offline preparation outputs."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from config import ExperimentConfig
from data_loading import load_feature_bundle
from feature_generation.config import ExtractionConfig
from feature_generation.inverse_config import InverseFoldingConfig
from feature_generation.inverse_runner import extract as extract_inverse
from feature_generation.runner import extract
from feature_generation.score_runner import score
from feature_generation.structure_inputs import load_structure, prepare_structure
from kernels import ProteinKernel
from tests.test_esm2_centered_score_pipeline import FakeCenteredScoreAdapter
from tests.test_esm2_nonoverlap_pipeline import FakeNonoverlapAdapter
from tests.test_esm2_pooling import PrefixAdapter
from tests.test_feature_generation import residue_values, write_csv
from tests.test_inverse_pipeline import FakeConditionalAdapter
from tests.test_structure_inputs import pdb_text


REPOSITORY = Path(__file__).resolve().parents[2]


class SequenceSitePresetTests(unittest.TestCase):
    def test_shipped_preparation_outputs_supply_all_selected_nkn_branches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            index = raw / "DMS_substitutions.csv"
            ids = ("wt", "first", "second", "double")
            mutants = ("WT", "A1V", "C2D", "A1V:C2D")
            sequences = ("ACD", "VCD", "ADD", "VDD")
            write_csv(index, ["DMS_id", "DMS_filename", "target_seq", "seq_len"],
                      [["tiny", "tiny.csv", "ACD", 3]])
            write_csv(raw / "tiny.csv", ["variant_id", "mutant", "mutated_sequence", "DMS_score"], [
                [variant_id, mutant, sequence, value]
                for value, (variant_id, mutant, sequence) in enumerate(zip(ids, mutants, sequences))
            ])
            output_root = root / "features"
            paths = dict(output_root=output_root, assay_index=index, device="cpu", model_dir=root / "models")
            centered = replace(ExtractionConfig.load(
                REPOSITORY / "configs/extraction/esm2_650m_centered.json"), **paths)
            pooled = replace(ExtractionConfig.load(
                REPOSITORY / "configs/extraction/esm2_650m.json"), **paths)
            inverse = replace(InverseFoldingConfig.load(
                REPOSITORY / "configs/extraction/proteinmpnn_v48_020_kermut.json"),
                **paths, structure_manifest_dir=root / "manifests")
            config = ExperimentConfig.load(REPOSITORY / "configs/esm2_proteinmpnn_nkn.json")
            self.assertEqual(config.feature_selection["sequence"], f"esm2/{centered.cache_id}")
            self.assertNotEqual(centered.cache_id, pooled.cache_id)
            self.assertEqual(centered.global_window, "non_overlapping")
            self.assertEqual(centered.truncation, "none")
            self.assertTrue(centered.cache_id.endswith("-mean-all-nonoverlap1022-centered-sites"))

            # Create the pooled leaf first, so only the explicit selection can
            # override the assay's existing global-only default representation.
            extract(pooled, ["tiny"], adapter_factory=PrefixAdapter)
            result = extract(centered, ["tiny"], adapter_factory=FakeNonoverlapAdapter)[0]
            self.assertEqual(result["shape"], [4, 2])
            self.assertEqual(result["site_shape"], [4, 2])
            score_result = score(centered, ["tiny"], adapter_factory=FakeCenteredScoreAdapter)[0]
            self.assertEqual(config.score_sources, [score_result["score_source"]])

            pdb = root / "tiny.pdb"
            pdb.write_text(pdb_text(), encoding="utf-8")
            manifest = prepare_structure(index, "tiny", pdb, output_dir=root / "manifests")
            structure = load_structure(manifest, "ACD")
            extract_inverse(inverse, ["tiny"], adapter_factory=FakeConditionalAdapter)

            batch = load_feature_bundle(
                output_root / "tiny", feature_selection=config.feature_selection,
                score_sources=config.score_sources, pooling=config.pooling,
                enabled_branches=config.model.enabled_branches,
                inverse_folding_policy=config.inverse_folding_policy,
                structure_manifest_dir=root / "manifests",
            )
            expected = torch.stack([residue_values(sequence) for sequence in sequences]).double()
            self.assertEqual(batch.variant_ids, ids)
            self.assertEqual(batch.owners.tolist(), [1, 2, 3, 3])
            self.assertEqual(batch.mutation_positions.tolist(), [1, 2, 1, 2])
            self.assertEqual(set(batch.local_features), {"sequence_site", "inverse_folding", "structure_distance"})
            torch.testing.assert_close(batch.local_features["structure_distance"],
                                       structure.coords[batch.mutation_positions - 1, 1].double())
            torch.testing.assert_close(batch.global_features, expected.mean(dim=1))
            torch.testing.assert_close(batch.local_features["sequence_site"], expected[[1, 2, 3, 3], [0, 1, 0, 1]])
            self.assertTrue((batch.local_features["inverse_folding"].sum(-1) < 1).all())

            kernel = ProteinKernel(config.model, batch)
            self.assertEqual(set(kernel.branches), {"sequence_global", "sequence_site", "inverse_folding", "structure_distance"})
            covariance = kernel(batch)
            self.assertTrue(torch.isfinite(covariance).all())
            torch.testing.assert_close(covariance, covariance.T)
            changed_sites = batch.local_features["sequence_site"].clone()
            changed_sites[0] = torch.tensor([-25., 75.], dtype=torch.float64)
            changed = replace(batch, local_features={**batch.local_features, "sequence_site": changed_sites})
            self.assertFalse(torch.allclose(kernel(changed), covariance, rtol=1e-10, atol=1e-10))
            covariance.square().sum().backward()
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                for parameter in kernel.branches["sequence_site"].parameters()))


if __name__ == "__main__":
    unittest.main()
