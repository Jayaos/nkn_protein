"""Fragment-local inference stitches only explicitly covered reference positions."""

from dataclasses import replace
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from data_loading import load_feature_bundle
from feature_generation.cli import main
from feature_generation.inputs import prepare_assay, sha256_file, stable_json
from feature_generation.inverse_config import InverseFoldingConfig
from feature_generation.inverse_runner import extract
from feature_generation.inverse_storage import validate_inverse_cache
from feature_generation.score_storage import make_score_config, write_score_cache
from feature_generation.structure_fragments import load_inverse_structure, prepare_structure_fragments
from tests.test_inverse_pipeline import FakeConditionalAdapter, write_csv
from tests.test_structure_inputs import pdb_text


ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def fragment_log_probs(sequence):
    centers = torch.tensor([ALPHABET.index(aa) for aa in sequence], dtype=torch.float32)
    # Local context changes each residue's output. This makes incorrect global
    # inference and off-by-one stitching visible in the saved probabilities.
    centers += torch.arange(len(sequence), dtype=torch.float32) / 10
    logits = -(torch.arange(20, dtype=torch.float32)[None, :] - centers[:, None]).abs()
    return logits.log_softmax(-1) - 0.25


class FragmentAdapter(FakeConditionalAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.calls = []

    def predict(self, structure):
        self.predict_count += 1
        if torch.is_grad_enabled():
            raise AssertionError("Inference must disable gradients")
        sequence = structure.reference_sequence
        if structure.coords.shape != (len(sequence), 4, 3):
            raise AssertionError("Each model call needs its own local fragment coordinates")
        self.calls.append(sequence)
        logs = fragment_log_probs(sequence)
        return {"probabilities": logs.exp(), "log_probs": logs}


class InverseFragmentPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reference = "ACDVVADV"
        self.index = self.root / "index.csv"
        write_csv(self.index, ["DMS_id", "DMS_filename", "target_seq", "seq_len"],
                  [["tiny", "tiny.csv", self.reference, len(self.reference)]])
        self.rows = [self.variant("WT"), self.variant("C2A"), self.variant("D3A"),
                     self.variant("V4A"), self.variant("V5A"), self.variant("A6C"),
                     self.variant("C2A:V5A")]
        self.write_variants(self.rows)
        self.manifests = self.root / "manifests"
        self.register_fragments(((1, 2), (3, 4), (5, 6)))
        self.config = InverseFoldingConfig(
            output_root=self.root / "features", assay_index=self.index,
            structure_manifest_dir=self.manifests, model_dir=self.root / "models",
            cache_id="fragment-conditional", conditioning_mode="sequence_conditional",
            device="cpu", seed=37,
        )
        self.adapters = []

    def variant(self, mutant):
        sequence = list(self.reference)
        if mutant != "WT":
            for token in mutant.split(":"):
                sequence[int(token[1:-1]) - 1] = token[-1]
        return [mutant, mutant, "".join(sequence), float(len(mutant))]

    def write_variants(self, rows):
        write_csv(self.root / "tiny.csv", ["variant_id", "mutant", "mutated_sequence", "DMS_score"], rows)

    def register_fragments(self, ranges, directory=None):
        files = []
        for start, end in ranges:
            path = self.root / f"fragment_{start}_{end}.pdb"
            path.write_text(pdb_text(self.reference[start - 1:end]), encoding="utf-8")
            files.append(path)
        return prepare_structure_fragments(
            self.index, "tiny", files, ranges, output_dir=directory or self.manifests,
        )

    def factory(self, config):
        adapter = FragmentAdapter(config)
        self.adapters.append(adapter)
        return adapter

    @property
    def assay_root(self):
        return self.config.output_root / "tiny"

    @property
    def leaf(self):
        return self.assay_root / "inverse_folding/proteinmpnn" / self.config.cache_id

    def run_extraction(self, **kwargs):
        return extract(self.config, ["tiny"], adapter_factory=self.factory, **kwargs)

    def add_mean_scores(self):
        inputs = prepare_assay(self.index, "tiny")
        config = make_score_config("test", "esm2", {
            "scoring_recipe_version": 1, "score_definition": "sum_masked_marginal_log_odds",
            "conditioning_mode": "masked_wild_type", "inference_method": "masked_marginals",
            "normalization": "model_vocabulary", "output_dtype": "float32",
        })
        write_score_cache(self.assay_root, "esm2", "test", inputs, config,
                          torch.arange(len(inputs.variant_ids), dtype=torch.float32),
                          {"uses_dms_labels": False})

    def load(self):
        return load_feature_bundle(
            self.assay_root, enabled_branches=["inverse_folding"],
            inverse_folding_policy="sequence_conditional", score_sources=["scores/esm2/test"],
        )

    def test_separate_inference_calls_stitch_partial_positions_and_preserve_raw_mass(self):
        result = self.run_extraction()[0]
        self.assertEqual(result["shape"], [6, 20])
        self.assertEqual(result["covered_residues"], 6)
        self.assertEqual(result["num_structure_fragments"], 3)
        self.assertEqual(self.adapters[0].load_count, 1)
        self.assertEqual(self.adapters[0].calls, ["AC", "DV", "VA"])
        payload = torch.load(self.leaf / "features.pt", weights_only=True)
        self.assertEqual(payload["reference_sequence"], self.reference)
        self.assertEqual(payload["positions"].tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(payload["probabilities"].shape, (6, 20))
        expected_logs = torch.cat([fragment_log_probs(sequence) for sequence in ("AC", "DV", "VA")])
        torch.testing.assert_close(payload["log_probs"], expected_logs, rtol=0, atol=0)
        torch.testing.assert_close(payload["probabilities"], expected_logs.exp(), rtol=0, atol=0)
        self.assertTrue((payload["probabilities"].sum(-1) < 1).all())
        config = json.loads((self.leaf / "config.json").read_text())
        self.assertEqual(config["structure"]["schema_version"], 2)
        self.assertEqual(config["structure"]["reference_sequence"], self.reference)
        self.assertEqual(len(config["structure"]["fragments"]), 3)

        report = validate_inverse_cache(self.assay_root, "proteinmpnn", self.config.cache_id)
        self.assertEqual(report["reference_length"], 8)
        self.assertEqual(report["covered_residues"], 6)
        before = {path.name: path.read_bytes() for path in self.leaf.iterdir() if path.is_file()}
        self.assertEqual(self.run_extraction(resume=True)[0]["status"], "skipped")
        self.assertEqual(sum(adapter.predict_count for adapter in self.adapters), 3)
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 1)
        self.assertEqual({path.name: path.read_bytes() for path in self.leaf.iterdir() if path.is_file()}, before)

    def test_loader_gathers_boundary_and_multi_mutant_sites_without_truncating_sequences(self):
        self.run_extraction()
        self.add_mean_scores()
        batch = self.load()
        payload = torch.load(self.leaf / "features.pt", weights_only=True)
        self.assertEqual(batch.reference_sequence, self.reference)
        self.assertTrue(all(len(sequence) == 8 for sequence in batch.sequences))
        self.assertEqual(batch.mutation_positions.tolist(), [2, 3, 4, 5, 6, 2, 5])
        expected = payload["probabilities"][[1, 2, 3, 4, 5, 1, 4]].double()
        torch.testing.assert_close(batch.local_features["inverse_folding"], expected, rtol=0, atol=0)
        self.assertEqual(batch.select([0]).local_features["inverse_folding"].shape, (0, 20))

    def test_internal_uncovered_position_does_not_shift_later_feature_rows(self):
        self.rows = [self.variant("WT"), self.variant("C2A"), self.variant("V4A"), self.variant("A6C")]
        self.write_variants(self.rows)
        manifests = self.root / "gapped_manifests"
        self.register_fragments(((1, 2), (4, 4), (5, 6)), manifests)
        self.config = replace(self.config, structure_manifest_dir=manifests)
        self.run_extraction()
        self.assertEqual(self.adapters[0].calls, ["AC", "V", "VA"])
        self.add_mean_scores()
        payload = torch.load(self.leaf / "features.pt", weights_only=True)
        self.assertEqual(payload["positions"].tolist(), [1, 2, 4, 5, 6])
        expected = payload["probabilities"][[1, 2, 4]].double()
        torch.testing.assert_close(self.load().local_features["inverse_folding"], expected, rtol=0, atol=0)

    def test_uncovered_mutation_rejected_before_inference_and_by_prediction_loader(self):
        self.write_variants([*self.rows, self.variant("D7A")])
        with self.assertRaisesRegex(ValueError, "cover|position"):
            self.run_extraction()
        self.assertEqual(sum(adapter.predict_count for adapter in self.adapters), 0)
        self.assertFalse(self.leaf.exists())

        self.write_variants(self.rows)
        self.run_extraction()
        self.add_mean_scores()
        # A new target outside cached coverage must not index another site or
        # silently fabricate its distribution, even without a DMS label.
        write_csv(self.assay_root / "variants.csv", ["variant_id", "mutant", "mutated_sequence"],
                  [self.variant("D7A")[:3]])
        with self.assertRaisesRegex(ValueError, "cover|position"):
            self.load()

    def test_cache_validation_rejects_resealed_covered_position_tampering(self):
        self.run_extraction()
        path = self.leaf / "features.pt"
        payload = torch.load(path, weights_only=True)
        payload["positions"][-1] = 8
        torch.save(payload, path)
        completion_path = self.leaf / "completion.json"
        completion = json.loads(completion_path.read_text())
        completion["files_sha256"]["features.pt"] = sha256_file(path)
        completion_path.write_text(json.dumps(completion), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "position|cover|mapping"):
            validate_inverse_cache(self.assay_root, "proteinmpnn", self.config.cache_id)

    def test_wrong_assay_manifest_rejected_before_loading_model(self):
        path = self.manifests / "tiny.json"
        metadata = json.loads(path.read_text())
        metadata["assay_id"] = "another_assay"
        for fragment in metadata["fragments"]:
            fragment["structure"]["assay_id"] = "another_assay"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        # The structure and sequence signatures remain valid: it is the
        # association with this assay that must fail before GPU model loading.
        load_inverse_structure(path, self.reference)
        with self.assertRaisesRegex(ValueError, "[Ss]tructure.*different assay"):
            self.run_extraction()
        self.assertEqual(sum(adapter.load_count for adapter in self.adapters), 0)
        self.assertEqual(sum(adapter.predict_count for adapter in self.adapters), 0)
        self.assertFalse(self.leaf.exists())

    def test_resealed_cache_for_wrong_structure_assay_rejected_by_validator_and_loader(self):
        self.run_extraction()
        self.add_mean_scores()
        config_path = self.leaf / "config.json"
        config = json.loads(config_path.read_text())
        config["structure"]["assay_id"] = "another_assay"
        for fragment in config["structure"]["fragments"]:
            fragment["structure"]["assay_id"] = "another_assay"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        signature = hashlib.sha256(stable_json(config).encode("utf-8")).hexdigest()
        payload_path = self.leaf / "features.pt"
        payload = torch.load(payload_path, weights_only=True)
        payload["config_signature"] = signature
        torch.save(payload, payload_path)
        completion_path = self.leaf / "completion.json"
        completion = json.loads(completion_path.read_text())
        completion["config_signature"] = signature
        for path in (config_path, payload_path):
            completion["files_sha256"][path.name] = sha256_file(path)
        completion_path.write_text(json.dumps(completion), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "[Ss]tructure.*different assay"):
            validate_inverse_cache(self.assay_root, "proteinmpnn", self.config.cache_id)
        with self.assertRaisesRegex(ValueError, "[Ss]tructure.*different assay"):
            self.load()

    def test_changed_fragment_bytes_block_resume_without_new_model_calls(self):
        self.run_extraction()
        with (self.root / "fragment_3_4.pdb").open("a", encoding="utf-8") as handle:
            handle.write("REMARK changed\n")
        with self.assertRaisesRegex(ValueError, "checksum|changed"):
            self.run_extraction(resume=True)
        self.assertEqual(sum(adapter.predict_count for adapter in self.adapters), 3)

    def invoke(self, arguments):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            status = main([str(argument) for argument in arguments])
        return status, output.getvalue(), errors.getvalue()

    def test_cli_registers_repeated_structure_and_reference_range_options(self):
        destination = self.root / "cli_manifests"
        arguments = ["prepare-structure", "--assay", "tiny", "--assay-index", self.index,
                     "--manifest-dir", destination]
        for start, end in ((1, 2), (3, 4), (5, 6)):
            arguments += ["--structure", self.root / f"fragment_{start}_{end}.pdb",
                          "--reference-range", f"{start}-{end}"]
        status, output, errors = self.invoke(arguments)
        self.assertEqual(status, 0, errors)
        manifest = Path(json.loads(output)["manifest"])
        structure = load_inverse_structure(manifest, self.reference)
        self.assertEqual(structure.positions.tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual([item.reference_sequence for item in structure.fragments], ["AC", "DV", "VA"])
        self.assertFalse(hasattr(structure, "coords"))

    def test_cli_requires_one_valid_explicit_range_for_each_fragment(self):
        destination = self.root / "invalid_manifests"
        arguments = ["prepare-structure", "--assay", "tiny", "--assay-index", self.index,
                     "--manifest-dir", destination,
                     "--structure", self.root / "fragment_1_2.pdb",
                     "--structure", self.root / "fragment_3_4.pdb"]
        for ranges in ([], ["--reference-range", "1-2"],
                       ["--reference-range", "1-2", "--reference-range", "3:4"]):
            with self.subTest(ranges=ranges):
                status, _, errors = self.invoke([*arguments, *ranges])
                self.assertEqual(status, 1)
                self.assertIn("--reference-range", errors)
                self.assertFalse((destination / "tiny.json").exists())


if __name__ == "__main__":
    unittest.main()
