"""ProteinGym-style ESM-IF1 scores, cache integrity, and NKN mean inputs offline."""

import contextlib
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from data_loading import load_feature_bundle
from feature_generation.adapters.esm_if1 import ESMIF1Adapter
from feature_generation.cli import _progress
from feature_generation.inverse_config import ALPHABET, ESM_IF1_SCORE_RECIPE, InverseFoldingConfig
from feature_generation.inverse_runner import extract
from feature_generation.score_runner import default_cache_id, score
from feature_generation.score_storage import make_score_config, validate_score_cache
from gp.preprocessing import FoldTransform
from tests.test_inverse_pipeline import structure_metadata, refresh_structure_signature, write_csv


class TinyAlphabet:
    tokens = ("<bos>", *reversed(ALPHABET), "<unk>", "<pad>")
    padding_idx = len(tokens) - 1

    def __len__(self):
        return len(self.tokens)

    def get_idx(self, token):
        return self.tokens.index(token)

    def get_tok(self, index):
        return self.tokens[index]


def tokens_for(sequence):
    return torch.tensor([TinyAlphabet().get_idx(aa) for aa in sequence])


class PrefixModel(torch.nn.Module):
    """Position scores depend on the entire causal prefix, including special mass."""

    def __init__(self):
        super().__init__()
        self.calls = []

    @staticmethod
    def logits(previous):
        vocab = torch.arange(len(TinyAlphabet()), dtype=torch.float32)[None, :, None]
        centers = (previous.cumsum(-1) % 20 + 1)[:, None, :]
        return -(vocab - centers).square() / 30 + 6 * (vocab == 21)

    def forward(self, coords, padding, confidence, previous):
        assert not torch.is_grad_enabled()
        self.calls.append((coords.clone(), previous.clone()))
        return self.logits(previous), None


def fake_structure(sequence="ACD"):
    coords = torch.arange(len(sequence) * 12).reshape(len(sequence), 4, 3).float()
    coords[:, 3] = float("nan")  # ESM-IF1 does not need oxygen.
    return SimpleNamespace(reference_sequence=sequence, coords=coords)


def adapter_for(config):
    adapter = ESMIF1Adapter(config)
    adapter._device = torch.device("cpu")
    adapter._alphabet = TinyAlphabet()
    adapter._model = PrefixModel().eval()
    adapter.load = Mock()
    adapter.metadata = lambda: {
        "model_id": config.model_id, "revision": config.revision,
        "checkpoint_sha256": "c" * 64, "seed": config.seed, "max_length": config.max_length,
        "conditioning_mode": "reference_prefix", "normalization": "canonical_amino_acids_only",
        "prefix_sequence": "assay_reference", "noncanonical_tokens": "excluded_then_renormalized",
    }

    def converter(batch, device):
        sequences = [item[2] for item in batch]
        length = len(sequences[0])
        coords = torch.full((len(batch), length + 2, 3, 3), float("inf"))
        for index, (backbone, confidence, sequence) in enumerate(batch):
            assert confidence is None and backbone.shape == (length, 3, 3)
            coords[index, 1:-1] = torch.from_numpy(backbone)
        tokens = torch.stack([torch.cat((torch.zeros(1, dtype=torch.long), tokens_for(seq))) for seq in sequences])
        return coords, torch.ones(len(batch), length + 2), sequences, tokens, torch.zeros(len(batch), length + 2, dtype=torch.bool)

    adapter._converter = converter
    return adapter


def expected_score(sequence):
    targets = tokens_for(sequence)
    previous = torch.cat((torch.zeros(1, dtype=torch.long), targets[:-1]))[None]
    logits = PrefixModel.logits(previous)[0]
    # Explicit native-vocabulary log probability, independent of cross_entropy.
    return torch.stack([logits[token, i] - torch.logsumexp(logits[:, i], 0)
                        for i, token in enumerate(targets)]).mean()


class ESMIF1ScoreAdapterTests(unittest.TestCase):
    def test_native_formula_variant_prefixes_full_backbone_and_batch_invariance(self):
        sequences = ["ACD", "VCD", "ADD", "VDD"]
        structure = fake_structure()
        expected = torch.stack([expected_score(sequence) for sequence in sequences])
        for size in (1, 3, 8):
            adapter = adapter_for(InverseFoldingConfig(model_family="esm_if1", device="cpu"))
            state = torch.get_rng_state().clone()
            batches = list(adapter.iter_scores(structure, sequences, batch_size=size))
            self.assertEqual([index for indices, _ in batches for index in indices], list(range(4)))
            actual = torch.cat([values for _, values in batches])
            torch.testing.assert_close(actual, expected)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            self.assertLess(actual[0].item(), 0)  # No WT subtraction.
            previous = torch.cat([call[1] for call in adapter._model.calls])
            self.assertEqual(previous.tolist(), [[0, *tokens_for(seq[:-1]).tolist()] for seq in sequences])
            for coords, _ in adapter._model.calls:
                torch.testing.assert_close(coords[:, 1:-1], structure.coords[None, :, :3].expand(len(coords), -1, -1, -1))
        wt_logits = PrefixModel.logits(torch.tensor([[0, *tokens_for("AC").tolist()]]))[0]
        canonical = [TinyAlphabet().get_idx(aa) for aa in ALPHABET]
        renormalized = wt_logits[canonical].log_softmax(0)
        kernel_lookup = torch.stack([renormalized[ALPHABET.index(aa), i] for i, aa in enumerate("ACD")]).mean()
        self.assertGreater(abs((kernel_lookup - expected[0]).item()), 0.1)
        wt_prefix_variant = wt_logits.log_softmax(0).gather(0, tokens_for("VCD")[None]).mean()
        self.assertGreater(abs((wt_prefix_variant - expected[1]).item()), 0.01)

    def test_long_sequence_scores_include_every_residue(self):
        sequence = "ACD" * 621
        adapter = adapter_for(InverseFoldingConfig(model_family="esm_if1"))
        batches = list(adapter.iter_scores(fake_structure(sequence), [sequence]))
        torch.testing.assert_close(batches[0][1][0], expected_score(sequence))
        self.assertEqual(adapter._model.calls[0][1].shape, (1, 1863))

    def test_invalid_sequences_and_limits_fail_before_loading(self):
        for sequences in ([], None, "ACD", ["AC"], ["ACX"], ["ACD", "ACDE"]):
            adapter = adapter_for(InverseFoldingConfig(model_family="esm_if1"))
            with self.subTest(sequences=sequences), self.assertRaises(ValueError):
                list(adapter.iter_scores(fake_structure(), sequences))
            adapter.load.assert_not_called()
        adapter = adapter_for(InverseFoldingConfig(model_family="esm_if1", max_length=2))
        with self.assertRaisesRegex(ValueError, "max_length"):
            list(adapter.iter_scores(fake_structure(), ["ACD"]))
        adapter.load.assert_not_called()

    def test_bad_native_outputs_and_token_alignment_are_rejected(self):
        for logits in (torch.zeros(1, 20, 3), torch.full((1, 23, 3), float("nan"))):
            adapter = adapter_for(InverseFoldingConfig(model_family="esm_if1"))
            adapter._model = lambda *args: (logits, None)
            with self.subTest(shape=logits.shape), self.assertRaisesRegex(ValueError, "native logits"):
                list(adapter.iter_scores(fake_structure(), ["ACD"]))
        adapter = adapter_for(InverseFoldingConfig(model_family="esm_if1"))
        converter = adapter._converter
        def wrong_tokens(*args, **kwargs):
            converted = converter(*args, **kwargs)
            converted[3][0, -1] = adapter._alphabet.padding_idx
            return converted
        adapter._converter = wrong_tokens
        with self.assertRaisesRegex(ValueError, "complete variant sequences"):
            list(adapter.iter_scores(fake_structure(), ["ACD"]))


class ESMIF1ScorePipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = self.root / "index.csv"
        self.sequences = ["ACD", "VCD", "ADD", "VDD"]
        write_csv(self.index, ["DMS_id", "DMS_filename", "target_seq", "seq_len"], [["tiny", "tiny.csv", "ACD", 3]])
        write_csv(self.root / "tiny.csv", ["variant_id", "mutant", "mutated_sequence", "DMS_score"],
                  [[str(i), mutant, sequence, i] for i, (mutant, sequence) in enumerate(
                      zip(("WT", "A1V", "C2D", "A1V:C2D"), self.sequences))])
        self.config = InverseFoldingConfig(model_family="esm_if1", device="cpu", assay_index=self.index,
                                          output_root=self.root / "features", structure_manifest_dir=self.root / "manifests")
        self.structure = fake_structure()
        self.structure.metadata = structure_metadata()
        for row in self.structure.metadata["residue_mapping"]:
            row["backbone_atoms"] = ["N", "CA", "C"]
        refresh_structure_signature(self.structure.metadata)
        self.structure.input_signature = self.structure.metadata["input_signature"]
        self.structure.structure_sha256 = self.structure.metadata["structure_sha256"]
        self.adapters = []
        loader = patch("feature_generation.score_runner.load_structure", return_value=self.structure)
        self.loader = loader.start()
        self.addCleanup(loader.stop)
        self.assay_root = self.config.output_root / "tiny"
        self.cache_id = default_cache_id(self.config)
        self.leaf = self.assay_root / "scores/esm_if1" / self.cache_id

    def factory(self, config):
        adapter = adapter_for(config)
        self.adapters.append(adapter)
        return adapter

    def run_score(self, **kwargs):
        return score(self.config, ["tiny"], adapter_factory=self.factory, **kwargs)

    def test_roundtrip_resume_and_nkn_standardization_preserve_kernel_caches(self):
        with patch("feature_generation.inverse_runner.load_structure", return_value=self.structure):
            extract(self.config, ["tiny"], adapter_factory=self.factory)
        kernel_files = {p: p.read_bytes() for p in self.assay_root.rglob("*") if p.is_file()}
        progress = io.StringIO()
        with contextlib.redirect_stderr(progress):
            result = self.run_score(progress=_progress)[0]
        self.assertIn("ESM-IF1 native log-likelihood", progress.getvalue())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.adapters[-1]._model.calls), 4)  # Default batch size is one.
        self.assertFalse(self.loader.call_args.kwargs["require_oxygen"])
        validate_score_cache(self.assay_root, "esm_if1", self.cache_id)
        self.assertEqual({p: p.read_bytes() for p in kernel_files}, kernel_files)
        batch = load_feature_bundle(self.assay_root, enabled_branches=["inverse_folding", "inverse_folding_site"],
                                    score_sources=[result["score_source"]])
        torch.testing.assert_close(batch.scores[:, 0], torch.stack([expected_score(s) for s in self.sequences]).double(),
                                   rtol=1e-5, atol=1e-6)
        train = batch.select([0, 1])
        transform = FoldTransform.fit(train)
        torch.testing.assert_close(transform.apply(batch).scores,
                                   (batch.scores - train.scores.mean(0)) / train.scores.std(0, unbiased=False))
        original = {p: p.read_bytes() for p in self.leaf.iterdir()}
        self.config = replace(self.config, max_length=4096)
        self.assertEqual(self.run_score(batch_size=3, resume=True)[0]["status"], "skipped")
        self.adapters[-1].load.assert_not_called()
        self.assertEqual({p: p.read_bytes() for p in original}, original)
        metadata = json.loads((self.leaf / "config.json").read_text())
        for key, value in ESM_IF1_SCORE_RECIPE.items():
            self.assertEqual(metadata[key], value)
        self.assertFalse(json.loads((self.leaf / "provenance.json").read_text())["uses_dms_labels"])

    def test_recipe_structure_and_file_corruption_are_rejected(self):
        self.run_score()
        metadata = self.adapters[-1].score_metadata()
        for change in ({"normalization": "canonical_amino_acids_only"}, {"prefix_sequence": "assay_reference"},
                       {"wild_type_subtraction": True}, {"residue_aggregation": "sum_over_mutations"}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "ESM-IF1 mean-score recipe"):
                make_score_config("bad", "esm_if1", {**metadata, **change}, self.structure.metadata)
        partial = copy.deepcopy(self.structure.metadata)
        partial["residue_mapping"].pop()
        with self.assertRaisesRegex(ValueError, "every reference position"):
            make_score_config("partial", "esm_if1", metadata, partial)
        self.config = replace(self.config, seed=1)
        with self.assertRaisesRegex(ValueError, "new cache ID"):
            self.run_score(resume=True)
        path = self.leaf / "mean_score.csv"
        path.write_text(path.read_text().replace("0,", "missing,"))
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_score_cache(self.assay_root, "esm_if1", self.cache_id)

    def test_controls_and_incomplete_structures_fail_before_model_loading(self):
        for kwargs in ({"num_orders": 2}, {"batch_size": 0}, {"batch_size": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_score(**kwargs)
        self.loader.assert_not_called()
        self.loader.side_effect = ValueError("Fragment collections do not define a shared coordinate frame")
        result = self.run_score(continue_on_error=True)[0]
        self.assertEqual(result["status"], "failed")
        self.adapters[-1].load.assert_not_called()
        self.assertFalse(self.leaf.exists())


if __name__ == "__main__":
    unittest.main()
