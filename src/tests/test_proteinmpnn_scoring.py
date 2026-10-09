"""Official sequence-score semantics, independent of downloaded model weights."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from feature_generation.adapters.proteinmpnn import ProteinMPNNAdapter
from feature_generation.inverse_config import InverseFoldingConfig


NATIVE_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def structure(sequence="ACD"):
    return SimpleNamespace(
        reference_sequence=sequence,
        coords=torch.arange(len(sequence) * 12).reshape(len(sequence), 4, 3).float(),
    )


class ContextModel(torch.nn.Module):
    """A native-shaped decoder whose distribution depends on earlier residues."""

    def __init__(self):
        super().__init__()
        self.calls = []
        self.outputs = []

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding, randn):
        self.calls.append(tuple(value.clone() for value in (X, S, mask, chain_M, residue_idx, chain_encoding, randn)))
        assert not torch.is_grad_enabled()
        rank = randn.abs().argsort(-1).argsort(-1)
        preceding = rank[:, None, :] < rank[:, :, None]
        context = (preceding * S[:, None, :]).sum(-1)
        alphabet = torch.arange(21, device=S.device, dtype=torch.float32)
        logits = -(alphabet - (2 + context % 11).unsqueeze(-1)).abs() / 4
        logits[..., 20] = 5  # X keeps substantial probability mass.
        output = logits.log_softmax(-1)
        self.outputs.append(output.clone())
        return output


def adapter(seed=0, model=None):
    value = ProteinMPNNAdapter(InverseFoldingConfig(device="cpu", seed=seed))
    value._device = torch.device("cpu")
    value._model = model if model is not None else ContextModel()
    value.load = Mock()
    return value


def scores(value, sequences, **kwargs):
    batches = list(value.iter_scores(structure(), sequences, **kwargs))
    return torch.cat([batch for _, batch in batches])


class ProteinMPNNScoringTests(unittest.TestCase):
    def test_native_forward_receives_actual_sequences_and_full_chain_masks(self):
        value = adapter()
        batches = list(value.iter_scores(structure(), ["ACD", "WYG", "CDE"], batch_size=2))
        self.assertEqual([indices for indices, _ in batches], [[0, 1], [2]])
        X, S, mask, chain_M, residue_idx, chain_encoding, randn = value._model.calls[0]
        torch.testing.assert_close(X, structure().coords[None].expand(2, -1, -1, -1))
        self.assertEqual(S.tolist(), [[0, 1, 2], [18, 19, 5]])
        self.assertEqual(S.dtype, torch.long)
        torch.testing.assert_close(mask, torch.ones(2, 3))
        torch.testing.assert_close(chain_M, mask)
        torch.testing.assert_close(residue_idx, torch.tensor([[0, 1, 2], [0, 1, 2]]))
        torch.testing.assert_close(chain_encoding, torch.ones(2, 3, dtype=torch.long))
        torch.testing.assert_close(randn[0], randn[1])
        for _, result in batches:
            self.assertEqual(result.device.type, "cpu")
            self.assertEqual(result.dtype, torch.float32)
            self.assertTrue(result.is_contiguous())
            self.assertFalse(result.requires_grad)

    def test_score_is_exact_negative_official_nll_without_canonical_renormalization(self):
        value = adapter()
        result = scores(value, ["ACD", "WYG"])
        _, S, mask, *_ = value._model.calls[0]
        log_probs = value._model.outputs[0]
        # Upstream protein_mpnn_utils._scores verbatim mathematical operations.
        criterion = torch.nn.NLLLoss(reduction="none")
        loss = criterion(log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)).view(S.size())
        official_nll = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
        torch.testing.assert_close(result, -official_nll, rtol=0, atol=0)
        renormalized = log_probs[..., :20].log_softmax(-1).gather(-1, S[..., None]).squeeze(-1).mean(-1)
        self.assertTrue(torch.all((result - renormalized).abs() > 1))
        self.assertLess(result[0].item(), 0)  # WT is not subtracted.

    def test_full_sequence_context_affects_unchanged_positions(self):
        value = adapter()
        result = scores(value, ["AAA", "CAA"])
        log_probs = value._model.outputs[0]
        # With seed 0 the decoding order is [1, 0, 2]: C at position 0
        # changes probability at unchanged position 2 through sequence context.
        self.assertNotEqual(log_probs[0, 2, 0].item(), log_probs[1, 2, 0].item())
        site_only_difference = log_probs[1, 0, 1] - log_probs[0, 0, 0]
        self.assertNotAlmostEqual((3 * (result[1] - result[0])).item(), site_only_difference.item())
        self.assertFalse(torch.equal(value._model.calls[0][1][0], value._model.calls[0][1][1]))

    def test_orders_are_shared_and_repeats_average_independently_of_batch_size(self):
        sequences = ["ACD", "CDE", "WYG", "AAA", "CCC"]
        first, second = adapter(seed=17), adapter(seed=17)
        result = scores(first, sequences, batch_size=2, num_orders=3)
        expected = scores(second, sequences, batch_size=4, num_orders=3)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        generated = torch.randn((3, 3), generator=torch.Generator().manual_seed(17))
        for call_index, call in enumerate(first._model.calls):
            torch.testing.assert_close(call[-1], generated[call_index % 3][None].expand(call[1].shape[0], -1))
        first_batch_scores = []
        for call, output in zip(first._model.calls[:3], first._model.outputs[:3]):
            first_batch_scores.append(output.gather(-1, call[1][..., None]).squeeze(-1).mean(-1))
        torch.testing.assert_close(result[:2], torch.stack(first_batch_scores).mean(0))
        other_seed = scores(adapter(seed=18), sequences, batch_size=2, num_orders=3)
        self.assertFalse(torch.equal(result, other_seed))

    def test_scoring_preserves_global_random_states(self):
        value = adapter()
        cpu_state = torch.get_rng_state().clone()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        scores(value, ["ACD", "CDE"], num_orders=4)
        self.assertTrue(torch.equal(cpu_state, torch.get_rng_state()))
        if cuda_states:
            self.assertTrue(all(torch.equal(old, new) for old, new in zip(cuda_states, torch.cuda.get_rng_state_all())))

    def test_invalid_parameters_and_sequences_fail_before_model_loading(self):
        for kwargs in ({"batch_size": 0}, {"batch_size": True}, {"batch_size": 1.5},
                       {"num_orders": 0}, {"num_orders": False}, {"num_orders": 1.5}):
            value = adapter()
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                list(value.iter_scores(structure(), ["ACD"], **kwargs))
            value.load.assert_not_called()
        for sequences in ([], None, "ACD", [None], ["AC"], ["ACDE"], ["ACX"], ["acd"], ["ACD", "AC-"]):
            value = adapter()
            with self.subTest(sequences=sequences), self.assertRaises(ValueError):
                list(value.iter_scores(structure(), sequences))
            value.load.assert_not_called()
        value = adapter()
        inputs = structure()
        inputs.coords[0, 3] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            list(value.iter_scores(inputs, ["ACD"]))
        value.load.assert_not_called()

    def test_rejects_bad_native_shapes_and_nonfinite_log_probabilities(self):
        for output in (torch.zeros(1, 3, 20), torch.zeros(1, 2, 21), torch.full((1, 3, 21), float("nan")),
                       torch.full((1, 3, 21), float("inf"))):
            value = adapter(model=Mock(return_value=output))
            with self.subTest(shape=tuple(output.shape)), self.assertRaisesRegex(ValueError, "native log-probabilities"):
                scores(value, ["ACD"])

    def test_metadata_overrides_backbone_distribution_semantics(self):
        value = adapter(seed=5)
        backbone_metadata = {
            "feature_type": "inverse_folding", "conditioning_mode": "backbone_only",
            "inference_method": "unconditional_probs", "decoding_order": "none",
            "alphabet": NATIVE_ALPHABET[:-1], "normalization": "canonical_amino_acids_only",
            "noncanonical_tokens": "excluded_then_renormalized",
            "precision_policy": "float32_log_softmax_reject_probability_underflow", "seed": 5,
        }
        with patch.object(value, "metadata", return_value=backbone_metadata):
            metadata = value.score_metadata(num_orders=3)
        self.assertEqual(metadata["feature_type"], "scores")
        self.assertEqual(metadata["conditioning_mode"], "sequence_autoregressive")
        self.assertEqual(metadata["inference_method"], "forward")
        self.assertEqual(metadata["normalization"], "native_21_amino_acid_tokens")
        self.assertEqual(metadata["noncanonical_tokens"], "retained_in_normalization")
        self.assertEqual(metadata["alphabet"], NATIVE_ALPHABET)
        self.assertEqual(metadata["num_orders"], 3)
        self.assertEqual(metadata["decoding_order_seed"], 5)
        self.assertEqual(metadata["residue_aggregation"], "mean_over_all_reference_residues")
        self.assertEqual(metadata["order_aggregation"], "arithmetic_mean_of_scores")
        self.assertEqual(metadata["scoring_recipe_version"], 1)
        self.assertEqual(metadata["score_direction"], "higher_is_more_likely")
        self.assertFalse(metadata["wild_type_subtraction"])
        for num_orders in (0, True, 1.5):
            with self.subTest(num_orders=num_orders), self.assertRaisesRegex(ValueError, "num_orders"):
                value.score_metadata(num_orders=num_orders)


if __name__ == "__main__":
    unittest.main()
