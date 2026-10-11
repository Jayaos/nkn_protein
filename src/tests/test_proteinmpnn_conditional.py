"""Kermut conditional feature semantics and parity with pinned native inference."""

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from feature_generation.adapters.proteinmpnn import ProteinMPNNAdapter
from feature_generation.inverse_checkpoints import checkpoint_path
from feature_generation.inverse_config import InverseFoldingConfig


def structure(sequence="ACDE"):
    return SimpleNamespace(
        reference_sequence=sequence,
        coords=torch.randn((len(sequence), 4, 3), generator=torch.Generator().manual_seed(4)),
    )


class ConditionalModel:
    """Keep sequence context, order randomness, and X probability observable."""

    def __init__(self):
        self.calls = []
        self.outputs = []

    def conditional_probs(self, X, S, mask, chain_M, residue_idx, chain_encoding, randn, *, backbone_only):
        assert not torch.is_grad_enabled()
        self.calls.append((X.clone(), S.clone(), mask.clone(), chain_M.clone(), residue_idx.clone(),
                           chain_encoding.clone(), randn.clone(), backbone_only))
        # The target token is excluded from this fake conditional context.
        context = S.sum(-1, keepdim=True) - S
        alphabet = torch.arange(21, device=S.device, dtype=torch.float32)
        logits = -(alphabet - context.unsqueeze(-1)).abs() + randn.unsqueeze(-1) * alphabet / 3
        logits[..., 20] = 6
        result = logits.log_softmax(-1)
        self.outputs.append(result.clone())
        return result


def adapter(*, seed=37, num_orders=10):
    value = ProteinMPNNAdapter(InverseFoldingConfig(
        device="cpu", conditioning_mode="sequence_conditional", seed=seed, num_orders=num_orders,
    ))
    value._device = torch.device("cpu")
    value._model = ConditionalModel()
    value.load = Mock()
    return value


class ProteinMPNNConditionalTests(unittest.TestCase):
    def test_config_keeps_default_and_separates_conditional_cache_identities(self):
        baseline = InverseFoldingConfig()
        self.assertEqual(baseline.cache_id, f"v_48_020-backbone_only-{baseline.revision[:12]}")
        first = InverseFoldingConfig(conditioning_mode="sequence_conditional", seed=37)
        self.assertEqual(first.num_orders, 10)
        self.assertIn("sequence_conditional", first.cache_id)
        self.assertIn("orders10-seed37", first.cache_id)
        identities = {
            baseline.cache_id, first.cache_id,
            InverseFoldingConfig(conditioning_mode="sequence_conditional", seed=38).cache_id,
            InverseFoldingConfig(conditioning_mode="sequence_conditional", seed=37, num_orders=1).cache_id,
        }
        self.assertEqual(len(identities), 4)
        self.assertEqual(InverseFoldingConfig(**first.to_dict()), first)
        for kwargs in ({"num_orders": 0}, {"num_orders": True}, {"num_orders": 1.5},
                       {"model_family": "esm_if1", "conditioning_mode": "sequence_conditional"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                InverseFoldingConfig(**kwargs)

    def test_supplies_wild_type_and_ten_separate_batch_size_one_orders(self):
        value = adapter()
        result = value.predict(structure())
        self.assertEqual(len(value._model.calls), 10)
        generator = torch.Generator().manual_seed(37)
        for X, S, mask, chain_M, residue_idx, chains, randn, backbone_only in value._model.calls:
            torch.testing.assert_close(X, structure().coords[None])
            self.assertEqual(S.tolist(), [[0, 1, 2, 3]])
            self.assertEqual(S.dtype, torch.long)
            torch.testing.assert_close(mask, torch.ones(1, 4))
            torch.testing.assert_close(chain_M, mask)
            torch.testing.assert_close(residue_idx, torch.tensor([[0, 1, 2, 3]]))
            torch.testing.assert_close(chains, torch.ones(1, 4, dtype=torch.long))
            torch.testing.assert_close(randn, torch.randn((1, 4), generator=generator))
            self.assertFalse(backbone_only)
        self.assertEqual(result["alphabet"], "ACDEFGHIKLMNPQRSTVWY")
        for key in ("probabilities", "log_probs"):
            self.assertEqual(result[key].shape, (4, 20))
            self.assertEqual(result[key].dtype, torch.float32)
            self.assertEqual(result[key].device.type, "cpu")
            self.assertTrue(result[key].is_contiguous())
            self.assertFalse(result[key].requires_grad)

    def test_exact_released_numpy_geometric_mean_and_no_renormalization(self):
        value = adapter()
        result = value.predict(structure())
        native = np.concatenate([entry.numpy() for entry in value._model.outputs], axis=0)
        expected_logs = native.mean(axis=0)
        expected_probs = np.exp(expected_logs)[:, :20]
        np.testing.assert_array_equal(result["probabilities"].numpy(), expected_probs)
        np.testing.assert_array_equal(result["log_probs"].numpy(), expected_logs[:, :20])
        self.assertTrue(torch.all(result["probabilities"].sum(-1) < 0.9))
        arithmetic_mean = np.exp(native).mean(axis=0)[:, :20]
        self.assertGreater(np.abs(expected_probs - arithmetic_mean).max(), 1e-4)

    def test_reference_context_changes_distributions_and_target_token_is_withheld(self):
        value = adapter(num_orders=1)
        first = value.predict(structure("ACDE"))["probabilities"]
        second = value.predict(structure("VCDE"))["probabilities"]
        torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
        self.assertFalse(torch.equal(first[1:], second[1:]))

    def test_orders_reset_per_structure_and_global_rng_is_preserved(self):
        value = adapter()
        cpu_state = torch.get_rng_state().clone()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        first = value.predict(structure())["probabilities"]
        value.predict(structure("AC"))
        repeated = value.predict(structure())["probabilities"]
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
        self.assertTrue(torch.equal(cpu_state, torch.get_rng_state()))
        if cuda_states:
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(cuda_states, torch.cuda.get_rng_state_all())))
        changed_seed = adapter(seed=38).predict(structure())["probabilities"]
        self.assertFalse(torch.equal(first, changed_seed))

    def test_cpu_orders_continue_from_post_initialization_rng_state(self):
        value = adapter(num_orders=1)
        generator = torch.Generator().manual_seed(37)
        torch.randn(129, generator=generator)
        value._initialization_rng_state = generator.get_state()
        value.predict(structure())
        torch.testing.assert_close(value._model.calls[0][-2], torch.randn((1, 4), generator=generator))

    def test_rejects_invalid_native_outputs(self):
        for output in (torch.zeros(1, 4, 20), torch.full((1, 4, 21), float("nan")),
                       torch.full((1, 4, 21), float("inf")), torch.ones(1, 4, 21),
                       torch.zeros(1, 4, 21, dtype=torch.float64), torch.full((1, 4, 21), -10000.)):
            value = adapter(num_orders=1)
            value._model.conditional_probs = Mock(return_value=output)
            with self.subTest(shape=output.shape, dtype=output.dtype), self.assertRaises(ValueError):
                value.predict(structure())

    def test_metadata_describes_conditional_features_without_changing_score_recipe(self):
        value = adapter()
        base = {"conditioning_mode": "sequence_conditional", "seed": 37}
        with patch.object(value, "_backbone_metadata", return_value=base):
            metadata = value.metadata()
            scores = value.score_metadata(num_orders=2)
        self.assertEqual(metadata["inference_method"], "conditional_probs")
        self.assertEqual(metadata["normalization"], "native_21_amino_acid_tokens")
        self.assertEqual(metadata["noncanonical_tokens"], "excluded_without_renormalization")
        self.assertEqual(metadata["order_aggregation"], "geometric_mean_of_probabilities")
        self.assertEqual(metadata["num_orders"], 10)
        self.assertEqual(metadata["decoding_order_seed"], 37)
        self.assertEqual(metadata["extraction_batch_size"], 1)
        self.assertEqual(scores["conditioning_mode"], "sequence_autoregressive")
        self.assertEqual(scores["num_orders"], 2)
        self.assertEqual(scores["scoring_recipe_version"], 1)
        for key in ("conditional_recipe_version", "extraction_batch_size", "decoding_order_rng_initialization"):
            self.assertNotIn(key, scores)


class ProteinMPNNNativeConditionalParityTests(unittest.TestCase):
    """Use locally verified weights if present; never download test assets."""

    def _check_native_parity(self, device):
        config = InverseFoldingConfig(device=device, conditioning_mode="sequence_conditional", seed=37)
        path = checkpoint_path(config)
        if not (path / "v_48_020.pt").is_file() or not (path / "source/protein_mpnn_utils.py").is_file():
            self.skipTest("Pinned local ProteinMPNN checkpoint is unavailable")
        value = ProteinMPNNAdapter(config)
        inputs = structure()
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            actual = value.predict(inputs)
            upstream = sys.modules[f"_nkn_proteinmpnn_{config.revision}"]
            checkpoint = torch.load(path / "v_48_020.pt", map_location=device, weights_only=True)
            cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
                torch.manual_seed(37)
                # Recreate protein_mpnn_run.py initialization independently,
                # retaining its default dropout (eval disables it).
                model = upstream.ProteinMPNN(
                    ca_only=False, num_letters=21, node_features=128, edge_features=128,
                    hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                    augment_eps=0.0, k_neighbors=48,
                )
                model.to(device).load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                coords = inputs.coords.numpy()
                protein = {"name": "parity", "seq": inputs.reference_sequence,
                           "seq_chain_A": inputs.reference_sequence,
                           "coords_chain_A": {f"{atom}_chain_A": coords[:, i].tolist()
                                              for i, atom in enumerate(("N", "CA", "C", "O"))}}
                batch = upstream.tied_featurize([protein], torch.device(device), None)
                X, S, mask, _, chain_M, chain_encoding = batch[:6]
                chain_M_pos, residue_idx = batch[10], batch[12]
                native_runs = []
                for _ in range(10):
                    randn = torch.randn(chain_M.shape, device=X.device)
                    result = model.conditional_probs(X, S, mask, chain_M * chain_M_pos,
                                                     residue_idx, chain_encoding, randn, False)
                    native_runs.append(result.cpu().numpy())
                mean_log_p = np.concatenate(native_runs, axis=0).mean(axis=0)
                expected = np.exp(mean_log_p)[:, :20]
            np.testing.assert_array_equal(actual["probabilities"].numpy(), expected)
            np.testing.assert_array_equal(actual["log_probs"].numpy(), mean_log_p[:, :20])
        finally:
            torch.set_num_threads(old_threads)

    def test_cpu_matches_native_initialization_featurization_and_released_postprocessing(self):
        self._check_native_parity("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_matches_native_initialization_featurization_and_released_postprocessing(self):
        self._check_native_parity("cuda")


if __name__ == "__main__":
    unittest.main()
