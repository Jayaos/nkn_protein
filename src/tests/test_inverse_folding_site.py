"""Mutation-specific ProteinMPNN features compose, aggregate, and reload correctly."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from baselines.kermut.kernel import _mutant_log_probabilities
from config import ModelConfig, TrainingConfig
from data_loading import load_feature_bundle
from data_loading.features import ALPHABET
from gp.training import FittedGP, fit_gp
from kernels import ProteinKernel
from kernels.nkn import AdjacentProduct
from tests.test_data import make_bundle, write_csv
from tests.test_nkn_conditional import RECIPE


SITE = "inverse_folding_site"
MODES = ((1,), (4, 2, 1))


def site_config(widths, branches=(SITE,)):
    return ModelConfig(
        enabled_branches=branches,
        branches={name: {"widths": widths} for name in branches},
        final_widths=(4, 2, 1),
    )


class InverseFoldingSiteTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        self.ids, self.sequences, _ = make_bundle(self.root)
        self.ids.append("A1G")
        self.sequences.append("GCD")
        write_csv(
            self.root / "variants.csv",
            ["variant_id", "mutant", "mutated_sequence", "DMS_score"],
            [[variant, variant, sequence, index / 3]
             for index, (variant, sequence) in enumerate(zip(self.ids, self.sequences))],
        )
        self.leaf = self.root / "inverse_folding/synthetic/v1"
        write_csv(self.leaf / "zero_shot.csv", ["variant_id", "score"],
                  [[variant, index / 5] for index, variant in enumerate(self.ids)])
        # Site-only and distribution+site loading must not read these unused caches.
        for name in ("sequence", "structure"):
            (self.root / name / "synthetic/v1/features.pt").write_bytes(b"unused cache")

    def load(self, branches=(SITE,), **kwargs):
        return load_feature_bundle(self.root, enabled_branches=branches, **kwargs)

    def test_site_only_reuses_cache_and_matches_kermut_for_each_mutation(self):
        site_only = self.load()
        both = self.load(("inverse_folding", SITE))
        self.assertIsNone(site_only.global_features)
        self.assertEqual(tuple(site_only.local_features), (SITE,))
        self.assertEqual(site_only.score_names, ("inverse_folding/synthetic/v1",))
        self.assertEqual(both.score_names, site_only.score_names)
        self.assertEqual(site_only.owners.tolist(), [1, 2, 3, 3, 4])
        self.assertEqual(site_only.mutation_positions.tolist(), [1, 2, 1, 2, 1])
        actual = site_only.local_features[SITE]
        self.assertEqual(actual.shape, (5, 1))
        torch.testing.assert_close(actual[:, 0], _mutant_log_probabilities(both), rtol=0, atol=0)
        torch.testing.assert_close(actual, both.local_features[SITE], rtol=0, atol=0)
        # A1V and A1G share a distribution but have different mutation likelihoods.
        torch.testing.assert_close(both.local_features["inverse_folding"][0],
                                   both.local_features["inverse_folding"][-1])
        self.assertNotEqual(actual[0].item(), actual[-1].item())
        chosen = site_only.select([4, 0, 3]).to("cpu")
        torch.testing.assert_close(chosen.local_features[SITE], actual[[4, 2, 3]])
        self.assertEqual(chosen.owners.tolist(), [0, 2, 2])
        self.assertEqual(chosen.select([1]).local_features[SITE].shape, (0, 1))

    def test_conditional_site_logs_preserve_probability_mass_and_cache_provenance(self):
        conditional = self.root / "inverse_folding/proteinmpnn/conditional"
        conditional.mkdir(parents=True)
        payload = torch.load(self.leaf / "features.pt", weights_only=True)
        payload["probabilities"] *= torch.tensor([0.8, 0.9, 0.7])[:, None]
        payload["log_probs"] = payload["probabilities"].log()
        torch.save(payload, conditional / "features.pt")
        config = json.loads((self.leaf / "config.json").read_text())
        config.update(RECIPE, checkpoint="conditional")
        (conditional / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (conditional / "zero_shot.csv").write_bytes((self.leaf / "zero_shot.csv").read_bytes())
        kwargs = {"feature_selection": {"inverse_folding": "proteinmpnn/conditional"}}
        with self.assertRaisesRegex(ValueError, "explicit sequence_conditional"):
            self.load(**kwargs)
        batch = self.load(inverse_folding_policy="sequence_conditional", **kwargs)
        columns = [ALPHABET.index(aa) for aa in ("V", "D", "V", "D", "G")]
        expected = payload["probabilities"].double()[[0, 1, 0, 1, 0], columns].log()
        torch.testing.assert_close(batch.local_features[SITE][:, 0], expected, rtol=0, atol=0)
        self.assertEqual(json.loads(batch.feature_signature["inverse_folding.probability_recipe"]), RECIPE)
        self.assertEqual(batch.feature_signature["inverse_folding"], "inverse_folding/proteinmpnn/conditional")
        self.assertEqual(batch.num_scores, 1)
        # This provenance check is required even without the distribution branch.
        kernel = ProteinKernel(site_config((1,)), batch)
        changed_signature = dict(batch.feature_signature)
        changed_signature.pop("inverse_folding.probability_recipe")
        with self.assertRaisesRegex(ValueError, "recipe differs"):
            kernel(batch, replace(batch, feature_signature=changed_signature))

    def test_selected_zero_probability_is_rejected_without_clipping(self):
        path = self.leaf / "features.pt"
        payload = torch.load(path, weights_only=True)
        payload["probabilities"][0, ALPHABET.index("V")] = 0
        payload["probabilities"] /= payload["probabilities"].sum(-1, keepdim=True)
        payload["log_probs"] = payload["probabilities"].log()
        torch.save(payload, path)
        # A normalized distribution can contain zeros, but log(0) is not a site feature.
        distribution = self.load(("inverse_folding",))
        self.assertEqual(distribution.local_features["inverse_folding"][0, ALPHABET.index("V")], 0)
        with self.assertRaisesRegex(ValueError, "positive"):
            self.load()

    def test_nonfinite_probabilities_fail_before_constructing_site_features(self):
        path = self.leaf / "features.pt"
        original = torch.load(path, weights_only=True)
        for invalid in (float("nan"), float("inf"), -0.1):
            with self.subTest(invalid=invalid):
                payload = copy.deepcopy(original)
                payload.pop("log_probs")
                payload["probabilities"][0, ALPHABET.index("V")] = invalid
                torch.save(payload, path)
                with self.assertRaises(ValueError):
                    self.load()

    def test_both_composition_modes_feed_final_kernel_and_keep_finite_gradients(self):
        batch = self.load(("inverse_folding", SITE))
        for widths in MODES:
            with self.subTest(widths=widths):
                torch.manual_seed(31)
                kernel = ProteinKernel(site_config(widths, ("inverse_folding", SITE)), batch)
                branch = kernel.branches[SITE]
                self.assertEqual({primitive.name for primitive in branch.primitives}, {"exp", "rbf", "matern32"})
                self.assertEqual(sum(isinstance(layer, AdjacentProduct) for layer in branch.nkn.layers),
                                 0 if widths == (1,) else 1)
                self.assertEqual(kernel.final_nkn.input_width, 2)
                covariance = kernel(batch)
                torch.testing.assert_close(covariance, covariance.T, rtol=1e-12, atol=1e-12)
                self.assertGreaterEqual(torch.linalg.eigvalsh(covariance).min().item(), -1e-10)
                self.assertGreater(covariance[1, 1].item(), covariance[1, 4].item())
                covariance.square().sum().backward()
                for name, parameter in kernel.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                torch.testing.assert_close(kernel(batch, batch.select([4, 0])),
                                           kernel(batch.select([4, 0]), batch).T)

    def test_site_composition_precedes_multimutant_aggregation_in_both_modes(self):
        batch = self.load()
        values = batch.local_features[SITE]
        for widths in MODES:
            with self.subTest(widths=widths):
                kernel = ProteinKernel(site_config(widths), batch)
                local = kernel.branches[SITE](values)
                aggregated = torch.zeros(batch.num_variants, batch.num_variants, dtype=torch.float64)
                for i, owner_i in enumerate(batch.owners):
                    for j, owner_j in enumerate(batch.owners):
                        aggregated[owner_i, owner_j] += local[i, j]
                torch.testing.assert_close(kernel(batch), kernel.final_nkn(aggregated[..., None]))
                # The double mutant is exactly the sum of its component mutations.
                torch.testing.assert_close(aggregated[3], aggregated[1] + aggregated[2])
                torch.testing.assert_close(aggregated[0], torch.zeros(5, dtype=torch.float64))

    def test_cached_dense_and_triangle_covariances_and_gradients_match(self):
        batch = self.load(("inverse_folding", SITE))
        for widths in MODES:
            reference = ProteinKernel(site_config(widths, ("inverse_folding", SITE)), batch)
            expected = reference(batch)
            expected.square().sum().backward()
            gradients = {name: parameter.grad.clone() for name, parameter in reference.named_parameters()}
            for triangle in (False, True):
                with self.subTest(widths=widths, triangle=triangle):
                    cached = copy.deepcopy(reference)
                    cached.zero_grad(set_to_none=True)
                    cached.prepare_training_geometry(batch, use_triangle=triangle)
                    self.assertEqual(cached._training_triangle is not None, triangle)
                    actual = cached(batch)
                    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                    actual.square().sum().backward()
                    for name, parameter in cached.named_parameters():
                        torch.testing.assert_close(parameter.grad, gradients[name], rtol=1e-10, atol=1e-10)

    def test_fitted_site_only_models_reload_with_identical_predictions_in_both_modes(self):
        batch = self.load()
        train, test = batch.select([1, 2, 4]), replace(batch.select([0, 3]), targets=None)
        training = TrainingConfig(max_steps=3, min_steps=1, patience=1, tolerance=0,
                                  seeds=(7,), lengthscale_multipliers=(1.0,))
        for widths in MODES:
            with self.subTest(widths=widths):
                fitted = fit_gp(train, site_config(widths), training)
                expected = fitted.predict(test)
                path = self.root / f"site_{len(widths)}.pt"
                fitted.save(path)
                restored = FittedGP.load(path)
                self.assertEqual(restored.model_config.enabled_branches, (SITE,))
                self.assertEqual(restored.model_config.branches[SITE].widths, widths)
                torch.testing.assert_close(restored.train.local_features[SITE], train.local_features[SITE])
                self.assertEqual(restored.train.feature_signature, train.feature_signature)
                actual = restored.predict(test)
                self.assertTrue(torch.isfinite(actual.mean).all())
                torch.testing.assert_close(actual.mean, expected.mean, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(actual.observation_covariance, expected.observation_covariance,
                                           rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
