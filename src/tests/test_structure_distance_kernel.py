"""Physical distance kernels preserve geometry and local-before-global composition."""

import copy
from types import SimpleNamespace
import unittest

import torch

from config import ModelConfig
from kernels import PrimitiveKernel, ProteinKernel, aggregate_mutations


DTYPE = torch.float64


def distance_batch():
    # Three reference sites, shared by single and double mutants; variant 3 is WT.
    sites = torch.tensor([[0., 0., 0.], [3., 4., 0.], [0., 0., 6.]], dtype=DTYPE)
    positions = torch.tensor([1, 1, 2, 2, 3], dtype=torch.long)
    return SimpleNamespace(
        num_variants=4,
        global_features=None,
        local_features={"structure_distance": sites[positions - 1]},
        owners=torch.tensor([0, 1, 1, 2, 2], dtype=torch.long),
        mutation_positions=positions,
    )


def distance_config(copies=None):
    branch = {"widths": (4, 2, 1)}
    if copies is not None:
        branch["copies"] = copies
    return ModelConfig(enabled_branches=["structure_distance"],
                       branches={"structure_distance": branch}, final_widths=(4, 2, 1))


class StructureDistanceKernelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.batch = distance_batch()
        self.kernel = ProteinKernel(distance_config(), self.batch)

    def test_rigid_rotation_translation_invariance_and_psd(self):
        rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=DTYPE))
        moved = copy.deepcopy(self.batch)
        moved.local_features["structure_distance"] = (
            moved.local_features["structure_distance"] @ rotation
            + torch.tensor([12., -7., 21.], dtype=DTYPE)
        )
        covariance = self.kernel(self.batch)
        torch.testing.assert_close(self.kernel(moved), covariance, rtol=1e-12, atol=1e-12)
        self.assertGreaterEqual(torch.linalg.eigvalsh(covariance).min().item(), -1e-10)
        # Initializing from transformed coordinates also preserves median-distance scales.
        transformed = ProteinKernel(distance_config(), moved)
        for first, second in zip(self.kernel.branches["structure_distance"].primitives,
                                 transformed.branches["structure_distance"].primitives):
            torch.testing.assert_close(first.lengthscale, second.lengthscale, rtol=1e-12, atol=1e-12)

    def test_local_nkn_then_mutation_sums_then_final_nkn(self):
        branch = self.kernel.branches["structure_distance"]
        coords = self.batch.local_features["structure_distance"]
        primitives = torch.stack([primitive(coords) for primitive in branch.primitives], dim=-1)
        local = branch.nkn(primitives)
        expected_branch = torch.zeros(4, 4, dtype=DTYPE)
        for i, owner_i in enumerate(self.batch.owners):
            for j, owner_j in enumerate(self.batch.owners):
                expected_branch[owner_i, owner_j] += local[i, j]
        expected = self.kernel.final_nkn(expected_branch[..., None])
        torch.testing.assert_close(self.kernel(self.batch), expected)
        torch.testing.assert_close(expected_branch[3], torch.zeros(4, dtype=DTYPE))
        # The nonlinear branch cannot be moved after summing primitive values.
        wrong_branch = branch.nkn(torch.stack([
            aggregate_mutations(primitives[..., i], self.batch.owners, self.batch.owners, 4, 4)
            for i in range(primitives.shape[-1])
        ], dim=-1))
        self.assertFalse(torch.allclose(expected_branch, wrong_branch))
        # Nor can the final NKN be applied to occurrence pairs before summation.
        wrong_final = aggregate_mutations(self.kernel.final_nkn(local[..., None]),
                                          self.batch.owners, self.batch.owners, 4, 4)
        self.assertFalse(torch.allclose(expected, wrong_final))

    def test_uncached_dense_and_triangle_outputs_and_parameter_gradients_match(self):
        expected = self.kernel(self.batch)
        expected.square().sum().backward()
        gradients = {name: parameter.grad.clone() for name, parameter in self.kernel.named_parameters()}
        for triangle in (False, True):
            with self.subTest(triangle=triangle):
                cached = copy.deepcopy(self.kernel)
                cached.zero_grad(set_to_none=True)
                cached.prepare_training_geometry(self.batch, use_triangle=triangle)
                self.assertEqual(cached._training_triangle is not None, triangle)
                actual = cached(self.batch)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                actual.square().sum().backward()
                for name, parameter in cached.named_parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    torch.testing.assert_close(parameter.grad, gradients[name], rtol=1e-10, atol=1e-10)
                self.assertEqual(tuple(cached.state_dict()), tuple(self.kernel.state_dict()))

    def test_exp_only_branch_cached_geometry_and_coordinate_gradients(self):
        # An exp-only branch must prepare Euclidean geometry even without RBF/Matérn.
        kernel = ProteinKernel(distance_config({"exp": 1}), self.batch)
        expected = kernel(self.batch)
        kernel.prepare_training_geometry(self.batch)
        torch.testing.assert_close(kernel(self.batch), expected)
        self.assertIn("euclidean_distance", kernel.branches["structure_distance"]._geometry_cache.values)
        for name in ("exp", "rbf", "matern32"):
            with self.subTest(primitive=name):
                coords = self.batch.local_features["structure_distance"].clone().requires_grad_()
                primitive = PrimitiveKernel(name, 2.)
                primitive(coords).square().sum().backward()
                self.assertTrue(torch.isfinite(coords.grad).all())
                self.assertTrue(torch.isfinite(primitive.raw_lengthscale.grad))

    def test_rejects_non_coordinates_at_initialization_and_prediction(self):
        invalid = [torch.zeros(5, 2, dtype=DTYPE), torch.zeros(5, 4, dtype=DTYPE)]
        for value in (float("nan"), float("inf")):
            coords = self.batch.local_features["structure_distance"].clone()
            coords[0, 0] = value
            invalid.append(coords)
        for coords in invalid:
            with self.subTest(coords=coords):
                changed = copy.deepcopy(self.batch)
                changed.local_features["structure_distance"] = coords
                with self.assertRaisesRegex(ValueError, "finite C-alpha coordinates"):
                    ProteinKernel(distance_config(), changed)
                with self.assertRaisesRegex(ValueError, "finite C-alpha coordinates"):
                    self.kernel(self.batch, changed)


if __name__ == "__main__":
    unittest.main()
