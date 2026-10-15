"""Branch selection preserves covariance behavior without unused representations."""

from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from config import (
    BRANCH_NAMES, DEFAULT_BRANCHES, IF_PRIMITIVES, STRUCTURE_DISTANCE_PRIMITIVES,
    ExperimentConfig, ModelConfig, validate_enabled_branches,
)
from kernels import ProteinKernel


DTYPE = torch.float64


def selected_batch(names, *, empty=False):
    owners = torch.tensor([] if empty else [0, 1, 1, 2], dtype=torch.long)
    vectors = torch.tensor([[1., 0.], [0., 1.], [0., 1.], [1., 1.]], dtype=DTYPE)
    probabilities = torch.softmax(torch.arange(80, dtype=DTYPE).reshape(4, 20) % 7, dim=1)
    local = {
        "inverse_folding": probabilities,
        "structure": vectors,
        "sequence_site": vectors + 0.25,
        "structure_distance": torch.tensor([[0., 0., 0.], [3., 4., 0.], [3., 4., 0.], [0., 0., 6.]], dtype=DTYPE),
        "inverse_folding_site": torch.tensor([[-1.], [-2.], [-3.], [-1.]], dtype=DTYPE),
    }
    return SimpleNamespace(
        num_variants=4,
        # Deliberately omit global vectors for local-only runs; no placeholder is needed.
        global_features=vectors if "sequence_global" in names else None,
        local_features={name: values[:0] if empty else values for name, values in local.items() if name in names},
        owners=owners,
        mutation_positions=torch.tensor([] if empty else [1, 2, 2, 3], dtype=torch.long),
    )


def small_config(names):
    return ModelConfig(
        enabled_branches=names,
        branches={name: {"widths": (4, 2, 1)} for name in names},
        final_widths=(4, 2, 1),
    )


class BranchConfigurationTests(unittest.TestCase):
    def test_invalid_selection(self):
        for values in ([], (), "sequence_global", None, {"structure"}, ["unknown"],
                       ["sequence_global", "sequence_global"], [1], ["structure", []]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ModelConfig(enabled_branches=values)

    def test_canonical_selection_and_serialization(self):
        config = ExperimentConfig(model={"enabled_branches": ["sequence_global", "inverse_folding"]})
        expected = ("inverse_folding", "sequence_global")
        self.assertEqual(config.model.enabled_branches, expected)
        self.assertEqual(validate_enabled_branches(list(reversed(BRANCH_NAMES))), BRANCH_NAMES)
        restored = ExperimentConfig.from_dict(json.loads(json.dumps(config.to_dict())))
        self.assertEqual(restored.model.enabled_branches, expected)
        self.assertEqual(restored.to_dict(), config.to_dict())

    def test_partial_hyperparameters_keep_branch_specific_defaults(self):
        config = ModelConfig(enabled_branches=["inverse_folding"], branches={
            "inverse_folding": {"widths": [4, 2, 1]},
        })
        self.assertEqual(config.branches["inverse_folding"].copies, dict.fromkeys(IF_PRIMITIVES, 1))
        self.assertEqual(config.branches["inverse_folding"].widths, (4, 2, 1))
        self.assertEqual(set(config.branches), set(BRANCH_NAMES))
        with self.assertRaisesRegex(ValueError, "Unknown branch configurations"):
            ModelConfig(branches={"typo": {}})
        with self.assertRaisesRegex(ValueError, "Unsupported primitives"):
            ModelConfig(enabled_branches=["inverse_folding"], branches={"inverse_folding": {"copies": {"cosine": 1}}})

    def test_structure_distance_defaults_and_allowed_primitives(self):
        config = ModelConfig(enabled_branches=["structure_distance"], branches={
            "structure_distance": {"widths": [4, 2, 1]},
        })
        self.assertEqual(config.branches["structure_distance"].copies, dict.fromkeys(STRUCTURE_DISTANCE_PRIMITIVES, 1))
        for branch, primitive in (("structure_distance", "cosine"), ("structure_distance", "hellinger_exp"),
                                  ("sequence_site", "exp")):
            with self.subTest(branch=branch, primitive=primitive), self.assertRaisesRegex(ValueError, "Unsupported primitives"):
                ModelConfig(branches={branch: {"copies": {primitive: 1}}})

    def test_manifest_path_serializes_and_legacy_defaults_stay_four_branches(self):
        self.assertEqual(ModelConfig().enabled_branches, DEFAULT_BRANCHES)
        config = ExperimentConfig(structure_manifest_dir=Path("custom/manifests"))
        restored = ExperimentConfig.from_dict(json.loads(json.dumps(config.to_dict())))
        self.assertEqual(restored.structure_manifest_dir, str(Path("custom/manifests")))
        for invalid in (None, "", "   ", 3, []):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "structure_manifest_dir"):
                ExperimentConfig(structure_manifest_dir=invalid)


class SelectedKernelTests(unittest.TestCase):
    def test_all_branch_combinations_covariance_gradients_and_parameters(self):
        cases = 0
        for count in range(1, len(BRANCH_NAMES) + 1):
            for names in combinations(BRANCH_NAMES, count):
                with self.subTest(names=names):
                    cases += 1
                    torch.manual_seed(31)
                    inputs = selected_batch(names)
                    kernel = ProteinKernel(small_config(names), inputs)
                    self.assertEqual(tuple(kernel.branches), names)
                    self.assertEqual(kernel.final_nkn.input_width, count)
                    self.assertEqual(kernel.final_nkn.layers[0].raw_weight.shape[1], count)
                    self.assertEqual(set(kernel.feature_dimensions), set(names))
                    for disabled in set(BRANCH_NAMES) - set(names):
                        self.assertFalse(any(key.startswith(f"branches.{disabled}.") for key in kernel.state_dict()))
                    covariance = kernel(inputs)
                    self.assertEqual(covariance.shape, (4, 4))
                    self.assertTrue(torch.isfinite(covariance).all())
                    torch.testing.assert_close(covariance, covariance.T)
                    self.assertGreaterEqual(torch.linalg.eigvalsh(covariance).min().item(), -1e-10)
                    covariance.square().sum().backward()
                    self.assertTrue(all(
                        parameter.grad is not None and torch.isfinite(parameter.grad).all()
                        for parameter in kernel.parameters()
                    ))
                    empty_mutations = selected_batch(names, empty=True)
                    cross = kernel(inputs, empty_mutations)
                    self.assertTrue(torch.isfinite(cross).all())
                    torch.testing.assert_close(cross, kernel(empty_mutations, inputs).T)
        self.assertEqual(cases, 2 ** len(BRANCH_NAMES) - 1)

    def test_missing_enabled_features_fail_at_initialization_and_prediction(self):
        for name in BRANCH_NAMES:
            with self.subTest(name=name):
                config = small_config([name])
                valid = selected_batch([name])
                kernel = ProteinKernel(config, valid)
                missing = selected_batch([])
                with self.assertRaisesRegex(ValueError, f"Missing features for enabled branch: {name}"):
                    ProteinKernel(config, missing)
                with self.assertRaisesRegex(ValueError, f"Missing features for enabled branch: {name}"):
                    kernel(valid, missing)

    def test_ignores_disabled_arrays_but_checks_active_dimensions(self):
        inputs = selected_batch(["sequence_global"])
        inputs.local_features = {"inverse_folding": "unused", "structure": torch.tensor(float("nan"))}
        kernel = ProteinKernel(small_config(["sequence_global"]), inputs)
        self.assertTrue(torch.isfinite(kernel(inputs)).all())
        wrong = selected_batch(["sequence_global"])
        wrong.global_features = torch.ones(4, 3, dtype=DTYPE)
        with self.assertRaisesRegex(ValueError, "differs from its training dimension"):
            kernel(inputs, wrong)

    def test_default_and_legacy_config_keep_state_layout_and_covariance(self):
        inputs = selected_batch(DEFAULT_BRANCHES)
        default = ModelConfig()
        legacy_values = asdict(default)
        legacy_values.pop("enabled_branches")
        legacy_values["branches"].pop("structure_distance")
        legacy_dataclass = ModelConfig(**legacy_values)
        # Existing consumers can still provide the pre-selection config interface.
        legacy_object = SimpleNamespace(branches={name: default.branches[name] for name in DEFAULT_BRANCHES},
                                        final_widths=default.final_widths)
        configs = (default, legacy_dataclass, legacy_object,
                   ModelConfig(enabled_branches=list(reversed(DEFAULT_BRANCHES))))
        models = []
        for config in configs:
            torch.manual_seed(44)
            models.append(ProteinKernel(config, inputs))
        for model in models[1:]:
            self.assertEqual(tuple(model.state_dict()), tuple(models[0].state_dict()))
            for key, value in models[0].state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
            torch.testing.assert_close(model(inputs), models[0](inputs), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
