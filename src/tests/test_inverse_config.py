"""Inverse-folding length policy without model dependencies."""

from pathlib import Path
import unittest

from feature_generation.inverse_config import InverseFoldingConfig


class InverseConfigLengthTests(unittest.TestCase):
    def test_defaults_and_shipped_presets_are_uncapped(self):
        for family in ("proteinmpnn", "esm_if1"):
            with self.subTest(family=family):
                self.assertIsNone(InverseFoldingConfig(model_family=family).max_length)
        configs = Path(__file__).resolve().parents[2] / "configs/extraction"
        for name in ("esm_if1_142m.json", "proteinmpnn_v48_020.json", "proteinmpnn_v48_020_kermut.json"):
            with self.subTest(preset=name):
                self.assertIsNone(InverseFoldingConfig.load(configs / name).max_length)

    def test_optional_limits_accept_null_and_any_positive_integer(self):
        for family in ("proteinmpnn", "esm_if1"):
            for cap in (None, 1, 1022, 1023, 1863, 4096):
                with self.subTest(family=family, cap=cap):
                    config = InverseFoldingConfig(model_family=family, max_length=cap)
                    self.assertEqual(config.to_dict()["max_length"], cap)

    def test_invalid_limits_are_rejected(self):
        for family in ("proteinmpnn", "esm_if1"):
            for cap in (0, -1, True, False, 1022.0, "1022"):
                with self.subTest(family=family, cap=cap), self.assertRaisesRegex(ValueError, "null or a positive integer"):
                    InverseFoldingConfig(model_family=family, max_length=cap)


if __name__ == "__main__":
    unittest.main()
