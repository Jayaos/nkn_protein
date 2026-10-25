"""Collection isolation without changing structure or model calculations."""

from pathlib import Path
import tempfile
import unittest

from feature_generation.config import PROJECT_ROOT
from feature_generation.inverse_config import InverseFoldingConfig
from feature_generation.prosst_config import ProSSTConfig
from feature_generation.score_runner import default_cache_id


class StructureSourceTests(unittest.TestCase):
    def test_collections_have_independent_paths_and_automatic_cache_names(self):
        for cls in (InverseFoldingConfig, ProSSTConfig):
            with self.subTest(config=cls.__name__):
                official = cls(structure_source="proteingym")
                alternative = cls(structure_source="protssn_proteingym")
                for config in (official, alternative):
                    root = PROJECT_ROOT / "data/structures" / config.structure_source
                    self.assertEqual(config.structure_manifest_dir, root / "manifests")
                    self.assertTrue(config.cache_id.startswith(config.structure_source + "--"))
                    self.assertTrue(default_cache_id(config).startswith(config.structure_source + "--"))
                    if isinstance(config, ProSSTConfig):
                        self.assertEqual(config.structure_token_dir, root / "tokens/prosst")
                self.assertNotEqual(official.cache_id, alternative.cache_id)
                self.assertNotEqual(default_cache_id(official), default_cache_id(alternative))

    def test_crossed_collections_and_unknown_source_are_rejected(self):
        for cls in (InverseFoldingConfig, ProSSTConfig):
            with self.subTest(config=cls.__name__):
                with self.assertRaisesRegex(ValueError, "structure_source"):
                    cls(structure_source="../outside")
                with self.assertRaisesRegex(ValueError, "belongs to"):
                    cls(structure_source="protssn_proteingym",
                        structure_manifest_dir="data/structures/proteingym/manifests")
                with self.assertRaisesRegex(ValueError, "belongs to"):
                    cls(structure_source="protssn_proteingym", cache_id="proteingym--copied_recipe")
        with self.assertRaisesRegex(ValueError, "belongs to"):
            ProSSTConfig(structure_source="protssn_proteingym",
                         structure_token_dir="data/structures/proteingym/tokens/prosst")

    def test_custom_directories_and_explicit_legacy_cache_ids_remain_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            for cls in (InverseFoldingConfig, ProSSTConfig):
                config = cls(structure_manifest_dir=directory, cache_id="custom_recipe")
                self.assertEqual(config.structure_manifest_dir, Path(directory).resolve())
                self.assertEqual(config.cache_id, "custom_recipe")


if __name__ == "__main__":
    unittest.main()
