"""Catalog layouts shared by extraction and Slurm assay selection."""

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from data_loading import AssayCatalog
from feature_generation.cli import _parser
from feature_generation.config import ExtractionConfig, PROJECT_ROOT
from feature_generation.inverse_config import InverseFoldingConfig
from feature_generation.msa_config import MSAConfig
from feature_generation.prosst_config import ProSSTConfig

spec = importlib.util.spec_from_file_location(
    "preparation_assays", Path(__file__).resolve().parents[2] / "sbatch_run/run_prepare_rep/list_assays.py",
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class CatalogPathTests(unittest.TestCase):
    def test_preparation_defaults_use_top_level_catalog(self):
        expected = PROJECT_ROOT / "data/DMS_ProteinGym_substitutions/DMS_substitutions.csv"
        msa = MSAConfig("features", "scores")
        for config in (ExtractionConfig(), InverseFoldingConfig(),
                       InverseFoldingConfig(model_family="esm_if1"), ProSSTConfig(), msa):
            with self.subTest(config=type(config).__name__):
                self.assertEqual(config.assay_index, expected.resolve())
        self.assertEqual(msa.msa_reference_csv, expected.resolve())
        for arguments in (["download-structure", "--assay", "tiny"],
                          ["prepare-structure", "--assay", "tiny", "--structure", "tiny.pdb"]):
            self.assertEqual(_parser().parse_args(arguments).assay_index, expected)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = self.root / "DMS_substitutions.csv"
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({"assay_index": str(self.index)}))
        self.write_index("tiny.csv")

    def write_index(self, filename):
        with self.index.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["DMS_id", "DMS_filename", "target_seq"])
            writer.writerow(["tiny", filename, "ACD"])

    def check_layout(self, directory):
        directory.mkdir(exist_ok=True)
        path = directory / "tiny.csv"
        path.write_text("mutant,mutated_sequence,DMS_score\nA1C,CCD,1\n")
        catalog = AssayCatalog(self.index)
        self.assertEqual(catalog.get("tiny").path, path.resolve())
        self.assertEqual(len(catalog.load_variants("tiny")), 1)
        self.assertEqual(helper.list_assays(self.config), ["tiny"])

    def test_csv_beside_catalog(self):
        self.check_layout(self.root)

    def test_csv_in_raw_substitutions(self):
        self.check_layout(self.root / "raw_substitutions")

    def test_explicit_relative_subdirectory(self):
        self.write_index("raw_substitutions/tiny.csv")
        self.check_layout(self.root / "raw_substitutions")

    def test_missing_csv_and_directory_escape_rejected(self):
        with self.assertRaises(FileNotFoundError):
            AssayCatalog(self.index)
        with self.assertRaises(ValueError):
            helper.list_assays(self.config)
        self.write_index("../tiny.csv")
        with self.assertRaises(ValueError):
            AssayCatalog(self.index)
        with self.assertRaises(ValueError):
            helper.list_assays(self.config)
