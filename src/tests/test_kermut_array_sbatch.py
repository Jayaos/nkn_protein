"""Apply the shared array-dispatch contract to the Kermut launcher."""

from pathlib import Path
import unittest

from . import test_nkn_array_sbatch as array_tests


class KermutArraySbatchTests(array_tests.NknArraySbatchTests):
    script_path = Path("sbatch_run/run_kermut/run_kermut_assays.sbatch")
    shared_runner = Path("sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn.sbatch")
    config_variable = "KERMUT_CONFIG"
    default_config = "configs/kermut_esm2_proteinmpnn.json"


if __name__ == "__main__":
    unittest.main()
