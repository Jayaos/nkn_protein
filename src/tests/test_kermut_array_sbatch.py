"""Apply the shared array-dispatch contract to the Kermut launcher."""

from pathlib import Path
import unittest

from . import test_nkn_array_sbatch as array_tests


class KermutArraySbatchTests(array_tests.NknArraySbatchTests):
    assays = (
        "TPMT_HUMAN_Matreyek_2018",
        "RASK_HUMAN_Weng_2022_binding-DARPin_K55",
        "FKBP3_HUMAN_Tsuboyama_2023_2KFV",
        "DYR_ECOLI_Thompson_2019",
        "A0A1I9GEU1_NEIME_Kennouche_2019",
    )
    script_path = Path("sbatch_run/run_kermut/run_kermut_assays.sbatch")
    shared_runner = Path("sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn.sbatch")
    config_variable = "KERMUT_CONFIG"
    default_config = "configs/kermut_esm2_proteinmpnn.json"


if __name__ == "__main__":
    unittest.main()
