"""Run Kermut's shell orchestration with the same scheduling checks as NKN."""

from src.tests import test_nkn_sbatch as sbatch_tests


class KermutSbatchTests(sbatch_tests.NknSbatchTests):
    script = sbatch_tests.REPOSITORY / 'sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn.sbatch'
    config_file = 'configs/kermut_esm2_proteinmpnn.json'
    config_variable = 'KERMUT_CONFIG'
    cli_module = 'baselines.kermut.cli'
    cli_file = 'baselines/kermut/cli.py'
    config_sources = ('src/config.py', 'baselines/kermut/config.py')
    config_type = ('baselines.kermut.config', 'KermutConfig')

    def setUp(self):
        super().setUp()
        # Load the real configuration without importing the scientific runtime
        # from package __init__, since shell orchestration uses cheap stubs.
        for package in ('baselines', 'baselines/kermut'):
            (self.root / package / '__init__.py').touch()
