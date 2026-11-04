"""Verify the full-catalog hardware launchers select ProSST/ESM-IF1 runs."""

from collections import Counter
import csv
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from . import test_gpu_array_sbatch as gpu_tests


ROOT = gpu_tests.ROOT
PRESET = Path("configs/prosst_esm_if1_nkn_four_branches_noise_initial_0p5.json")
GPU_SCRIPTS = {
    "run_nkn_gpu_v100_16G.sbatch": 61,
    "run_nkn_gpu_v100_32G.sbatch": 14,
    "run_nkn_gpu_a100_40G.sbatch": 7,
    "run_nkn_gpu_a100_80G.sbatch": 4,
}
CATALOG = next((ROOT / path for path in (
    "data/DMS_substitutions.csv",
    "data/DMS_ProteinGym_substitutions/DMS_substitutions.csv",
    "data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv",
) if (ROOT / path).is_file()), None)


def launcher(name):
    return ROOT / "sbatch_run/run_nkn" / name


def assays(path):
    return re.findall(r'^  "([^"]+)"$', path.read_text(encoding="utf-8"), re.MULTILINE)


class HardwareCoverageTests(unittest.TestCase):
    @unittest.skipUnless(CATALOG, "Local ProteinGym catalog is required")
    def test_five_launchers_cover_the_catalog_once(self):
        with CATALOG.open(encoding="utf-8-sig", newline="") as handle:
            expected = {row["DMS_id"] for row in csv.DictReader(handle)}
        counts = Counter()
        for name, size in {"run_nkn_cpu.sbatch": 131, **GPU_SCRIPTS}.items():
            path = launcher(name)
            selected = assays(path)
            self.assertEqual(len(selected), size)
            self.assertIn(f"#SBATCH --array=0-{size - 1}\n", path.read_text())
            counts.update(selected)
        self.assertEqual(set(counts), expected)
        self.assertTrue(all(count == 1 for count in counts.values()))


@unittest.skipUnless(gpu_tests.BASH and Path(gpu_tests.BASH).is_file(), "Bash required")
class HardwareGpuDispatchTests(unittest.TestCase):
    def setUp(self):
        # Reuse the cheap interpreter fixture; exercise each real hardware script.
        self.fixture = gpu_tests.HumanAssayArrayTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.fixture.env.pop("NKN_RUN_PREFIX", None)

    def invoke(self, name, index, *arguments):
        self.fixture.record.unlink(missing_ok=True)
        return self.fixture.invoke(index, *arguments, script=launcher(name))

    def parsed_record(self):
        record = json.loads(self.fixture.record.read_text())
        self.assertEqual(record["args"][:3], ["-u", "-m", "experiments.large_assay_gpu"])
        return gpu_tests._parser().parse_args(record["args"][3:]), record

    def test_defaults_and_result_names_at_array_boundaries(self):
        for name, size in GPU_SCRIPTS.items():
            selected = assays(launcher(name))
            for index in (0, size - 1):
                with self.subTest(script=name, index=index):
                    result = self.invoke(name, index, "singles_random")
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    parsed, record = self.parsed_record()
                    self.assertEqual(parsed.config, PRESET)
                    self.assertEqual(parsed.assay, selected[index])
                    self.assertEqual(parsed.selection, "singles_random")
                    self.assertEqual(parsed.output, self.fixture.root / "results" / selected[index]
                                     / f"nkn_prosst_esmif_singles_random_gpu_4242_{index}")
                    self.assertEqual(record["cuda"], "3")

    def test_config_overrides_prefix_and_results_group_are_preserved(self):
        for name in GPU_SCRIPTS:
            for options, expected in (((), "configs/environment.json"),
                                      (("--config", "configs/custom config.json"), "configs/custom config.json")):
                with self.subTest(script=name, options=options), patch.dict(self.fixture.env, {
                    "NKN_CONFIG": "configs/environment.json", "NKN_RUN_PREFIX": "nkn_custom",
                }):
                    result = self.invoke(name, 0, "singles_modulo", "--experiment-suffix", "comparison", *options)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    parsed, _ = self.parsed_record()
                    self.assertEqual(parsed.config, Path(expected))
                    self.assertEqual(parsed.output, self.fixture.root / "results/comparison"
                                     / assays(launcher(name))[0] / "nkn_custom_singles_modulo_gpu_4242_0")

    def test_invalid_prefix_fails_before_dispatch(self):
        for name in GPU_SCRIPTS:
            with self.subTest(script=name), patch.dict(self.fixture.env, {"NKN_RUN_PREFIX": "../outside"}):
                result = self.invoke(name, 0)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("NKN_RUN_PREFIX", result.stderr)
                self.assertFalse(self.fixture.record.exists())

    def test_help_describes_prosst_esmif_without_dispatch(self):
        for name in GPU_SCRIPTS:
            with self.subTest(script=name):
                result = self.invoke(name, 0, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(PRESET.as_posix(), result.stdout)
                self.assertIn("Result prefix: nkn_prosst_esmif", result.stdout)
                self.assertIn("ProSST-2048 and ESM-IF1", result.stdout)
                self.assertFalse(self.fixture.record.exists())


if __name__ == "__main__":
    unittest.main()
