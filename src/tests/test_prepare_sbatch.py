"""Exercise preparation target selection and batch dispatch without GPU work."""

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


REPOSITORY = Path(__file__).resolve().parents[2]
PREPARATION = Path("sbatch_run/run_prepare_rep")
CONFIGS = ("esm2_650m.json", "esm2_650m_residue.json", "esm2_650m_centered.json",
           "proteinmpnn_v48_020_kermut.json", "proteinmpnn_v48_020.json")
SCRIPTS = ("prepare_esm2.sbatch", "prepare_proteinmpnn.sbatch")
SELECTED_ASSAYS = ["DYR_ECOLI_Thompson_2019", "TPMT_HUMAN_Matreyek_2018",
                   "RASK_HUMAN_Weng_2022_binding-DARPin_K55"]
ASSAYS = [SELECTED_ASSAYS[1], "protein_Z_assay_1", SELECTED_ASSAYS[2],
          SELECTED_ASSAYS[0], "protein_Z_assay_2"]


def available_bash():
    if os.name == "nt":
        for candidate in (Path("C:/Program Files/Git/usr/bin/bash.exe"),
                          Path("C:/Program Files/Git/bin/bash.exe")):
            if candidate.is_file():
                return str(candidate)
        return None
    return shutil.which("bash")


BASH = available_bash()


class PreparationFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.root = self.workspace / "repository with spaces"
        self.raw = self.root / "data/DMS_ProteinGym_substitutions/raw_substitutions"
        for directory in (PREPARATION, Path("configs/extraction"), Path("src/feature_generation"),
                          Path("etc/profile.d"), self.raw.relative_to(self.root)):
            (self.root / directory).mkdir(parents=True)
        (self.root / "src/feature_generation/cli.py").touch()
        self.helper = self.root / PREPARATION / "list_assays.py"
        shutil.copyfile(REPOSITORY / PREPARATION / "list_assays.py", self.helper)
        for filename in CONFIGS:
            shutil.copyfile(REPOSITORY / "configs/extraction" / filename,
                            self.root / "configs/extraction" / filename)
        for filename in SCRIPTS:
            shutil.copyfile(REPOSITORY / PREPARATION / filename, self.root / PREPARATION / filename)
        self.rows = [
            {"DMS_id": assay, "DMS_filename": f"{assay}.csv", "target_seq": "ACD",
             "UniProt_ID": assay.rsplit("_assay_", 1)[0]}
            for assay in ASSAYS
        ]
        self.write_catalog()
        for row in self.rows:
            (self.raw / row["DMS_filename"]).write_text(
                "mutant,mutated_sequence,DMS_score\nA1C,CCD,1.0\n", encoding="utf-8")

    def write_catalog(self):
        with (self.raw / "DMS_substitutions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("DMS_id", "DMS_filename", "target_seq", "UniProt_ID"))
            writer.writeheader()
            writer.writerows(self.rows)


class PreparationTargetTests(PreparationFixture):
    def invoke_helper(self, *arguments):
        return subprocess.run(
            [sys.executable, str(self.helper), "--config",
             str(self.root / "configs/extraction/esm2_650m.json"), *arguments],
            cwd=self.workspace, capture_output=True, text=True, timeout=20,
        )

    def test_targets_keep_catalog_order_and_distinct_assays_for_same_protein(self):
        (self.raw / "unregistered.csv").touch()
        result = self.invoke_helper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ASSAYS)
        result = self.invoke_helper("--count")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(len(ASSAYS)))

    def test_array_selection_is_zero_based_and_rejects_invalid_indices(self):
        for index, assay in enumerate(ASSAYS):
            with self.subTest(index=index):
                result = self.invoke_helper("--task-index", str(index))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), [assay])
        for index in ("-1", str(len(ASSAYS)), "not-an-index"):
            with self.subTest(index=index):
                result = self.invoke_helper("--task-index", index)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")

    def test_explicit_targets_preserve_requested_order_and_scope_count_and_array(self):
        arguments = [value for assay in SELECTED_ASSAYS for value in ("--assay", assay)]
        result = self.invoke_helper(*arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), SELECTED_ASSAYS)
        result = self.invoke_helper(*arguments, "--count")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(len(SELECTED_ASSAYS)))
        for index, assay in enumerate(SELECTED_ASSAYS):
            with self.subTest(index=index):
                result = self.invoke_helper(*arguments, "--task-index", str(index))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), [assay])
        result = self.invoke_helper(*arguments, "--task-index", str(len(SELECTED_ASSAYS)))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_explicit_unknown_or_duplicate_targets_are_rejected(self):
        cases = (
            ("--assay", "missing_assay"),
            ("--assay", SELECTED_ASSAYS[0], "--assay", SELECTED_ASSAYS[0]),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = self.invoke_helper(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn("error:", result.stderr)

    def test_catalog_rejects_missing_unsafe_duplicate_and_empty_targets(self):
        original = [row.copy() for row in self.rows]
        for problem in ("missing_file", "outside_directory", "duplicate_id", "empty"):
            with self.subTest(problem=problem):
                self.rows = [row.copy() for row in original]
                if problem == "missing_file":
                    self.rows[1]["DMS_filename"] = "missing.csv"
                elif problem == "outside_directory":
                    self.rows[1]["DMS_filename"] = "../outside.csv"
                elif problem == "duplicate_id":
                    self.rows[1]["DMS_id"] = self.rows[0]["DMS_id"]
                else:
                    self.rows = []
                self.write_catalog()
                result = self.invoke_helper()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn("error:", result.stderr)


@unittest.skipUnless(BASH, "Bash is required for preparation launcher tests")
class PreparationSbatchTests(PreparationFixture):
    def setUp(self):
        super().setUp()
        self.records = self.root / "calls.jsonl"
        self.bootstrap = self.workspace / "bootstrap.sh"
        self.bootstrap.write_text(textwrap.dedent("""\
            export PATH="/usr/bin:/bin:$PATH"
            conda() {
              if [[ "$1" == info ]]; then printf '%s\\n' "$TEST_ROOT"; fi
              if [[ "$1" == activate && "$2" != nkn_protein ]]; then return 20; fi
            }
            """), encoding="utf-8", newline="\n")
        (self.root / "etc/profile.d/conda.sh").write_text(
            'python() { "$TEST_PYTHON" "$TEST_ROOT/record.py" "$@"; }\n',
            encoding="utf-8", newline="\n")
        (self.root / "record.py").write_text(textwrap.dedent("""\
            import csv
            import json
            import os
            from pathlib import Path
            import runpy
            import sys
            from types import ModuleType, SimpleNamespace

            root = Path(os.environ['TEST_ROOT'])
            args = [arg for arg in sys.argv[1:] if arg != '-u']
            if args[0].endswith('list_assays.py'):
                # Match Linux stdout when the harness runs with Windows Python.
                sys.stdout.reconfigure(newline='\\n')
                sys.argv = args
                runpy.run_path(args[0], run_name='__main__')
                raise SystemExit(0)

            def record(kind, arguments):
                with (root / 'calls.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps({'kind': kind, 'args': arguments}) + '\\n')

            if args[:2] == ['-m', 'feature_generation.cli']:
                record(args[2], args[3:])
                raise SystemExit(7 if os.environ.get('TEST_FAIL_STAGE') == args[2] else 0)

            assert args[0] == '-', args
            body = sys.stdin.read()
            record('inline', args[1:])

            # Run the real inline control flow, replacing GPU/config/download I/O.
            class FakeConfig:
                @classmethod
                def load(cls, path):
                    values = json.loads(Path(path).read_text())
                    for key in ('assay_index', 'structure_manifest_dir', 'model_dir'):
                        if key in values:
                            values[key] = root / values[key]
                    return SimpleNamespace(**values)

            class FakeCatalog:
                def __init__(self, path):
                    with Path(path).open(newline='', encoding='utf-8') as handle:
                        self.rows = {row['DMS_id']: row for row in csv.DictReader(handle)}

                def get(self, assay):
                    return SimpleNamespace(reference_sequence=self.rows[assay]['target_seq'])

            def download_structure(index, assay, *, manifest_dir, **kwargs):
                record('structure-download', [assay])
                if os.environ.get('TEST_FAIL_STRUCTURE_ASSAY') == assay:
                    raise ValueError('fixture structure unavailable')
                return manifest_dir / (assay + '.json')

            def load_structure(manifest, reference):
                record('structure-verify', [Path(manifest).stem, reference])

            locked_files = []

            def lock_download(descriptor, operation):
                assert operation == 2, 'Model download must hold an exclusive lock'
                locked_files.append(descriptor)
                record('download-lock', [])

            def download_model(config):
                assert locked_files, 'Model download must acquire its lock first'
                os.fstat(locked_files[-1])  # The lock file must remain open during download.
                record('download', [str(config.model_dir)])
                if os.environ.get('TEST_FAIL_STAGE') == 'download':
                    raise RuntimeError('fixture model download failed')
                return 'fixture model'

            def module(name, **attributes):
                stub = ModuleType(name)
                stub.__dict__.update(attributes)
                sys.modules[name] = stub

            module('torch', __version__='fixture', version=SimpleNamespace(cuda='fixture'),
                   cuda=SimpleNamespace(is_available=lambda: True,
                                        get_device_name=lambda index: 'fixture GPU'))
            module('feature_generation')
            module('feature_generation.config', ExtractionConfig=FakeConfig)
            module('feature_generation.inverse_config', InverseFoldingConfig=FakeConfig)
            module('feature_generation.inverse_checkpoints', download=download_model)
            module('feature_generation.structure_inputs',
                   download_proteingym_structure=download_structure, load_structure=load_structure)
            module('feature_generation.structure_fragments', load_inverse_structure=load_structure)
            module('fcntl', LOCK_EX=2, flock=lock_download)
            module('data_loading')
            module('data_loading.catalog', AssayCatalog=FakeCatalog)
            sys.argv = args
            exec(compile(body, '<preparation inline>', 'exec'), {'__name__': '__main__'})
            """), encoding="utf-8")

    def invoke(self, script, environment=None, arguments=()):
        self.records.unlink(missing_ok=True)
        env = os.environ.copy()
        for name in ("SLURM_ARRAY_TASK_ID", "TEST_FAIL_STAGE", "TEST_FAIL_STRUCTURE_ASSAY", "ESM2_CONFIG"):
            env.pop(name, None)
        env.update(BASH_ENV=self.bootstrap.as_posix(), TEST_ROOT=self.root.as_posix(),
                   TEST_PYTHON=Path(sys.executable).as_posix(),
                   SLURM_SUBMIT_DIR=(self.root / "src/feature_generation").as_posix())
        env.update(environment or {})
        result = subprocess.run(
            [BASH, str(self.root / PREPARATION / script), *arguments], env=env, cwd=self.workspace,
            capture_output=True, text=True, timeout=30,
        )
        records = [json.loads(line) for line in self.records.read_text().splitlines()] if self.records.exists() else []
        return result, records

    def assert_targets(self, records, expected):
        for stage in ("extract", "score"):
            calls = [record for record in records if record["kind"] == stage]
            self.assertEqual(len(calls), 1, records)
            args = calls[0]["args"]
            selected = [args[index + 1] for index, value in enumerate(args) if value == "--assay"]
            self.assertEqual(selected, expected)
            self.assertIn("--resume", args)
            self.assertIn("--singles-only", args)
            self.assertIn("--continue-on-error", args)
            self.assertEqual(args[args.index("--device") + 1], "cuda")

    def test_default_dispatches_only_requested_assays_and_preserves_model_recipes(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                result, records = self.invoke(script)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assert_targets(records, SELECTED_ASSAYS)
                stages = [record["kind"] for record in records if record["kind"] in ("download", "extract", "score")]
                self.assertEqual(stages, ["download", "extract", "score"])
                scoring = next(record["args"] for record in records if record["kind"] == "score")
                self.assertEqual(scoring[scoring.index("--batch-size") + 1],
                                 "1" if "proteinmpnn" in script else "8")
                if "proteinmpnn" in script:
                    locking = [record["kind"] for record in records
                               if record["kind"] in ("download-lock", "download")]
                    self.assertEqual(locking, ["download-lock", "download"])
                    self.assertEqual(scoring[scoring.index("--num-orders") + 1], "1")
                    self.assertEqual(scoring[scoring.index("--config") + 1],
                                     "configs/extraction/proteinmpnn_v48_020.json")
                    self.assertEqual(
                        [record["args"][0] for record in records if record["kind"] == "structure-download"],
                        SELECTED_ASSAYS,
                    )
                else:
                    self.assert_esm2_config(records, "configs/extraction/esm2_650m_centered.json")

    def test_brca2_extracts_fragments_and_only_other_assays_receive_proteinmpnn_scores(self):
        assay = "BRCA2_HUMAN_Erwood_2022_HEK293T"
        self.rows.append({"DMS_id": assay, "DMS_filename": f"{assay}.csv", "target_seq": "ACD", "UniProt_ID": "BRCA2_HUMAN"})
        self.write_catalog()
        (self.raw / f"{assay}.csv").write_text("mutant,mutated_sequence,DMS_score\nA1C,CCD,1.0\n", encoding="utf-8")
        script = self.root / PREPARATION / SCRIPTS[1]
        script.write_text(script.read_text().replace(f"--assay {SELECTED_ASSAYS[0]}", f"--assay {assay}"),
                          encoding="utf-8", newline="\n")
        for array in (False, True):
            with self.subTest(brca2_only=array):
                result, records = self.invoke(SCRIPTS[1], {"SLURM_ARRAY_TASK_ID": "0"} if array else {})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("use ESM2 mean scores", result.stdout)
                extracted = next(record["args"] for record in records if record["kind"] == "extract")
                targets = [extracted[index + 1] for index, value in enumerate(extracted) if value == "--assay"]
                self.assertEqual(targets, [assay] if array else [assay, *SELECTED_ASSAYS[1:]])
                verified = [record["args"][0] for record in records if record["kind"] == "structure-verify"]
                self.assertEqual(verified, targets)
                scores = [record["args"] for record in records if record["kind"] == "score"]
                if array:
                    self.assertEqual(scores, [])
                else:
                    self.assertEqual(len(scores), 1)
                    scored = [scores[0][index + 1] for index, value in enumerate(scores[0]) if value == "--assay"]
                    self.assertEqual(scored, SELECTED_ASSAYS[1:])

    def assert_esm2_config(self, records, expected):
        for stage in ("download", "extract", "score"):
            calls = [record for record in records if record["kind"] == stage]
            self.assertEqual(len(calls), 1, records)
            arguments = calls[0]["args"]
            self.assertEqual(arguments[arguments.index("--config") + 1], expected)
        self.assertEqual([record["args"] for record in records if record["kind"] == "inline"], [[expected]])

    def test_esm2_config_override_selects_catalog_without_broadening_requested_assays(self):
        # A distinct catalog exposes any stage that accidentally keeps the default.
        catalog = self.raw / "subset.csv"
        with catalog.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(reversed(self.rows))
        # The default catalog lacks the requested assays, so ignoring the override fails.
        self.rows = [row for row in self.rows if row["DMS_id"] not in SELECTED_ASSAYS]
        self.write_catalog()
        config = json.loads((self.root / "configs/extraction/esm2_650m.json").read_text())
        config["assay_index"] = catalog.relative_to(self.root).as_posix()
        selected = "configs/extraction/custom pooled config.json"
        (self.root / selected).write_text(json.dumps(config), encoding="utf-8")
        cases = (
            ({"ESM2_CONFIG": selected}, ()),
            ({"ESM2_CONFIG": "does-not-exist.json"}, ("--config", selected)),
            ({"ESM2_CONFIG": "does-not-exist.json"}, (f"--config={selected}",)),
        )
        for environment, arguments in cases:
            with self.subTest(environment=environment, arguments=arguments):
                result, records = self.invoke(SCRIPTS[0], environment, arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assert_targets(records, SELECTED_ASSAYS)
                self.assert_esm2_config(records, selected)

    def test_esm2_config_override_missing_requested_assay_stops_before_model_work(self):
        catalog = self.raw / "incomplete.csv"
        with catalog.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(row for row in self.rows if row["DMS_id"] != SELECTED_ASSAYS[1])
        config = json.loads((self.root / "configs/extraction/esm2_650m.json").read_text())
        config["assay_index"] = catalog.relative_to(self.root).as_posix()
        selected = "configs/extraction/incomplete.json"
        (self.root / selected).write_text(json.dumps(config), encoding="utf-8")
        result, records = self.invoke(SCRIPTS[0], arguments=("--config", selected))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(SELECTED_ASSAYS[1], result.stderr)
        self.assertEqual(records, [])

    def test_esm2_invalid_arguments_stop_before_model_work(self):
        for arguments in (("--config",), ("--config=",), ("--config", ""),
                          ("--config", "--resume"), ("--unknown",), ("unexpected.json",)):
            with self.subTest(arguments=arguments):
                result, records = self.invoke(SCRIPTS[0], arguments=arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("error:", result.stderr)
                self.assertEqual(records, [])

    def test_array_task_selects_only_its_requested_assay(self):
        for script in SCRIPTS:
            for index, assay in enumerate(SELECTED_ASSAYS):
                with self.subTest(script=script, index=index):
                    result, records = self.invoke(script, {"SLURM_ARRAY_TASK_ID": str(index)})
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assert_targets(records, [assay])
                    downloads = [record["args"][0] for record in records if record["kind"] == "structure-download"]
                    if "proteinmpnn" in script:
                        self.assertEqual(downloads, [assay])

    def test_selection_failure_stops_before_model_or_structure_download(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                result, records = self.invoke(script, {"SLURM_ARRAY_TASK_ID": str(len(SELECTED_ASSAYS))})
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(records, [])
        (self.raw / f"{SELECTED_ASSAYS[1]}.csv").unlink()
        for script in SCRIPTS:
            with self.subTest(script=script, missing_file=True):
                result, records = self.invoke(script)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(records, [])

    def test_missing_requested_catalog_assay_stops_before_model_work(self):
        self.rows = [row for row in self.rows if row["DMS_id"] != SELECTED_ASSAYS[1]]
        self.write_catalog()
        for script in SCRIPTS:
            with self.subTest(script=script):
                result, records = self.invoke(script)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(SELECTED_ASSAYS[1], result.stderr)
                self.assertEqual(records, [])

    def test_failed_extraction_still_scores_and_both_stage_failures_reach_exit_status(self):
        for script in SCRIPTS:
            for stage in ("extract", "score"):
                with self.subTest(script=script, stage=stage):
                    result, records = self.invoke(script, {"TEST_FAIL_STAGE": stage})
                    self.assertNotEqual(result.returncode, 0)
                    self.assert_targets(records, SELECTED_ASSAYS)

    def test_failed_model_download_stops_before_assay_processing(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                result, records = self.invoke(script, {"TEST_FAIL_STAGE": "download"})
                self.assertNotEqual(result.returncode, 0)
                stages = [record["kind"] for record in records
                          if record["kind"] in ("download", "structure-download", "extract", "score")]
                self.assertEqual(stages, ["download"])

    def test_failed_structure_continues_other_assays_and_preserves_registered_manifests(self):
        manifests = self.root / "data/structures/manifests"
        manifests.mkdir(parents=True)
        (manifests / f"{SELECTED_ASSAYS[1]}.json").write_text("registered fixture", encoding="utf-8")
        result, records = self.invoke(SCRIPTS[1], {"TEST_FAIL_STRUCTURE_ASSAY": SELECTED_ASSAYS[0]})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"{SELECTED_ASSAYS[0]}: structure failed", result.stderr)
        downloads = [record["args"][0] for record in records if record["kind"] == "structure-download"]
        self.assertEqual(downloads, [SELECTED_ASSAYS[0], *SELECTED_ASSAYS[2:]])
        verified = [record["args"][0] for record in records if record["kind"] == "structure-verify"]
        self.assertEqual(verified, SELECTED_ASSAYS[1:])
        self.assert_targets(records, SELECTED_ASSAYS)


if __name__ == "__main__":
    unittest.main()
