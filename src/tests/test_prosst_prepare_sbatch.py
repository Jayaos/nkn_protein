"""Exercise the catalog-wide ProSST job with real configs and mocked GPU/model I/O."""

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


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path("sbatch_run/run_prepare_rep/prepare_prosst.sbatch")
CONFIG = Path("configs/extraction/prosst_2048_singles.json")
ASSAY = "FKBP3_HUMAN_Tsuboyama_2023_2KFV"
BASH = next((str(path) for path in (
    Path("C:/Program Files/Git/usr/bin/bash.exe"), Path("C:/Program Files/Git/bin/bash.exe"),
) if path.is_file()), None) if os.name == "nt" else shutil.which("bash")


@unittest.skipUnless(BASH, "Bash is required for preparation launcher tests")
class ProsstPreparationSbatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.root = self.workspace / "repository with spaces"
        files = (SCRIPT, CONFIG, Path("src/feature_generation/__init__.py"),
                 Path("src/feature_generation/config.py"),
                 Path("src/feature_generation/prosst_config.py"),
                 Path("src/feature_generation/prosst_artifacts.json"))
        for path in files:
            (self.root / path.parent).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / path, self.root / path)
        (self.root / "src/feature_generation/cli.py").touch()
        raw = self.root / "data/DMS_ProteinGym_substitutions/raw_substitutions"
        raw.mkdir(parents=True)
        with (raw.parent / "DMS_substitutions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("DMS_id", "DMS_filename", "target_seq"))
            writer.writeheader()
            writer.writerow(dict(DMS_id=ASSAY, DMS_filename=f"{ASSAY}.csv", target_seq="ACD"))
        (raw / f"{ASSAY}.csv").write_text("mutant,mutated_sequence,DMS_score\nA1C,CCD,1.0\n")
        self.records = self.root / "calls.jsonl"
        self.bootstrap = self.workspace / "bootstrap.sh"
        self.bootstrap.write_text(textwrap.dedent("""\
            export PATH="/usr/bin:/bin:$PATH"
            conda() {
              if [[ "$1" == info ]]; then printf '%s\\n' "$TEST_ROOT"; fi
              if [[ "$1" == activate && "$2" != nkn_protein ]]; then return 20; fi
            }
            """), encoding="utf-8", newline="\n")
        profile = self.root / "etc/profile.d"
        profile.mkdir(parents=True)
        (profile / "conda.sh").write_text(
            'python() { "$TEST_PYTHON" "$TEST_ROOT/record.py" "$@"; }\n',
            encoding="utf-8", newline="\n",
        )
        (self.root / "record.py").write_text(textwrap.dedent("""\
            import csv
            import json
            import os
            from pathlib import Path
            import sys
            from types import ModuleType, SimpleNamespace

            root = Path(os.environ['TEST_ROOT'])
            sys.path.insert(0, str(root / 'src'))
            args = [arg for arg in sys.argv[1:] if arg != '-u']
            assert args[0] == '-', args
            body = sys.stdin.read()

            def record(kind, **values):
                with (root / 'calls.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(dict(kind=kind, **values)) + '\\n')
                if (os.environ.get('TEST_FAIL_STAGE') == kind and
                    (not os.environ.get('TEST_FAIL_ASSAY') or
                     values.get('assays') == [os.environ['TEST_FAIL_ASSAY']])):
                    raise RuntimeError('fixture failure: ' + kind)

            class Catalog:
                def __init__(self, path):
                    with Path(path).open(newline='', encoding='utf-8') as handle:
                        self.rows = {r['DMS_id']: r for r in csv.DictReader(handle)}
                def get(self, assay):
                    return SimpleNamespace(reference_sequence=self.rows[assay]['target_seq'])

            def prepare_assay(index, assay, singles_only=False):
                record('inputs', assay=assay, singles_only=singles_only)
                reference = Catalog(index).get(assay).reference_sequence
                return SimpleNamespace(length=len(reference), variant_ids=['A1C'])

            def runtime():
                record('runtime')
                return {'scikit-learn': '1.2.2'}

            locks = []
            def flock(fd, operation):
                assert operation == 2
                os.fstat(fd)
                locks.append(fd)
                record('lock')

            def require_lock():
                assert locks
                os.fstat(locks[-1])

            def download(config):
                require_lock()
                record('download', model=config.model_id, model_dir=str(config.model_dir))
                return config.model_dir

            def download_structure(index, assay, *, manifest_dir):
                require_lock()
                record('structure-download', assay=assay, manifest_dir=str(manifest_dir))
                path = manifest_dir / (assay + '.json')
                path.write_text('{}')
                return path

            def load_structure(manifest, reference, *, require_oxygen):
                require_lock()
                record('structure-verify', path=str(manifest), require_oxygen=require_oxygen)

            def extract(config, assays, *, singles_only, resume, progress):
                require_lock()
                record('extract', assays=assays, singles_only=singles_only, resume=resume,
                       device=config.device, output_root=str(config.output_root),
                       cache_id=config.cache_id, max_tokens_per_batch=config.max_tokens_per_batch,
                       token_dir=str(config.structure_token_dir), layer=config.layer)
                return [{'status': os.environ.get('TEST_EXTRACT_STATUS', 'completed')}]

            def validate(path, cache_id, *, model_family):
                require_lock()
                record('validate', path=str(path), cache_id=cache_id, model_family=model_family)
                return {'path': str(path), 'num_variants': 1, 'embedding_dim': 768}

            def score(config, assays, *, singles_only, resume, progress):
                require_lock()
                record('score', assays=assays, singles_only=singles_only, resume=resume)
                return [{'status': os.environ.get('TEST_SCORE_STATUS', 'completed')}]

            def validate_scores(path, model_family, cache_id):
                require_lock()
                record('validate-scores', path=str(path), model_family=model_family, cache_id=cache_id)
                return {'path': str(path), 'num_variants': 1}

            def module(name, **attributes):
                stub = ModuleType(name)
                stub.__dict__.update(attributes)
                sys.modules[name] = stub

            module('torch', __version__='fixture', version=SimpleNamespace(cuda='fixture'),
                   cuda=SimpleNamespace(is_available=lambda: not os.environ.get('TEST_NO_CUDA'),
                                        get_device_name=lambda index: 'fixture GPU'))
            module('fcntl', LOCK_EX=2, flock=flock)
            module('data_loading')
            module('data_loading.catalog', AssayCatalog=Catalog)
            module('feature_generation.inputs', prepare_assay=prepare_assay)
            module('feature_generation.prosst_structure', _runtime_versions=runtime)
            module('feature_generation.prosst_checkpoints', download=download)
            module('feature_generation.structure_inputs',
                   download_proteingym_structure=download_structure, load_structure=load_structure)
            module('feature_generation.prosst_runner', extract=extract)
            module('feature_generation.score_runner', score=score, default_cache_id=lambda config: 'score-recipe')
            module('feature_generation.score_storage', validate_score_cache=validate_scores)
            module('feature_generation.storage', validate_sequence_cache=validate)
            module('feature_generation.cli', _progress=lambda event: None)
            sys.argv = args
            exec(compile(body, '<ProSST preparation>', 'exec'), {'__name__': '__main__'})
            """), encoding="utf-8")

    def invoke(self, arguments=(), environment=None):
        self.records.unlink(missing_ok=True)
        env = os.environ.copy()
        for name in ("PROSST_CONFIG", "SLURM_ARRAY_TASK_ID", "TEST_FAIL_STAGE",
                     "TEST_NO_CUDA", "TEST_EXTRACT_STATUS", "TEST_FAIL_ASSAY", "TEST_SCORE_STATUS"):
            env.pop(name, None)
        env.update(BASH_ENV=self.bootstrap.as_posix(), TEST_ROOT=self.root.as_posix(),
                   TEST_PYTHON=Path(sys.executable).as_posix(),
                   SLURM_SUBMIT_DIR=(self.root / "src/feature_generation").as_posix())
        env.update(environment or {})
        result = subprocess.run([BASH, str(self.root / SCRIPT), *arguments],
                                env=env, cwd=self.workspace, capture_output=True, text=True, timeout=30)
        records = [json.loads(line) for line in self.records.read_text().splitlines()] if self.records.exists() else []
        return result, records

    def test_complete_pipeline_uses_singles_cuda_and_matching_validation(self):
        result, records = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        stages = [r['kind'] for r in records]
        self.assertEqual(stages, ['inputs', 'runtime', 'lock', 'download', 'lock',
                                  'structure-download', 'structure-verify', 'extract', 'validate',
                                  'score', 'validate-scores'])
        extraction = next(r for r in records if r['kind'] == 'extract')
        self.assertEqual(extraction['assays'], [ASSAY])
        self.assertTrue(extraction['singles_only'])
        self.assertTrue(extraction['resume'])
        self.assertEqual(extraction['device'], 'cuda')
        self.assertEqual(Path(extraction['output_root']), self.root / 'data/features/pretrained/singles')
        self.assertEqual(extraction['layer'], 12)
        validation = next(r for r in records if r['kind'] == 'validate')
        self.assertEqual(validation['cache_id'], extraction['cache_id'])
        self.assertEqual(validation['model_family'], 'prosst')
        self.assertEqual(Path(validation['path']), Path(extraction['output_root']) / ASSAY)
        self.assertTrue(next(r for r in records if r['kind'] == 'structure-verify')['require_oxygen'])

    def test_config_precedence_preserves_registered_structure_and_paths(self):
        custom = self.root / 'configs/extraction/custom paths.json'
        # Exercise real ProSSTConfig defaults, including omitted assay_index.
        custom.write_text(json.dumps({'model_family': 'prosst', 'layer': 10,
                                     'model_dir': 'custom models', 'output_root': 'custom outputs',
                                     'structure_manifest_dir': 'custom structures',
                                     'structure_token_dir': 'custom tokens', 'max_tokens_per_batch': 4096}))
        manifests = self.root / 'custom structures'
        manifests.mkdir()
        manifest = manifests / f'{ASSAY}.json'
        manifest.write_text('registered structure')
        cases = (((), {'PROSST_CONFIG': str(custom)}),
                 (('--config', str(custom)), {'PROSST_CONFIG': 'missing.json'}),
                 ((f'--config={custom}',), {'PROSST_CONFIG': 'missing.json'}))
        for arguments, environment in cases:
            with self.subTest(arguments=arguments):
                result, records = self.invoke(arguments, environment)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn('structure-download', [r['kind'] for r in records])
                extraction = next(r for r in records if r['kind'] == 'extract')
                self.assertEqual(extraction['layer'], 10)
                self.assertEqual(extraction['max_tokens_per_batch'], 4096)
                self.assertEqual(Path(extraction['output_root']), self.root / 'custom outputs')
                self.assertEqual(Path(extraction['token_dir']), self.root / 'custom tokens')
                self.assertEqual(manifest.read_text(), 'registered structure')

    def test_stage_failures_stop_later_stages_and_return_nonzero(self):
        order = ['runtime', 'download', 'structure-verify', 'extract', 'validate', 'score', 'validate-scores']
        for index, stage in enumerate(order):
            with self.subTest(stage=stage):
                result, records = self.invoke(environment={'TEST_FAIL_STAGE': stage})
                self.assertNotEqual(result.returncode, 0)
                kinds = [r['kind'] for r in records]
                self.assertIn(stage, kinds)
                self.assertFalse(set(order[index + 1:]).intersection(kinds))

    def test_resumed_cache_is_still_validated_and_failed_result_is_not(self):
        result, records = self.invoke(environment={'TEST_EXTRACT_STATUS': 'skipped'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(records[-1]['kind'], 'validate-scores')
        result, records = self.invoke(environment={'TEST_EXTRACT_STATUS': 'failed'})
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('validate', [r['kind'] for r in records])

    def test_preflight_rejects_unavailable_cuda_and_insufficient_resource_guards(self):
        result, records = self.invoke(environment={'TEST_NO_CUDA': '1'})
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('download', [r['kind'] for r in records])
        custom = self.root / 'configs/extraction/invalid.json'
        for override in ({'max_length': 2}, {'max_tokens_per_batch': 2},
                         {'max_output_gib': 1e-10}, {'model_id': 'wrong-model'}):
            with self.subTest(override=override):
                custom.write_text(json.dumps(override))
                result, records = self.invoke(('--config', str(custom)))
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('download', [r['kind'] for r in records])

    def test_invalid_arguments_and_array_indices_stop_before_model_work(self):
        for arguments in (('--config',), ('--config=',), ('--config', '--resume'),
                          ('--config', 'missing.json'), ('--all-assays',), ('--assay', 'other')):
            with self.subTest(arguments=arguments):
                result, records = self.invoke(arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(records, [])
        for index in ('-1', '1', 'invalid'):
            result, records = self.invoke(environment={'SLURM_ARRAY_TASK_ID': index})
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(records, [])
        result, _ = self.invoke(environment={'SLURM_ARRAY_TASK_ID': '0'})
        self.assertEqual(result.returncode, 0, result.stderr)

    def add_second_assay(self):
        raw = self.root / "data/DMS_ProteinGym_substitutions/raw_substitutions"
        with (raw.parent / "DMS_substitutions.csv").open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(["SECOND_ASSAY", "SECOND_ASSAY.csv", "ACD"])
        (raw / "SECOND_ASSAY.csv").write_text("mutant,mutated_sequence,DMS_score\nA1C,CCD,1.0\n")

    def test_all_catalog_assays_run_in_order_with_one_model_download(self):
        self.add_second_assay()
        result, records = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([r['assays'] for r in records if r['kind'] == 'extract'],
                         [[ASSAY], ['SECOND_ASSAY']])
        self.assertEqual(sum(r['kind'] == 'download' for r in records), 1)
        self.assertEqual(sum(r['kind'] == 'validate' for r in records), 2)

    def test_array_selects_second_assay(self):
        self.add_second_assay()
        result, records = self.invoke(environment={'SLURM_ARRAY_TASK_ID': '1'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([r['assays'] for r in records if r['kind'] == 'extract'],
                         [['SECOND_ASSAY']])

    def test_failed_assay_does_not_prevent_later_assays(self):
        self.add_second_assay()
        result, records = self.invoke(environment={'TEST_FAIL_STAGE': 'extract',
                                                   'TEST_FAIL_ASSAY': ASSAY})
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([r['assays'] for r in records if r['kind'] == 'extract'],
                         [[ASSAY], ['SECOND_ASSAY']])
        self.assertEqual(records[-1]['kind'], 'validate-scores')
        self.assertEqual(Path(records[-1]['path']).name, 'SECOND_ASSAY')
        self.assertIn('Failed assays (1)', result.stderr)

    def test_failed_score_is_reported_without_score_validation(self):
        result, records = self.invoke(environment={'TEST_SCORE_STATUS': 'failed'})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('validate', [r['kind'] for r in records])
        self.assertNotIn('validate-scores', [r['kind'] for r in records])

    def test_help_requires_no_runtime_or_downloads(self):
        result, records = self.invoke(('--help',))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("all assays", result.stdout)
        self.assertEqual(records, [])


if __name__ == '__main__':
    unittest.main()
