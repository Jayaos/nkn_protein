# NKN protein fitness

An independent PyTorch implementation of the compositional-kernel exact GP in
`nkn_protein_variant_prediction.pdf`. The local Kermut and Neural-Kernel-Network
clones are references only: this package does not import them or require them
on `PYTHONPATH`. Implementation and tests are in this `src/` directory;
JSON configuration presets are in the repository-root `configs/` directory.
Data loading is consolidated in `data_loading/`.

The independent [Kermut baseline](../baselines/kermut/README.md) shares these
cached inputs, ProteinGym splits and metrics. Its configuration is
`configs/kermut_esm2_proteinmpnn.json`; run it with
`python -m baselines.kermut.cli` from the repository root.
The NKN batch preset and Kermut both select ESM2 650M embeddings/scores and
the ProteinMPNN Kermut recipe (`configs/extraction/proteinmpnn_v48_020_kermut.json`): ten
sequence-conditioned runs with seed 37, averaging log probabilities and
removing `X` without renormalization. The ProteinMPNN preparation job generates
this sequence-conditioned cache and the separate variant likelihood scores.
NKN uses both ESM2 and ProteinMPNN score tables in its GP mean.

## Installation and first run

To create the Conda environment, run these commands from the repository root:

```powershell
conda env create -f env.yaml
conda activate kernel_protein_fitness
$env:PYTHONPATH = (Resolve-Path ./src).Path + [IO.Path]::PathSeparator + $env:PYTHONPATH
python -m experiments.cli --help
```

The environment installs Python 3.10, NumPy 1.26.4, PyTorch 2.5.1, the
CUDA 12.1 runtime, and inference dependencies for ESM-2, ProSST, ProteinMPNN, and ESM-IF1.
Dependencies are declared in `env.yaml` and `requirements.txt` (for a Python 3.10 venv).
The `PYTHONPATH` command makes the source packages available
in the current PowerShell session so the CLI can run directly from the repository.
The PyTorch/CUDA combination follows the
[official PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/#v251).
GPU execution requires a compatible NVIDIA GPU and driver. The experiment CLI defaults
to CPU; extraction presets select CUDA when available. Use `--device cuda` for explicit GPU execution. For a CPU-only Conda environment,
replace `pytorch-cuda=12.1` with `cpuonly` and use the matching CPU
`torch-scatter` wheel/index in `env.yaml` before creating it.
The code runs directly from `src/`; no project installation is required.
After activating your environment, set `PYTHONPATH` as shown above in each new
PowerShell session. If you already have the named Conda environment, update its
dependencies with `conda env update -f env.yaml`. Then run the fixture checks:

```powershell
python -m experiments.cli validate --features data/features/A0A1I9GEU1_NEIME_Kennouche_2019 --assay-index data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv --allow-subset --output results/fixture_validation
python -m unittest discover -s src/tests -p test_pipeline.py -v
```

Run experiments with `python -m experiments.cli` and extraction with
`python -m feature_generation.cli`. Experiment commands require a new or empty
output directory.
The validation example's cached representations and zero-shot scores are
synthetic, while its labels are real. The pipeline tests use small synthetic
inputs and explicit fold assignments for integration checks.

Without `--config`, training uses the project's 5,000-step maximum, 1,000-step
minimum before early stopping, 500-step comparison window, and one initialization
per training split.
Device defaults to CPU. Use `--device cuda` with a suitable PyTorch installation.
The experiment CLI consumes saved features. The separate feature-generation
CLI prepares structures, downloads checkpoints, extracts frozen ESM-2 and structure-conditioned ProSST embeddings,
and computes ProteinMPNN/ESM-IF1 amino-acid distributions.
See [the extraction guide](feature_generation/README.md) for installation,
checkpoint downloads, structure mapping, resumable extraction and standalone cache validation.
Notebooks are optional smoke tests; the CLI is the default extraction workflow.

## Repository organization

| Path | Responsibility |
| --- | --- |
| `src/config.py` | Validated model, optimizer, evaluation, and runtime settings |
| `src/data_loading/` | Assay catalog, cache/score alignment, PyTorch dataset, packed mutations, folds |
| `src/kernels/` | Primitive kernels, NKN layers, mutation sums, final kernel |
| `src/gp/` | Fold transforms, dense Cholesky, exact GP, restarts, checkpoints |
| `src/evaluation/` | Spearman, interval ECE, PairECE, fold aggregation |
| `src/experiments/` | Fit/CV/prediction execution, CLI, result artifacts |
| `src/tests/` | Numerical, data-contract, leakage, and integration tests |
| `configs/smoke.json` | Short fixture-validation run |
| `src/feature_generation/` | Pinned checkpoints, structure mapping, ESM-2/ProSST embeddings, ProteinMPNN/ESM-IF1 distributions, cache validation |
| `configs/extraction/` | ESM-2 8M/650M, ProSST-2048, ProteinMPNN v48_020, and ESM-IF1 142M recipes |

The Python packages listed above are directly under `src/`.
Import them by their actual names, for example `from gp import fit_gp` or
`from config import ExperimentConfig`. Add `src/` to `PYTHONPATH` before
using these imports and module commands. Source edits take effect directly;
there is no installed project package to rebuild or reinstall.

The former `data` and `dataset` packages are combined in `data_loading`:

```python
from data_loading import ProteinFitnessDataset, load_feature_bundle
```

`ProteinFitnessDataset` provides item-by-item access to synthetic fixture
representations. `load_feature_bundle` prepares the packed inputs used by the GP
pipeline. The assay catalog and split utilities are exported from the same package.

## Inputs and data scope

The cache adapter accepts the existing schema-version-2/layout-version-1 format:

```text
features/<assay>/
  assay.json
  variants.csv
  sequence/<family>/<checkpoint>/{config.json,features.pt}
  structure/<family>/<checkpoint>/{config.json,features.pt}
  inverse_folding/<family>/<checkpoint>/{config.json,features.pt}
  scores/<family>/<recipe>/{config.json,mean_score.csv,provenance.json,completion.json}
```

`variants.csv` identifies the requested variants. Sequence cache IDs and hashes
must match its complete sequences. Structural embeddings and inverse-folding
probabilities are shared wild-type arrays indexed by one-based reference position.
Working tensors are converted to float64; cache files are not rewritten.
The current adapter uses mean pooling over all sequence residues for the global
branch. Model-specific preprocessing and other pooling can later be supplied
through the model-independent `FeatureBatch` API. The old
`ProteinFitnessDataset.as_feature_matrices` demo transformations are not used.

Mean score sources default to the distinct representation directories required
by enabled branches, for compatibility with existing fixtures. For generated
scores, set `score_sources` explicitly to `scores/family/recipe`. Scores are joined
by ID and standardized on training variants only. `mean_score` in the score config
declares the CSV file and ID/value columns; legacy `zero_shot` descriptors are still
readable. ProteinMPNN scores are negative mean sequence NLL; ESM2 scores sum
masked-WT mutation log-odds. Both record their direction in the config. The loader
uses values as stored; it does not automatically flip score direction. Source
identity and order are saved in checkpoints and validated during prediction.

The ProteinGym catalog reads `DMS_id`, `DMS_filename`, and `target_seq` from its
index. It validates processed mutation positions directly against the reference;
it does not reapply raw-data, MSA, or structure offsets. Cross-assay identities
include the assay ID. No feature extraction is inferred from a raw assay CSV.

When `--assay-index` is supplied, complete indexed assay coverage is required
unless `--allow-subset` is explicit. The current fixture contains 16 of its
assay's 922 variants, so the validation example uses that flag. With
`--singles-only` on validation or prediction, completeness refers to all indexed
single substitutions. Training commands validate full assay coverage before
selecting the mutation depths required by `--setting`. Rows are not silently
removed to match an incomplete feature cache.

## Fold inputs and experiments

Raw assay CSVs and the assay index are under
`data/DMS_ProteinGym_substitutions/raw_substitutions/`. They contain no fold
assignments. Downloaded ProteinGym benchmark CSVs with assignments are under
`data/DMS_ProteinGym_substitutions/cv_folds_singles_substitutions/` and
`data/DMS_ProteinGym_substitutions/cv_folds_multiples_substitutions/`.
Both `fit` and `cv` require `--setting`. The first five settings below use
published five-fold assignments and run with `cv`, which evaluates all five
folds automatically and requires a fold CSV through `--folds`. `fit` accepts only
`--setting single_to_double` for the single extrapolation experiment. Full-data
fitting is not an experiment mode.

| Setting | Training variants | Evaluation variants | Fold CSV column |
| --- | --- | --- | --- |
| `singles_random` | 1M in four folds | 1M in the held-out fold | `fold_random_5` |
| `singles_modulo` | 1M in four folds | 1M in the held-out fold | `fold_modulo_5` |
| `singles_contiguous` | 1M in four folds | 1M in the held-out fold | `fold_contiguous_5` |
| `mixed_random_singles` | 1M and 2M in four folds | Held-out 1M | `fold_rand_multiples` |
| `mixed_random_doubles` | 1M and 2M in four folds | Held-out 2M | `fold_rand_multiples` |
| `single_to_double` | Sampled fraction of 1M (default 80%; configurable up to all singles) | All 2M | None |

1M and 2M mean exactly one and two substitutions. Mixed settings exclude wild
type and mutation depths above two. They use the same models and held-out
predictions: either mixed setting produces separate metrics for **both** domains
in `setting_metrics`; the chosen setting determines the top-level metrics.
Running both mixed setting commands separately is unnecessary.

Supply the singles archive for any `singles_*` setting, and the multiples archive
for either `mixed_*` setting. The multiples archive includes singles as well as
multiple substitutions. The loader accepts the published CSVs directly, including
their `mutant` ID column:

```text
variant_id,fold_random_5,fold_modulo_5,fold_contiguous_5
M7V,0,1,2
...
```

`mutant` can replace `variant_id`. Short column aliases `random`, `modulo`,
`contiguous`, and `random_multiples` are accepted, with `fold` as a fallback.
Every selected variant must have an assignment. Extra CSV IDs are permitted only
when they belong to known cached variants excluded by the setting; unknown IDs
are rejected. Folds are numbered `0` through `n_folds - 1`, with every fold
present after mutation-depth selection. Positional settings require disjoint
mutation positions across folds. Published ProteinGym protocols have five folds;
test configurations with explicit assignments can use fewer folds.

```powershell
python -m experiments.cli cv --features PATH_TO_CACHE --folds PATH_TO_SINGLES_FOLDS.csv --setting singles_random --output results/assay_random
python -m experiments.cli cv --features PATH_TO_CACHE --folds PATH_TO_SINGLES_FOLDS.csv --setting singles_modulo --output results/assay_modulo
python -m experiments.cli cv --features PATH_TO_MIXED_CACHE --folds PATH_TO_MULTIPLES_FOLDS.csv --setting mixed_random_singles --output results/assay_mixed
python -m experiments.cli fit --features PATH_TO_MIXED_CACHE --setting single_to_double --sample-ratio 0.8 --split-seed 2024 --output results/single_to_double
# Train on all singles and test on all doubles:
python -m experiments.cli fit --features PATH_TO_MIXED_CACHE --setting single_to_double --sample-ratio 1.0 --output results/single_to_double_all_singles
python -m experiments.cli predict --features PATH_TO_NEW_VARIANTS --checkpoint results/assay_modulo/fold_0/checkpoint.pt --output results/prediction
```

The extrapolation setting accepts two `fit` options: `--sample-ratio` (default
`0.8`) and `--split-seed` (default `2024`). The ratio must be finite and satisfy
`0 < sample_ratio <= 1`; the seed must be an integer from `0` through `2**32 - 1`.
For ratios below `1.0`, it samples `round(sample_ratio * n_singles)` singles
without replacement using NumPy's `RandomState` and the supplied seed. Python's
`round` rounds halfway cases to the nearest even integer. Matching the exact
sampled singles requires the same source single-mutant order. With
`--sample-ratio 1.0`, all singles are retained in their original order without
random sampling, so the seed does not affect the split.

The defaults follow Kermut's
[reproduction code](https://github.com/petergroth/kermut/blob/e6500221b5159a1896272bfc4a3c6e3896eb0027/src/experiments/proteingym_domains.py#L122-L139);
the paper does not specify the 80% training fraction. All double mutants are
tested for any ratio. Remaining singles are unused for this doubles-only
evaluation; wild type and depths above two are excluded. Saved partition
assignments identify the training, test, unused, and excluded variants. Metrics
and the run summary record `sample_ratio`, `split_seed`, and the legacy alias
`training_single_fraction`; the input manifest records
`extrapolation_sample_ratio` and `extrapolation_split_seed`. The recorded fraction
is the requested ratio; the actual training count follows the rounding above.
This setting takes no fold CSV and has no fold mean or standard deviation. At
least two sampled training singles and one test double are required. `cv` does
not accept either sampling option and always uses the published fold assignments.

`predict` can use an experiment checkpoint and does not need labels; omit
`DMS_score` or leave every value blank in the prediction cache's variant table.
The checkpoint includes training representations, standardized training
scores/targets, and fitted transformations. A prediction command uses the
adjacent `resolved_config.json` unless an explicit configuration is supplied.

### CPU batch job with published ProteinGym folds

The [ESM2 + ProteinMPNN job](../sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch)
defaults to the real shared cache for `A0A1I9GEU1_NEIME_Kennouche_2019` and
[`configs/esm2_proteinmpnn_nkn.json`](../configs/esm2_proteinmpnn_nkn.json).
It requests 15 CPUs and no GPU. ESM2 650M layer-33 embeddings and ProteinMPNN
sequence-conditioned distributions provide two kernel branches. Their GP mean
inputs are the ESM2 650M masked-marginal scores and the existing ProteinMPNN
sequence likelihoods (one order, seed 0). The inverse-folding cache uses ten
orders with seed 37 and retains the stored 20 amino-acid values without
renormalization through `inverse_folding_policy: "sequence_conditional"`.

Prepare these inputs before running NKN, and wait for both preparation jobs to
finish successfully:

```bash
sbatch sbatch_run/run_prepare_rep/prepare_esm2.sbatch
sbatch sbatch_run/run_prepare_rep/prepare_proteinmpnn.sbatch
```

The NKN launcher uses the cluster's existing `nkn_protein` Conda environment.

By default, the job runs `singles`: the three singles settings, five folds each.
This matches the default assay, whose 922 variants are all single mutants.
`all` requests all six settings on an assay with singles and doubles: three
singles CV runs, one mixed CV run reporting both domains, and one extrapolation
fit. This totals 21 fits, with one initialization per fit in the preset.
Every requested setting and fold file is checked before the first training run.
An assay without doubles fails when a mixed or extrapolation setting is requested;
settings are never silently skipped. The job requires published column names and
five folds, and never generates replacement assignments.

Use either assay-specific launcher from the repository root on the cluster:

```bash
sbatch sbatch_run/run_nkn/run_nkn_A0A1I9GEU1_NEIME_Kennouche_2019.sbatch
sbatch sbatch_run/run_nkn/run_nkn_FKBP3_HUMAN_Tsuboyama_2023_2KFV.sbatch

# Run all 15 singles fits concurrently for either assay:
sbatch sbatch_run/run_nkn/run_nkn_A0A1I9GEU1_NEIME_Kennouche_2019.sbatch singles --parallel
sbatch sbatch_run/run_nkn/run_nkn_FKBP3_HUMAN_Tsuboyama_2023_2KFV.sbatch singles --parallel
```

Each fixes its named assay and defaults to `data/features/pretrained/<assay>`.
Both use the shared NKN preset's ESM2 650M and sequence-conditioned ProteinMPNN
features, with both score sources in the GP mean. They forward settings,
`--parallel`, and `--config PATH` to the shared launcher; `FEATURES` and
`NKN_CONFIG` overrides remain available. Outputs use the assay-specific
`results/<assay>/` directory described below.

The generic launcher also remains available:

```bash
# Default assay, all three singles settings: 15 fold fits
sbatch sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch

# All 15 singles fits concurrently: 3 settings x 5 folds x 1 CPU thread
sbatch sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch singles --parallel

# One setting: 5 folds concurrently with 1 CPU thread each
sbatch --cpus-per-task=5 sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch singles_random --parallel

# One setting
sbatch sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch singles_modulo

# One setting with your own JSON hyperparameters
cp configs/esm2_proteinmpnn_nkn.json configs/my_nkn.json
# Edit configs/my_nkn.json before submitting:
sbatch sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch singles_random --config configs/my_nkn.json

# All six settings; prepare the complete ESM2 + ProteinMPNN cache first
ASSAY=A4_HUMAN_Seuma_2022 sbatch sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch all

# Only extrapolation, using an 80% sample of singles for training
ASSAY=A4_HUMAN_Seuma_2022 SAMPLE_RATIO=0.8 SPLIT_SEED=2024 sbatch sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch single_to_double

# Extrapolation with all singles for training
ASSAY=A4_HUMAN_Seuma_2022 SAMPLE_RATIO=1.0 sbatch sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch single_to_double
```

You can edit the default preset directly or copy it to keep separate experiments.
`--config PATH` can appear before or after the optional setting; omitting the
setting still runs `singles`. Alternatively, set `NKN_CONFIG=PATH` when submitting.
The command-line `--config` takes precedence over `NKN_CONFIG`, which takes
precedence over the default preset. Config paths may be absolute or relative to
the repository root. See the hyperparameter fields below.

Each submission creates
`results/<assay>/nkn_esm2_proteinmpnn_<selection>_<job-id>/`.
CV results are saved under `cv/<setting>/`; extrapolation is under
`fit/single_to_double/`. For `all`, both mixed-domain results are in
`cv/mixed_random_singles/metrics.json` under `setting_metrics`.
The run saves byte-for-byte `folds/singles.csv` and/or `folds/multiples.csv`
snapshots as needed, plus a `fold_manifest.json` with source paths, SHA-256
hashes and selected fold sizes. When extrapolation is requested, the manifest also
records `sample_ratio` and `split_seed`.
The top-level `config.json` records the expanded configuration actually used,
including defaults and the job's CPU and thread settings.

For the generic launcher, `ASSAY` chooses the assay. `FEATURES` overrides the default
`data/features/pretrained/<assay>` cache. `SINGLES_FOLDS_CSV` and
`MULTIPLES_FOLDS_CSV` override the corresponding published files. `SAMPLE_RATIO`
(default `0.8`) controls the fraction of singles used for extrapolation training;
`SPLIT_SEED` (default `2024`) selects the reproducible sample when the ratio is
below `1.0`. Both are used only when extrapolation is requested and are then
recorded in the batch manifest; neither changes published CV folds. Relative paths resolve from the repository
root. The former positional `cv`/`fit` modes and `FOLDS_CSV` variable are rejected.
Existing pretrained features and scores are reused without extraction.

### Parallel CPU folds

The optional `--parallel` batch flag runs settings concurrently and uses separate
CPU processes for their folds. With `singles` and `--cpus-per-task=15`, one Slurm
job runs three setting processes, each with five fold workers using one PyTorch
and BLAS thread per worker. Omitting `--parallel` preserves serial settings and
folds, with each fit using the full CPU allocation. Requesting more CPUs alone
does not enable concurrent folds.

Parallel worker counts and threads are derived from the allocation, with the
total number of fitting threads kept within `SLURM_CPUS_PER_TASK`. The batch job
prints and saves its execution plan. Each setting has its own log under
`logs/<setting>.log`; per-fold checkpoints, histories, predictions and metrics
keep their existing `cv/<setting>/fold_<n>/` paths. The parent process combines
results in fold and variant order after every fold succeeds. Worker failures
make the job fail rather than reporting an incomplete run as completed.

Every fold still fits its transformations and model using only that fold's
training variants. Fold assignments, seeds, hyperparameters and evaluation
protocols are unchanged. `all --parallel` retains the shared mixed CV models and
the single extrapolation fit; extrapolation is not split into folds.

Memory requested with `--mem` is shared by the entire job. Concurrent folds hold
separate models and dense covariance tensors, so memory needs increase with
concurrency; choose a larger allocation when needed for the assay. There is no
automatic resume of interrupted fits. Submit a new job to use parallel mode;
an already running job continues with its original execution mode.

For direct NKN CLI use, `cv --fold-workers N` enables up to `N` concurrent folds
(at most `evaluation.n_folds`). `num_threads` in the JSON is the thread count
**per worker**, so set it to `1` for five workers on five CPUs. Parallel folds
require `device: "cpu"`. The default is one worker. The shared runner also
supports importable module-level fit adapters, including the
[Kermut baseline](../baselines/kermut/README.md); local closures and lambdas are
supported only in serial execution.

## Configuration and numerical behavior

JSON configurations define `ExperimentConfig` fields. Unspecified fields use the
validated defaults in `config.py`; a custom JSON is not merged with the ESM2 +
ProteinMPNN preset. Copy that preset before editing to retain its two selected
branches, feature caches, and score sources.

The [batch preset](../configs/esm2_proteinmpnn_nkn.json) explicitly lists the current
training, enabled-kernel, and evaluation hyperparameters:

| JSON field | Controls |
| --- | --- |
| `inverse_folding_policy` | Require normalized distributions, or preserve declared sequence-conditioned ProteinMPNN values |
| `training.learning_rate`, `training.betas`, `training.epsilon` | Adam optimizer settings |
| `training.max_steps` | Maximum optimizer updates per initialization |
| `training.min_steps`, `training.patience`, `training.tolerance` | Earliest stopping step, improvement window, and minimum improvement in NLL per variant |
| `training.seeds`, `training.lengthscale_multipliers` | Paired initialization settings; keep `[0]` and `[1.0]` for one initialization |
| `training.noise_initial`, `training.noise_floor` | Initial observation-noise variance and its positive lower bound |
| `training.jitter_initial`, `training.jitter_max` | Numerical stabilization range for Cholesky factorization |
| `training.max_train_variants` | Optional training-set size guard; `null` disables the guard |
| `model.branches.<branch>.copies` | Number of copies of each primitive kernel in that branch |
| `model.branches.<branch>.widths`, `model.final_widths` | Branch and final NKN layer widths |
| `evaluation.coverage_levels` | Nominal interval coverage levels for Interval ECE |
| `evaluation.n_bins`, `evaluation.pair_block_size` | PairECE confidence bins and computation block size |
| `evaluation.n_folds` | Number of folds; the published-fold batch job requires `5` |

For example, change `training.learning_rate` to `0.0005` and
`training.max_steps` to `10000` in your copied JSON before submitting it. Other
values remain as listed in that file. Initialization lists must have equal
lengths; adding entries requests additional starts per split. NKN widths alternate
linear and product layers, ending with linear width `1`; each product width must
be half the preceding even linear width, as in `[8, 4, 4, 2, 1]`.

The CPU batch job sets `device` to `"cpu"` and `num_threads` to the Slurm CPU
allocation in serial mode, or the computed per-worker thread count with
`--parallel`, regardless of their JSON values. Extrapolation sampling remains
controlled by the `SAMPLE_RATIO` and `SPLIT_SEED` environment variables described
above; those are separate from model initialization seeds.

### Selecting NKN branches

Set `model.enabled_branches` to any nonempty combination of these names:

| Branch | Required feature cache | Representation |
| --- | --- | --- |
| `inverse_folding` | `inverse_folding` | Reference amino-acid distributions at mutation positions |
| `structure` | `structure` | Reference structural embeddings at mutation positions |
| `sequence_site` | `sequence` | Variant residue embeddings at mutation positions |
| `sequence_global` | `sequence` | Mean-pooled variant residue embeddings |

For exactly two branches using sequence and inverse-folding features:

```json
{
  "model": {
    "enabled_branches": ["sequence_global", "inverse_folding"]
  }
}
```

Add `sequence_site` to also use mutation-site sequence features; both sequence
branches share the same selected cache. Only enabled branches are constructed,
and the final NKN automatically receives that many inputs. Disabled feature
caches and default descriptors are not required. Any remaining enabled branch
still requires its actual features. Empty, duplicate and unknown branch selections
are rejected. Omit the setting to retain the original four-branch model.

`feature_selection` independently chooses the family/checkpoint for each feature
type. The NKN batch preset selects
`esm2/esm2_t33_650M_UR50D-layer33-08e4846e5371` for `sequence` and
`proteinmpnn/v_48_020-sequence_conditional-orders10-seed37-8907e6671bfb` for
`inverse_folding`. Enabling an NKN branch does not select a different pretrained
model or regenerate its representations.

The preset sets top-level `inverse_folding_policy` to `"sequence_conditional"`.
The loader verifies the declared ProteinMPNN recipe and retains its raw
probabilities and log probabilities. Each row's mass can be below one because
the recipe averages native log probabilities and removes `X` without
renormalization. NKN's Hellinger kernels retain their formulas; the
Bhattacharyya kernel's self-similarity equals the row's retained mass rather
than being forced to one. Omitting the policy keeps the default `"normalized"`
behavior for existing backbone-only and ESM-IF1 configurations.

For a direct CLI run using the same inputs as the batch job, supply its preset:

```powershell
python -m experiments.cli cv --features data/features/pretrained/A0A1I9GEU1_NEIME_Kennouche_2019 --config configs/esm2_proteinmpnn_nkn.json --folds data/DMS_ProteinGym_substitutions/cv_folds_singles_substitutions/A0A1I9GEU1_NEIME_Kennouche_2019.csv --setting singles_random --assay-index data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv --full-assay --output results/A0A1I9GEU1_NEIME_Kennouche_2019/nkn_650m_random
```

The GP mean requires at least one mean score source. `score_sources` remains
independent of kernel branches: when omitted, it uses the distinct feature caches
required by the enabled branches; when supplied, only the declared score tables
are read. The `feature_generation.cli score` command generates ESM2 masked-marginal
scores or ProteinMPNN sequence likelihoods in `scores/<family>/<recipe>/mean_score.csv`.
Select either family or both sources as independent GP mean columns. See
[ESM2 score preparation](feature_generation/README.md#esm2-scores-for-the-gp-mean)
and [ProteinMPNN score preparation](feature_generation/README.md#proteinmpnn-scores-for-the-gp-mean).
`configs/esm2_mean_score_smoke.json` selects only ESM2 representations/scores;
`configs/esm2_proteinmpnn_combined_mean_smoke.json` selects both modalities and
both scorers. Branch selection alone does not supply missing scores.

The branch selection is saved in resolved configs and GP checkpoints. Construction
uses a fixed canonical branch order, regardless of JSON ordering. Older configs
and checkpoints without the new setting retain all four branches. Prediction must
use the fitted checkpoint's branch selection; CLI prediction automatically reads
its adjacent `resolved_config.json` when no `--config` is supplied. When moving
a checkpoint, keep that config beside it or supply `--config` explicitly.

`configs/sequence_inverse_smoke.json` selects two branches for synthetic fixture
features and limits training to five updates per restart. Using this preset with
`cv` requires a fold CSV matching the selected fixture variants; the full-assay
ProteinGym fold CSV does not match the 16-variant fixture cache.

Per-branch kernel settings remain under `model.branches`. Overrides may specify
just the branches/settings being changed; omitted settings use defaults. For example:

```json
{
  "model": {
    "branches": {
      "inverse_folding": {"copies": {"hellinger_exp": 3, "hellinger_rbf": 2, "bhattacharyya": 1}, "widths": [8, 4, 4, 2, 1]},
      "structure": {"copies": {"cosine": 1, "rbf": 2, "matern32": 2}, "widths": [8, 4, 4, 2, 1]},
      "sequence_site": {"copies": {"cosine": 1, "rbf": 2, "matern32": 2}, "widths": [8, 4, 4, 2, 1]},
      "sequence_global": {"copies": {"cosine": 1, "rbf": 2, "matern32": 2}, "widths": [8, 4, 4, 2, 1]}
    },
    "final_widths": [8, 4, 4, 2, 1]
  },
  "evaluation": {"coverage_levels": [0.5, 0.8, 0.9, 0.95], "n_bins": 10},
  "training": {"max_train_variants": 5000},
  "device": "cpu"
}
```

Local primitive kernels are composed by their NKN before mutation-pair sums.
Cosine values retain their sign. Primitive amplitudes are one; every affine NKN
layer has positive softplus weights and constant-kernel biases. Copies have
independent lengthscales. No pretrained features, extra scale parameter,
weight-decay penalty, or parameter priors are optimized.

Training minimizes negative marginal likelihood per training variant. Adam uses
learning rate `1e-3`, betas `(0.9,0.999)`, epsilon `1e-8`, and no weight decay.
Target and score scales use population standard deviations (`ddof=0`). Constant
score columns become zero after centering with scale one; constant training
targets are rejected. Training inputs are detached and copied before fitting.
Lengthscale medians exclude zero distances but retain repeated occurrences;
even-sized medians average the middle two values. Training defaults to one
initialization with seed `0` and lengthscale multiplier `1.0`. Multiple starts
can be configured with matching `seeds` and `lengthscale_multipliers` lists.
NKN biases and noise start at `0.01`; the noise lower bound is `1e-4`.

The training matrix always includes jitter starting at `1e-6`, escalating by ten
through `1e-3` on factorization failure. A failed restart is ineligible for
selection. All failed restarts cause the fit to fail with recorded diagnostics.
The best evaluated checkpoint is restored, including when it precedes the last
optimizer step. Numerical jitter is not added to observation uncertainty.

The solver is dense and exact: covariance storage is quadratic and factorization
cost is cubic in training-set size. `max_train_variants` is an optional explicit
capacity guard. It never triggers a sparse approximation or silent subsampling.
PairECE calculations are blocked, but the current prediction API still returns
a dense held-out covariance matrix.

## Outputs and evaluation

Runs save resolved configuration, checkpoints, optimizer histories, selected
restart diagnostics, and CSV predictions in original and standardized units.
CLI runs also record input-cache provenance. CV outputs include per-fold
artifacts and an ID-ordered out-of-fold prediction table.

* `spearman_oof` uses combined original-unit out-of-fold means.
* `spearman_oof_standardized` pools targets and means standardized separately
  using each fold's training target mean and standard deviation, following
  Kermut's pooled Spearman aggregation pattern. The existing NKN transformations
  use population standard deviation (`ddof=0`), while Kermut's reference uses
  sample standard deviation; this is not identical numerical reproduction.
  Prediction CSVs retain both original and standardized targets and predictions.
* `avg_spearman` and `std_spearman` summarize the individual fold correlations,
  with equal fold weights and population standard deviation (`ddof=0`). Undefined
  correlations are excluded and `n_valid_spearman_folds` records the count. With
  no valid folds both summaries are JSON `null`; with one valid fold the standard
  deviation is zero. These fields appear in each CV `metrics.json` and CLI summary.
* Interval ECE is computed in each nonempty held-out fold and then equally
  averaged. For mixed runs it is computed separately within each mutation depth.
* PairECE uses the observation covariance for pairs within the same held-out
  fold, excludes exact observed and predicted ties, and then averages eligible
  fold-level values equally. Bins have equal width on `[0.5,1]`, are left-closed
  and right-open, and the last bin includes one. Empty eligible-pair sets produce
  JSON `null`, with contributing-fold and pair counts recorded.
* Uncertainty summaries average latent and observation variances in
  training-standardized units over held-out variants, including separate
  single/double summaries when present.

Mixed CV `setting_metrics` contains a complete summary for each of
`mixed_random_singles` and `mixed_random_doubles`, including subgroup per-fold
Spearman, pooled Spearman, interval ECE, PairECE, and variance summaries. PairECE
uses the covariance restricted to the subgroup; it never compares variants
across folds or mutation depths. A fold can contain no variants of one depth:
that subgroup's fold metrics are undefined and excluded from the corresponding
averages. Contributing counts are recorded separately for each metric; a
constant-target subgroup can still contribute calibration metrics even when
Spearman is undefined. Single-fold `fit` and fixed extrapolation report their
held-out metrics directly, without inventing a five-fold mean or deviation.

Labels supplied to prediction are only exported alongside predictions; they
are not used for conditioning, model selection, or automatic recalibration.

## Validation

After installing the package, run from the repository root:

```powershell
python -m unittest discover -s src/tests -v
```

The tests use independent small-matrix calculations, finite differences,
hand-crafted calibration cases, mutation-count cases, and the supplied fixture.
They check source/ID alignment, PSD and kernel composition order, gradients,
noise/jitter separation, held-out leakage, seeded restart selection, checkpoint
reload, and complete experiment artifacts. Test fitting uses short explicit
step budgets; production defaults remain unchanged.
