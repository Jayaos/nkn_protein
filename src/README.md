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
The generic NKN batch preset uses only ESM2 650M scores in its GP mean.
The five-assay array launcher defaults to `configs/esm2_proteinmpnn_nkn_two_scores.json`,
which includes both ESM2 and ProteinMPNN mean scores.

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

Without `--config`, training uses the project's 5,000-step maximum, zero-step
minimum before early stopping, 100-step comparison window, and one initialization
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
The global branch accepts residue embeddings and averages them, or loads
prepooled vectors directly. The NKN ESM2 preset saves mutation-site vectors from
centered windows and a separate full-sequence mean from fixed non-overlapping
windows, including for long references.
The Kermut preset reuses that compact cache's global vector and centered scores;
it does not use the mutation-site vectors. Optional legacy pooled ESM2 presets
average only the first 1,022 residues (all residues when shorter) and require
explicit cache selection in the experiment config.
Model-specific preprocessing and other pooling can be supplied through the
model-independent `FeatureBatch` API. The old
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
`--singles-only` on validation, prediction, or `cv --setting singles_*`,
completeness refers to all indexed single substitutions. In particular,
`--full-assay --singles-only` requires complete singles coverage. Without
`--singles-only`, training commands retain full raw-assay coverage checks before
selecting the mutation depths required by `--setting`. Mixed CV and
`single_to_double` require the full-assay inputs. Rows are not silently removed
to match an incomplete feature cache.

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
It requests 15 CPUs and no GPU. ESM2 650M layer-33 centered windows provide
`sequence_site` vectors at mutation positions, while `sequence_global` vectors
average every residue of each variant using fixed non-overlapping windows of
up to 1,022 residues, weighted by each window's residue count.
ProteinMPNN sequence-conditioned distributions
provide the third branch, `inverse_folding`. Reference Cα coordinates supply
the fourth branch, `structure_distance`, using the existing assay manifests in
`data/structures/manifests`. The GP mean uses only ESM2 650M
masked-marginal scores. The inverse-folding cache uses ten
orders with seed 37 and retains the stored 20 amino-acid values without
renormalization through `inverse_folding_policy: "sequence_conditional"`.

Prepare the selected assay inputs before running NKN, and wait for both
preparation jobs to finish successfully:

```bash
sbatch sbatch_run/run_prepare_rep/prepare_esm2.sbatch
sbatch sbatch_run/run_prepare_rep/prepare_proteinmpnn.sbatch
```

Both jobs prepare only `PIN1_HUMAN_Tsuboyama_2023_1I6C`,
`VKOR1_HUMAN_Chiasson_2020_activity`, then `BLAT_ECOLX_Jacquier_2013`.
Optional `--array=0-2` selects one of those
assays per task. They pass `--singles-only` to extraction and scoring and write to
`data/features/pretrained/singles/<assay>/`. Each single mutant is prepared once
for reuse across all singles folds and protocols. Existing full-assay caches
remain at their original paths; they are not migrated into the singles root.
For full-variant preparation, use the extraction/scoring CLI with
`--output-root data/features/pretrained` and omit `--singles-only` from both
stages. See the [preparation guide](feature_generation/README.md).
The NEIME and FKBP3 run examples below require their previously prepared caches;
the current preparation selection covers the three additional assays.

ESM2 preparation defaults to `configs/extraction/esm2_650m_centered.json`, with
192 GiB host RAM and a 32 GiB compact-output guard. It stores only the mutation
vectors and global means; it does not allocate every residue for every variant.
The new `-mean-all-nonoverlap1022-centered-sites` cache must be prepared before
using the NKN or Kermut preset. Existing prefix-mean and residue recipes remain
readable and separate. Centered masked-marginal scores retain their existing
recipe, so matching completed score caches can be reused.
`--resume` reuses completed assay caches; interrupted assays restart extraction.
Both ESM2 stages support sites anywhere in long references. Scoring uses
`scoring_window: "centered"`: it masks each required WT position inside a window
of up to 1,022 residues and sums mutant-minus-WT log probabilities per variant.
Short references use full WT context. Both NKN and Kermut presets select the
`-centered1022` score cache; older full-reference score caches remain separate.
ProteinMPNN's presets now use `max_length: null`, removing its sequence-length
guard while retaining full-reference structure requirements. Its preparation
job scores one complete variant per batch; GPU memory and runtime still limit
practical input sizes. ESM-IF1 retains its 1,022-residue limit.

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

To run five assays as one Slurm array, submit from the repository root on the cluster:

```bash
sbatch sbatch_run/run_nkn/run_nkn_assays.sbatch

# Run only FKBP3 or NEIME:
sbatch --array=4 sbatch_run/run_nkn/run_nkn_assays.sbatch
sbatch --array=3 sbatch_run/run_nkn/run_nkn_assays.sbatch

# Run both FKBP3 and NEIME:
sbatch --array=3,4 sbatch_run/run_nkn/run_nkn_assays.sbatch

# Optionally run the 15 folds within each assay concurrently as well:
sbatch sbatch_run/run_nkn/run_nkn_assays.sbatch singles --parallel

# Or limit the array to two concurrent assays:
sbatch --array=0-4%2 sbatch_run/run_nkn/run_nkn_assays.sbatch
```

The array has a fixed mapping:

| Array index | Assay |
| --- | --- |
| 0 | `PIN1_HUMAN_Tsuboyama_2023_1I6C` |
| 1 | `VKOR1_HUMAN_Chiasson_2020_activity` |
| 2 | `BLAT_ECOLX_Jacquier_2013` |
| 3 | `A0A1I9GEU1_NEIME_Kennouche_2019` |
| 4 | `FKBP3_HUMAN_Tsuboyama_2023_2KFV` |

Each task defaults to `data/features/pretrained/singles/<assay>` and
[`configs/esm2_proteinmpnn_nkn_two_scores.json`](../configs/esm2_proteinmpnn_nkn_two_scores.json).
The preset uses `[4, 2, 1]` in every enabled branch and the final NKN, and includes
both ESM2 650M masked-marginal and ProteinMPNN sequence-likelihood scores in the GP
mean. Each score is standardized using training variants and has its own learned
coefficient. Tasks default to all three singles protocols (15 fits per assay).
The requested 15 CPUs, 32 GB RAM, and 24-hour limit apply to each assay task.
Without `--parallel`, folds within an assay run sequentially; the array allows
different assays to run concurrently. Settings, `--parallel`, and `--config PATH`
are forwarded to the shared launcher; `FEATURES` and `NKN_CONFIG` overrides remain
available. Prepare features, both score caches, and the structure
manifest for every selected assay before submission.

Logs are `nkn_assays-<array-job-id>_<index>.out` in the submission directory;
each task writes its own
`results/<assay>/nkn_esm2_proteinmpnn_<selection>_<task-job-id>/` directory.
Slurm's `%A` and `%a` log tokens identify the array and task respectively;
see [Slurm job arrays](https://slurm.schedmd.com/job_array.html).

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
Add `--experiment-suffix NAME` to either the single-assay or array sbatch script
to save under `results/NAME/<assay>/<model-run>/` instead. Alternatively, set
`EXPERIMENT_SUFFIX=NAME` when submitting; the command-line option takes precedence.
The suffix must be one folder name. Omitting it preserves the existing paths.
Use the same suffix for NKN and Kermut submissions belonging to one experiment:

```bash
sbatch sbatch_run/run_nkn/run_nkn_assays.sbatch singles --experiment-suffix comparison_v1
sbatch sbatch_run/run_kermut/run_kermut_assays.sbatch singles --experiment-suffix comparison_v1
PYTHONPATH=src python -m reporting --results results/comparison_v1 --output results/comparison_v1/aggregate
```

CV results are saved under `cv/<setting>/`; extrapolation is under
`fit/single_to_double/`. For `all`, both mixed-domain results are in
`cv/mixed_random_singles/metrics.json` under `setting_metrics`.
The run saves byte-for-byte `folds/singles.csv` and/or `folds/multiples.csv`
snapshots as needed, plus a `fold_manifest.json` with source paths, SHA-256
hashes and selected fold sizes. When extrapolation is requested, the manifest also
records `sample_ratio` and `split_seed`.
The top-level `config.json` records the expanded configuration actually used,
including defaults and the job's CPU and thread settings.

For the generic launcher, `ASSAY` chooses the assay. Singles selections default to
`data/features/pretrained/singles/<assay>` and validate all indexed singles.
Mixed selections, `all`, and `single_to_double` retain
`data/features/pretrained/<assay>` and full-assay validation. `FEATURES` overrides
the selected default cache. `SINGLES_FOLDS_CSV` and
`MULTIPLES_FOLDS_CSV` override the corresponding published files. `SAMPLE_RATIO`
(default `0.8`) controls the fraction of singles used for extrapolation training;
`SPLIT_SEED` (default `2024`) selects the reproducible sample when the ratio is
below `1.0`. Both are used only when extrapolation is requested and are then
recorded in the batch manifest; neither changes published CV folds. Relative paths resolve from the repository
root. The former positional `cv`/`fit` modes and `FOLDS_CSV` variable are rejected.
Existing pretrained features and scores are reused without extraction.

### Optional validation-Spearman stopping

The default `training_nll` mode trains on four folds and selects checkpoints and
restarts by training marginal NLL. To use three training folds, one validation
fold, and one outer test fold, submit:

```bash
sbatch sbatch_run/run_nkn/run_nkn_assays.sbatch singles --parallel --stopping-criterion validation_spearman
```

The same flags work with `run_nkn_esm2_proteinmpnn.sbatch` and the direct NKN
`experiments.cli cv` command. This option supports five-fold `singles_random`,
`singles_modulo`, and `singles_contiguous` CV. For outer test fold `k`, validation
uses fold `(k + 1) % 5`; the other three folds train the model. Supplied fold
assignments are preserved, so a position never crosses partitions in positional
schemes. This trains on about 60% of variants, validates on 20%, and tests on 20%.

Training still optimizes marginal NLL. Target/score standardization and kernel
initialization use only the three training folds. Validation labels select the
checkpoint and restart with the highest validation Spearman; outer test labels
never select models. No refit on the validation fold is performed.

Defaults are `--validation-interval 100 --patience 100 --min-steps 0`.
Validation evaluates step 0, every interval, and the final maximum step if it
falls between intervals. Step 0 is eligible for selection. Patience counts
**optimizer steps**, not checks: with both interval and patience set to 100,
one evaluation without a strictly higher score stops training. Ties keep the
earlier checkpoint and do not reset patience. Stopping is checked only at
validation evaluations, after `min_steps`; checkpoint selection starts at step 0
even with a later minimum. The highest-scoring checkpoint is restored. The NLL
`tolerance` setting is unused in validation mode. Constant validation targets or
runs with no defined validation score fail explicitly rather than reverting to NLL.

These CLI flags override the corresponding `training` JSON fields:
`stopping_criterion`, `validation_interval`, `patience`, and `min_steps`.
The default minimum is now 0 in both modes and both NKN presets. To reproduce
the previous four-fold training scheme with its 1,000-step minimum:

```bash
sbatch sbatch_run/run_nkn/run_nkn_assays.sbatch singles --parallel --stopping-criterion training_nll --min-steps 1000
```

Each validation-mode fold records `partition_assignments.csv`,
`partition_summary.json`, and a streamed `validation_spearman.csv`.
`fit_summary.json`, restart summaries, and checkpoints record the selected step,
best validation Spearman, and training NLL at the selected checkpoint.
`best_nll_per_variant` remains the lowest training NLL reached by the selected
restart and can differ from `selected_nll_per_variant` in validation mode.
Validation scores are also saved in `history.json` and shown in progress logs.
Serial and parallel folds use the same partition and selection rules.

### Optional held-out Spearman diagnostics

Add `--held-out-interval 100` after either NKN sbatch script name to record
held-out Spearman every 100 optimizer updates:

```bash
sbatch sbatch_run/run_nkn/run_nkn_assays.sbatch singles --parallel --held-out-interval 100
# Or for the assay selected by ASSAY:
sbatch sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch singles --held-out-interval 100
```

The flag overrides `training.held_out_interval` in the JSON. Diagnostics are
disabled by default; use `--held-out-interval 0` to disable them explicitly.
The direct NKN `experiments.cli cv` and `fit` commands accept the same option.

Each initialization records step 0, every interval, and its final stopping step.
Rows contain `seed`, `step`, training `nll` and `nll_per_variant`,
`held_out_spearman`, `n_held_out`, and `held_out_error`. They are flushed as training
runs to `cv/<setting>/fold_<n>/held_out_spearman.csv`, or
`fit/single_to_double/held_out_spearman.csv` for extrapolation, under the run
directory. The same diagnostic fields appear in `history.json` and the checkpoint
history. Progress logs also show Spearman, including when settings/folds run in
parallel; `--quiet` suppresses console progress but still records the CSV.

Diagnostics evaluate the current model using the existing held-out fold (doubles
for single-to-double extrapolation). Mixed settings evaluate the requested
single- or double-mutant domain. Empty, one-variant, or constant-rank samples
have undefined Spearman, saved as an empty CSV cell / JSON `null`. Numerical
diagnostic failures are recorded in `held_out_error` without rejecting the fit.
Only predictive means are computed for diagnostics, but enabling them adds
prediction time. Held-out scores never change gradients, early stopping,
checkpoint selection, or restart selection. Those use training NLL by default,
or the separate validation partition in `validation_spearman` mode.
The final selected checkpoint may differ from the last diagnostic step.

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
ProteinMPNN preset. Copy that preset before editing to retain its four selected
branches, feature caches, and ESM2 score source.

The [batch preset](../configs/esm2_proteinmpnn_nkn.json) explicitly lists the current
training, enabled-kernel, and evaluation hyperparameters:

| JSON field | Controls |
| --- | --- |
| `inverse_folding_policy` | Require normalized distributions, or preserve declared sequence-conditioned ProteinMPNN values |
| `structure_manifest_dir` | Directory of `<assay_id>.json` reference structure manifests for `structure_distance`; relative paths resolve from the repository root |
| `training.learning_rate`, `training.betas`, `training.epsilon` | Adam optimizer settings |
| `training.max_steps` | Maximum optimizer updates per initialization |
| `training.stopping_criterion` | `training_nll` (default) or `validation_spearman` (five-fold singles CV) |
| `training.min_steps`, `training.patience` | Earliest stopping step (default `0`) and patience in optimizer steps (default `100`) |
| `training.tolerance` | Minimum improvement in NLL per variant over the patience window; used only by `training_nll` |
| `training.validation_interval` | Validation-Spearman evaluation interval in optimizer steps (default `100`) |
| `training.held_out_interval` | NKN diagnostic Spearman interval in optimizer steps; `0` disables it (default) |
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
be half the preceding even linear width, as in `[4, 2, 1]`.
Both `esm2_proteinmpnn_nkn.json` and `esm2_proteinmpnn_nkn_two_scores.json`
use `[4, 2, 1]` for every enabled branch and the final NKN, with one copy of
each selected primitive kernel.

For additive information branches followed by a deeper final NKN, select
[`esm2_proteinmpnn_nkn_two_scores_additive_branches.json`](../configs/esm2_proteinmpnn_nkn_two_scores_additive_branches.json):

| Component | Widths | Operation |
| --- | --- | --- |
| Each of the four enabled branches | `[1]` | One learned nonnegative weighted sum of primitive kernels, plus a positive bias |
| Final NKN | `[8, 4, 4, 2, 1]` | Linear(8), Product(4), Linear(4), Product(2), Linear(1) |

The branch and final widths are independently configurable through
`model.branches.<branch>.widths` and `model.final_widths`. A single width `[1]`
creates an additive branch with no product layer. The two product layers in
the final NKN allow terms up to degree four in the branch kernels, including
three-way interactions and powers of individual branches. Additive branches
do not prohibit such powers in the final NKN. All layers and primitive
parameters are still trained jointly.

This preset retains the two-score preset's features, mean scores, training,
and evaluation settings. Submit the assay array with:

```bash
sbatch sbatch_run/run_nkn/run_nkn_assays.sbatch singles --parallel \
  --config configs/esm2_proteinmpnn_nkn_two_scores_additive_branches.json
```

The same `--config` option works with the single-assay launcher. Use
`configs/esm2_proteinmpnn_nkn_two_scores.json` to select the original
`[4, 2, 1]` branches and final NKN. Stopping overrides work with either preset.

The CPU batch job sets `device` to `"cpu"` and `num_threads` to the Slurm CPU
allocation in serial mode, or the computed per-worker thread count with
`--parallel`, regardless of their JSON values. Extrapolation sampling remains
controlled by the `SAMPLE_RATIO` and `SPLIT_SEED` environment variables described
above; those are separate from model initialization seeds.

### Selecting NKN branches

Set `model.enabled_branches` to any nonempty combination of these names:

| Branch | Required input | Representation |
| --- | --- | --- |
| `inverse_folding` | `inverse_folding` | Reference amino-acid distributions at mutation positions |
| `structure` | `structure` | Reference structural embeddings at mutation positions |
| `sequence_site` | `sequence` | Variant residue embeddings at mutation positions |
| `sequence_global` | `sequence` | Mean-pooled variant residue embeddings |
| `structure_distance` | Reference structure manifest and PDB/mmCIF | Reference Cα coordinates at mutation positions, in angstroms |

The ESM2 + ProteinMPNN NKN preset enables four branches:

```json
{
  "model": {
    "enabled_branches": ["sequence_site", "sequence_global", "inverse_folding", "structure_distance"]
  }
}
```

`sequence_site` accepts a full residue cache or the compact centered-site cache;
the global-only pooled ESM2 presets cannot supply it. Both sequence branches
share the same selected cache. For compact caches, top-level `pooling: "mean"`
loads the stored global average and does not average the mutation-site vectors.
The current NKN preset covers the full sequence; older compact recipes cover
only the prefix.
For full residue caches, it averages all residues. Only enabled branches are constructed,
and the final NKN automatically receives that many inputs. Disabled feature
caches and default descriptors are not required. Any remaining enabled branch
still requires its actual features. Empty, duplicate and unknown branch selections
are rejected. Omit the setting to retain the original four-branch model.

`structure_distance` uses its own NKN with exponential (`exp`), RBF (`rbf`),
and Matérn-3/2 (`matern32`) primitives on Euclidean Cα distances. Each primitive
copy has an independent positive lengthscale in angstroms. These kernels are
invariant to rigid translation and rotation of the reference coordinates.
The branch NKN acts on mutation pairs before their covariances are summed into
variant pairs; the final NKN then combines this branch with the other enabled
branches. Copy counts and layer widths use the same configuration fields as
the existing branches.

The loader validates full reference coverage, residue mapping, structure checksum,
and any selected structure-conditioned cache provenance. The structure identity
is saved in the feature signature and checked during prediction. This branch
does not require a learned `structure` embedding cache. The separate `structure`
branch continues to represent learned embeddings; enabling it as well gives
the five-branch architecture described in the method. Existing configurations
and checkpoints keep their previous branch selections. To run the previous
three-branch ESM2 + ProteinMPNN preset, remove `structure_distance` from
`model.enabled_branches`.
Older configs and checkpoints using `physical_distance` are accepted on load;
new configurations and checkpoints use the name `structure_distance`.

For `BRCA2_HUMAN_Erwood_2022_HEK293T`, the experiment CLI and batch launcher
automatically remove `structure_distance` before loading features. Resolved run
configs and checkpoints record the remaining branches. The three independently
folded ProteinGym structures do not share a validated coordinate frame.
Programmatic callers should use `config = config.for_assay(assay_id)` before
loading their feature batch; experiment runners reject a BRCA2 batch that still
contains this branch. Other assays retain their configured branch selection.

BRCA2 inverse-folding extraction runs each fragment separately and stitches its
probability rows at reference positions 1–1000, 1001–2085, and 2086–2832. The
full reference remains 3418 residues long. All current assay mutation sites lie
inside the covered region; requesting an uncovered site raises an error. The
default ESM2 mean scores remain applicable. See
[fragment preparation](feature_generation/README.md#brca2-fragment-preparation)
for extraction commands and cache details.

`feature_selection` independently chooses the family/checkpoint for each feature
type. The NKN batch preset selects
`esm2/esm2_t33_650M_UR50D-layer33-08e4846e5371-mean-all-nonoverlap1022-centered-sites` for `sequence` and
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
python -m experiments.cli cv --features data/features/pretrained/singles/A0A1I9GEU1_NEIME_Kennouche_2019 --config configs/esm2_proteinmpnn_nkn.json --folds data/DMS_ProteinGym_substitutions/cv_folds_singles_substitutions/A0A1I9GEU1_NEIME_Kennouche_2019.csv --setting singles_random --assay-index data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv --full-assay --singles-only --output results/A0A1I9GEU1_NEIME_Kennouche_2019/nkn_650m_random
```

The GP mean requires at least one mean score source. `score_sources` remains
independent of kernel branches: when omitted, it uses the distinct feature caches
required by the enabled branches; when supplied, only the declared score tables
are read. The `feature_generation.cli score` command generates ESM2 masked-marginal
scores or ProteinMPNN sequence likelihoods in `scores/<family>/<recipe>/mean_score.csv`.
Select either family or both sources as independent GP mean columns. See
[ESM2 score preparation](feature_generation/README.md#esm2-scores-for-the-gp-mean)
and [ProteinMPNN score preparation](feature_generation/README.md#proteinmpnn-scores-for-the-gp-mean).
Branch selection alone does not supply missing scores.
For a `structure_distance`-only model, select at least one existing score source
explicitly; coordinates do not provide a zero-shot score.

The branch selection is saved in resolved configs and GP checkpoints. Construction
uses a fixed canonical branch order, regardless of JSON ordering. Older configs
and checkpoints without the new setting retain all four branches. Prediction must
use the fitted checkpoint's branch selection; CLI prediction automatically reads
its adjacent `resolved_config.json` when no `--config` is supplied. When moving
a checkpoint, keep that config beside it or supply `--config` explicitly.

Per-branch kernel settings remain under `model.branches`. Overrides may specify
just the branches/settings being changed; omitted settings use defaults. For example:

```json
{
  "model": {
    "branches": {
      "inverse_folding": {"copies": {"hellinger_exp": 3, "hellinger_rbf": 2, "bhattacharyya": 1}, "widths": [8, 4, 4, 2, 1]},
      "structure": {"copies": {"cosine": 1, "rbf": 2, "matern32": 2}, "widths": [8, 4, 4, 2, 1]},
      "sequence_site": {"copies": {"cosine": 1, "rbf": 2, "matern32": 2}, "widths": [8, 4, 4, 2, 1]},
      "sequence_global": {"copies": {"cosine": 1, "rbf": 2, "matern32": 2}, "widths": [8, 4, 4, 2, 1]},
      "structure_distance": {"copies": {"exp": 1, "rbf": 2, "matern32": 2}, "widths": [8, 4, 4, 2, 1]}
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
Each fit caches their fixed pairwise distances, cosine similarities, and
Bhattacharyya products in float64, sharing distances between primitive kernels.
For this prepared training batch, branch and final NKNs evaluate only the upper
triangle and mirror the result, including each diagonal once. The inverse-folding
branch additionally evaluates pairs of exactly identical feature rows only once
per training step, including repeated ProteinMPNN probabilities. Equality is
checked on feature vectors, not residue positions. Values are mapped back to all
mutation occurrences before the unchanged mutation aggregation and final NKN;
autograd accumulates every repeated pair's contribution. Initialization retains
the full occurrence-weighted lengthscale median. These optimizations reduce
repeated pair arithmetic without changing the model or approximating the dense
GP; floating-point reduction order can produce small rounding differences.
Lengthscales, NKN parameters, learned covariances, and Cholesky factors still
update at every training step. Full geometry is retained for fallback, alongside
packed triangular geometry and index maps. These caches use quadratic memory,
are excluded from checkpoints, and are bypassed for different or modified inputs
and for features requiring gradients. Direct kernel calls remain uncached unless
`prepare_training_geometry(train)` is called explicitly.
For numerical comparisons,
`prepare_training_geometry(train, use_unique_features=False)` retains triangular
evaluation without feature deduplication, while `use_triangle=False` keeps
geometry caching but evaluates every NKN pair. Both optimizations are automatic
in normal fitting; no JSON or Slurm option is required. Clearing caches or moving
the kernel to a different device/dtype discards all prepared geometry and maps.
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

Saved interval and pairwise calibration metrics can be plotted with
`sbatch sbatch_run/run_plotting/plot_nkn_calibration.sbatch`. By default, PNG/PDF
figures and a JSON plotting summary go to each `results/<protein>/<model-run>/plots/`
directory. See [plotting instructions](../sbatch_run/run_plotting/README.md) for
custom output locations, assay selection, dependencies, and aggregation details.

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
