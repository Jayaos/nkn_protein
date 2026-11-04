# Kermut baseline

An independent PyTorch implementation of the full Kermut model, sharing this
project's cached inputs, ProteinGym splits, evaluation and result formats. The
top-level `kermut/` checkout is reference material only: this baseline does not
import it or require GPyTorch. No GPU is required after feature extraction.

## Defaults and publication setup

The kernel and optimization follow the publication's
[reproduction implementation](https://github.com/petergroth/kermut/tree/e6500221b5159a1896272bfc4a3c6e3896eb0027)
and [paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/34547650b2ca69d91f3b3c3ae8b21962-Paper-Conference.pdf):

- A learned weighted sum of the mutation kernel and an RBF kernel on mean-pooled
  sequence embeddings. The mutation kernel sums over all pairs of substituted
  sites, combining Hellinger distances between amino-acid distributions,
  distances between reference CA atoms, and differences in the substituted
  amino acids' log probabilities. Coordinates retain their Angstrom units.
- A learned constant plus a learned coefficient for each raw mean-score source.
  Scores and embeddings are not standardized. Each training split standardizes
  its targets using its own mean and population standard deviation (`ddof=0`).
  This matches NKN and ProteinNPT's supervised ProteinGym preprocessing; it
  deliberately differs from the original Kermut implementation's sample
  standard deviation (`ddof=1`).
- One initialization, 150 AdamW updates, learning rate 0.1, weight decay 0.01,
  betas 0.9/0.999, epsilon 1e-8, and float32. Optimization minimizes negative
  log marginal likelihood minus the Half-Cauchy log prior on noise variance,
  divided by the training set size. The prior scale is 0.1.
- The saved state is the final iterate. There is no early stopping or selection
  of a best initialization. Seed 2024 is recorded; current initialization and
  dense optimization are deterministic. Numerical jitter is added only if
  Cholesky factorization needs it and is recorded in training history.

The default selects ESM2 **650M** layer-33 embeddings and masked-marginal scores,
plus ProteinMPNN `v_48_020` distributions prepared with Kermut's extraction recipe.
It reuses the default ESM2 preparation outputs selected by NKN: the compact
`-mean-all-nonoverlap1022-centered-sites` cache and `-centered1022` scores.
Kermut reads only the saved global embedding from that compact cache; it does
not use its mutation-site vectors. Global embeddings average every variant
residue using fixed non-overlapping windows of at most 1,022 residues. Scores
use WT windows centered on each masked position. For the short NEIME and FKBP3
references, both the current and legacy prefix recipes use full-sequence
context. For longer references, the current global vector includes residues
beyond 1,022, whereas the legacy prefix vector excludes them; window boundaries
still limit attention context. ProteinMPNN's presets have no sequence-length
cap. Registered CAS9/P53 mappings and ZIKV/HCV domain structures are supported;
BRCA2 uses its independently prepared fragment features with the distance factor disabled.
The GP mean combines ESM2 and ProteinMPNN variant scores; the publication's
mean uses ESM2 alone. ProteinMPNN variant scoring uses one order and seed 0.
For each reference position, ProteinMPNN conditions on the backbone and the
other wild-type residues, using ten randomized decoding orders with seed 37.
Extraction averages log probabilities, exponentiates, and removes the unknown
amino-acid category `X` without renormalizing. The resulting 20 values can sum
to less than one; the loader and kernel preserve those values.

This matches the ProteinMPNN extraction recipe for the pipeline's supported
single-chain structures. It does not establish numerical
reproduction of the publication: matching structure inputs, model/runtime versions, assay folds,
and benchmark aggregation still matters. The baseline implementation also
needs comparison against the authors' released predictions before claiming
the same reported results. The data adapter verifies reference structure
mapping and its consistency with inverse-folding cache provenance.

## Configuration

Start from [`configs/kermut_esm2_proteinmpnn.json`](../../configs/kermut_esm2_proteinmpnn.json).
`feature_selection.sequence` chooses embeddings; `feature_selection.inverse_folding`
chooses amino-acid distributions. `structure_manifest_dir` locates per-assay
`<assay_id>.json` manifests and resolves relative to the repository root.

`score_sources` independently selects the GP mean inputs. The shared JSON
default combines both sources:

```json
"score_sources": [
  "scores/esm2/esm2_t33_650M_UR50D-masked_marginals-float32-08e4846e5371-centered1022",
  "scores/proteinmpnn/v_48_020-sequence_log_likelihood-orders1-seed0-8907e6671bfb"
]
```

Both Kermut presets select recipe-only `v_48_020-*` ProteinMPNN feature and score
directories. ProSST cache names also omit the structure collection; configuration
and saved provenance identify each cache's input structures.

The Slurm launchers prefer the catalog at
`data/DMS_ProteinGym_substitutions/DMS_substitutions.csv`, falling back to
`data/DMS_substitutions.csv` and then the legacy `raw_substitutions` location.
Set `ASSAY_INDEX` to explicitly select a catalog. The selected path is printed
in the job log and used for validation, CV, and extrapolation.

Keeping only the first entry uses the publication's ESM2-only mean; keeping
only the second uses a ProteinMPNN-only mean. Multiple entries
give `mean(x) = intercept + sum(coefficient[j] * score[j](x))`; all coefficients
start at one and are trained jointly with the kernel. Scores remain in their
raw units. Duplicate sources and unknown configuration fields are rejected.

The same assay restrictions as NKN are resolved automatically before feature
loading, Slurm preflight, and saving run configurations. For
`A0A140D2T1_ZIKV_Sourisseau_2019`, `POLG_HCVJF_Qi_2014`, and
`BRCA2_HUMAN_Erwood_2022_HEK293T`, unavailable ProteinMPNN mean scores are
removed from the selected sources; the default preset therefore uses ESM2
scores alone. A preset containing only ProteinMPNN scores is rejected with a
request to configure ESM2 scores. Other missing caches remain errors.

For BRCA2 only, `model.use_structure_distance` resolves to `false`. The
mutation kernel retains its Hellinger and substituted-amino-acid probability
factors, but omits the spatial factor. No shared-frame coordinates are loaded
or fabricated, and the distance parameter is not trained. ZIKV/HCV keep the
distance factor using their mapped domain coordinates. CAS9 and the three P53
assays retain both scores and the distance factor. The effective settings are
saved in run configurations and checkpoints and preserved during prediction.
Existing complete-structure configurations and checkpoints default to
`use_structure_distance: true`.

`model.raw_*` values are initial unconstrained kernel parameters. Positive
parameters use softplus and `raw_alpha` uses sigmoid. `training` controls the
optimizer, numerical precision and noise prior. `training.max_train_variants`
can optionally cap dense-GP training size; it is unset by default. Training
uses quadratic memory and cubic Cholesky work in the number of variants.
`evaluation` has the same coverage levels, pairwise bins and five-fold setting
as NKN. Interval ECE defaults to 11 nominal coverage levels from `0.0` through
`1.0` in steps of `0.1`, including both endpoints. The complete accepted schema
is in [`config.py`](config.py).

## Run on CPU

Prepare the selected caches first. On the cluster, submit both preparation
jobs and wait for them to finish successfully before submitting Kermut:

```bash
sbatch sbatch_run/run_prepare_rep/prepare_esm2.sbatch
sbatch sbatch_run/run_prepare_rep/prepare_proteinmpnn.sbatch
```

Both jobs prepare only `PIN1_HUMAN_Tsuboyama_2023_1I6C`,
`VKOR1_HUMAN_Chiasson_2020_activity`, then `BLAT_ECOLX_Jacquier_2013`.
Optional `--array=0-2` selects one of those
assays per task. They pass `--singles-only` for extraction and scoring and save to
`data/features/pretrained/singles/<assay>/`, reusing each single mutant's outputs
across folds. Existing full-assay caches are not moved or modified. For mixed or
extrapolation experiments, prepare full-variant caches through the CLI with
`--output-root data/features/pretrained` and omit `--singles-only` consistently
from extraction and scoring.
The NEIME run examples below require its previously prepared cache; the current
preparation selection covers the three additional assays.

The ESM2 preparation default supplies both the NKN and Kermut presets. Existing
matching centered caches can be reused without preparing a separate pooled cache.
The legacy prefix recipe remains optional: prepare with
`--config configs/extraction/esm2_650m.json`, then explicitly select
`esm2/esm2_t33_650M_UR50D-layer33-08e4846e5371-mean-first1022` in
`feature_selection.sequence` and its full-reference ESM2 score source (without
the `-centered1022` suffix) in a copied Kermut experiment JSON. Pass that JSON
to the run's `--config`. This scoring recipe requires references no longer than
1,022 residues. Both cache recipes can coexist and share the downloaded checkpoint.

The ProteinMPNN job writes the sequence-conditioned cache selected by Kermut
and retains the existing separate variant-score recipe. It scores one variant
per batch to reduce GPU memory; long references can still exceed available
memory or runtime. Existing backbone-only
outputs cannot satisfy Kermut's cache selection. To extract only the Kermut
distributions from prepared model and structure assets, run:

```bash
PYTHONPATH=src python -m feature_generation.cli extract --config configs/extraction/proteinmpnn_v48_020_kermut.json --assay A0A1I9GEU1_NEIME_Kennouche_2019 --singles-only --device cuda --resume
```

From the repository root in PowerShell, activate the existing environment and
make the shared source packages available:

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path + [IO.Path]::PathSeparator + $env:PYTHONPATH
python -m baselines.kermut.cli --help
```

For the current assay's five-fold random single-mutant evaluation:

```powershell
python -m baselines.kermut.cli cv --features data/features/pretrained/singles/A0A1I9GEU1_NEIME_Kennouche_2019 --config configs/kermut_esm2_proteinmpnn.json --folds data/DMS_ProteinGym_substitutions/cv_folds_singles_substitutions/A0A1I9GEU1_NEIME_Kennouche_2019.csv --setting singles_random --assay-index data/DMS_ProteinGym_substitutions/DMS_substitutions.csv --full-assay --singles-only --output results/A0A1I9GEU1_NEIME_Kennouche_2019/kermut_random
```

Output directories must be new or empty. `validate` checks inputs without
training. `predict --checkpoint <fold_dir>/checkpoint.pt` reloads a saved
posterior, supports unlabeled cached variants and defaults to the adjacent
resolved configuration. Prediction verifies assay, reference, feature sources,
structure identity and ordered score sources against the checkpoint.

To run the 57 assays with **1-1,000 single mutants (inclusive)** in the local
ProteinGym catalog, use the dedicated small-assay launcher:

```bash
sbatch sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn_small_assays.sbatch

# At most four assays at once; parallelize fits within each assay:
sbatch --array=0-56%4 sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn_small_assays.sbatch singles --parallel

# Group outputs with related NKN runs:
sbatch sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn_small_assays.sbatch singles --experiment-suffix comparison_v1

# Show the task-to-assay mapping without submitting a job:
bash sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn_small_assays.sbatch --help
```

This delegates to `run_kermut_esm2_proteinmpnn.sbatch`, preserving its model,
configuration, feature/score inputs, fold handling, and result paths. It requests
15 CPUs, 16 GB RAM, and 24 hours per assay. All arguments, including `--config`,
`--parallel`, and `--experiment-suffix`, are forwarded. The selection is a fixed
catalog snapshot in catalog order; update the list and array bounds together
when updating the catalog. Assays with no single mutants are excluded. FKBP3
has 1,237 single mutants and is therefore outside this subset.

For the same small-assay setup using **ProSST instead of ESM2**:

```bash
sbatch sbatch_run/run_kermut/run_kermut_prosst_proteinmpnn_small_assays.sbatch

sbatch --array=0-56%4 sbatch_run/run_kermut/run_kermut_prosst_proteinmpnn_small_assays.sbatch singles --parallel --experiment-suffix comparison_v1
```

The preset `configs/kermut_prosst_proteinmpnn.json` replaces the ESM2 global
embeddings and masked-marginal scores with ProSST-2048 layer-12 embeddings and
ProSST scores from `configs/extraction/prosst_2048_singles.json`. Kermut uses the
pooled global vectors from that cache. ProteinMPNN features/scores, reference
coordinates, training hyperparameters, evaluation settings, the 57 assay indices,
and Slurm resources match the ESM2 version. Prepare the ProSST caches before
submission; the default singles scope is `data/features/pretrained/singles`.
The presets select `prosst/ProSST-2048-...` and `proteinmpnn/v_48_020-...`
feature paths, with the same recipe-only naming under `scores/`. If prepared
ProSST or ESM-IF1 caches still have `protssn_proteingym--` prefixes, complete
`sbatch sbatch_run/run_prepare_rep/migrate_structure_cache_names.sbatch` before
training. The runner reads custom JSON selections exactly as supplied; it does
not infer a different cache from its structure collection. A custom ESM-IF1
selection uses `esm_if1/esm_if1_gvp4_t16_142M_UR50-...` and needs matching
`structure_manifest_dir` coordinates when the structure distance factor is enabled.
Full-assay selections require matching full-assay caches and double mutants.
Results use `kermut_prosst_proteinmpnn_<selection>_<job-id>` directories, allowing
the summarizer to distinguish configurations. The shared runner accepts
`KERMUT_RUN_PREFIX` to customize this prefix, which must start with `kermut_`.
`--config`, `KERMUT_CONFIG`, `--parallel`, and `--experiment-suffix` work as usual.

For the same 57 assays using **MSA Transformer embeddings and ProteinMPNN**:

```bash
sbatch sbatch_run/run_kermut/run_kermut_msa_proteinmpnn_small_assays.sbatch
sbatch --array=0-56%4 sbatch_run/run_kermut/run_kermut_msa_proteinmpnn_small_assays.sbatch singles --parallel --experiment-suffix comparison_v1
```

The preset `configs/kermut_msa_proteinmpnn.json` uses MSA Transformer global
embeddings and masked-marginal scores from `configs/extraction/msa_transformer.json`,
with ProteinMPNN inverse-folding features and mean scores. Prepare both models'
caches under `data/features/pretrained/singles/<assay>` before submitting.
ProteinMPNN inputs, reference coordinates, training settings, and Slurm resources
match the ESM2 launcher. Results use `kermut_msa_proteinmpnn_<selection>_<job-id>`.
The usual `--config`, `KERMUT_CONFIG`, `KERMUT_RUN_PREFIX`, `--parallel`, and
`--experiment-suffix` overrides are supported. BRCA2 retains the MSA score while
disabling the ProteinMPNN score and structure distance factor, as in the other
Kermut presets.

For MSA Transformer embeddings with **ESM-IF1 inverse-folding features and both
models' mean scores**, use:

```bash
sbatch sbatch_run/run_kermut/run_kermut_msa_esmif_small_assays.sbatch
sbatch --array=0-56%4 sbatch_run/run_kermut/run_kermut_msa_esmif_small_assays.sbatch singles --parallel --experiment-suffix comparison_v1
```

The preset `configs/kermut_msa_esmif.json` selects the prepared MSA Transformer
global embeddings and masked-marginal scores, plus ESM-IF1 normalized WT-prefix
probabilities and native sequence-log-likelihood scores. Cache names match
`configs/extraction/msa_transformer.json` and `configs/extraction/esm_if1_142m.json`
under `data/features/pretrained/singles/<assay>`. Structure-distance coordinates
come from `data/structures/protssn_proteingym/manifests` to match ESM-IF1's recorded
backbone. Training settings, evaluation, the 57-assay mapping, and resources
match the ESM2/ProteinMPNN reference. BRCA2 disables the structure-distance
factor while retaining both mean scores. Results use
`kermut_msa_esmif_<selection>_<job-id>`; the standard config, result-prefix,
parallelism, and experiment-suffix overrides remain available.

For **ProSST-2048 embeddings and ESM-IF1 inverse-folding features**, with both
models' mean scores:

```bash
sbatch sbatch_run/run_kermut/run_kermut_prosst_esmif_small_assays.sbatch
sbatch --array=0-56%4 sbatch_run/run_kermut/run_kermut_prosst_esmif_small_assays.sbatch singles --parallel --experiment-suffix comparison_v1
```

The preset `configs/kermut_prosst_esmif.json` selects ProSST layer-12 pooled global
embeddings and masked-marginal scores from `configs/extraction/prosst_2048_singles.json`,
plus ESM-IF1 normalized WT-prefix probabilities and native sequence-log-likelihood
scores from `configs/extraction/esm_if1_142m.json`. Cache names match the prepared
example under `data/features/pretrained/singles/<assay>` and omit the structure
collection prefix. Structure-distance coordinates use the matching
`data/structures/protssn_proteingym/manifests`. The 57-assay mapping, training and
evaluation settings, and Slurm resources match the ESM2/ProteinMPNN reference.
BRCA2 retains both mean scores with structure distance disabled. Results use
`kermut_prosst_esmif_<selection>_<job-id>`; the usual config, result-prefix,
parallelism, and experiment-suffix overrides are supported.

To run the same ProSST/ESM-IF model on **all 217 assays** in the local ProteinGym
substitutions catalog, use
[`run_kermut_prosst_esmif_all_assays.sbatch`](../../sbatch_run/run_kermut/run_kermut_prosst_esmif_all_assays.sbatch):

```bash
sbatch sbatch_run/run_kermut/run_kermut_prosst_esmif_all_assays.sbatch singles --experiment-suffix kermut_prosst_esmif_all_assays

# At most four assays running at once:
sbatch --array=0-216%4 sbatch_run/run_kermut/run_kermut_prosst_esmif_all_assays.sbatch singles --experiment-suffix kermut_prosst_esmif_all_assays

# Inspect the full task-to-assay mapping:
bash sbatch_run/run_kermut/run_kermut_prosst_esmif_all_assays.sbatch --help
```

The default array is `0-216`, with one assay per task and no assay-size filter.
Indices follow the full catalog order and differ from the small-assay launcher.
The list is a fixed catalog snapshot; update it and the array bounds together
when changing catalogs. Each task defaults to random, modulo, and contiguous
singles evaluation. All assays need prepared ProSST/ESM-IF caches and published
fold CSVs. The preset, options, and resources (15 CPUs, 16 GB RAM, 24 hours)
match the small-assay job; larger assays may require `sbatch --mem=...` and
`--time=...` overrides. `--parallel` also increases concurrent memory use.
With the example suffix, outputs go to
`results/kermut_prosst_esmif_all_assays/<assay>/kermut_prosst_esmif_singles_<job-id>/`.

Use the five-assay CPU array launcher from the repository root on the cluster:

```bash
sbatch sbatch_run/run_kermut/run_kermut_assays.sbatch

# Only FKBP3 or NEIME:
sbatch --array=2 sbatch_run/run_kermut/run_kermut_assays.sbatch
sbatch --array=4 sbatch_run/run_kermut/run_kermut_assays.sbatch

# Both FKBP3 and NEIME:
sbatch --array=2,4 sbatch_run/run_kermut/run_kermut_assays.sbatch

# Also parallelize the 15 folds within each assay:
sbatch sbatch_run/run_kermut/run_kermut_assays.sbatch singles --parallel

# Limit the array to two concurrent assays:
sbatch --array=0-4%2 sbatch_run/run_kermut/run_kermut_assays.sbatch
```

The fixed array mapping matches the NKN array:

| Array index | Assay |
| --- | --- |
| 0 | `TPMT_HUMAN_Matreyek_2018` |
| 1 | `RASK_HUMAN_Weng_2022_binding-DARPin_K55` |
| 2 | `FKBP3_HUMAN_Tsuboyama_2023_2KFV` |
| 3 | `DYR_ECOLI_Thompson_2019` |
| 4 | `A0A1I9GEU1_NEIME_Kennouche_2019` |

Each task selects its assay by array index and defaults to
`data/features/pretrained/singles/<assay>` for singles selections.
The default `configs/kermut_esm2_proteinmpnn.json` uses ESM2 650M layer-33 embeddings, sequence-conditioned ProteinMPNN
distributions (ten orders, seed 37), and a GP mean combining ESM2 650M
masked-marginal scores with ProteinMPNN sequence likelihoods (one order, seed 0).
The array forwards setting arguments, `--parallel`, and `--config PATH` to the
shared launcher; `FEATURES` and `KERMUT_CONFIG` overrides remain available.
Prepare features and the applicable score caches before submission. The
intentional ProteinMPNN score omissions for BRCA2, ZIKV, and HCV are supported;
BRCA2 also skips loading shared-frame coordinates. Other assays require a
registered structure manifest. Each task defaults to all three singles CV protocols
(15 fits) and requests 15 CPUs, 16 GB RAM, and a 24-hour limit. Assays may run
concurrently; within each assay, fits run sequentially unless `--parallel` is set.
Logs are `kermut_assays-<array-job-id>_<index>.out` in the submission directory.
Results are separated by assay under
`results/<assay>/kermut_esm2_proteinmpnn_<selection>_<task-job-id>/`.

The generic CPU Slurm launcher remains available and follows the existing NKN
job conventions:

```bash
sbatch sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn.sbatch
sbatch sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn.sbatch singles --parallel
sbatch sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn.sbatch singles_random --config configs/kermut_esm2_proteinmpnn.json
```

Default assay: `FKBP3_HUMAN_Tsuboyama_2023_2KFV`. Default selection: all three
singles protocols, for 15 fits. The launcher accepts `ASSAY`, `FEATURES`,
`KERMUT_CONFIG` (overridden by `--config`), `SINGLES_FOLDS_CSV`, and
`MULTIPLES_FOLDS_CSV`. It snapshots published fold files and the resolved JSON.
Results default to `results/<assay>/kermut_esm2_proteinmpnn_<selection>_<job_id>/`.
All Kermut training sbatch launchers accept `--output-root PATH` (or `--output-root=PATH`)
to choose the base directory. Precedence is **`--output-root`, then
`KERMUT_OUTPUT_ROOT`, then `results/`**. Absolute paths are used directly;
relative paths are resolved from the repository root, including when submitting
from a subdirectory. Quote paths containing spaces.

Add `--experiment-suffix NAME` to save under
`<output-root>/NAME/<assay>/<model-run>/`. This overrides `EXPERIMENT_SUFFIX=NAME`
if set when submitting. The suffix must be one folder name. For example:

```bash
sbatch sbatch_run/run_kermut/run_kermut_prosst_esmif_all_assays.sbatch \
  singles --output-root /scratch/your_username/results --experiment-suffix comparison_v1
```

This saves runs under
`/scratch/your_username/results/comparison_v1/<assay>/kermut_prosst_esmif_singles_<job-id>/`.
Omitting both options preserves the existing paths. Existing run directories
are never overwritten. Slurm `.out` files still follow the launcher's
`#SBATCH --output` setting in the submission directory.
Use the same output root and suffix across related NKN and Kermut submissions,
then pass that group's path to `python -m reporting --results`.
Like the NKN launcher, it activates the cluster's `nkn_protein` Conda environment.

Singles selections use `data/features/pretrained/singles/<assay>` and require
all indexed singles with `--full-assay --singles-only`. Mixed selections, `all`,
and `single_to_double` retain `data/features/pretrained/<assay>` and full-assay
validation. `FEATURES` overrides either default root. Direct `cv` commands accept
`--singles-only` only with a `singles_*` setting; omitting it retains the original
full raw-assay coverage check.

The launcher requests **15 CPUs by default**. Add `--parallel` to run all 15
singles fits concurrently: three setting processes, each with five fold workers
and one CPU thread per worker. Without `--parallel`, settings and folds remain
sequential and each fit uses the full CPU allocation. To run only one setting
with five simultaneous single-threaded folds:

```bash
sbatch --cpus-per-task=5 sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn.sbatch singles_random --parallel
```

The launcher derives worker and thread counts from `SLURM_CPUS_PER_TASK`, keeping
the total fitting threads within the allocation. It saves `execution_plan.json`
and writes parallel setting logs to `logs/<setting>.log`. Fold checkpoints and
metrics retain their existing paths. Each worker runs the Kermut model with the
same configured precision, fixed AdamW update count, raw mean scores and
training-only target transform; the final iterate is still saved. Both mixed
domains still share five fits when using `all --parallel`.

For direct CLI use, add `cv --fold-workers N` (at most the configured number of
folds), use `device: "cpu"`, and set JSON `num_threads` to the thread count **per
worker**. For example, five workers on five CPUs need `num_threads: 1`. The
default remains one fold worker. Parallel workers preserve progress and failure
artifacts; a failed fold prevents aggregate results from being reported as
complete.

The default `--mem=16G` is shared by the entire job. Concurrent fits hold separate
models and covariance tensors, so increase `--mem` if the assay needs more
memory. Parallel mode applies to a new submission, and does not resume or change
an already running job.

## Six experiment settings

| Setting | Training and evaluation |
| --- | --- |
| `singles_random` | Five published random folds of singles |
| `singles_modulo` | Five published position-modulo folds of singles |
| `singles_contiguous` | Five published contiguous-position folds of singles |
| `mixed_random_singles` | Train on the other four mixed folds; evaluate held-out singles |
| `mixed_random_doubles` | Train on the other four mixed folds; evaluate held-out doubles |
| `single_to_double` | One fit on sampled singles; evaluate all doubles |

Use `cv --setting ... --folds ...` for the first five settings, and
`fit --setting single_to_double --sample-ratio 0.8 --split-seed 2024` for the
last. A sample ratio of 1 uses all singles. There is no full-data fit command.
Use the same input row order, sampling ratio and seed as NKN to reproduce its
extrapolation training set.

Both mixed evaluation domains are reported from the same fold fits. Slurm
selection `all` therefore runs 21 fits: 15 singles, five mixed, and one
extrapolation. The current assay has only singles, so mixed/extrapolation
settings require a different assay with both singles and doubles. The launcher
checks every requested setting before training starts. `SAMPLE_RATIO` and
`SPLIT_SEED` control Slurm extrapolation sampling.

## Saved results and verification

Each training partition saves a self-contained `checkpoint.pt`, resolved
configuration, history, fit summary and predictions with latent and observation
variances. The common `restart_summaries.json` artifact contains one final run.
No pretrained models or structure files are needed to load the checkpoint
itself; new predictions require compatible cached inputs.

New fits use `ddof=0` and record the target-normalization convention in their
checkpoints and fit summaries. Historical checkpoints retain their saved target
mean, scale, and standardized training labels when loaded, including the earlier
`ddof=1` convention. Existing predictions and metrics are not recomputed by this
change or by rerunning reporting; rerun training to obtain results from the new
normalization.

Evaluation reuses the NKN pipeline: per-fold Spearman, equal-fold average and
population standard deviation of defined Spearman values, pooled out-of-fold
Spearman, MSE, interval ECE, PairECE, and standardized variance summaries. MSE
is saved per fold as `mse` and `mse_standardized`, and pooled across all held-out
variants as `mse_oof` and `mse_oof_standardized`. The standardized version uses
each fold's training target mean and population standard deviation (`ddof=0`)
for new fits, with pooled held-out predictions. MSE weights variants equally;
mixed runs report each mutation depth separately. Calibration
uses the full observation covariance within each held-out fold; covariances
are never pooled across folds. Extrapolation reports one held-out evaluation
without a cross-fold average or standard deviation, including `mse` and
`mse_standardized`. The MSE fields also appear in the shared reporting CSVs.

Interval ECE averages absolute coverage errors over the 11 default levels
(`0.0, 0.1, ..., 0.9, 1.0`) within each fold, then averages folds equally.
Existing saved ECE values and coverage curves retain their original grid, as do
plots generated from those values. Reporting and plotting do not migrate the
grid; older predictions must be reevaluated at the new levels to update them.

Error calibration reports `ence` and `uncertainty_cv` in every held-out fold
and fixed extrapolation evaluation. ENCE uses five variance-quantile bins by
default (`evaluation.error_n_bins`); uncertainty CV uses sample standard
deviation of predictive standard deviations divided by their mean. Both use
observation-predictive variances, including observation noise. CV summaries
average folds equally and include `std_ence`, `std_uncertainty_cv`, and valid
fold counts. `error_bin_stats` retains RMV/RMSE bin statistics in original and
training-standardized units. The reporting CSVs include the new scalar metrics.
For constant positive variance across at least two variants, the shared
NKN/Kermut evaluator reports the calculated `ence = abs(RMV - RMSE) / RMV`,
one populated bin (`n_error_bins: 1`), and exactly `uncertainty_cv: 0.0`.
The requested `error_n_bins` remains 5 by default. Both counts are saved per
fold in `metrics.json` and exported to `folds.csv`; singleton CV remains null.
The separate [`plot_kermut_calibration.sbatch`](../../sbatch_run/run_plotting/plot_kermut_calibration.sbatch) plotting job also saves
`ence.png`/`ence.pdf` with RMV/RMSE curves and ENCE/cv annotations when per-fold
`error_bin_stats` are available. For older runs, missing bins are reconstructed
from saved per-fold targets, predicted means, and observation variances in
`predictions.csv`; source results remain unchanged. Training does not generate
images. Runs without bins or usable predictions skip ENCE with an explicit reason.

The uncertainty CV follows Appendix J.2's equation. The released Kermut code
places the mean inside the square root, producing different values; see the
[shared evaluation notes](../../src/README.md#outputs-and-evaluation) for
formulas, binning edge cases, and historical-result compatibility.

```powershell
python -m unittest discover -s baselines/kermut/tests -v
```

Tests cover the independent kernel formula, exact likelihood/prior and Gaussian
conditioning, final-iterate AdamW updates, training-only transforms, frozen
features, configurable means, checkpoint round trips and all six settings.

## Check completion across all assays

Pass an experiment results directory containing `<assay>/<run>/` directories to
the CPU-only checker:

```bash
sbatch sbatch_run/run_kermut/check_kermut_results.sbatch results/kermut_prosst_esmif_all_assays
# Optional machine-readable report:
sbatch sbatch_run/run_kermut/check_kermut_results.sbatch results/kermut_prosst_esmif_all_assays --report reports/kermut_completion.json
```

By default it checks every assay in the ProteinGym catalog for a complete
`kermut_prosst_esmif` run with `singles_random`, `singles_modulo`, and
`singles_contiguous` results. Any one complete matching run satisfies the assay;
an older failed attempt does not invalidate a successful rerun. All requested
settings must be complete within that same run. The audit checks saved metadata,
prediction coverage, populated fold artifacts, and failure markers without
deserializing checkpoint tensors or rerunning training. It leaves results
unchanged, reports each assay and missing/incomplete catalog indices, and exits
with status 0 when all assays are complete, 1 when any are missing/incomplete,
or 2 for invalid inputs.
The optional JSON report path must not already exist.

Use `--run-prefix NAME` for another Kermut experiment, `--settings NAME...` for a
subset of the three singles settings, and `--assay-index PATH` for a different
catalog. Catalog discovery otherwise uses `ASSAY_INDEX`, then the same standard
catalog locations as the training launcher. Relative results, catalog, and
report paths resolve from the repository root. Submit from the repository or a
subdirectory; direct `bash` execution also locates the repository from the
script path. `bash sbatch_run/run_kermut/check_kermut_results.sbatch --help`
prints options without activating Conda.
