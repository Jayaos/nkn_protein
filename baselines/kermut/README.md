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
  its targets using its own mean and sample standard deviation.
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
The GP mean combines ESM2 and ProteinMPNN variant scores; the publication's
mean uses ESM2 alone. ProteinMPNN variant scoring uses one order and seed 0.
For each reference position, ProteinMPNN conditions on the backbone and the
other wild-type residues, using ten randomized decoding orders with seed 37.
Extraction averages log probabilities, exponentiates, and removes the unknown
amino-acid category `X` without renormalizing. The resulting 20 values can sum
to less than one; the loader and kernel preserve those values.

This matches the ProteinMPNN extraction recipe for the pipeline's supported
single-chain structures of up to 1,022 residues. It does not establish numerical
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
  "scores/esm2/esm2_t33_650M_UR50D-masked_marginals-float32-08e4846e5371",
  "scores/proteinmpnn/v_48_020-sequence_log_likelihood-orders1-seed0-8907e6671bfb"
]
```

Keeping only the first entry uses the publication's ESM2-only mean; keeping
only the second uses a ProteinMPNN-only mean. Multiple entries
give `mean(x) = intercept + sum(coefficient[j] * score[j](x))`; all coefficients
start at one and are trained jointly with the kernel. Scores remain in their
raw units. Duplicate sources and unknown configuration fields are rejected.

`model.raw_*` values are initial unconstrained kernel parameters. Positive
parameters use softplus and `raw_alpha` uses sigmoid. `training` controls the
optimizer, numerical precision and noise prior. `training.max_train_variants`
can optionally cap dense-GP training size; it is unset by default. Training
uses quadratic memory and cubic Cholesky work in the number of variants.
`evaluation` has the same coverage levels, pairwise bins and five-fold setting
as NKN. The complete accepted schema is in [`config.py`](config.py).

## Run on CPU

Prepare the selected caches first. On the cluster, submit both preparation
jobs and wait for them to finish successfully before submitting Kermut:

```bash
sbatch sbatch_run/run_prepare_rep/prepare_esm2.sbatch
sbatch sbatch_run/run_prepare_rep/prepare_proteinmpnn.sbatch
```

The ProteinMPNN job writes the sequence-conditioned cache selected by Kermut
and retains the existing separate variant-score recipe. Existing backbone-only
outputs cannot satisfy Kermut's cache selection. To extract only the Kermut
distributions from prepared model and structure assets, run:

```bash
PYTHONPATH=src python -m feature_generation.cli extract --config configs/extraction/proteinmpnn_v48_020_kermut.json --assay A0A1I9GEU1_NEIME_Kennouche_2019 --device cuda --resume
```

From the repository root in PowerShell, activate the existing environment and
make the shared source packages available:

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path + [IO.Path]::PathSeparator + $env:PYTHONPATH
python -m baselines.kermut.cli --help
```

For the current assay's five-fold random single-mutant evaluation:

```powershell
python -m baselines.kermut.cli cv --features data/features/pretrained/A0A1I9GEU1_NEIME_Kennouche_2019 --config configs/kermut_esm2_proteinmpnn.json --folds data/DMS_ProteinGym_substitutions/cv_folds_singles_substitutions/A0A1I9GEU1_NEIME_Kennouche_2019.csv --setting singles_random --assay-index data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv --full-assay --output results/A0A1I9GEU1_NEIME_Kennouche_2019/kermut_random
```

Output directories must be new or empty. `validate` checks inputs without
training. `predict --checkpoint <fold_dir>/checkpoint.pt` reloads a saved
posterior, supports unlabeled cached variants and defaults to the adjacent
resolved configuration. Prediction verifies assay, reference, feature sources,
structure identity and ordered score sources against the checkpoint.

Use either assay-specific CPU launcher:

```bash
sbatch sbatch_run/run_kermut/run_kermut_A0A1I9GEU1_NEIME_Kennouche_2019.sbatch
sbatch sbatch_run/run_kermut/run_kermut_FKBP3_HUMAN_Tsuboyama_2023_2KFV.sbatch
```

Each fixes its named assay and defaults to `data/features/pretrained/<assay>`.
Both use ESM2 650M layer-33 embeddings, sequence-conditioned ProteinMPNN
distributions (ten orders, seed 37), and a GP mean combining ESM2 650M
masked-marginal scores with ProteinMPNN sequence likelihoods (one order, seed 0).
They forward setting arguments, `--parallel`, and `--config PATH` to the shared
launcher; `FEATURES` and configuration overrides remain available. Results
are separated by assay under
`results/<assay>/kermut_esm2_proteinmpnn_<selection>_<job_id>/`.

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
Results go under `results/<assay>/kermut_esm2_proteinmpnn_<selection>_<job_id>/`.
Like the NKN launcher, it activates the cluster's `nkn_protein` Conda environment.

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

Evaluation reuses the NKN pipeline: per-fold Spearman, equal-fold average and
population standard deviation of defined Spearman values, pooled out-of-fold
Spearman, interval ECE, PairECE, and standardized variance summaries. Calibration
uses the full observation covariance within each held-out fold; covariances
are never pooled across folds. Extrapolation reports one held-out evaluation
without a cross-fold average or standard deviation.

```powershell
python -m unittest discover -s baselines/kermut/tests -v
```

Tests cover the independent kernel formula, exact likelihood/prior and Gaussian
conditioning, final-iterate AdamW updates, training-only transforms, frozen
features, configurable means, checkpoint round trips and all six settings.
