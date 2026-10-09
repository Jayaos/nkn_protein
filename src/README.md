# NKN protein fitness

An independent PyTorch implementation of the compositional-kernel exact GP in
`nkn_protein_variant_prediction.pdf`. The local Kermut and Neural-Kernel-Network
clones are references only: this package does not import them or require them
on `PYTHONPATH`. All implementation, configuration examples, and tests are in
this `src/` directory. Data loading is consolidated in `data_loading/`.

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
python -m experiments.cli validate --features data/features/A0A1I9GEU1_NEIME_Kennouche_2019 --assay-index data/DMS_ProteinGym_substitutions/DMS_substitutions.csv --allow-subset --output src/runs/fixture_validation
python -m experiments.cli cv --features data/features/A0A1I9GEU1_NEIME_Kennouche_2019 --config src/configs/smoke.json --development-splits --scheme random --output src/runs/fixture_smoke
```

Run experiments with `python -m experiments.cli` and extraction with
`python -m feature_generation.cli`. Experiment commands require a new or empty
output directory.
The smoke configuration uses five Adam updates for each of three restarts; it is
an integration check, not a completed scientific fit. The example's cached
representations and zero-shot scores are synthetic, while its labels are real.

Without `--config`, training uses the paper's 5,000-step maximum, 1,000-step
minimum before early stopping, 500-step comparison window, and three restarts.
Device defaults to CPU. Use `--device cuda` with a suitable PyTorch installation.
The experiment CLI consumes saved features. The separate feature-generation
CLI prepares structures, downloads checkpoints, extracts frozen ESM-2 and structure-conditioned ProSST embeddings,
and computes ProteinMPNN/ESM-IF1 amino-acid distributions.
See [the extraction guide](feature_generation/README.md) for installation,
checkpoint downloads, structure mapping, resumable extraction and standalone cache validation.
Notebooks are optional smoke tests; the CLI is the default extraction workflow.

## Source organization

| Path | Responsibility |
| --- | --- |
| `config.py` | Validated model, optimizer, evaluation, and runtime settings |
| `data_loading/` | Assay catalog, cache/score alignment, PyTorch dataset, packed mutations, folds |
| `kernels/` | Primitive kernels, NKN layers, mutation sums, final kernel |
| `gp/` | Fold transforms, dense Cholesky, exact GP, restarts, checkpoints |
| `evaluation/` | Spearman, interval ECE, PairECE, fold aggregation |
| `experiments/` | Fit/CV/prediction execution, CLI, result artifacts |
| `tests/` | Numerical, data-contract, leakage, and integration tests |
| `configs/smoke.json` | Short fixture-validation run |
| `feature_generation/` | Pinned checkpoints, structure mapping, ESM-2/ProSST embeddings, ProteinMPNN/ESM-IF1 distributions, cache validation |
| `configs/extraction/` | ESM-2 8M/650M, ProSST-2048, ProteinMPNN v48_020, and ESM-IF1 142M recipes |

These directories are independent Python packages directly under `src/`.
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
  sequence/<family>/<checkpoint>/{config.json,features.pt,zero_shot.csv}
  structure/<family>/<checkpoint>/{config.json,features.pt,zero_shot.csv}
  inverse_folding/<family>/<checkpoint>/{config.json,features.pt,zero_shot.csv}
```

`variants.csv` identifies the requested variants. Sequence cache IDs and hashes
must match its complete sequences. Structural embeddings and inverse-folding
probabilities are shared wild-type arrays indexed by one-based reference position.
Working tensors are converted to float64; cache files are not rewritten.
The current adapter uses mean pooling over all sequence residues for the global
branch. Model-specific preprocessing and other pooling can later be supplied
through the model-independent `FeatureBatch` API. The old
`ProteinFitnessDataset.as_feature_matrices` demo transformations are not used.

Zero-shot sources default to the three selected representation directories.
`score_sources` may instead list any selected score directories in a fixed order,
using `feature_type/family/checkpoint`. Scores are joined by ID and standardized
on training variants only. Score direction must be `higher_is_better` or
`lower_is_better`, as declared in its configuration. Source identity and order
are saved in checkpoints and validated during prediction.

The ProteinGym catalog reads `DMS_id`, `DMS_filename`, and `target_seq` from its
index. It validates processed mutation positions directly against the reference;
it does not reapply raw-data, MSA, or structure offsets. Cross-assay identities
include the assay ID. No feature extraction is inferred from a raw assay CSV.

When `--assay-index` is supplied, complete indexed assay coverage is required
unless `--allow-subset` is explicit. The current fixture contains 16 of its
assay's 922 variants, so the validation example uses that flag. With
`--singles-only`, completeness refers to all indexed single substitutions. Rows
are not silently removed to match an incomplete feature cache.

## Fold inputs and experiments

The raw ProteinGym CSVs in this workspace contain no official fold assignments.
For benchmark CV supply a separate ID-aligned CSV:

```text
variant_id,fold_random_5,fold_modulo_5,fold_contiguous_5
M7V,0,1,2
...
```

`mutant` can replace `variant_id`. A column named `random`, `modulo`, or
`contiguous` is also accepted, and a generic `fold` column is the fallback.
IDs must exactly cover the selected variants; folds are numbered `0` through
`n_folds - 1`, with every fold present. The chosen `--scheme` identifies the
protocol. Positional schemes require single substitutions and disjoint mutation
positions across folds. Multi-mutant positional protocols remain deferred.

```powershell
python -m experiments.cli cv --features PATH_TO_CACHE --folds PATH_TO_FOLDS.csv --scheme random --output src/runs/assay_random
python -m experiments.cli fit --features PATH_TO_CACHE --folds PATH_TO_FOLDS.csv --scheme random --test-fold 0 --output src/runs/fold_0
python -m experiments.cli fit --features PATH_TO_CACHE --output src/runs/full_fit
python -m experiments.cli predict --features PATH_TO_NEW_VARIANTS --checkpoint src/runs/full_fit/checkpoint.pt --output src/runs/prediction
```

`fit` without a split fits all requested variants. `predict` does not need labels;
omit `DMS_score` or leave every value blank in the prediction cache's variant
table. The checkpoint includes training representations, standardized training
scores/targets, and fitted transformations. A prediction command uses the
adjacent `resolved_config.json` unless an explicit configuration is supplied.

Development splits require the explicit `--development-splits` flag and are
labelled accordingly in the outputs. They are suitable for fixture tests, not
substitutes for supplied benchmark assignments. Random splits can include mixed
mutation counts. For the single-to-double experiment:

```powershell
python -m experiments.cli fit --features PATH_TO_MIXED_CACHE --single-to-double --output src/runs/single_to_double
```

This fits all single substitutions and evaluates double substitutions once.
Higher mutation counts are excluded and counted. It is a fixed train/test
experiment, not five-fold cross-validation.

## Configuration and numerical behavior

JSON configurations can override any `ExperimentConfig` field. Unspecified
fields use the validated defaults in `config.py`. If overriding `model.branches`,
supply all four branches. For example:

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
even-sized medians average the middle two values. Seeds default to `0,1,2` and
lengthscale multipliers to `0.5,1,2`. NKN biases and noise start at `0.01`; the
noise lower bound is `1e-4`.

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

* Spearman uses combined original-unit out-of-fold means.
* Interval ECE is computed in each fold and then equally averaged.
* PairECE uses the observation covariance for pairs within the same held-out
  fold, excludes exact observed and predicted ties, and then averages eligible
  fold-level values equally. Bins have equal width on `[0.5,1]`, are left-closed
  and right-open, and the last bin includes one. Empty eligible-pair sets produce
  JSON `null`, with contributing-fold and pair counts recorded.
* Uncertainty summaries average observation variances in training-standardized
  units, including separate single/double summaries when present.

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
