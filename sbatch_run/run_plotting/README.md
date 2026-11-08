# NKN and Kermut plots

## Predictive variance comparison (Figure 2)

To compare Kermut and NKN side by side for **1M -> 1M random, modulo, and
contiguous** schemes, submit:

```bash
sbatch sbatch_run/run_plotting/plot_predictive_variance.sbatch \
  --kermut-results-dir results/kermut_experiment \
  --nkn-results-dir results/nkn_experiment \
  --output-dir plots/kermut_vs_nkn_variance
```

Each input root has the layout `<root>/<assay>/{kermut,nkn}_*/cv/`. If both
models are in one collection, pass the same path to both options (for example,
`results/hierarchical_nkn`). Each root defaults to `results`; discovery does
not recursively combine result collections.

The grouped violins follow [Figure 2 of the Kermut paper](https://arxiv.org/pdf/2407.00002),
with adjacent Kermut and NKN distributions for each of the three schemes.
They use **observation predictive variance**, including observation noise,
in saved training-standardized target units. Each plotted value is the mean across
an assay's held-out single-mutant variants, pooled across its CV folds. Each
assay then contributes one value to its model/scheme distribution, regardless
of assay size. Use `--variance-kind latent` to compare latent variances instead.
The code reads `variance_summary.single_mutants` from saved aggregate
`metrics.json` files; when that summary is absent, it weights the saved fold
means by their variant counts. It does not load models or recompute predictions.
Saved units are preserved: legacy Kermut runs may use training sample SD,
while current runs use population SD; the plotter does not rescale those runs.

Dots represent assays, thick bars span the 25th to 75th percentiles, black ticks
mark medians, and whiskers extend to the most extreme values within 1.5 IQR.
Singleton or constant groups use points and summary bars without a density.

Only assays with complete results for both models in all three schemes enter
the comparison; the same assay cohort is used throughout. Mixed-mutant and
single-to-double settings are excluded. This uses all eligible shared assays,
without requiring the double-mutant data used to select assays in the paper.
Saved variant/fold assignments must agree when available; otherwise matching
uses fold counts and the summary records this limitation.
The JSON summary records the included assays, omitted results, source paths,
and aggregation choices. Multiple runs for the same model and assay require
explicit selection with repeatable `--kermut-run-dir` or `--nkn-run-dir`; these
options bypass discovery for that model. Use repeatable `--assay` to restrict
the comparison to particular assays.

The job writes directly to `--output-dir` (default `plots/predictive_variance`):

```text
predictive_variance.png
predictive_variance.pdf
predictive_variance_per_assay.csv
predictive_variance_summary.json
```

Use `--formats png pdf svg` and `--dpi 300` to customize exports. Rerunning at
the same destination replaces these outputs, so choose a separate output
directory for each model configuration comparison. Source results are read only.
The launcher uses the existing `nkn_protein` environment with Matplotlib, one
CPU, 2 GB RAM, and 10 minutes. `PLOT_KERMUT_RESULTS_DIR`,
`PLOT_NKN_RESULTS_DIR`, and `PLOT_OUTPUT_DIR` provide environment defaults;
explicit command-line values take precedence. Relative paths use the repository
root. Launcher help works without Conda or Slurm:

```bash
bash sbatch_run/run_plotting/plot_predictive_variance.sbatch --help
PYTHONPATH=src python -m plotting.predictive_variance \
  --kermut-results-dir results/hierarchical_nkn \
  --nkn-results-dir results/hierarchical_nkn \
  --output-dir plots/hierarchical_nkn_variance
```

## Calibration plots

Submit from the repository root or a subdirectory:

```bash
sbatch sbatch_run/run_plotting/plot_nkn_calibration.sbatch \
  --results-dir results/hierarchical_nkn
sbatch sbatch_run/run_plotting/plot_kermut_calibration.sbatch \
  --results-dir results/kermut_experiment
```

The NKN launcher discovers only `<results-dir>/<assay>/nkn_*` runs; the Kermut
launcher discovers only `<results-dir>/<assay>/kermut_*` runs. Both require `cv/`
or `fit/` results. A directory containing both model types is filtered to the
launcher's model. An explicit `--run-dir` for the wrong model fails with an error.
Each job reads saved calibration metrics without loading checkpoints or running
training. Each requests one CPU, 2 GB RAM, and 10 minutes, using the same account
and QoS as the NKN jobs.

The existing `nkn_protein` Conda environment needs Matplotlib. It is included in
`env.yaml`; for an existing environment, install it once before submitting:

```bash
conda activate nkn_protein
python -m pip install "matplotlib>=3.8,<3.11"
```

## Output locations

By default the batch job writes under the repository's top-level `plots/`:

```text
plots/<results-directory-name>/<assay>/
    interval_ece.png
    interval_ece.pdf
    pair_ece.png
    pair_ece.pdf
    ence.png
    ence.pdf
    calibration_summary.json
```

The ENCE figure uses saved `error_bin_stats` when available. For older runs,
the plotter reconstructs missing bins from `fold_*/predictions.csv`, falling
back to the setting's pooled `predictions.csv` split by fold ID. Recovery uses
targets, predicted means, and **observation variance**, including observation
noise. It never pools folds when computing ENCE and filters singles/doubles
for mixed settings. It uses the saved `error_n_bins` configuration, or five bins
for older configurations. Standardized plot coordinates use the training scale
recovered from raw and standardized observation variances.

Recovery requires PyTorch, already included in `nkn_protein`. Source metrics
and predictions are not modified. `calibration_summary.json` records the
recovered metrics, source files, and per-fold recovery status. Missing or invalid
prediction data produce an explicit reason in that summary and the job log;
if no fold has usable ENCE data, only the interval and pairwise figures are saved.
Training itself saves numeric metrics only; this plotting job creates images.

The group name is the final folder name of `--results-dir`. For example,
`--results-dir results/kermut_experiment` saves to
`plots/kermut_experiment/<assay>/`; `--results-dir results/hierarchical_nkn`
saves to `plots/hierarchical_nkn/<assay>/`. The default input `results` saves
under `plots/results/<assay>/`.

If an assay directory contains multiple model runs, filenames include the run
name, such as `kermut_singles_123__ence.png` and
`kermut_singles_123__calibration_summary.json`, to avoid overwriting other runs.
This also applies when selecting only one of those runs with `--run-dir`.
For `--run-dir`, the group name comes from the run's grandparent directory:
`<group>/<assay>/<model-run>`.

Set a custom output root while preserving the results-group and assay folders:

```bash
sbatch sbatch_run/run_plotting/plot_kermut_calibration.sbatch \
  --results-dir results/kermut_experiment --output-dir /path/to/figures
# Saves to /path/to/figures/kermut_experiment/<assay>/
```

Relative paths are resolved from the repository root. Rerunning replaces plot
files and the plotting summary at the selected destination. Original metrics
are only read.

## Selecting results and formats

```bash
# One protein (repeat --assay to select several):
sbatch sbatch_run/run_plotting/plot_nkn_calibration.sbatch \
  --assay A0A1I9GEU1_NEIME_Kennouche_2019

# One exact model run (repeat --run-dir to select several):
sbatch sbatch_run/run_plotting/plot_nkn_calibration.sbatch \
  --run-dir results/FKBP3_HUMAN_Tsuboyama_2023_2KFV/nkn_esm2_proteinmpnn_singles_13951633

# Different input root and export settings:
sbatch sbatch_run/run_plotting/plot_nkn_calibration.sbatch \
  --results-dir /path/to/results --formats png pdf svg --dpi 300

# A results collection organized as collection/assay/model-run:
sbatch sbatch_run/run_plotting/plot_nkn_calibration.sbatch \
  --results-dir results/hierarchical_nkn

# A Kermut collection uses the dedicated Kermut launcher:
sbatch sbatch_run/run_plotting/plot_kermut_calibration.sbatch \
  --results-dir results/kermut_experiment
```

Repeat `--results-dir` for multiple input roots of the launcher's model. Each
collection gets its own figures under `plots/<results-directory-name>/<assay>/`.
The separate jobs above write to `plots/hierarchical_nkn/` and
`plots/kermut_experiment/`, respectively. Duplicate inputs are
processed once. `--assay` filters every input root; each selected root must contain
matching runs of the selected model. Roots must have distinct final folder names when sharing an output
root, so their plots cannot overwrite each other.

`--run-dir` bypasses discovery and cannot be combined with `--assay`.
`PLOT_RESULTS_DIR` and `PLOT_OUTPUT_DIR` can supply defaults; command-line
arguments take precedence. Explicit `--results-dir` arguments replace the
`PLOT_RESULTS_DIR` default. Run `bash
sbatch_run/run_plotting/plot_nkn_calibration.sbatch --help` or the corresponding
Kermut script for launcher help.

For a local run with the same output layout:

```bash
PYTHONPATH=src python -m plotting.calibration --model-family nkn --results-dir results/hierarchical_nkn --output-dir plots
PYTHONPATH=src python -m plotting.calibration --model-family kermut --results-dir results/kermut_experiment --output-dir plots
```

Without `--output-dir`, the Python command saves inside each model run's `plots/`
directory. The shared Python command supports both model families when
`--model-family` is omitted; the batch launchers always select their own model.

## Reading the figures

- **Interval ECE:** nominal interval coverage versus observed coverage, with an
  ideal diagonal, individual fold curves, and equal-fold means with one sample
  standard deviation across folds. New evaluations default to 11 nominal
  coverage levels from `0.0` through `1.0` in steps of `0.1`, including both
  endpoints. Plots use the saved coverage grid and ECE: older results retain
  their original grid until their predictions are reevaluated. Rerunning the
  plotting job alone does not migrate those saved metrics.
- **Pair ECE:** mean ordering confidence versus ordering accuracy in each saved
  confidence bin. Points average the folds with pairs in that bin equally;
  horizontal and vertical error bars show one sample standard deviation. The
  histogram sums within-fold pair counts. Empty bins remain empty.
- **ECE annotations:** arithmetic mean and sample standard deviation of the
  saved fold ECEs. ECE is not recalculated from an averaged curve. Within each
  fold, the saved PairECE already weights bin errors by pair counts. Fold SD
  describes variation across folds; it is not a confidence interval. With just
  one contributing fold, SD is undefined and omitted.
- **ENCE:** RMV versus RMSE with the ideal diagonal, individual fold curves,
  and mean curves with one sample SD on both axes. Coordinates use saved
  training-standardized units when available. ENCE and cv annotations show
  the mean **±2 sample SD** across folds, following the paper's presentation;
  they are not confidence intervals. Metrics average defined folds equally.
  Mean curves require matching full quantile grids; collapsed or incomplete
  bins retain individual curves. Missing ENCE data are labelled explicitly.
  cv uses the saved `uncertainty_cv` field (the Appendix J.2 equation).

Aggregate `metrics.json` embedded fold metrics take precedence over separate
fold files, avoiding duplicate counting. If aggregate metrics are absent,
completed `fold_*/metrics.json` files can still produce a partial plot. Missing
settings are labelled unavailable; they do not prevent plotting other settings.
The summary JSON records source paths, contributing counts, aggregation rules,
and partial/unavailable reasons. The job fails if no evaluation data are usable.

For the currently supplied results, all three NEIME settings and FKBP3's random
and modulo settings are available. FKBP3's contiguous setting has a failure
record and no completed fold metrics, so its panel is labelled unavailable.
