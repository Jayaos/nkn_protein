# NKN calibration plots

Submit from the repository root or a subdirectory:

```bash
sbatch sbatch_run/run_plotting/plot_nkn_calibration.sbatch
```

The job discovers every `results/<protein>/nkn_*` run containing `cv/` or
`fit/` results. It reads saved calibration metrics without loading checkpoints
or running training. It requests one CPU, 2 GB RAM, and 10 minutes, using the
same account and QoS as the NKN jobs.

The existing `nkn_protein` Conda environment needs Matplotlib. It is included in
`env.yaml`; for an existing environment, install it once before submitting:

```bash
conda activate nkn_protein
python -m pip install "matplotlib>=3.8,<3.11"
```

## Output locations

By default each run writes to its own directory:

```text
results/<protein>/<model-run>/plots/
    interval_ece.png
    interval_ece.pdf
    pair_ece.png
    pair_ece.pdf
    calibration_summary.json
```

Set a custom output root while preserving separate protein and model folders:

```bash
sbatch sbatch_run/run_plotting/plot_nkn_calibration.sbatch --output-dir /path/to/figures
# Saves to /path/to/figures/<protein>/<model-run>/
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
```

`--run-dir` bypasses discovery and cannot be combined with `--assay`.
`PLOT_RESULTS_DIR` and `PLOT_OUTPUT_DIR` can supply defaults; command-line
arguments take precedence. Run `bash
sbatch_run/run_plotting/plot_nkn_calibration.sbatch --help` for launcher help.

For a local run with the same options:

```bash
PYTHONPATH=src python -m plotting.calibration --results-dir results
```

## Reading the figures

- **Interval ECE:** nominal interval coverage versus observed coverage, with an
  ideal diagonal, individual fold curves, and equal-fold means with one sample
  standard deviation across folds.
- **Pair ECE:** mean ordering confidence versus ordering accuracy in each saved
  confidence bin. Points average the folds with pairs in that bin equally;
  horizontal and vertical error bars show one sample standard deviation. The
  histogram sums within-fold pair counts. Empty bins remain empty.
- **ECE annotations:** arithmetic mean and sample standard deviation of the
  saved fold ECEs. ECE is not recalculated from an averaged curve. Within each
  fold, the saved PairECE already weights bin errors by pair counts. Fold SD
  describes variation across folds; it is not a confidence interval. With just
  one contributing fold, SD is undefined and omitted.

Aggregate `metrics.json` embedded fold metrics take precedence over separate
fold files, avoiding duplicate counting. If aggregate metrics are absent,
completed `fold_*/metrics.json` files can still produce a partial plot. Missing
settings are labelled unavailable; they do not prevent plotting other settings.
The summary JSON records source paths, contributing counts, aggregation rules,
and partial/unavailable reasons. The job fails if no evaluation data are usable.

For the currently supplied results, all three NEIME settings and FKBP3's random
and modulo settings are available. FKBP3's contiguous setting has a failure
record and no completed fold metrics, so its panel is labelled unavailable.
