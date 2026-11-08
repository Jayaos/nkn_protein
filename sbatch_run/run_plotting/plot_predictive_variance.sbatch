#!/bin/bash
#SBATCH --job-name=predictive_variance_plots
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%j.out

# Compare saved Kermut/NKN predictive variances without loading models.
# Submit from the repository or a subdirectory, or run this file with bash.
set -eo pipefail

for argument in "$@"; do
  case "$argument" in
    -h|--help)
      cat <<'HELP'
Usage: sbatch sbatch_run/run_plotting/plot_predictive_variance.sbatch [OPTIONS]

Figure 2-like Kermut/NKN violins for 1M -> 1M random, modulo, and contiguous CV.
The same shared assays must have complete results in all three schemes.
  --kermut-results-dir PATH  Kermut collection (default: PLOT_KERMUT_RESULTS_DIR
                            or results), containing assay/kermut_*/cv/.
  --nkn-results-dir PATH     NKN collection (default: PLOT_NKN_RESULTS_DIR
                            or results), containing assay/nkn_*/cv/.
  --kermut-run-dir PATH      Exact Kermut run; repeat for several assays.
  --nkn-run-dir PATH         Exact NKN run; repeat for several assays.
  --assay ID                Restrict assays; repeat for several.
  --output-dir PATH          Destination (default: PLOT_OUTPUT_DIR or
                            plots/predictive_variance).
  --variance-kind KIND       observation (default, includes noise) or latent.
  --formats png pdf         One or more formats: png, pdf, svg.
  --dpi INTEGER             Raster resolution (default: 200).

Observation variance is in training-standardized units. Each violin contains
one mean per assay, averaging all held-out variants before comparing assays.
Explicit run directories bypass that model's collection discovery. Multiple
runs for an assay require explicit selection; no rerun is chosen silently.
Command-line paths override environment defaults. Relative paths use the
repository root. Files are written directly into --output-dir; choose a
different destination for each comparison to preserve earlier figures.

Example (both families in the same collection):
  sbatch sbatch_run/run_plotting/plot_predictive_variance.sbatch \
    --kermut-results-dir results/hierarchical_nkn \
    --nkn-results-dir results/hierarchical_nkn
HELP
      exit 0 ;;
  esac
done

find_project_root() {
  local candidate
  candidate="$(cd -- "$1" 2>/dev/null && pwd -P)" || return 1
  while [[ ! -f "$candidate/src/plotting/predictive_variance.py" || ! -f "$candidate/src/experiments/cli.py" ]]; do
    [[ "$candidate" != "/" ]] || return 1
    candidate="$(dirname -- "$candidate")"
  done
  printf '%s\n' "$candidate"
}

if project_root="$(find_project_root "${SLURM_SUBMIT_DIR:-$PWD}")"; then
  :
elif project_root="$(find_project_root "$(dirname -- "${BASH_SOURCE[0]}")")"; then
  :
else
  echo "error: Submit from the nkn_protein repository or run its plotting script with bash." >&2
  exit 1
fi
cd -- "$project_root"

# These are singular argparse options: later command-line values take precedence.
plot_defaults=(
  --kermut-results-dir "${PLOT_KERMUT_RESULTS_DIR:-results}"
  --nkn-results-dir "${PLOT_NKN_RESULTS_DIR:-results}"
  --output-dir "${PLOT_OUTPUT_DIR:-plots/predictive_variance}"
)

export PYTHONNOUSERSITE=1
if ! command -v conda >/dev/null 2>&1; then
  module load anaconda3
fi
conda_base="$(conda info --base)"
source "$conda_base/etc/profile.d/conda.sh"
conda activate nkn_protein
set -u

export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
plot_cache_dir="$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/variance-plot.XXXXXXXX")"
trap 'rm -rf -- "$plot_cache_dir"' EXIT
export MPLCONFIGDIR="$plot_cache_dir/matplotlib"
export XDG_CACHE_HOME="$plot_cache_dir/cache"
mkdir -- "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

printf 'Job: %s\nRepository: %s\n' "${SLURM_JOB_ID:-manual}" "$project_root"
python -u -m plotting.predictive_variance "${plot_defaults[@]}" "$@"
