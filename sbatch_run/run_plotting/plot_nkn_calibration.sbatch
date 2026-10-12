#!/bin/bash
#SBATCH --job-name=nkn_calibration_plots
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%j.out

# Plot saved NKN calibration metrics with the existing nkn_protein environment.
# Submit from the repository or a subdirectory, or run this file with bash.
# Relative input/output paths use the repository root. No model fitting occurs.
set -eo pipefail

for argument in "$@"; do
  case "$argument" in
    -h|--help)
      cat <<'HELP'
Usage: sbatch sbatch_run/run_plotting/plot_nkn_calibration.sbatch [OPTIONS]

Plot calibration diagnostics from saved NKN model runs.
  --results-dir PATH   Results root (default: PLOT_RESULTS_DIR or results).
  --assay ID           Restrict discovery to an assay; repeat for several.
  --run-dir PATH       Select a model-run directory; repeat for several.
  --output-dir PATH    Output root (default: PLOT_OUTPUT_DIR, or each run/plots).
  --formats png pdf    One or more output formats.
  --dpi INTEGER       Raster resolution (default: 200).

Command-line paths override environment defaults. Relative paths use the
repository root. A custom output root keeps separate assay/run directories.
This job reads existing results and does not run training.
HELP
      exit 0 ;;
  esac
done

find_project_root() {
  local candidate
  candidate="$(cd -- "$1" 2>/dev/null && pwd -P)" || return 1
  while [[ ! -f "$candidate/src/plotting/calibration.py" || ! -f "$candidate/src/experiments/cli.py" ]]; do
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

# Argparse uses the last supplied value, so explicit CLI options take precedence.
plot_defaults=(--results-dir "${PLOT_RESULTS_DIR:-results}")
if [[ -n "${PLOT_OUTPUT_DIR:-}" ]]; then
  plot_defaults+=(--output-dir "$PLOT_OUTPUT_DIR")
fi

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
plot_cache_dir="$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/nkn-plot.XXXXXXXX")"
trap 'rm -rf -- "$plot_cache_dir"' EXIT
export MPLCONFIGDIR="$plot_cache_dir/matplotlib"
export XDG_CACHE_HOME="$plot_cache_dir/cache"
mkdir -- "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

printf 'Job: %s\nRepository: %s\n' "${SLURM_JOB_ID:-manual}" "$project_root"
python -u -m plotting.calibration "${plot_defaults[@]}" "$@"
