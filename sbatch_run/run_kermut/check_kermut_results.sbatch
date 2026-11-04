#!/bin/bash
#SBATCH --job-name=check_kermut_results
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%j.out

# Check saved Kermut results without training or loading checkpoint tensors.
# Default: all catalog assays, kermut_prosst_esmif runs, all three singles settings.
# Any one complete matching run satisfies an assay; older failed attempts do not
# invalidate a successful rerun. Results are never modified.
# Relative results/catalog/report paths are resolved from the repository root.
#
#   sbatch sbatch_run/run_kermut/check_kermut_results.sbatch results/kermut_prosst_esmif_all_assays
#   sbatch sbatch_run/run_kermut/check_kermut_results.sbatch /path/to/results --report /path/to/check.json
#   bash sbatch_run/run_kermut/check_kermut_results.sbatch --help
set -eo pipefail

usage() {
  cat <<'EOF'
Usage: sbatch sbatch_run/run_kermut/check_kermut_results.sbatch RESULTS_DIR [OPTIONS]

Check each catalog assay for a complete matching Kermut run under RESULTS_DIR.
Defaults: --run-prefix kermut_prosst_esmif and all three singles CV settings.

Options:
  --assay-index PATH  Catalog override; otherwise ASSAY_INDEX or standard catalog locations.
  --run-prefix NAME   Match this run-name prefix (default kermut_prosst_esmif).
  --settings NAME... Select singles_random, singles_modulo, and/or singles_contiguous.
  --report PATH      Save the audit as a new JSON file; existing files are not overwritten.
  -h, --help         Print this help without activating Conda.

Relative paths use the repository root. This job uses one CPU and no GPU.
Exit status: 0 all complete; 1 missing/incomplete assays; 2 invalid inputs.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  project_root="$(cd -- "$SLURM_SUBMIT_DIR" && pwd -P)"
else
  # Direct Bash execution can start outside the repository.
  project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
fi
while [[ ! -f "$project_root/sbatch_run/run_kermut/check_kermut_results.py" ]]; do
  if [[ "$project_root" == "/" ]]; then
    echo "error: Submit this job from the nkn_protein repository or a subdirectory." >&2
    exit 2
  fi
  project_root="$(dirname -- "$project_root")"
done
cd -- "$project_root"

export PYTHONNOUSERSITE=1
if ! command -v conda >/dev/null 2>&1; then
  module load anaconda3
fi
conda_base="$(conda info --base)"
source "$conda_base/etc/profile.d/conda.sh"
conda activate nkn_protein
set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

printf 'Repository: %s\n' "$project_root"
python -u sbatch_run/run_kermut/check_kermut_results.py "$@"
