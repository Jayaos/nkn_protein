#!/bin/bash
#SBATCH --job-name=validate_prepared
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

# Submit after both preparation jobs finish:
# sbatch sbatch_run/run_prepare_rep/validate_prepared.sbatch [--esm2-config PATH]
# Validate the assets selected by prepare_esm2.sbatch and prepare_proteinmpnn.sbatch.
# CPU only, no inference/downloads; details print to the Slurm log.
# One CSV summary: results/preparation-check-<job-id>-<timestamp>.csv
# Override its destination with --summary-csv PATH (must be a new file).
# ESM2_CONFIG and --esm2-config select the ESM2 extraction recipe used to prepare.
set -eo pipefail

project_root="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
while [[ ! -f "$project_root/src/feature_generation/readiness.py" || ! -f "$project_root/sbatch_run/run_prepare_rep/list_assays.py" ]]; do
  if [[ "$project_root" == "/" ]]; then
    echo "error: Submit this job from the nkn_protein repository or a subdirectory." >&2
    exit 1
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

export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export CUDA_VISIBLE_DEVICES=""

printf 'Repository: %s\n' "$project_root"
python -u -m feature_generation.readiness "$@"
