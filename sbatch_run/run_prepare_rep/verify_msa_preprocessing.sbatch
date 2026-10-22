#!/bin/bash
#SBATCH --job-name=verify_msa_brca1
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%j.out

# CPU-only BRCA1 pilot. Requires the manually installed step-1 environment.
# Submit from the repository or a subdirectory. No checkpoint inference is run.
# Optional: sbatch .../verify_msa_preprocessing.sbatch --reference-csv /path/to/DMS_substitutions.csv
set -eo pipefail
project_root="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
while [[ ! -f "$project_root/env_msa_transformer.yaml" ]]; do
  if [[ "$project_root" == / ]]; then
    echo "error: Submit from the nkn_protein repository or a subdirectory." >&2
    exit 1
  fi
  project_root="$(dirname -- "$project_root")"
done
cd -- "$project_root"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CONDA_NUMBER_CHANNEL_NOTICES=0
unset PYTHONPATH
if ! command -v conda >/dev/null 2>&1; then
  module load anaconda3
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nkn_msa_transformer
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"
python -u "$project_root/sbatch_run/run_prepare_rep/verify_msa_preprocessing.py" \
  --config "$project_root/configs/setup/msa_preprocessing_brca1.json" \
  --cpus "$OMP_NUM_THREADS" "$@"
