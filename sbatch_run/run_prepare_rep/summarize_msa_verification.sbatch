#!/bin/bash
#SBATCH --job-name=summary_msa_verification
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%j.out

# Usage: sbatch sbatch_run/run_prepare_rep/summarize_msa_verification.sbatch
# Optional: append --job-id ARRAY_JOB_ID to summarize one verification array.
# To wait for an array (including failed tasks), submit with
# sbatch --dependency=afterany:ARRAY_JOB_ID <this-file> --job-id ARRAY_JOB_ID
# Other summary helper options are forwarded unchanged.
set -eo pipefail
submission_dir="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
project_root="$submission_dir"
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
python -u "$project_root/sbatch_run/run_prepare_rep/summarize_msa_verification.py" \
  --logs-dir "$submission_dir" "$@"
