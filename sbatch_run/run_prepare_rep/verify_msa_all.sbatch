#!/bin/bash
#SBATCH --job-name=verify_msa_all
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%A_%a.out

# Pass --array=0-(catalog row count minus one)%4 when submitting.
# Reuse the CPU launcher; SBATCH directives in that child script are not applied.
set -euo pipefail
project_root="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
while [[ ! -f "$project_root/env_msa_transformer.yaml" ]]; do
  if [[ "$project_root" == / ]]; then
    echo "error: Submit from the nkn_protein repository or a subdirectory." >&2
    exit 1
  fi
  project_root="$(dirname -- "$project_root")"
done
: "${SLURM_ARRAY_TASK_ID:?Submit this job with --array}"
bash "$project_root/sbatch_run/run_prepare_rep/verify_msa_preprocessing.sbatch" \
  --assay-index "$SLURM_ARRAY_TASK_ID" --specialized-only "$@"
