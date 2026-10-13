#!/bin/bash
#SBATCH --job-name=nkn_FKBP3
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

# NKN on FKBP3_HUMAN_Tsuboyama_2023_2KFV using the shared NKN runner.
# Defaults: ESM2 650M and sequence-conditioned ProteinMPNN caches under
# data/features/pretrained/singles/FKBP3_HUMAN_Tsuboyama_2023_2KFV, with only ESM2 650M
# scores feeding the GP mean via configs/esm2_proteinmpnn_nkn.json.
# Submit from the repository root or a subdirectory. Arguments and optional
# cache/config overrides are handled by the shared launcher; ASSAY is fixed here.
# Usage: sbatch sbatch_run/run_nkn/run_nkn_FKBP3_HUMAN_Tsuboyama_2023_2KFV.sbatch [singles|SETTING] [--config PATH] [--parallel]
# Default: the three published singles CV protocols (15 fold fits).
# Default training: up to 10,000 steps per initialization, with early stopping.
# This assay has only singles; settings requiring doubles are unavailable.
set -eo pipefail

project_root="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
while [[ ! -f "$project_root/sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch" ]]; do
  if [[ "$project_root" == "/" ]]; then
    echo "error: Submit this job from the nkn_protein repository or a subdirectory." >&2
    exit 1
  fi
  project_root="$(dirname -- "$project_root")"
done

export ASSAY="FKBP3_HUMAN_Tsuboyama_2023_2KFV"
exec bash "$project_root/sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch" "$@"
