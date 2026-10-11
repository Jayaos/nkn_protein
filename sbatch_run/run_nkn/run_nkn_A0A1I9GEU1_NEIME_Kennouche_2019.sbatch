#!/bin/bash
#SBATCH --job-name=nkn_NEIME
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

# NKN on A0A1I9GEU1_NEIME_Kennouche_2019 using the shared NKN runner.
# Defaults: ESM2 650M and sequence-conditioned ProteinMPNN caches under
# data/features/pretrained/A0A1I9GEU1_NEIME_Kennouche_2019, with both scores
# feeding the GP mean via configs/esm2_proteinmpnn_nkn.json.
# Submit from the repository root or a subdirectory. Arguments and optional
# cache/config overrides are handled by the shared launcher; ASSAY is fixed here.
# Usage: sbatch sbatch_run/run_nkn/run_nkn_A0A1I9GEU1_NEIME_Kennouche_2019.sbatch [singles|SETTING] [--config PATH] [--parallel]
# Default: the three published singles CV protocols (15 fold fits).
set -eo pipefail

project_root="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
while [[ ! -f "$project_root/sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch" ]]; do
  if [[ "$project_root" == "/" ]]; then
    echo "error: Submit this job from the nkn_protein repository or a subdirectory." >&2
    exit 1
  fi
  project_root="$(dirname -- "$project_root")"
done

export ASSAY="A0A1I9GEU1_NEIME_Kennouche_2019"
exec bash "$project_root/sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch" "$@"
