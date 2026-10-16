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

# Submit from the repository or a subdirectory after preparation finishes:
# sbatch sbatch_run/run_prepare_rep/validate_prepared.sbatch [--config PATH]
# CPU only: validate all catalog assays against the NKN experiment preset,
# including complete single-mutant coverage, features, scores, and structures.
# BRCA2 uses the shared experiment resolver (no distance kernel or ProteinMPNN score).
# No inference, training, or downloads. Feature caches are read without changes.
# Results and errors go directly to the Slurm log; no report files are created.
# A failed assay is logged without stopping the remaining checks; exit 1 if any fail.
set -eo pipefail

project_root="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
while [[ ! -f "$project_root/src/experiments/cli.py" || ! -f "$project_root/sbatch_run/run_prepare_rep/list_assays.py" ]]; do
  if [[ "$project_root" == "/" ]]; then
    echo "error: Submit this job from the nkn_protein repository or a subdirectory." >&2
    exit 1
  fi
  project_root="$(dirname -- "$project_root")"
done
cd -- "$project_root"

config="configs/esm2_proteinmpnn_nkn_two_scores_additive_four_branches_noise_initial_0p5.json"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
        echo "error: --config requires an NKN experiment JSON file path." >&2
        exit 1
      fi
      config="$2"
      shift 2 ;;
    --config=*)
      config="${1#--config=}"
      shift ;;
    -h|--help)
      printf 'Usage: sbatch sbatch_run/run_prepare_rep/validate_prepared.sbatch [--config PATH]\n'
      printf 'Validates all catalog single mutants on CPU using an NKN experiment config.\n'
      exit 0 ;;
    *)
      echo "error: Unknown argument: $1" >&2
      exit 1 ;;
  esac
done
if [[ ! -f "$config" ]]; then
  echo "error: NKN experiment configuration does not exist: $config" >&2
  exit 1
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export CUDA_VISIBLE_DEVICES=""

assay_index="data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv"
features_root="data/features/pretrained/singles"
printf 'Repository: %s\nConfig: %s\n' "$project_root" "$config"

# Fail shared setup before producing a misleading readiness summary.
assay_list="$(python sbatch_run/run_prepare_rep/list_assays.py \
  --config configs/extraction/esm2_650m_centered.json)"

ready=0
failed=0
index=0
while IFS= read -r assay; do
  printf '\nValidating array index %s: %s\n' "$index" "$assay"
  if python -u -m experiments.cli validate \
    --features "$features_root/$assay" \
    --config "$config" \
    --assay-index "$assay_index" \
    --full-assay --singles-only --device cpu --no-output; then
    status=READY
    ready=$((ready + 1))
  else
    status=FAILED
    failed=$((failed + 1))
  fi
  printf '%s\t%s\t%s\n' "$index" "$assay" "$status"
  index=$((index + 1))
done <<< "$assay_list"

printf '\nReady: %s / %s; failed: %s\n' "$ready" "$index" "$failed"
if (( failed )); then
  echo 'error: Some assays are not ready; see validation errors above.' >&2
  exit 1
fi
