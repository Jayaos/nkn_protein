#!/bin/bash
#SBATCH --job-name=nkn_s22a1_gpu
#SBATCH --account=gts-yxie77-paid
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

# Real assay: S22A1_HUMAN_Yee_2023_activity (human OCT1; 10,094 single mutants).
# Same default four-branch NKN preset as run_nkn_assays.sbatch, in float64.
# Default: all three singles settings and published five-fold assignments.
# Settings and folds run sequentially on one GPU, using all prepared singles.
# Training limits come from the preset unless explicitly overridden.
# Cholesky is the default; --inference-backend bbmm selects iterative inference.
# A small synthetic numerical preflight precedes training.
# Usage: sbatch sbatch_run/run_nkn/run_nkn_gpu.sbatch [singles|SETTING] [OPTIONS]
# One setting: ... singles_random
# BBMM: ... singles_random --inference-backend bbmm
# Short timing run (all variants): ... singles_random --max-steps 2 --min-steps 0
# Override GPU hardware/resources using sbatch options before the script path.
# NKN_PYTHON may name an existing interpreter; otherwise activate nkn_protein.
set -eo pipefail

for argument in "$@"; do
  if [[ "$argument" == "--help" || "$argument" == "-h" ]]; then
    cat <<'HELP'
Usage: sbatch sbatch_run/run_nkn/run_nkn_gpu.sbatch [singles|SETTING] [OPTIONS]
SETTINGS: singles_random, singles_modulo, singles_contiguous
Default singles runs all three, sequentially, with five folds each.
Assay: S22A1_HUMAN_Yee_2023_activity (human OCT1; 10,094 single mutants).
Options: --config PATH --max-steps N --min-steps N --log-every N
         --num-threads N --output PATH --features PATH --folds PATH --assay-index PATH
         --inference-backend {cholesky,bbmm}
         --bbmm-cg-tolerance FLOAT --bbmm-max-cg-iterations N
         --bbmm-num-trace-samples N --bbmm-max-lanczos-iterations N
         --bbmm-preconditioner-size N --bbmm-probe-seed N
This launcher uses CUDA, prepared real features, and published folds.
Cholesky is the default. BBMM requires linear_operator==0.5.2 in the environment.
Short timing run (all variants, five folds):
  sbatch sbatch_run/run_nkn/run_nkn_gpu.sbatch singles_random --max-steps 2 --min-steps 0
BBMM with the same assay, folds, and GPU resources:
  sbatch sbatch_run/run_nkn/run_nkn_gpu.sbatch singles_random --inference-backend bbmm
HELP
    exit 0
  fi
done

if [[ "${SLURM_ARRAY_TASK_ID:-0}" != "0" ]]; then
  echo "error: This single-assay job uses array index 0 only." >&2
  exit 1
fi
project_root="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
while [[ ! -f "$project_root/src/experiments/large_assay_gpu.py" ]]; do
  if [[ "$project_root" == "/" ]]; then
    echo "error: Submit from the nkn_protein repository or a subdirectory." >&2
    exit 1
  fi
  project_root="$(dirname -- "$project_root")"
done
cd -- "$project_root"

export PYTHONNOUSERSITE=1
if [[ -n "${NKN_PYTHON:-}" ]]; then
  python_exec="$NKN_PYTHON"
else
  if ! command -v conda >/dev/null 2>&1; then
    module load anaconda3
  fi
  conda_base="$(conda info --base)"
  source "$conda_base/etc/profile.d/conda.sh"
  conda activate nkn_protein
  python_exec="$(command -v python)"
fi
set -u

cpu_count="${SLURM_CPUS_PER_TASK:-8}"
if [[ ! "$cpu_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: SLURM_CPUS_PER_TASK must be a positive integer." >&2
  exit 1
fi
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$cpu_count"
export MKL_NUM_THREADS="$cpu_count"
export OPENBLAS_NUM_THREADS="$cpu_count"
# Preserve the GPU allocation exposed through CUDA_VISIBLE_DEVICES by Slurm.
run_id="${SLURM_JOB_ID:-manual_$(date -u +%Y%m%dT%H%M%SZ)}"
selection="singles"
case "${1:-}" in
  singles|singles_random|singles_modulo|singles_contiguous) selection="$1" ;;
esac
run_root="results/S22A1_HUMAN_Yee_2023_activity/nkn_esm2_proteinmpnn_${selection}_gpu_${run_id}"
config="${NKN_CONFIG:-configs/esm2_proteinmpnn_nkn_two_scores_additive_four_branches_noise_initial_1p0.json}"
printf 'Job: %s\nRepository: %s\nPython: %s\n' "$run_id" "$project_root" "$python_exec"
exec "$python_exec" -u -m experiments.large_assay_gpu \
  --config "$config" --output "$run_root" --num-threads "$cpu_count" \
  "$@"
