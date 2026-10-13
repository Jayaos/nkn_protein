#!/bin/bash
#SBATCH --job-name=kermut_assays
#SBATCH --account=gts-yxie77-paid
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

# One assay per array task, matching run_nkn_assays.sbatch's index order.
# Submit from the repository or a subdirectory; each task delegates to the
# shared Kermut runner with both ESM2 and ProteinMPNN GP mean scores.
# Default: singles (random, modulo, contiguous; five folds each), using
# configs/kermut_esm2_proteinmpnn.json and data/features/pretrained/singles.
# Resources above are per assay task. The array parallelizes assays; --parallel
# additionally divides each task's CPUs between its settings and folds.
# Usage: sbatch sbatch_run/run_kermut/run_kermut_assays.sbatch [singles|all|SETTING] [--config PATH] [--parallel]
# Limit assay concurrency: sbatch --array=0-4%2 sbatch_run/run_kermut/run_kermut_assays.sbatch
# Logs: kermut_assays-<array-job-id>_<task-index>.out in the submission directory.
# Results: results/<assay>/kermut_esm2_proteinmpnn_<selection>_<task-job-id>/
# Config precedence: --config PATH, KERMUT_CONFIG, then the default preset.
set -eo pipefail

assays=(
  "TPMT_HUMAN_Matreyek_2018"
  "RASK_HUMAN_Weng_2022_binding-DARPin_K55"
  "FKBP3_HUMAN_Tsuboyama_2023_2KFV"
  "DYR_ECOLI_Thompson_2019"
  "A0A1I9GEU1_NEIME_Kennouche_2019"
)

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf 'Usage: sbatch sbatch_run/run_kermut/run_kermut_assays.sbatch [singles|all|SETTING] [--config PATH] [--parallel]\n'
  printf 'Default array: 0-4; each task runs one assay with both GP mean scores.\n'
  for index in "${!assays[@]}"; do
    printf '  %s: %s\n' "$index" "${assays[$index]}"
  done
  exit 0
fi

task_index="${SLURM_ARRAY_TASK_ID:-}"
case "$task_index" in
  0|1|2|3|4) ;;
  *)
    printf 'error: SLURM_ARRAY_TASK_ID must be 0-4 (got "%s"). Submit this script with sbatch; its default array is 0-4.\n' "$task_index" >&2
    exit 1 ;;
esac

project_root="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
while [[ ! -f "$project_root/sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn.sbatch" ]]; do
  if [[ "$project_root" == "/" ]]; then
    echo "error: Submit this job from the nkn_protein repository or a subdirectory." >&2
    exit 1
  fi
  project_root="$(dirname -- "$project_root")"
done

export ASSAY="${assays[$task_index]}"
export KERMUT_CONFIG="${KERMUT_CONFIG:-configs/kermut_esm2_proteinmpnn.json}"
printf 'Array task: %s_%s; assay: %s\n' "${SLURM_ARRAY_JOB_ID:-manual}" "$task_index" "$ASSAY"
exec bash "$project_root/sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn.sbatch" "$@"
