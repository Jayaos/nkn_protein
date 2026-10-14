#!/bin/bash
#SBATCH --job-name=nkn_assays
#SBATCH --account=gts-yxie77-paid
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

# One assay per array task, in the order below. Submit from the repository or
# a subdirectory; each task delegates to the shared NKN runner.
# Default: singles (random, modulo, contiguous; five folds each), with the
# [4, 2, 1] two-score preset and caches under data/features/pretrained/singles.
# Resources above are per assay task. The array parallelizes assays; --parallel
# additionally divides each task's CPUs between its settings and folds.
# Usage: sbatch sbatch_run/run_nkn/run_nkn_assays.sbatch [singles|SETTING] [--config PATH] [--parallel] [--held-out-interval N]
# Add --held-out-interval 100 to record diagnostic Spearman every 100 updates.
# Add --stopping-criterion validation_spearman for three train / one validation /
# one test folds. --validation-interval 100 --patience 100 --min-steps 0 are defaults.
# Limit assay concurrency: sbatch --array=0-4%2 sbatch_run/run_nkn/run_nkn_assays.sbatch
# Logs: nkn_assays-<array-job-id>_<task-index>.out in the submission directory.
# Results: results/<assay>/nkn_esm2_proteinmpnn_<selection>_<task-job-id>/
# Config precedence: --config PATH, NKN_CONFIG, then the two-score preset.
set -eo pipefail

assays=(
  "PIN1_HUMAN_Tsuboyama_2023_1I6C"
  "VKOR1_HUMAN_Chiasson_2020_activity"
  "BLAT_ECOLX_Jacquier_2013"
  "A0A1I9GEU1_NEIME_Kennouche_2019"
  "FKBP3_HUMAN_Tsuboyama_2023_2KFV"
)

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf 'Usage: sbatch sbatch_run/run_nkn/run_nkn_assays.sbatch [singles|SETTING] [--config PATH] [--parallel] [--held-out-interval N]\n'
  printf 'Default array: 0-4; each task runs one assay with both GP mean scores.\n'
  printf 'Stopping: --stopping-criterion training_nll|validation_spearman (default training_nll)\n'
  printf 'Overrides: --validation-interval 100 --patience 100 --min-steps 0 (optimizer steps)\n'
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
while [[ ! -f "$project_root/sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch" ]]; do
  if [[ "$project_root" == "/" ]]; then
    echo "error: Submit this job from the nkn_protein repository or a subdirectory." >&2
    exit 1
  fi
  project_root="$(dirname -- "$project_root")"
done

export ASSAY="${assays[$task_index]}"
export NKN_CONFIG="${NKN_CONFIG:-configs/esm2_proteinmpnn_nkn_two_scores.json}"
printf 'Array task: %s_%s; assay: %s\n' "${SLURM_ARRAY_JOB_ID:-manual}" "$task_index" "$ASSAY"
exec bash "$project_root/sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch" "$@"
