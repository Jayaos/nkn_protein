#!/bin/bash
#SBATCH --job-name=kermut_prosst_esmif_five_fix
#SBATCH --account=gts-yxie77-paid
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

# Rerun the five assays repaired with migrate_structure_cache_names --repair-renamed.
# Same resources, config, shared runner, and result prefix as
# run_kermut_prosst_esmif_all_assays.sbatch; one assay per task, in catalog order.
# This array uses 0-4, corresponding to full-catalog tasks 24, 29, 30, 31, 47.
# Default: singles (random, modulo, contiguous), using prepared ProSST-2048 and
# ESM-IF1 caches and configs/kermut_prosst_esmif.json. CPU only.
# ProSST pooled global embeddings and ESM-IF1 inverse-folding probabilities feed
# the kernel; both models supply GP mean scores. Structure manifests are selected
# by the config, matching data/structures/protssn_proteingym/manifests by default.
# Prepare both models' caches under data/features/pretrained/singles/<assay>.
# All arguments and feature/fold/config environment overrides reach the runner.
# KERMUT_RUN_PREFIX overrides the default kermut_prosst_esmif result prefix.
# --parallel enables concurrent fits within each assay's CPU allocation.
# all/mixed/single_to_double additionally require double mutants and full caches.
#
# Submit from the repository root or a subdirectory:
#   sbatch sbatch_run/run_kermut/run_kermut_prosst_esmif_five_fix_assays.sbatch
# Match the earlier parallel run and its experiment group:
#   sbatch sbatch_run/run_kermut/run_kermut_prosst_esmif_five_fix_assays.sbatch singles --parallel --experiment-suffix kermut_prosst_esmif_all_assays
# Limit concurrent assay tasks to four:
#   sbatch --array=0-4%4 sbatch_run/run_kermut/run_kermut_prosst_esmif_five_fix_assays.sbatch singles --parallel
# Print the task-to-assay mapping without submitting or training:
#   bash sbatch_run/run_kermut/run_kermut_prosst_esmif_five_fix_assays.sbatch --help
# Results: <output-root>/[EXPERIMENT_SUFFIX/]<assay>/kermut_prosst_esmif_<selection>_<job-id>/
# --output-root PATH overrides KERMUT_OUTPUT_ROOT (default results/).
# Relative output roots are resolved from the repository root.
set -eo pipefail

assays=(
  "POLG_CXB3N_Mattenberger_2021"
  "POLG_DEN26_Suphatrakul_2023"
  "A4D664_9INFA_Soh_2019"
  "HMDH_HUMAN_Jiang_2019"
  "MSH2_HUMAN_Jia_2020"
)

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf 'Output root: --output-root PATH, then KERMUT_OUTPUT_ROOT, then results/. Relative paths use the repository root.\n'
  printf 'Usage: sbatch sbatch_run/run_kermut/run_kermut_prosst_esmif_five_fix_assays.sbatch [singles|all|SETTING] [--config PATH] [--parallel] [--output-root PATH] [--experiment-suffix NAME]\n'
  printf 'Default array: 0-4; five repaired assays, one assay per task (full-catalog indices 24,29,30,31,47).\n'
  printf 'Uses run_kermut_esm2_proteinmpnn.sbatch with all arguments forwarded.\n'
  printf 'Default config: configs/kermut_prosst_esmif.json; --config overrides KERMUT_CONFIG.\n'
  printf 'Inputs: ProSST-2048 global embeddings and ESM-IF1 inverse-folding probabilities; both models supply GP mean scores.\n'
  printf 'Structure manifests: data/structures/protssn_proteingym/manifests, matching ProSST and ESM-IF1 features.\n'
  printf 'Results use kermut_prosst_esmif_<selection>_<job-id> (KERMUT_RUN_PREFIX overrides).\n'
  for index in "${!assays[@]}"; do
    printf '  %s: %s\n' "$index" "${assays[$index]}"
  done
  exit 0
fi

task_index="${SLURM_ARRAY_TASK_ID:-}"
case "$task_index" in
  [0-4]) ;;
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
export KERMUT_CONFIG="${KERMUT_CONFIG:-configs/kermut_prosst_esmif.json}"
export KERMUT_RUN_PREFIX="${KERMUT_RUN_PREFIX:-kermut_prosst_esmif}"
printf 'Array task: %s_%s; assay: %s\n' "${SLURM_ARRAY_JOB_ID:-manual}" "$task_index" "$ASSAY"
exec bash "$project_root/sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn.sbatch" "$@"
