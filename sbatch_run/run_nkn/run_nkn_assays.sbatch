#!/bin/bash
#SBATCH --job-name=nkn_assays
#SBATCH --account=gts-yxie77-paid
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=192G
#SBATCH --time=2:00:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

# Timing test: A0A2Z5U3Z0_9INFA_Doud_2016 (10,715 single mutants), array index 0.
# Submit from the repository or a subdirectory; the task delegates to the
# shared NKN runner with four branches and initial noise variance 1.0.
# Default: singles (random, modulo, contiguous; populated published folds), with the
# additive two-score preset and caches under data/features/pretrained/singles.
# Branches: [1]; final NKN: [8, 4, 4, 2, 1]. Minimum optimizer steps: 1000.
# Observation-noise variance prior: none. Initial noise variance: 1.0.
# Enabled: sequence_site, sequence_global, inverse_folding, structure_distance.
# Resources above are for this assay. --parallel divides the task's CPUs
# between its settings and folds.
# Usage: sbatch sbatch_run/run_nkn/run_nkn_assays.sbatch [singles|SETTING] [--config PATH] [--parallel] [--held-out-interval N]
# Add --held-out-interval 100 to record diagnostic Spearman every 100 updates.
# Add --stopping-criterion validation_spearman for three train / one validation /
# one test folds. --validation-interval 100 --patience 100 --min-steps 1000 are defaults.
# For one random-CV timing run, pass singles_random instead of the default singles.
# Logs: nkn_assays-<array-job-id>_<task-index>.out in the submission directory.
# Results: results/<assay>/nkn_esm2_proteinmpnn_<selection>_<task-job-id>/
# Add --experiment-suffix NAME (or EXPERIMENT_SUFFIX=NAME) to insert NAME under results/.
# Config precedence: --config PATH, NKN_CONFIG, then the four-branch additive
# noise_initial_1p0 preset for every assay.
# Four-branch alternative with the original noise_initial=0.01:
#   --config configs/esm2_proteinmpnn_nkn_two_scores_additive_four_branches.json
# Optional: --noise-prior half_cauchy --noise-prior-scale 0.1.
set -eo pipefail

# This timing-test selection has its own index; preparation uses the full catalog.
assays=(
  "BBC1_YEAST_Tsuboyama_2023_1TG0"
)

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf 'Results grouping: --experiment-suffix NAME (overrides EXPERIMENT_SUFFIX); omitted keeps existing paths.\n'
  printf 'Usage: sbatch sbatch_run/run_nkn/run_nkn_assays.sbatch [singles|SETTING] [--config PATH] [--parallel] [--held-out-interval N]\n'
  printf 'Timing test: array index 0; A0A2Z5U3Z0_9INFA_Doud_2016 (10,715 single mutants).\n'
  printf 'Default config for every assay: configs/esm2_proteinmpnn_nkn_two_scores_additive_four_branches_noise_initial_1p0.json\n'
  printf 'Initial noise variance: 1.0 (explicit --config or NKN_CONFIG overrides the default).\n'
  printf 'Branches: sequence_site, sequence_global, inverse_folding, structure_distance\n'
  printf 'Stopping: --stopping-criterion training_nll|validation_spearman (default training_nll)\n'
  printf 'Noise prior: none; override with --noise-prior none|half_cauchy --noise-prior-scale SCALE\n'
  printf 'Overrides: --validation-interval 100 --patience 100 --min-steps 1000 (optimizer steps)\n'
  for index in "${!assays[@]}"; do
    printf '  %s: %s\n' "$index" "${assays[$index]}"
  done
  exit 0
fi

task_index="${SLURM_ARRAY_TASK_ID:-}"
case "$task_index" in
  0) ;;
  *)
    printf 'error: SLURM_ARRAY_TASK_ID must be 0 (got "%s"). Submit this script with sbatch; its default array is 0.\n' "$task_index" >&2
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
default_config="configs/esm2_proteinmpnn_nkn_two_scores_additive_four_branches_noise_initial_1p0.json"
export NKN_CONFIG="${NKN_CONFIG:-$default_config}"
printf 'Array task: %s_%s; assay: %s\n' "${SLURM_ARRAY_JOB_ID:-manual}" "$task_index" "$ASSAY"
exec bash "$project_root/sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch" "$@"
