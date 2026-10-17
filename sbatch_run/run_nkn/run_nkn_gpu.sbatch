#!/bin/bash
#SBATCH --job-name=nkn_gpu
#SBATCH --account=gts-yxie77-paid
#SBATCH --array=0-57
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

# Same default four-branch NKN preset as run_nkn_assays.sbatch, in float64.
# Load prepared features/scores and published five-fold singles assignments.
# Default: all three singles settings; assay-specific config rules still apply.
# Settings and folds run sequentially on one GPU per task.
# Training limits come from the preset unless explicitly overridden.
# Cholesky is the default; --inference-backend bbmm selects iterative inference.
# Usage: sbatch sbatch_run/run_nkn/run_nkn_gpu.sbatch [singles|SETTING] [OPTIONS]
# One setting: ... singles_random
# BBMM: ... singles_random --inference-backend bbmm
# Results grouping: --experiment-suffix NAME (or EXPERIMENT_SUFFIX=NAME).
# Inserts NAME below NKN_OUTPUT_ROOT (default results), as in the CPU launcher.
# Short timing run: ... singles_random --max-steps 2 --min-steps 0
# Override GPU hardware/resources using sbatch options before the script path.
# NKN_PYTHON may name an existing interpreter; otherwise activate nkn_protein.
set -eo pipefail

assays=(
  "A0A1I9GEU1_NEIME_Kennouche_2019"
  "A4_HUMAN_Seuma_2022"
  "AICDA_HUMAN_Gajula_2014_3cycles"
  "AMFR_HUMAN_Tsuboyama_2023_4G3O"
  "B2L11_HUMAN_Dutta_2010_binding-Mcl-1"
  "BCHB_CHLTE_Tsuboyama_2023_2KRU"
  "BLAT_ECOLX_Jacquier_2013"
  "BRCA2_HUMAN_Erwood_2022_HEK293T"
  "CAPSD_AAV2S_Sinai_2021"
  "CBX4_HUMAN_Tsuboyama_2023_2K28"
  "CUE1_YEAST_Tsuboyama_2023_2MYX"
  "ENV_HV1B9_DuenasDecamp_2016"
  "ERBB2_HUMAN_Elazar_2016"
  "F7YBW8_MESOW_Ding_2023"
  "F7YBW8_MESOW_Aakre_2015"
  "GCN4_YEAST_Staller_2018"
  "GLPA_HUMAN_Elazar_2016"
  "HIS7_YEAST_Pokusaeva_2019"
  "ISDH_STAAW_Tsuboyama_2023_2LHR"
  "KCNH2_HUMAN_Kozek_2020"
  "LYAM1_HUMAN_Elazar_2016"
  "MAFG_MOUSE_Tsuboyama_2023_1K1V"
  "MYO3_YEAST_Tsuboyama_2023_2BTT"
  "NPC1_HUMAN_Erwood_2022_HEK293T"
  "NPC1_HUMAN_Erwood_2022_RPE1"
  "NRAM_I33A0_Jiang_2016"
  "ODP2_GEOSE_Tsuboyama_2023_1W4G"
  "OPSD_HUMAN_Wan_2019"
  "OTU7A_HUMAN_Tsuboyama_2023_2L2D"
  "PIN1_HUMAN_Tsuboyama_2023_1I6C"
  "PITX2_HUMAN_Tsuboyama_2023_2L7M"
  "POLG_PESV_Tsuboyama_2023_2MXD"
  "Q837P4_ENTFA_Meier_2023"
  "Q837P5_ENTFA_Meier_2023"
  "RAD_ANTMA_Tsuboyama_2023_2CJJ"
  "RAF1_HUMAN_Zinkus-Boltz_2019"
  "RBP1_HUMAN_Tsuboyama_2023_2KWH"
  "RCD1_ARATH_Tsuboyama_2023_5OAO"
  "RD23A_HUMAN_Tsuboyama_2023_1IFY"
  "RFAH_ECOLI_Tsuboyama_2023_2LCL"
  "RPC1_LAMBD_Li_2019_high-expression"
  "RPC1_LAMBD_Li_2019_low-expression"
  "SAV1_MOUSE_Tsuboyama_2023_2YSB"
  "SCN5A_HUMAN_Glazer_2019"
  "SDA_BACSU_Tsuboyama_2023_1PV0"
  "SPG1_STRSG_Wu_2016"
  "SQSTM_MOUSE_Tsuboyama_2023_2RRU"
  "SR43C_ARATH_Tsuboyama_2023_2N88"
  "TCRG1_MOUSE_Tsuboyama_2023_1E0L"
  "THO1_YEAST_Tsuboyama_2023_2WQG"
  "TPOR_HUMAN_Bridgford_2020"
  "UBE4B_MOUSE_Starita_2013"
  "VG08_BPP22_Tsuboyama_2023_2GP8"
  "VKOR1_HUMAN_Chiasson_2020_activity"
  "YAIA_ECOLI_Tsuboyama_2023_2KVT"
  "YAP1_HUMAN_Araya_2012"
  "YNZC_BACSU_Tsuboyama_2023_2JVD"
  "FKBP3_HUMAN_Tsuboyama_2023_2KFV"
)

assay_count="${#assays[@]}"
last_index=$((assay_count - 1))
for ((index = 0; index < assay_count; index++)); do
  assay="${assays[$index]}"
  for ((previous = 0; previous < index; previous++)); do
    if [[ "$assay" == "${assays[$previous]}" ]]; then
      echo "error: Duplicate assay ID: $assay" >&2
      exit 1
    fi
  done
done

for argument in "$@"; do
  if [[ "$argument" == "--help" || "$argument" == "-h" ]]; then
    cat <<'HELP'
Usage: sbatch sbatch_run/run_nkn/run_nkn_gpu.sbatch [singles|SETTING] [OPTIONS]
SETTINGS: singles_random, singles_modulo, singles_contiguous
Default singles runs all three, sequentially, with five folds each.
Edit assays=(...) in this file to choose among the four supported human IDs.
Set the Slurm array range to match the list; current entries are shown below.
Prepared features and scores: data/features/pretrained/singles/<assay>/
Published folds: data/DMS_ProteinGym_substitutions/cv_folds_singles_substitutions/<assay>.csv
Options: --config PATH --max-steps N --min-steps N --log-every N
         --experiment-suffix NAME (overrides EXPERIMENT_SUFFIX)
         --num-threads N --features PATH --folds PATH --assay-index PATH
         --inference-backend {cholesky,bbmm}
         --bbmm-cg-tolerance FLOAT --bbmm-max-cg-iterations N
         --bbmm-num-trace-samples N --bbmm-max-lanczos-iterations N
         --bbmm-preconditioner-size N --bbmm-probe-seed N
This launcher uses real prepared inputs and CUDA; missing assets fail explicitly.
Supported IDs: A4_HUMAN_Seuma_2022, BRCA2_HUMAN_Erwood_2022_HEK293T,
               FKBP3_HUMAN_Tsuboyama_2023_2KFV, S22A1_HUMAN_Yee_2023_activity.
NKN_OUTPUT_ROOT changes the results root (default results).
Results grouping: --experiment-suffix NAME inserts one folder below that root.
Without a suffix, existing output paths are preserved.
Assay, device, and per-task output paths are managed by this launcher.
Cholesky is the default. BBMM requires linear_operator==0.5.2 in the environment.
Short timing run (configured human assays, five folds each):
  sbatch sbatch_run/run_nkn/run_nkn_gpu.sbatch singles_random --max-steps 2 --min-steps 0
BBMM with the same prepared data, published folds, and GPU resources:
  sbatch sbatch_run/run_nkn/run_nkn_gpu.sbatch singles_random --inference-backend bbmm
HELP
    printf '\nConfigured human assays: %s; submit with --array=0-%s\n' "$assay_count" "$last_index"
    for index in "${!assays[@]}"; do
      printf '  %s: %s\n' "$index" "${assays[$index]}"
    done
    exit 0
  fi
  case "$argument" in
    --assay|--assay=*|--output|--output=*|--device|--device=*)
      echo "error: The array manages assay IDs, output paths, and CUDA. Use NKN_OUTPUT_ROOT to change the results root." >&2
      exit 1 ;;
  esac
done

experiment_suffix="${EXPERIMENT_SUFFIX:-}"
runner_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-suffix)
      if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
        echo "error: --experiment-suffix requires a folder name." >&2
        exit 1
      fi
      experiment_suffix="$2"
      shift 2 ;;
    --experiment-suffix=*)
      set -- --experiment-suffix "${1#*=}" "${@:2}" ;;
    *)
      runner_args+=("$1")
      shift ;;
  esac
done
if [[ "$experiment_suffix" == *[/\\]* || "$experiment_suffix" == "." || "$experiment_suffix" == ".." ]]; then
  echo "error: Experiment suffix must be a single folder name, not a path." >&2
  exit 1
fi
set -- "${runner_args[@]}"

task_index="${SLURM_ARRAY_TASK_ID:-0}"
# Check the decimal spelling and length before arithmetic to reject both
# shell arithmetic expressions and integer overflow from invalid indices.
if [[ ! "$task_index" =~ ^(0|[1-9][0-9]*)$ ]] \
    || (( ${#task_index} > ${#last_index} )) \
    || (( task_index > last_index )); then
  echo "error: SLURM_ARRAY_TASK_ID must be an integer from 0 to $last_index for the configured human assays." >&2
  exit 1
fi
assay="${assays[$task_index]}"
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
output_root="${NKN_OUTPUT_ROOT:-results}"
run_root="$output_root/${experiment_suffix:+$experiment_suffix/}$assay/nkn_esm2_proteinmpnn_${selection}_gpu_${run_id}_${task_index}"
config="${NKN_CONFIG:-configs/esm2_proteinmpnn_nkn_two_scores_additive_four_branches_noise_initial_1p0.json}"
printf 'Job: %s\nRepository: %s\nPython: %s\n' "$run_id" "$project_root" "$python_exec"
printf 'Array task: %s_%s; assay: %s\n' \
  "${SLURM_ARRAY_JOB_ID:-manual}" "$task_index" "$assay"
printf 'Results: %s\n' "$run_root"
exec "$python_exec" -u -m experiments.large_assay_gpu \
  --config "$config" --num-threads "$cpu_count" \
  "$@" --assay "$assay" --output "$run_root"
