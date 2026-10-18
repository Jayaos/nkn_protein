#!/bin/bash
#SBATCH --job-name=nkn_gpu
#SBATCH --account=gts-yxie77-paid
#SBATCH --array=0-85
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:V100:1
#SBATCH --constraint=gpu-v100-32GB
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

# 86 assays with more than 2,000 single-mutant variants; array indices 0-85.
# Same default four-branch NKN preset as run_nkn_assays.sbatch:
# initial noise variance 0.5, in float64.
# Load prepared features/scores and published singles assignments; run populated folds.
# Default: all three singles settings; assay-specific config rules still apply.
# Settings and folds run sequentially on one GPU per task.
# Training limits come from the preset unless explicitly overridden.
# Cholesky is the default; --gp-computation-backend bbmm applies to training and prediction.
# Usage: sbatch sbatch_run/run_nkn/run_nkn_gpu.sbatch [NAME] [singles|SETTING] [OPTIONS]
# One setting: ... singles_random
# BBMM: ... singles_random --gp-computation-backend bbmm
# Results grouping: --experiment-suffix NAME (or EXPERIMENT_SUFFIX=NAME).
# A leading NAME is also accepted: ... gpu_bbmm_test --gp-computation-backend bbmm
# Inserts NAME below the output root, as in the CPU launcher.
# --output-root PATH overrides NKN_OUTPUT_ROOT (default results/).
# Relative output roots are resolved from the repository root.
# Short timing run: ... singles_random --max-steps 2 --min-steps 0
# Override GPU hardware/resources using sbatch options before the script path.
# NKN_PYTHON may name an existing interpreter; otherwise activate nkn_protein.
set -eo pipefail

# DMS_number_single_mutants > 2000, in catalog order, from
# data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv.
assays=(
  "A0A140D2T1_ZIKV_Sourisseau_2019"
  "A0A192B1T2_9HIV1_Haddox_2018"
  "A0A2Z5U3Z0_9INFA_Doud_2016"
  "A0A2Z5U3Z0_9INFA_Wu_2014"
  "A4D664_9INFA_Soh_2019"
  "A4GRB6_PSEAI_Chen_2020"
  "ACE2_HUMAN_Chan_2020"
  "ADRB2_HUMAN_Jones_2020"
  "AMIE_PSEAE_Wrenbeck_2017"
  "ANCSZ_Hobbs_2022"
  "BLAT_ECOLX_Deng_2012"
  "BLAT_ECOLX_Firnberg_2014"
  "BLAT_ECOLX_Stiffler_2015"
  "C6KNH7_9INFA_Lee_2018"
  "CAR11_HUMAN_Meitlis_2020_gof"
  "CAR11_HUMAN_Meitlis_2020_lof"
  "CAS9_STRP1_Spencer_2017_positive"
  "CBS_HUMAN_Sun_2020"
  "CCR5_HUMAN_Gill_2023"
  "CD19_HUMAN_Klesmith_2019_FMC_singles"
  "CP2C9_HUMAN_Amorosi_2021_abundance"
  "CP2C9_HUMAN_Amorosi_2021_activity"
  "DYR_ECOLI_Nguyen_2023"
  "DYR_ECOLI_Thompson_2019"
  "ENV_HV1BR_Haddox_2016"
  "ESTA_BACSU_Nutschel_2020"
  "HEM3_HUMAN_Loggerenberg_2023"
  "HMDH_HUMAN_Jiang_2019"
  "HSP82_YEAST_Cote-Hammarlof_2020_growth-H2O2"
  "HSP82_YEAST_Flynn_2019"
  "HSP82_YEAST_Mishra_2016"
  "HXK4_HUMAN_Gersing_2022_activity"
  "HXK4_HUMAN_Gersing_2023_abundance"
  "I6TAH8_I68A0_Doud_2015"
  "KCNE1_HUMAN_Muhammad_2023_expression"
  "KCNE1_HUMAN_Muhammad_2023_function"
  "KCNJ2_MOUSE_Coyote-Maestas_2022_function"
  "KCNJ2_MOUSE_Coyote-Maestas_2022_surface"
  "KKA2_KLEPN_Melnikov_2014"
  "LGK_LIPST_Klesmith_2015"
  "MET_HUMAN_Estevam_2023"
  "MK01_HUMAN_Brenan_2016"
  "MLAC_ECOLI_MacRae_2023"
  "MSH2_HUMAN_Jia_2020"
  "MTHR_HUMAN_Weile_2021"
  "NCAP_I34A1_Doud_2015"
  "NUD15_HUMAN_Suiter_2020"
  "OXDA_RHOTO_Vanella_2023_activity"
  "OXDA_RHOTO_Vanella_2023_expression"
  "P53_HUMAN_Giacomelli_2018_Null_Etoposide"
  "P53_HUMAN_Giacomelli_2018_Null_Nutlin"
  "P53_HUMAN_Giacomelli_2018_WT_Nutlin"
  "PAI1_HUMAN_Huttinger_2021"
  "PHOT_CHLRE_Chen_2023"
  "POLG_CXB3N_Mattenberger_2021"
  "POLG_DEN26_Suphatrakul_2023"
  "PPARG_HUMAN_Majithia_2016"
  "PPM1D_HUMAN_Miller_2022"
  "PRKN_HUMAN_Clausen_2023"
  "PTEN_HUMAN_Matreyek_2021"
  "PTEN_HUMAN_Mighell_2018"
  "Q2N0S5_9HIV1_Haddox_2018"
  "Q53Z42_HUMAN_McShan_2019_binding-TAPBPR"
  "Q53Z42_HUMAN_McShan_2019_expression"
  "Q59976_STRSQ_Romero_2015"
  "R1AB_SARS2_Flynn_2022"
  "RASH_HUMAN_Bandaru_2017"
  "RASK_HUMAN_Weng_2022_abundance"
  "RASK_HUMAN_Weng_2022_binding-DARPin_K55"
  "RDRP_I33A0_Li_2023"
  "REV_HV1H2_Fernandes_2016"
  "RNC_ECOLI_Weeks_2023"
  "S22A1_HUMAN_Yee_2023_abundance"
  "S22A1_HUMAN_Yee_2023_activity"
  "SC6A4_HUMAN_Young_2021"
  "SHOC2_HUMAN_Kwon_2022"
  "SPIKE_SARS2_Starr_2020_binding"
  "SPIKE_SARS2_Starr_2020_expression"
  "SRC_HUMAN_Ahler_2019"
  "SRC_HUMAN_Chakraborty_2023_binding-DAS_25uM"
  "SRC_HUMAN_Nguyen_2022"
  "SYUA_HUMAN_Newberry_2020"
  "TPK1_HUMAN_Weile_2017"
  "TPMT_HUMAN_Matreyek_2018"
  "UBC9_HUMAN_Weile_2017"
  "VKOR1_HUMAN_Chiasson_2020_abundance"
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
Usage: sbatch sbatch_run/run_nkn/run_nkn_gpu.sbatch [NAME] [singles|SETTING] [OPTIONS]
SETTINGS: singles_random, singles_modulo, singles_contiguous
NAME: optional leading results-group folder name (must not be a setting).
Default singles runs all three sequentially, using each scheme's populated folds.
Default config: configs/esm2_proteinmpnn_nkn_two_scores_additive_four_branches_noise_initial_0p5.json
Initial noise variance: 0.5; same model/training defaults as run_nkn_assays.sbatch.
Config precedence: --config PATH, NKN_CONFIG, then the default above.
Default: 86 ProteinGym assays with more than 2,000 single-mutant variants.
Set the Slurm array range to match the list; current entries are shown below.
Prepared features and scores: data/features/pretrained/singles/<assay>/
Published folds: data/DMS_ProteinGym_substitutions/cv_folds_singles_substitutions/<assay>.csv
Options: --config PATH --max-steps N --min-steps N --log-every N
         --output-root PATH (overrides NKN_OUTPUT_ROOT; default results/)
         --experiment-suffix NAME (overrides EXPERIMENT_SUFFIX)
         --num-threads N --features PATH --folds PATH --assay-index PATH
         --gp-computation-backend {cholesky,bbmm}
         --bbmm-cg-tolerance FLOAT --bbmm-max-cg-iterations N
         --bbmm-num-trace-samples N --bbmm-max-lanczos-iterations N
         --bbmm-preconditioner-size N --bbmm-probe-seed N
This launcher uses real prepared inputs and CUDA; missing assets fail explicitly.
Assays are selected in catalog order by DMS_number_single_mutants > 2000.
Output root precedence: --output-root PATH, NKN_OUTPUT_ROOT, then results/.
Relative output roots are resolved from the repository root.
Results grouping: --experiment-suffix NAME inserts one folder below that root.
Precedence: --experiment-suffix, leading NAME, then EXPERIMENT_SUFFIX.
Without a suffix, existing output paths are preserved.
Assay, device, and per-task output paths are managed by this launcher.
Cholesky is the default. BBMM requires linear_operator==0.5.2 in the environment.
Short timing run (configured assays, all populated folds):
  sbatch sbatch_run/run_nkn/run_nkn_gpu.sbatch singles_random --max-steps 2 --min-steps 0
BBMM with the same prepared data, published folds, and GPU resources:
  sbatch sbatch_run/run_nkn/run_nkn_gpu.sbatch singles_random --gp-computation-backend bbmm
HELP
    printf '\nConfigured assays: %s; submit with --array=0-%s\n' "$assay_count" "$last_index"
    for index in "${!assays[@]}"; do
      printf '  %s: %s\n' "$index" "${assays[$index]}"
    done
    exit 0
  fi
  case "$argument" in
    --assay|--assay=*|--output|--output=*|--device|--device=*)
      echo "error: The array manages assay IDs, output paths, and CUDA. Use --output-root PATH or NKN_OUTPUT_ROOT to change the results root." >&2
      exit 1 ;;
  esac
done

experiment_suffix="${EXPERIMENT_SUFFIX:-}"
output_root="${NKN_OUTPUT_ROOT:-results}"
# Consume a leading results-group name before forwarding runner arguments.
# Known selections and options retain their existing meaning.
case "${1:-}" in
  ""|-*|singles|singles_random|singles_modulo|singles_contiguous) ;;
  *) experiment_suffix="$1"; shift ;;
esac
runner_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
        echo "error: --output-root requires a directory path." >&2
        exit 1
      fi
      output_root="$2"
      shift 2 ;;
    --output-root=*)
      set -- --output-root "${1#*=}" "${@:2}" ;;
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
  echo "error: SLURM_ARRAY_TASK_ID must be an integer from 0 to $last_index for the configured assays." >&2
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
run_root="$output_root/${experiment_suffix:+$experiment_suffix/}$assay/nkn_esm2_proteinmpnn_${selection}_gpu_${run_id}_${task_index}"
config="${NKN_CONFIG:-configs/esm2_proteinmpnn_nkn_two_scores_additive_four_branches_noise_initial_0p5.json}"
printf 'Job: %s\nRepository: %s\nPython: %s\n' "$run_id" "$project_root" "$python_exec"
printf 'Array task: %s_%s; assay: %s\n' \
  "${SLURM_ARRAY_JOB_ID:-manual}" "$task_index" "$assay"
printf 'Results: %s\n' "$run_root"
exec "$python_exec" -u -m experiments.large_assay_gpu \
  --config "$config" --num-threads "$cpu_count" \
  "$@" --assay "$assay" --output "$run_root"
