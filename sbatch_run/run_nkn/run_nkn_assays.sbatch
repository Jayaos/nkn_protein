#!/bin/bash
#SBATCH --job-name=nkn_assays
#SBATCH --account=gts-yxie77-paid
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

# Default array: NEIME only (index 0), four branches and initial noise variance 0.1.
# One assay per array task, in the order below. Submit from the repository or
# a subdirectory; each task delegates to the shared NKN runner.
# Default: singles (random, modulo, contiguous; five folds each), with the
# additive two-score preset and caches under data/features/pretrained/singles.
# Branches: [1]; final NKN: [8, 4, 4, 2, 1]. Minimum optimizer steps: 1000.
# Observation-noise variance prior: none. Other assays retain noise_initial=0.01.
# NEIME enables: sequence_site, sequence_global, inverse_folding, structure_distance.
# Other assays additionally enable inverse_folding_site.
# Resources above are per assay task. The array parallelizes assays; --parallel
# additionally divides each task's CPUs between its settings and folds.
# Usage: sbatch sbatch_run/run_nkn/run_nkn_assays.sbatch [singles|SETTING] [--config PATH] [--parallel] [--held-out-interval N]
# Add --held-out-interval 100 to record diagnostic Spearman every 100 updates.
# Add --stopping-criterion validation_spearman for three train / one validation /
# one test folds. --validation-interval 100 --patience 100 --min-steps 1000 are defaults.
# Explicitly run the full catalog: sbatch --array=0-57%2 sbatch_run/run_nkn/run_nkn_assays.sbatch
# Logs: nkn_assays-<array-job-id>_<task-index>.out in the submission directory.
# Results: results/<assay>/nkn_esm2_proteinmpnn_<selection>_<task-job-id>/
# Add --experiment-suffix NAME (or EXPERIMENT_SUFFIX=NAME) to insert NAME under results/.
# Config precedence: --config PATH, NKN_CONFIG, then the assay-specific default.
# NEIME uses the four-branch additive noise_initial_0p1 preset; other assays use
# the original five-branch additive preset.
# Four-branch alternative with the original noise_initial=0.01:
#   --config configs/esm2_proteinmpnn_nkn_two_scores_additive_four_branches.json
# Optional: --noise-prior half_cauchy --noise-prior-scale 0.1.
set -eo pipefail

# 57 assays with DMS_number_single_mutants < 1000, in catalog order,
# matching prepare_esm2.sbatch and prepare_proteinmpnn.sbatch at indices 0-56.
# FKBP3_HUMAN_Tsuboyama_2023_2KFV is appended at index 57.
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

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf 'Results grouping: --experiment-suffix NAME (overrides EXPERIMENT_SUFFIX); omitted keeps existing paths.\n'
  printf 'Usage: sbatch sbatch_run/run_nkn/run_nkn_assays.sbatch [singles|SETTING] [--config PATH] [--parallel] [--held-out-interval N]\n'
  printf 'Default array: 0 (NEIME only); four additive branches and both GP mean scores.\n'
  printf 'NEIME default config: configs/esm2_proteinmpnn_nkn_two_scores_additive_four_branches_noise_initial_0p1.json\n'
  printf 'Other assay indices require an explicit --array override and retain configs/esm2_proteinmpnn_nkn_two_scores_additive_branches.json.\n'
  printf 'Initial noise variance: NEIME 0.1; other assays 0.01 (explicit --config or NKN_CONFIG overrides these defaults).\n'
  printf 'NEIME branches: sequence_site, sequence_global, inverse_folding, structure_distance\n'
  printf 'Other assays additionally enable inverse_folding_site.\n'
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
  [0-9]|[1-4][0-9]|5[0-7]) ;;
  *)
    printf 'error: SLURM_ARRAY_TASK_ID must be 0-57 (got "%s"). Submit this script with sbatch; its default array is 0 (NEIME only).\n' "$task_index" >&2
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
default_config="configs/esm2_proteinmpnn_nkn_two_scores_additive_branches.json"
if [[ "$ASSAY" == "A0A1I9GEU1_NEIME_Kennouche_2019" ]]; then
  default_config="configs/esm2_proteinmpnn_nkn_two_scores_additive_four_branches_noise_initial_0p1.json"
fi
export NKN_CONFIG="${NKN_CONFIG:-$default_config}"
printf 'Array task: %s_%s; assay: %s\n' "${SLURM_ARRAY_JOB_ID:-manual}" "$task_index" "$ASSAY"
exec bash "$project_root/sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch" "$@"
