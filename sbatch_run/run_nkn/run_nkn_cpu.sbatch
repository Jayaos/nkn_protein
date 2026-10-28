#!/bin/bash
#SBATCH --job-name=nkn_assays
#SBATCH --account=gts-yxie77-paid
#SBATCH --array=0-130
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

# 131 assays with fewer than 2,000 single-mutant variants; array indices 0-130.
# Submit from the repository or a subdirectory; the task delegates to the
# shared NKN runner with four branches and initial noise variance 0.5.
# Default: singles (random, modulo, contiguous; populated published folds), with the
# additive two-score preset and caches under data/features/pretrained/singles.
# Branches: [1]; final NKN: [8, 4, 4, 2, 1]. Minimum optimizer steps: 1000.
# Observation-noise variance prior: none. Initial noise variance: 0.5.
# Enabled: sequence_site, sequence_global, inverse_folding, structure_distance.
# Resources above apply to each assay. --parallel divides the task's CPUs
# between its settings and folds.
# Usage: sbatch sbatch_run/run_nkn/run_nkn_assays.sbatch [singles|SETTING] [--config PATH] [--parallel] [--held-out-interval N]
# Add --held-out-interval 100 to record diagnostic Spearman every 100 updates.
# Add --stopping-criterion validation_spearman for three train / one validation /
# one test folds. --validation-interval 100 --patience 100 --min-steps 1000 are defaults.
# For random CV only, pass singles_random instead of the default singles.
# Logs: nkn_assays-<array-job-id>_<task-index>.out in the submission directory.
# Results: results/<assay>/nkn_esm2_proteinmpnn_<selection>_<task-job-id>/
# --output-root PATH overrides NKN_OUTPUT_ROOT (default results/).
# Relative output roots are resolved from the repository root.
# Add --experiment-suffix NAME (or EXPERIMENT_SUFFIX=NAME) under that output root.
# Config precedence: --config PATH, NKN_CONFIG, then the four-branch additive
# noise_initial_0p5 preset for every assay.
# Four-branch alternative with the original noise_initial=0.01:
#   --config configs/esm2_proteinmpnn_nkn_two_scores_additive_four_branches.json
# Optional: --noise-prior half_cauchy --noise-prior-scale 0.1.
set -eo pipefail

# DMS_number_single_mutants < 2000, in catalog order, from
# data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv.
# This selection has its own indices; preparation uses the full catalog.
assays=(
  "A0A1I9GEU1_NEIME_Kennouche_2019"
  "A0A247D711_LISMN_Stadelmann_2021"
  "A4_HUMAN_Seuma_2022"
  "AACC1_PSEAI_Dandage_2018"
  "AICDA_HUMAN_Gajula_2014_3cycles"
  "AMFR_HUMAN_Tsuboyama_2023_4G3O"
  "ARGR_ECOLI_Tsuboyama_2023_1AOY"
  "B2L11_HUMAN_Dutta_2010_binding-Mcl-1"
  "BBC1_YEAST_Tsuboyama_2023_1TG0"
  "BCHB_CHLTE_Tsuboyama_2023_2KRU"
  "BLAT_ECOLX_Jacquier_2013"
  "BRCA1_HUMAN_Findlay_2018"
  "BRCA2_HUMAN_Erwood_2022_HEK293T"
  "CALM1_HUMAN_Weile_2017"
  "CAPSD_AAV2S_Sinai_2021"
  "CASP3_HUMAN_Roychowdhury_2020"
  "CASP7_HUMAN_Roychowdhury_2020"
  "CATR_CHLRE_Tsuboyama_2023_2AMI"
  "CBPA2_HUMAN_Tsuboyama_2023_1O6X"
  "CBX4_HUMAN_Tsuboyama_2023_2K28"
  "CCDB_ECOLI_Adkar_2012"
  "CCDB_ECOLI_Tripathi_2016"
  "CSN4_MOUSE_Tsuboyama_2023_1UFM"
  "CUE1_YEAST_Tsuboyama_2023_2MYX"
  "D7PM05_CLYGR_Somermeyer_2022"
  "DLG4_HUMAN_Faure_2021"
  "DLG4_RAT_McLaughlin_2012"
  "DN7A_SACS2_Tsuboyama_2023_1JIC"
  "DNJA1_HUMAN_Tsuboyama_2023_2LO1"
  "DOCK1_MOUSE_Tsuboyama_2023_2M0Y"
  "ENV_HV1B9_DuenasDecamp_2016"
  "ENVZ_ECOLI_Ghose_2023"
  "EPHB2_HUMAN_Tsuboyama_2023_1F0M"
  "ERBB2_HUMAN_Elazar_2016"
  "F7YBW8_MESOW_Ding_2023"
  "F7YBW8_MESOW_Aakre_2015"
  "FECA_ECOLI_Tsuboyama_2023_2D1U"
  "FKBP3_HUMAN_Tsuboyama_2023_2KFV"
  "GAL4_YEAST_Kitzman_2015"
  "GCN4_YEAST_Staller_2018"
  "GDIA_HUMAN_Silverstein_2021"
  "GFP_AEQVI_Sarkisyan_2016"
  "GLPA_HUMAN_Elazar_2016"
  "GRB2_HUMAN_Faure_2021"
  "HCP_LAMBD_Tsuboyama_2023_2L6Q"
  "HECD1_HUMAN_Tsuboyama_2023_3DKM"
  "HIS7_YEAST_Pokusaeva_2019"
  "IF1_ECOLI_Kelsic_2016"
  "ILF3_HUMAN_Tsuboyama_2023_2L33"
  "ISDH_STAAW_Tsuboyama_2023_2LHR"
  "KCNH2_HUMAN_Kozek_2020"
  "LYAM1_HUMAN_Elazar_2016"
  "MAFG_MOUSE_Tsuboyama_2023_1K1V"
  "MBD11_ARATH_Tsuboyama_2023_6ACV"
  "MTH3_HAEAE_RockahShmuel_2015"
  "MYO3_YEAST_Tsuboyama_2023_2BTT"
  "NKX31_HUMAN_Tsuboyama_2023_2L9R"
  "NPC1_HUMAN_Erwood_2022_HEK293T"
  "NPC1_HUMAN_Erwood_2022_RPE1"
  "NRAM_I33A0_Jiang_2016"
  "NUSA_ECOLI_Tsuboyama_2023_1WCL"
  "NUSG_MYCTU_Tsuboyama_2023_2MI6"
  "OBSCN_HUMAN_Tsuboyama_2023_1V1C"
  "ODP2_GEOSE_Tsuboyama_2023_1W4G"
  "OPSD_HUMAN_Wan_2019"
  "OTC_HUMAN_Lo_2023"
  "OTU7A_HUMAN_Tsuboyama_2023_2L2D"
  "P53_HUMAN_Kotler_2018"
  "P84126_THETH_Chan_2017"
  "PA_I34A1_Wu_2015"
  "PABP_YEAST_Melamed_2013"
  "PIN1_HUMAN_Tsuboyama_2023_1I6C"
  "PITX2_HUMAN_Tsuboyama_2023_2L7M"
  "PKN1_HUMAN_Tsuboyama_2023_1URF"
  "POLG_HCVJF_Qi_2014"
  "POLG_PESV_Tsuboyama_2023_2MXD"
  "PR40A_HUMAN_Tsuboyama_2023_1UZC"
  "PSAE_PICP2_Tsuboyama_2023_1PSE"
  "Q6WV12_9MAXI_Somermeyer_2022"
  "Q837P4_ENTFA_Meier_2023"
  "Q837P5_ENTFA_Meier_2023"
  "Q8WTC7_9CNID_Somermeyer_2022"
  "RAD_ANTMA_Tsuboyama_2023_2CJJ"
  "RAF1_HUMAN_Zinkus-Boltz_2019"
  "RBP1_HUMAN_Tsuboyama_2023_2KWH"
  "RCD1_ARATH_Tsuboyama_2023_5OAO"
  "RCRO_LAMBD_Tsuboyama_2023_1ORC"
  "RD23A_HUMAN_Tsuboyama_2023_1IFY"
  "RFAH_ECOLI_Tsuboyama_2023_2LCL"
  "RL20_AQUAE_Tsuboyama_2023_1GYZ"
  "RL40A_YEAST_Mavor_2016"
  "RL40A_YEAST_Roscoe_2013"
  "RL40A_YEAST_Roscoe_2014"
  "RPC1_BP434_Tsuboyama_2023_1R69"
  "RPC1_LAMBD_Li_2019_high-expression"
  "RPC1_LAMBD_Li_2019_low-expression"
  "RS15_GEOSE_Tsuboyama_2023_1A32"
  "SAV1_MOUSE_Tsuboyama_2023_2YSB"
  "SBI_STAAM_Tsuboyama_2023_2JVG"
  "SCIN_STAAR_Tsuboyama_2023_2QFF"
  "SCN5A_HUMAN_Glazer_2019"
  "SDA_BACSU_Tsuboyama_2023_1PV0"
  "SERC_HUMAN_Xie_2023"
  "SOX30_HUMAN_Tsuboyama_2023_7JJK"
  "SPA_STAAU_Tsuboyama_2023_1LP1"
  "SPG1_STRSG_Olson_2014"
  "SPG1_STRSG_Wu_2016"
  "SPG2_STRSG_Tsuboyama_2023_5UBS"
  "SPTN1_CHICK_Tsuboyama_2023_1TUD"
  "SQSTM_MOUSE_Tsuboyama_2023_2RRU"
  "SR43C_ARATH_Tsuboyama_2023_2N88"
  "SRBS1_HUMAN_Tsuboyama_2023_2O2W"
  "SUMO1_HUMAN_Weile_2017"
  "TADBP_HUMAN_Bolognesi_2019"
  "TAT_HV1BR_Fernandes_2016"
  "TCRG1_MOUSE_Tsuboyama_2023_1E0L"
  "THO1_YEAST_Tsuboyama_2023_2WQG"
  "TNKS2_HUMAN_Tsuboyama_2023_5JRT"
  "TPOR_HUMAN_Bridgford_2020"
  "TRPC_SACS2_Chan_2017"
  "TRPC_THEMA_Chan_2017"
  "UBE4B_HUMAN_Tsuboyama_2023_3L1X"
  "UBE4B_MOUSE_Starita_2013"
  "UBR5_HUMAN_Tsuboyama_2023_1I2T"
  "VG08_BPP22_Tsuboyama_2023_2GP8"
  "VILI_CHICK_Tsuboyama_2023_1YU5"
  "VKOR1_HUMAN_Chiasson_2020_activity"
  "VRPI_BPT7_Tsuboyama_2023_2WNM"
  "YAIA_ECOLI_Tsuboyama_2023_2KVT"
  "YAP1_HUMAN_Araya_2012"
  "YNZC_BACSU_Tsuboyama_2023_2JVD"
)

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf 'Output root: --output-root PATH, then NKN_OUTPUT_ROOT, then results/. Relative paths use the repository root.\n'
  printf 'Results grouping: --experiment-suffix NAME (overrides EXPERIMENT_SUFFIX); omitted keeps existing paths.\n'
  printf 'Usage: sbatch sbatch_run/run_nkn/run_nkn_assays.sbatch [singles|SETTING] [--config PATH] [--parallel] [--held-out-interval N]\n'
  printf 'Default array: 0-130; 131 assays with fewer than 2,000 single-mutant variants.\n'
  printf 'Default config for every assay: configs/esm2_proteinmpnn_nkn_two_scores_additive_four_branches_noise_initial_0p5.json\n'
  printf 'Initial noise variance: 0.5 (explicit --config or NKN_CONFIG overrides the default).\n'
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
  [0-9]|[1-9][0-9]|1[0-2][0-9]|130) ;;
  *)
    printf 'error: SLURM_ARRAY_TASK_ID must be 0-130 (got "%s"). Submit this script with sbatch; its default array is 0-130.\n' "$task_index" >&2
    exit 1 ;;
esac

project_root="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
while [[ ! -f "$project_root/sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn_small_assays.sbatch" ]]; do
  if [[ "$project_root" == "/" ]]; then
    printf 'error: Cannot find sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn_small_assays.sbatch under submission directory "%s" or its parents. Submit from a complete nkn_protein checkout.\n' "${SLURM_SUBMIT_DIR:-$PWD}" >&2
    exit 1
  fi
  project_root="$(dirname -- "$project_root")"
done

export ASSAY="${assays[$task_index]}"
export NKN_ASSAY_OVERRIDE="$ASSAY"
default_config="configs/esm2_proteinmpnn_nkn_two_scores_additive_four_branches_noise_initial_0p5.json"
export NKN_CONFIG="${NKN_CONFIG:-$default_config}"
printf 'Array task: %s_%s; assay: %s\n' "${SLURM_ARRAY_JOB_ID:-manual}" "$task_index" "$ASSAY"
exec bash "$project_root/sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn_small_assays.sbatch" "$@"
