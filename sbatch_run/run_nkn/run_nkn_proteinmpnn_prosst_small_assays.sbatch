#!/bin/bash
#SBATCH --job-name=nkn_proteinmpnn_prosst_small_assays
#SBATCH --account=gts-yxie77-paid
#SBATCH --array=0-56
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

# One assay per array task: 57 assays with 1 <= DMS_number_single_mutants <= 1000.
# This is a snapshot of the local ProteinGym catalog, in catalog order:
# data/DMS_substitutions.csv.
# The shared runner auto-detects the catalog; ASSAY_INDEX selects an explicit path.
# Zero-single-mutant assays cannot run singles CV and are excluded.
# These subset indices differ from the full-catalog representation-preparation indices.
# Regenerate the list and array bounds together if the catalog changes.
#
# Submit from the repository root or a subdirectory:
#   sbatch sbatch_run/run_nkn/run_nkn_proteinmpnn_prosst_small_assays.sbatch
# Limit concurrent assay jobs (resources above are per assay):
#   sbatch --array=0-56%4 sbatch_run/run_nkn/run_nkn_proteinmpnn_prosst_small_assays.sbatch
# All arguments are forwarded to run_nkn_esm2_proteinmpnn.sbatch, for example:
#   sbatch sbatch_run/run_nkn/run_nkn_proteinmpnn_prosst_small_assays.sbatch singles_random --parallel
# Default: singles (random, modulo, contiguous; populated published folds).
# Kernel inputs: ProSST-2048 layer-12 centered-site and full-sequence mean embeddings
# for sequence_site and sequence_global; sequence-conditioned ProteinMPNN
# distributions and mutant log probabilities for inverse_folding and inverse_folding_site.
# ProteinMPNN features and scores use the existing unprefixed v_48_020-* directories
# under data/features/pretrained/singles/<assay> (no proteingym-- prefix).
# ProSST caches match configs/extraction/prosst_2048_singles.json (protssn_proteingym).
# Each cache is checked against its own recorded structure (structure_consistency=per_cache).
# ProteinMPNN and CA distances use proteingym; ProSST uses protssn_proteingym.
# structure_distance uses the reference CA coordinates. The GP mean uses both
# ProSST masked-marginal scores and ProteinMPNN sequence log-likelihood scores.
# BRCA2 retains its existing ProteinMPNN-score exclusion and uses ProSST scores only.
# Default config: configs/proteinmpnn_prosst_nkn.json, with the same precedence:
# --config PATH, then NKN_CONFIG, then the default. Prepared features and
# published folds for each selected assay must already exist.
# "all" and mixed/double settings require doubles and do not work for every assay.
# --parallel additionally divides each assay task's CPUs between settings/folds.
# Results: results/<assay>/nkn_proteinmpnn_prosst_<selection>_<task-job-id>/
# NKN_RUN_PREFIX overrides the model name in result folders.
# --output-root and --experiment-suffix are forwarded to the shared runner.
# Logs: nkn_proteinmpnn_prosst_small_assays-<array-job-id>_<task-index>.out in the submission directory.
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
)

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf 'Usage: sbatch sbatch_run/run_nkn/run_nkn_proteinmpnn_prosst_small_assays.sbatch [singles|SETTING] [shared-runner options]\n'
  printf 'Default array: 0-56; 57 assays with 1-1000 single mutants (inclusive).\n'
  printf 'Catalog: auto-detected from data/DMS_substitutions.csv or the extracted-data folder; ASSAY_INDEX overrides.\n'
  printf 'Limit concurrent assays with sbatch --array=0-56%%4 ...\n'
  printf 'Default config: configs/proteinmpnn_prosst_nkn.json; --config overrides NKN_CONFIG.\n'
  printf 'Results: results/<assay>/nkn_proteinmpnn_prosst_<selection>_<task-job-id>/ (NKN_RUN_PREFIX overrides the prefix).\n'
  printf 'Kernel inputs: ProSST sequence_site/sequence_global; ProteinMPNN inverse_folding/inverse_folding_site; reference CA structure_distance.\n'
  printf 'GP mean scores: ProSST and ProteinMPNN (ProSST only for BRCA2).\n'
  printf 'Structure consistency: per_cache; ProSST and ProteinMPNN retain their independently validated structures.\n'
  printf 'ProteinMPNN features/scores: existing unprefixed v_48_020-* cache directories.\n'
  printf 'ProSST embeddings and scores must match configs/extraction/prosst_2048_singles.json.\n'
  printf 'Default settings: singles; --parallel also parallelizes settings and folds within each assay.\n'
  printf 'Resources per assay: 15 CPUs, 16G memory, 24 hours; prepared inputs are required.\n'
  for index in "${!assays[@]}"; do
    printf '  %s: %s\n' "$index" "${assays[$index]}"
  done
  exit 0
fi

task_index="${SLURM_ARRAY_TASK_ID:-}"
case "$task_index" in
  [0-9]|[1-4][0-9]|5[0-6]) ;;
  *)
    printf 'error: SLURM_ARRAY_TASK_ID must be 0-56 (got "%s"). Submit this script with sbatch; its default array is 0-56.\n' "$task_index" >&2
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
export NKN_CONFIG="${NKN_CONFIG:-configs/proteinmpnn_prosst_nkn.json}"
export NKN_RUN_PREFIX="${NKN_RUN_PREFIX:-nkn_proteinmpnn_prosst}"
printf 'Array task: %s_%s; assay: %s\n' "${SLURM_ARRAY_JOB_ID:-manual}" "$task_index" "$ASSAY"
exec bash "$project_root/sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch" "$@"

