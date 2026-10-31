#!/bin/bash
#SBATCH --job-name=kermut_msa_esmif_small
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

# One assay per task: 0 < DMS_number_single_mutants <= 1000, in catalog order.
# This list is a snapshot of the local ProteinGym substitutions catalog.
# Update the list, array bounds, and task-index validation together if it changes.
# Zero-single-mutant assays cannot run the default singles CV protocols.
# Resources and training setup match run_kermut_esm2_proteinmpnn.sbatch.
# All arguments and feature/fold/config environment overrides reach that runner.
# Default: singles (random, modulo, contiguous), using prepared MSA Transformer and
# ESM-IF1 caches and configs/kermut_msa_esmif.json. CPU only.
# MSA embeddings and scores use configs/extraction/msa_transformer.json.
# Kermut uses MSA pooled global embeddings and ESM-IF1 normalized inverse-folding probabilities.
# ESM-IF1 features/scores match configs/extraction/esm_if1_142m.json.
# Reference CA coordinates use the matching protssn_proteingym structure manifests.
# GP mean scores: MSA Transformer and ESM-IF1; both remain selected for BRCA2.
# BRCA2 disables the structure-distance factor under the shared assay policy.
# Prepare both models' caches under data/features/pretrained/singles/<assay>.
# KERMUT_RUN_PREFIX overrides the default kermut_msa_esmif result prefix.
# --parallel enables concurrent fits within each assay's CPU allocation.
# all/mixed/single_to_double additionally require double mutants and full caches.
#
# Submit from the repository root or a subdirectory:
#   sbatch sbatch_run/run_kermut/run_kermut_msa_esmif_small_assays.sbatch
# Limit concurrent assay tasks to four:
#   sbatch --array=0-56%4 sbatch_run/run_kermut/run_kermut_msa_esmif_small_assays.sbatch singles --parallel
# Forward a custom config and group results with related NKN runs:
#   sbatch sbatch_run/run_kermut/run_kermut_msa_esmif_small_assays.sbatch singles --config configs/kermut_msa_esmif.json --experiment-suffix comparison_v1
# Print the task-to-assay mapping without submitting or training:
#   bash sbatch_run/run_kermut/run_kermut_msa_esmif_small_assays.sbatch --help
# Results: results/[EXPERIMENT_SUFFIX/]<assay>/kermut_msa_esmif_<selection>_<job-id>/
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
  printf 'Usage: sbatch sbatch_run/run_kermut/run_kermut_msa_esmif_small_assays.sbatch [singles|all|SETTING] [--config PATH] [--parallel] [--experiment-suffix NAME]\n'
  printf 'Default array: 0-56; 1-1000 single mutants (inclusive), one assay per task.\n'
  printf 'Uses run_kermut_esm2_proteinmpnn.sbatch with all arguments forwarded.\n'
  printf 'Default config: configs/kermut_msa_esmif.json; --config overrides KERMUT_CONFIG.\n'
  printf 'Inputs: MSA Transformer global embeddings and ESM-IF1 inverse-folding probabilities; both models supply GP mean scores.\n'
  printf 'Structure manifests: data/structures/protssn_proteingym/manifests, matching ESM-IF1 features.\n'
  printf 'Both mean scores remain selected for BRCA2; its structure-distance factor is disabled.\n'
  printf 'Results use kermut_msa_esmif_<selection>_<job-id> (KERMUT_RUN_PREFIX overrides).\n'
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
while [[ ! -f "$project_root/sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn.sbatch" ]]; do
  if [[ "$project_root" == "/" ]]; then
    echo "error: Submit this job from the nkn_protein repository or a subdirectory." >&2
    exit 1
  fi
  project_root="$(dirname -- "$project_root")"
done

export ASSAY="${assays[$task_index]}"
export KERMUT_CONFIG="${KERMUT_CONFIG:-configs/kermut_msa_esmif.json}"
export KERMUT_RUN_PREFIX="${KERMUT_RUN_PREFIX:-kermut_msa_esmif}"
printf 'Array task: %s_%s; assay: %s\n' "${SLURM_ARRAY_JOB_ID:-manual}" "$task_index" "$ASSAY"
exec bash "$project_root/sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn.sbatch" "$@"
