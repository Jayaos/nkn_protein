#!/bin/bash
#SBATCH --job-name=esm2_650m_rep
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

# Requires the nkn_protein Conda environment and ProteinGym assay CSVs.
# This job downloads/verifies ESM2-650M, saves layer-33 mutation-site embeddings
# from centered windows and a full-sequence mean for NKN, and prepares mean_score.csv
# with the same model. The node needs HTTPS for the first download.
# The default compact preset covers mutation sites anywhere in the sequence with
# at most 1,022 residues per model input. Global means combine fixed non-overlapping
# windows covering the full variant, weighted by each window's residue count.
# Its 32 GiB output guard covers global + site vectors.
# Masked-marginal scoring also uses centered WT windows for long references.
# From the repository root: sbatch sbatch_run/run_prepare_rep/prepare_esm2.sbatch
# Select a different preset with --config PATH or ESM2_CONFIG=PATH; --config wins.
# Paths use the repository root. For Kermut's prefix-averaged embeddings, use
# --config configs/extraction/esm2_650m.json (a separate cache recipe).
# Process the 57 assays with DMS_number_single_mutants < 1000 in
# raw_substitutions/DMS_substitutions.csv, in catalog order (listed below).
# Extract and score only single mutants, saved under data/features/pretrained/singles/.
# An optional --array=0-56 selects one assay per task in the order below.
# Without an array, all 57 assays run in this job.
# Submission from a repository subdirectory is also supported; logs go to the
# submission directory. Slurm spools this script, so its own path is not used.
set -eo pipefail

project_root="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
while [[ ! -f "$project_root/src/feature_generation/cli.py" || ! -f "$project_root/configs/extraction/esm2_650m.json" ]]; do
  if [[ "$project_root" == "/" ]]; then
    echo "error: Submit this job from the nkn_protein repository or a subdirectory." >&2
    exit 1
  fi
  project_root="$(dirname -- "$project_root")"
done
cd -- "$project_root"

config="${ESM2_CONFIG:-configs/extraction/esm2_650m_centered.json}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
        echo "error: --config requires a JSON file path." >&2
        exit 1
      fi
      config="$2"
      shift 2 ;;
    --config=*)
      config="${1#--config=}"
      if [[ -z "$config" ]]; then
        echo "error: --config requires a JSON file path." >&2
        exit 1
      fi
      shift ;;
    -h|--help)
      printf 'Usage: sbatch %s [--config PATH]\n' "sbatch_run/run_prepare_rep/prepare_esm2.sbatch"
      printf 'Config precedence: --config, ESM2_CONFIG, configs/extraction/esm2_650m_centered.json\n'
      exit 0 ;;
    *)
      echo "error: Unknown argument: $1" >&2
      exit 1 ;;
  esac
done
if [[ ! -f "$config" ]]; then
  echo "error: ESM2 JSON configuration file does not exist: $config" >&2
  exit 1
fi

# Prevent ~/.local packages from shadowing the Conda environment.
export PYTHONNOUSERSITE=1
# Reuse Conda inherited from the submission shell. Reloading the module can run
# deactivation hooks before this noninteractive shell has initialized Conda.
if ! command -v conda >/dev/null 2>&1; then
  module load anaconda3
fi
conda_base="$(conda info --base)"
source "$conda_base/etc/profile.d/conda.sh"
conda activate nkn_protein
# Enable unset-variable checks after the module and Conda initialization scripts.
set -u

export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# Override the selected preset's batching budget; its window length is unchanged.
max_tokens_per_batch=8192

selection_args=(
  --config "$config"
  --assay A0A1I9GEU1_NEIME_Kennouche_2019
  --assay A4_HUMAN_Seuma_2022
  --assay AICDA_HUMAN_Gajula_2014_3cycles
  --assay AMFR_HUMAN_Tsuboyama_2023_4G3O
  --assay B2L11_HUMAN_Dutta_2010_binding-Mcl-1
  --assay BCHB_CHLTE_Tsuboyama_2023_2KRU
  --assay BLAT_ECOLX_Jacquier_2013
  --assay BRCA2_HUMAN_Erwood_2022_HEK293T
  --assay CAPSD_AAV2S_Sinai_2021
  --assay CBX4_HUMAN_Tsuboyama_2023_2K28
  --assay CUE1_YEAST_Tsuboyama_2023_2MYX
  --assay ENV_HV1B9_DuenasDecamp_2016
  --assay ERBB2_HUMAN_Elazar_2016
  --assay F7YBW8_MESOW_Ding_2023
  --assay F7YBW8_MESOW_Aakre_2015
  --assay GCN4_YEAST_Staller_2018
  --assay GLPA_HUMAN_Elazar_2016
  --assay HIS7_YEAST_Pokusaeva_2019
  --assay ISDH_STAAW_Tsuboyama_2023_2LHR
  --assay KCNH2_HUMAN_Kozek_2020
  --assay LYAM1_HUMAN_Elazar_2016
  --assay MAFG_MOUSE_Tsuboyama_2023_1K1V
  --assay MYO3_YEAST_Tsuboyama_2023_2BTT
  --assay NPC1_HUMAN_Erwood_2022_HEK293T
  --assay NPC1_HUMAN_Erwood_2022_RPE1
  --assay NRAM_I33A0_Jiang_2016
  --assay ODP2_GEOSE_Tsuboyama_2023_1W4G
  --assay OPSD_HUMAN_Wan_2019
  --assay OTU7A_HUMAN_Tsuboyama_2023_2L2D
  --assay PIN1_HUMAN_Tsuboyama_2023_1I6C
  --assay PITX2_HUMAN_Tsuboyama_2023_2L7M
  --assay POLG_PESV_Tsuboyama_2023_2MXD
  --assay Q837P4_ENTFA_Meier_2023
  --assay Q837P5_ENTFA_Meier_2023
  --assay RAD_ANTMA_Tsuboyama_2023_2CJJ
  --assay RAF1_HUMAN_Zinkus-Boltz_2019
  --assay RBP1_HUMAN_Tsuboyama_2023_2KWH
  --assay RCD1_ARATH_Tsuboyama_2023_5OAO
  --assay RD23A_HUMAN_Tsuboyama_2023_1IFY
  --assay RFAH_ECOLI_Tsuboyama_2023_2LCL
  --assay RPC1_LAMBD_Li_2019_high-expression
  --assay RPC1_LAMBD_Li_2019_low-expression
  --assay SAV1_MOUSE_Tsuboyama_2023_2YSB
  --assay SCN5A_HUMAN_Glazer_2019
  --assay SDA_BACSU_Tsuboyama_2023_1PV0
  --assay SPG1_STRSG_Wu_2016
  --assay SQSTM_MOUSE_Tsuboyama_2023_2RRU
  --assay SR43C_ARATH_Tsuboyama_2023_2N88
  --assay TCRG1_MOUSE_Tsuboyama_2023_1E0L
  --assay THO1_YEAST_Tsuboyama_2023_2WQG
  --assay TPOR_HUMAN_Bridgford_2020
  --assay UBE4B_MOUSE_Starita_2013
  --assay VG08_BPP22_Tsuboyama_2023_2GP8
  --assay VKOR1_HUMAN_Chiasson_2020_activity
  --assay YAIA_ECOLI_Tsuboyama_2023_2KVT
  --assay YAP1_HUMAN_Araya_2012
  --assay YNZC_BACSU_Tsuboyama_2023_2JVD
)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  selection_args+=(--task-index "$SLURM_ARRAY_TASK_ID")
fi
assay_list="$(python -u sbatch_run/run_prepare_rep/list_assays.py "${selection_args[@]}")"
mapfile -t assays <<< "$assay_list"
assay_args=()
for assay in "${assays[@]}"; do
  assay_args+=(--assay "$assay")
done
printf 'Job: %s\nRepository: %s\nConfig: %s\nSelected assays: %s\n' \
  "${SLURM_JOB_ID:-manual}" "$project_root" "$config" "${#assays[@]}"
printf '  %s\n' "${assays[@]}"
printf 'GPU token budget per batch: %s\n' "$max_tokens_per_batch"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  printf 'Array task: %s; assay: %s\n' "$SLURM_ARRAY_TASK_ID" "${assays[0]}"
fi
python -u - "$config" <<'PY'
import sys
import torch
from feature_generation.config import ExtractionConfig

config = ExtractionConfig.load(sys.argv[1])
print(f"Model: {config.model_id}; representation layer: {config.layer}")
print(f"Embeddings: pooling={config.pooling}; truncation={config.truncation}; max_length={config.max_length}")
print(f"Global embedding windows: {getattr(config, 'global_window', 'prefix')}")
print(f"Mutation-site embeddings: {getattr(config, 'site_embeddings', 'none')}")
print(f"Masked-marginal scoring windows: {getattr(config, 'scoring_window', 'full')}")
print(f"Cache: esm2/{config.cache_id}; output limit: {config.max_output_gib} GiB")
print(f"Assay catalog: {config.assay_index}")

print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("error: CUDA is unavailable; check the GPU allocation and Conda environment.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

# Download from the pinned Hugging Face revision, or verify existing local assets.
printf '[1/3] Downloading/verifying ESM2 checkpoint and tokenizer\n'
python -u -m feature_generation.cli download \
  --config "$config"

# Extract single mutants using the selected recipe (centered sites + full-sequence mean
# by default); --resume skips only a matching, completed assay cache. Legacy
# pooled and full-residue caches retain their separate IDs.
# The pipeline verifies the local checkpoint and validates the output cache.
# Extraction and scoring share the job's 8192-token batch budget.
# Continue across failed assays and still attempt scoring if extraction fails.
status=0
printf '[2/3] Extracting ESM2 representations\n'
if python -u -m feature_generation.cli extract \
  --config "$config" \
  "${assay_args[@]}" \
  --singles-only \
  --device cuda \
  --max-tokens-per-batch "$max_tokens_per_batch" \
  --resume \
  --continue-on-error; then
  :
else
  echo 'error: ESM2 extraction failed for one or more selected assays; continuing to scoring.' >&2
  status=1
fi

# Mask each relevant WT position once, then sum mutation log-odds per variant.
# The GP mean consumes these scores independently of kernel branch selection.
# The default preset scores long references in centered WT windows of up to
# 1,022 residues. Other presets retain full-reference scoring unless they set
# scoring_window="centered"; embedding truncation is an independent choice.
printf '[3/3] Preparing ESM2 masked-marginal mean_score.csv\n'
if python -u -m feature_generation.cli score \
  --config "$config" \
  "${assay_args[@]}" \
  --singles-only \
  --device cuda \
  --max-tokens-per-batch "$max_tokens_per_batch" \
  --batch-size 8 \
  --resume \
  --continue-on-error; then
  :
else
  echo 'error: ESM2 scoring failed for one or more selected assays.' >&2
  status=1
fi

exit "$status"
