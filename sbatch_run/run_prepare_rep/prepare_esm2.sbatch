#!/bin/bash
#SBATCH --job-name=esm2_650m_rep
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --time=02:00:00
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
# By default, process every assay in the configured raw_substitutions catalog.
# An optional Slurm array selects one assay per zero-based task index; derive its
# size with list_assays.py --config configs/extraction/esm2_650m_centered.json --count.
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

selection_args=(--config "$config")
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

# Extract all variants using the selected recipe (centered sites + full-sequence mean
# by default); --resume skips only a matching, completed assay cache. Legacy
# pooled and full-residue caches retain their separate IDs.
# The pipeline verifies the local checkpoint and validates the output cache.
# Extraction and scoring use the 650M config's 1024-token batch budget.
# Continue across failed assays and still attempt scoring if extraction fails.
status=0
printf '[2/3] Extracting ESM2 representations\n'
if python -u -m feature_generation.cli extract \
  --config "$config" \
  "${assay_args[@]}" \
  --device cuda \
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
  --device cuda \
  --batch-size 8 \
  --resume \
  --continue-on-error; then
  :
else
  echo 'error: ESM2 scoring failed for one or more selected assays.' >&2
  status=1
fi

exit "$status"
