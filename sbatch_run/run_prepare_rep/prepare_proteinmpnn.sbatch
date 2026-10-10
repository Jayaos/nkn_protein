#!/bin/bash
#SBATCH --job-name=proteinmpnn_rep
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

# Prerequisites: the nkn_protein Conda environment, the ProteinGym assay catalog
# and assay CSVs in data/DMS_ProteinGym_substitutions/raw_substitutions/, and outbound HTTPS access
# from the allocated compute node. No downloaded checkpoint or structure needed.
# This job downloads and verifies the assets before extraction and scoring.
# Submit: sbatch sbatch_run/run_prepare_rep/prepare_proteinmpnn.sbatch
# Submission from a repository subdirectory is also supported; logs go to the
# submission directory. Slurm spools this script, so its own path is not used.
set -eo pipefail

project_root="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
while [[ ! -f "$project_root/src/feature_generation/cli.py" || ! -f "$project_root/configs/extraction/proteinmpnn_v48_020.json" ]]; do
  if [[ "$project_root" == "/" ]]; then
    echo "error: Submit this job from the nkn_protein repository or a subdirectory." >&2
    exit 1
  fi
  project_root="$(dirname -- "$project_root")"
done
cd -- "$project_root"

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

config="configs/extraction/proteinmpnn_v48_020.json"
assay="A0A1I9GEU1_NEIME_Kennouche_2019"
printf 'Job: %s\nRepository: %s\nAssay: %s\n' "${SLURM_JOB_ID:-manual}" "$project_root" "$assay"
# Check the environment and raw assay inputs before starting any downloads.
python -u - "$config" "$assay" <<'PY'
import sys

from feature_generation.inputs import prepare_assay
from feature_generation.inverse_config import InverseFoldingConfig

config = InverseFoldingConfig.load(sys.argv[1])
inputs = prepare_assay(config.assay_index, sys.argv[2])
print(f"Assay CSV: {inputs.source_assay_path}")
print(f"Variants: {len(inputs.variant_ids)}; reference residues: {inputs.length}")

import torch

print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("error: CUDA is unavailable; check the GPU allocation and Conda environment.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

# Acquire the pinned checkpoint, native source, and reference structure. These
# commands also verify and reuse valid existing assets on later submissions.
printf '[1/4] Downloading/verifying ProteinMPNN checkpoint and source\n'
python -u -m feature_generation.cli download \
  --config "$config"

printf '[2/4] Downloading/verifying reference structure and chain A mapping\n'
python -u -m feature_generation.cli download-structure \
  --assay "$assay" \
  --chain A

# Compute backbone-only amino-acid probabilities and log_probs once per reference
# structure, with shape [L, 20]. Keep the full shared variant table for this assay.
# The pipeline verifies the checkpoint/structure and validates the output cache.
# --resume skips only a matching, completed ProteinMPNN cache.
printf '[3/4] Extracting reference amino-acid distributions\n'
python -u -m feature_generation.cli extract \
  --config "$config" \
  --assay "$assay" \
  --device cuda \
  --resume

# Score every variant using the full sequence and the fixed reference backbone.
# Store negative mean NLL (higher likelihood is larger) in a separate mean_score.csv.
# These scores use native autoregressive probabilities, independent of the [L,20] cache.
printf '[4/4] Preparing variant mean_score.csv\n'
python -u -m feature_generation.cli score \
  --config "$config" \
  --assay "$assay" \
  --device cuda \
  --batch-size 8 \
  --num-orders 1 \
  --resume
