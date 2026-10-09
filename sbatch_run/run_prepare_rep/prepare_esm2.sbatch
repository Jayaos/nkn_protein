#!/bin/bash
#SBATCH --job-name=esm2_8m_rep
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

# First download the checkpoint with the CLI's download command using esm2_8m.json.
# From the repository root: sbatch sbatch_run/run_prepare_rep/prepare_esm2.sbatch
# Submission from a repository subdirectory is also supported; logs go to the
# submission directory. Slurm spools this script, so its own path is not used.
set -eo pipefail

project_root="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
while [[ ! -f "$project_root/src/feature_generation/cli.py" || ! -f "$project_root/src/configs/extraction/esm2_8m.json" ]]; do
  if [[ "$project_root" == "/" ]]; then
    echo "error: Submit this job from the nkn_protein repository or a subdirectory." >&2
    exit 1
  fi
  project_root="$(dirname -- "$project_root")"
done
cd -- "$project_root"

# Prevent ~/.local packages from shadowing the Conda environment.
export PYTHONNOUSERSITE=1
module load anaconda3
conda_base="$(conda info --base)"
source "$conda_base/etc/profile.d/conda.sh"
conda activate nkn_protein
# Enable unset-variable checks after the module and Conda initialization scripts.
set -u

export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

config="src/configs/extraction/esm2_8m.json"
assay="A0A1I9GEU1_NEIME_Kennouche_2019"
printf 'Job: %s\nRepository: %s\nAssay: %s\n' "${SLURM_JOB_ID:-manual}" "$project_root" "$assay"
python -u - <<'PY'
import sys
import torch

print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("error: CUDA is unavailable; check the GPU allocation and Conda environment.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

# Extract all variants; --resume skips only a matching, completed cache.
# The pipeline verifies the local checkpoint and validates the output cache.
python -u -m feature_generation.cli extract \
  --config "$config" \
  --assay "$assay" \
  --device cuda \
  --max-tokens-per-batch 2048 \
  --resume
