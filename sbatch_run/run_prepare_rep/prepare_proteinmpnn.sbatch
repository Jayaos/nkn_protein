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

# Before the first submission, from the repository root with nkn_protein active:
# export PYTHONNOUSERSITE=1
# export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# python -m feature_generation.cli download --config src/configs/extraction/proteinmpnn_v48_020.json
# python -m feature_generation.cli download-structure --assay A0A1I9GEU1_NEIME_Kennouche_2019 --chain A
# Submit: sbatch sbatch_run/run_prepare_rep/prepare_proteinmpnn.sbatch
# Submission from a repository subdirectory is also supported; logs go to the
# submission directory. Slurm spools this script, so its own path is not used.
set -eo pipefail

project_root="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
while [[ ! -f "$project_root/src/feature_generation/cli.py" || ! -f "$project_root/src/configs/extraction/proteinmpnn_v48_020.json" ]]; do
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

config="src/configs/extraction/proteinmpnn_v48_020.json"
assay="A0A1I9GEU1_NEIME_Kennouche_2019"
structure_manifest="data/structures/manifests/$assay.json"
if [[ ! -f "$structure_manifest" ]]; then
  printf 'error: Missing reference structure manifest: %s\n' "$structure_manifest" >&2
  printf 'Prepare it before submission: python -m feature_generation.cli download-structure --assay %s --chain A\n' "$assay" >&2
  exit 1
fi

printf 'Job: %s\nRepository: %s\nAssay: %s\nStructure manifest: %s\n' "${SLURM_JOB_ID:-manual}" "$project_root" "$assay" "$structure_manifest"
python -u - <<'PY'
import sys
import torch

print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("error: CUDA is unavailable; check the GPU allocation and Conda environment.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

# Compute backbone-only amino-acid probabilities and log_probs once per reference
# structure, with shape [L, 20]. Keep the full shared variant table for this assay.
# The pipeline verifies the checkpoint/structure and validates the output cache.
# --resume skips only a matching, completed ProteinMPNN cache.
python -u -m feature_generation.cli extract \
  --config "$config" \
  --assay "$assay" \
  --device cuda \
  --resume
