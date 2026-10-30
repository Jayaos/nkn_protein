#!/bin/bash
#SBATCH --job-name=migrate_cache_names
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%j.out

# Remove protssn_proteingym-- from existing ProSST and ESM-IF1 features/scores.
# Scans every assay under data/features/pretrained/singles by default.
# CPU only; updates cache identities/checksums without model inference.
# Submit while preparation/training jobs using these caches are idle.
# Temporary space: one replacement cache at a time; no permanent duplicate.
# An interrupted migration is recovered automatically on the next apply run.
#
# Apply to all assays:
#   sbatch sbatch_run/run_prepare_rep/migrate_structure_cache_names.sbatch
# Preview only:
#   sbatch sbatch_run/run_prepare_rep/migrate_structure_cache_names.sbatch --dry-run
# Custom root or selected assay:
#   sbatch sbatch_run/run_prepare_rep/migrate_structure_cache_names.sbatch --output-root /path/to/singles --assay BLAT_ECOLX_Jacquier_2013
set -eo pipefail

project_root="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
while [[ ! -f "$project_root/src/feature_generation/storage.py" || ! -f "$project_root/sbatch_run/run_prepare_rep/migrate_structure_cache_names.py" ]]; do
  if [[ "$project_root" == "/" ]]; then
    echo "error: Submit this job from the nkn_protein repository or a subdirectory." >&2
    exit 1
  fi
  project_root="$(dirname -- "$project_root")"
done
cd -- "$project_root"

export PYTHONNOUSERSITE=1
if ! command -v conda >/dev/null 2>&1; then
  module load anaconda3
fi
conda_base="$(conda info --base)"
source "$conda_base/etc/profile.d/conda.sh"
conda activate nkn_protein
set -u

export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

printf 'Repository: %s\n' "$project_root"
python -u sbatch_run/run_prepare_rep/migrate_structure_cache_names.py "$@"
