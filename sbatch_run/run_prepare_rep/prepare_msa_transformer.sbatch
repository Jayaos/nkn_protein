#!/bin/bash
#SBATCH --job-name=msa_transformer_rep
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

# Requires completed setup_msa_transformer.sbatch and generated MSA/weight pairs.
# All assays in the extraction catalog, single mutants only; --assay ID selects
# a subset. Use --array=0-(N-1)%4 for one assay per task (catalog order).
# Count N with prepare_msa_transformer.py --count; do not hardcode catalog size.
# Without an array, runs the entire catalog sequentially. Resubmit to resume.
# Config precedence: --config PATH, MSA_TRANSFORMER_CONFIG, shipped preset.
# Default catalog: data/DMS_ProteinGym_substitutions/DMS_substitutions.csv;
# individual assay CSVs may be beside it or in raw_substitutions/ below it.
# ProteinGym-style sampling: 32 rows, weighted replacement, seed 1.
# Scores crop to the MSA interval and use upstream 1024-token optimal windows.
# NKN v3 global: fixed consecutive windows over the MSA interval, length-weighted.
# Sites: full-query centered windows, gap-padded homologs; drop all-gap rows,
# using query-only MSA Transformer input if no homolog residues remain.
# Multiple configured seeds are averaged for both embeddings and scores.
# Outputs: sequence_site, sequence_global and masked-marginal mean scores.
set -eo pipefail

# Set True (or export CLEANUP_AND_SAVE=True) to remove other cache names for
# this assay/model family after reuse or saving. Resume keeps valid current
# caches and regenerates incomplete/corrupted ones. Cleanup defaults to False.
cleanup_and_save="${CLEANUP_AND_SAVE:-False}"
cache_args=(--resume)
case "${cleanup_and_save,,}" in
  true|1) cache_args+=(--cleanup-and-save) ;;
  false|0) ;;
  *) echo "error: CLEANUP_AND_SAVE must be True or False (or 1/0)." >&2; exit 1 ;;
esac

project_root="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
while [[ ! -f "$project_root/src/feature_generation/msa_runner.py" || ! -f "$project_root/env_msa_transformer.yaml" ]]; do
  if [[ "$project_root" == "/" ]]; then
    echo "error: Submit from the nkn_protein repository or a subdirectory." >&2
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
conda activate nkn_msa_transformer
set -u

export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
printf 'Job: %s\nRepository: %s\nArray index: %s\n' \
  "${SLURM_JOB_ID:-manual}" "$project_root" "${SLURM_ARRAY_TASK_ID:-all}"
python -u sbatch_run/run_prepare_rep/prepare_msa_transformer.py "${cache_args[@]}" "$@"
