#!/bin/bash
#SBATCH --job-name=setup_msa_transformer
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%j.out

# Step 1 only: use a manually installed environment, download weights, and test inference.
# Before submission: conda env create --file env_msa_transformer.yaml
# Submit from the repository or a subdirectory. Requires outbound HTTPS.
set -eo pipefail

project_root="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
while [[ ! -f "$project_root/env_msa_transformer.yaml" ]]; do
  if [[ "$project_root" == / ]]; then
    echo "error: Submit from the nkn_protein repository or a subdirectory." >&2
    exit 1
  fi
  project_root="$(dirname -- "$project_root")"
done
cd -- "$project_root"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH

if ! command -v conda >/dev/null 2>&1; then
  module load anaconda3
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

env_name=nkn_msa_transformer
if ! conda run -n "$env_name" python -c 'import sys' >/dev/null 2>&1; then
  echo "error: Conda environment '$env_name' is missing or cannot run Python." >&2
  echo "Install it manually on Phoenix before submitting this job:" >&2
  printf '  conda env create --file %q\n' "$project_root/env_msa_transformer.yaml" >&2
  echo "If it already exists, activate it manually and check its Python installation." >&2
  exit 1
fi
conda activate "$env_name"
# Conda activation hooks can reference unset variables.
set -u

runtime_dir="$project_root/data/models/msa_transformer"
mkdir -p -- "$runtime_dir"
# Prevent two setup jobs from updating the source/checkpoint cache concurrently.
exec 9>"$runtime_dir/.setup.lock"
flock -n 9 || { echo "error: Another MSA setup job is running." >&2; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"

config="$project_root/configs/setup/msa_transformer_runtime.json"
commit="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["proteingym_commit"])' "$config")"
repository="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["proteingym_repository"])' "$config")"
source_dir="$runtime_dir/ProteinGym-$commit"
if [[ ! -d "$source_dir/.git" ]]; then
  if [[ -e "$source_dir" ]]; then
    echo "error: Source directory exists without Git metadata: $source_dir" >&2
    exit 1
  fi
  git init "$source_dir"
  git -C "$source_dir" remote add origin "$repository"
fi
if [[ "$(git -C "$source_dir" remote get-url origin)" != "$repository" ]]; then
  echo "error: Cached ProteinGym source has a different origin." >&2
  exit 1
fi
if ! git -C "$source_dir" rev-parse --verify HEAD >/dev/null 2>&1; then
  git -C "$source_dir" sparse-checkout init --cone
  git -C "$source_dir" sparse-checkout set proteingym/baselines/esm proteingym/utils environments scripts/scoring_DMS_zero_shot
  git -C "$source_dir" fetch --depth=1 --filter=blob:none origin "$commit"
  git -C "$source_dir" checkout --detach "$commit"
fi
if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$commit" ]] || [[ -n "$(git -C "$source_dir" status --porcelain)" ]]; then
  echo "error: ProteinGym source must be clean and at the pinned commit." >&2
  exit 1
fi

helper="$project_root/sbatch_run/run_prepare_rep/setup_msa_transformer.py"
python -u "$helper" download --runtime-dir "$runtime_dir"
python -u "$helper" validate --runtime-dir "$runtime_dir" --device cuda
conda list --explicit > "$runtime_dir/conda-explicit.txt"
python -m pip freeze > "$runtime_dir/pip-freeze.txt"
echo "Step 1 passed. Runtime report: $runtime_dir/runtime_report.json"
echo "MSA filtering and prepared NKN features have not been generated."
