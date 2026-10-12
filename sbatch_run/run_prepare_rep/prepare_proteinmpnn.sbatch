#!/bin/bash
#SBATCH --job-name=proteinmpnn_rep
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

# Prerequisites: the nkn_protein Conda environment, the ProteinGym assay catalog
# and assay CSVs in data/DMS_ProteinGym_substitutions/raw_substitutions/, and outbound HTTPS access
# from the allocated compute node. No downloaded checkpoint or structure needed.
# This job downloads and verifies the assets before extraction and scoring.
# It prepares Kermut's sequence-conditioned cache and variant mean scores.
# Feature extraction uses ten conditional passes over every reference position.
# The default presets have no sequence-length cap. Complete reference structures
# are required; long inputs still need sufficient GPU memory and execution time.
# Variant scoring uses one sequence per batch to reduce peak GPU memory.
# Submit: sbatch sbatch_run/run_prepare_rep/prepare_proteinmpnn.sbatch
# By default, process every assay in the configured raw-substitutions catalog.
# An explicitly submitted Slurm array processes one catalog assay per task.
# Failed assays are reported without stopping the remaining assays or stages.
# Submission from a repository subdirectory is also supported; logs go to the
# submission directory. Slurm spools this script, so its own path is not used.
set -eo pipefail

project_root="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
while [[ ! -f "$project_root/src/feature_generation/cli.py" || ! -f "$project_root/configs/extraction/proteinmpnn_v48_020_kermut.json" ]]; do
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

feature_config="configs/extraction/proteinmpnn_v48_020_kermut.json"
# Scoring uses its own seed-0 recipe, independently of feature conditioning.
score_config="configs/extraction/proteinmpnn_v48_020.json"
selection_args=(--config "$feature_config")
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  selection_args+=(--task-index "$SLURM_ARRAY_TASK_ID")
fi
assay_list="$(python sbatch_run/run_prepare_rep/list_assays.py "${selection_args[@]}")"
mapfile -t assays <<< "$assay_list"
assay_args=()
for assay in "${assays[@]}"; do
  assay_args+=(--assay "$assay")
done
printf 'Job: %s\nRepository: %s\nSelected assays: %s\n' "${SLURM_JOB_ID:-manual}" "$project_root" "${#assays[@]}"
printf '  %s\n' "${assays[@]}"
# Check the recipes and environment before starting any downloads.
python -u - "$feature_config" "$score_config" <<'PY'
import sys

from feature_generation.inverse_config import InverseFoldingConfig

feature_config = InverseFoldingConfig.load(sys.argv[1])
score_config = InverseFoldingConfig.load(sys.argv[2])
if feature_config.assay_index != score_config.assay_index:
    raise SystemExit("error: Feature and score configs must use the same assay catalog.")
if feature_config.structure_manifest_dir != score_config.structure_manifest_dir:
    raise SystemExit("error: Feature and score configs must use the same structure manifests.")
print(f"Assay catalog: {feature_config.assay_index}")
print(f"Kermut extraction: {feature_config.conditioning_mode}; "
      f"orders: {feature_config.num_orders}; seed: {feature_config.seed}")
print(f"Variant scoring seed: {score_config.seed}")
for stage, config in (("Feature extraction", feature_config), ("Variant scoring", score_config)):
    length_policy = "no sequence-length cap" if config.max_length is None else f"at most {config.max_length} residues"
    print(f"{stage} length policy: {length_policy}; complete reference required")
print("Variant scoring batch size: 1")

import torch

print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("error: CUDA is unavailable; check the GPU allocation and Conda environment.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

# Acquire the pinned checkpoint, native source, and reference structures. These
# commands also verify and reuse valid existing assets on later submissions.
printf '[1/4] Downloading/verifying ProteinMPNN checkpoint and source\n'
# Serialize array tasks while staging the shared source directory. Its strict
# inventory check must not see another task's unfinished download files.
python -u - "$feature_config" <<'PY'
import fcntl
import sys

from feature_generation.inverse_checkpoints import download
from feature_generation.inverse_config import InverseFoldingConfig

config = InverseFoldingConfig.load(sys.argv[1])
config.model_dir.mkdir(parents=True, exist_ok=True)
with (config.model_dir / ".download.lock").open("a+b") as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    print(download(config))
PY

failed=0
printf '[2/4] Downloading/verifying reference structures and full-sequence mappings\n'
if python -u - "$feature_config" "${assays[@]}" <<'PY'
import sys

from data_loading.catalog import AssayCatalog
from feature_generation.inverse_config import InverseFoldingConfig
from feature_generation.structure_inputs import download_proteingym_structure, load_structure

config = InverseFoldingConfig.load(sys.argv[1])
catalog = AssayCatalog(config.assay_index)
failures = []
for assay_id in sys.argv[2:]:
    try:
        reference = catalog.get(assay_id).reference_sequence
        manifest = config.structure_manifest_dir / f"{assay_id}.json"
        if not manifest.exists():
            # A sole polymer chain is selected automatically. Ambiguous chains
            # require a separately prepared manifest with an explicit chain.
            manifest = download_proteingym_structure(
                config.assay_index, assay_id,
                manifest_dir=config.structure_manifest_dir,
            )
        # Preserve valid registered structures, including custom source files.
        load_structure(manifest, reference)
        print(f"{assay_id}: structure verified: {manifest}")
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        failures.append(assay_id)
        print(f"{assay_id}: structure failed: {error}", file=sys.stderr)
print(f"Structures: {len(sys.argv[2:]) - len(failures)} verified; {len(failures)} failed")
raise SystemExit(bool(failures))
PY
then
  :
else
  failed=1
fi

# Match Kermut's recipe: each target is conditioned on the other WT residues,
# across ten randomized orders (seed 37). Average log probabilities, exponentiate,
# and remove X without renormalizing the remaining 20 values.
# --resume skips only a matching, completed sequence-conditioned cache.
printf '[3/4] Extracting Kermut sequence-conditioned amino-acid distributions\n'
if python -u -m feature_generation.cli extract \
  --config "$feature_config" \
  "${assay_args[@]}" \
  --device cuda \
  --resume \
  --continue-on-error; then
  :
else
  failed=1
fi

# Score every variant using the full sequence and the fixed reference backbone.
# Store negative mean NLL (higher likelihood is larger) in a separate mean_score.csv.
# These scores use native autoregressive probabilities, independent of the [L,20] cache.
# Retain the original score recipe/seed for existing NKN configurations.
# One variant per batch reduces GPU memory for long complete sequences.
printf '[4/4] Preparing variant mean_score.csv\n'
if python -u -m feature_generation.cli score \
  --config "$score_config" \
  "${assay_args[@]}" \
  --device cuda \
  --batch-size 1 \
  --num-orders 1 \
  --resume \
  --continue-on-error; then
  :
else
  failed=1
fi

if (( failed )); then
  echo "error: One or more ProteinMPNN stages failed; see assay errors above. Completed caches are retained for --resume." >&2
fi
exit "$failed"
