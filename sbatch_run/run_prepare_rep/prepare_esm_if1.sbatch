#!/bin/bash
#SBATCH --job-name=esm_if1_rep
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

# Prerequisites: the nkn_protein Conda environment and the ProteinGym catalog at
# data/DMS_ProteinGym_substitutions/DMS_substitutions.csv. Individual assay CSVs
# may be beside the catalog or in its raw_substitutions/ subdirectory. Requires
# outbound HTTPS access for first model downloads. The preset automatically registers
# local protssn_proteingym structures using that collection's source.json inventory.
# Uses the pinned esm_if1_142m.json recipe: WT reference-prefix conditioning,
# float32 inference, and normalized probabilities over the 20 canonical residues.
# Selects single mutants only, saved under data/features/pretrained/singles/.
# Cache directories use model recipes; structure collection is stored in metadata.
# Also prepares ProteinGym-style full-variant mean log-likelihoods for the GP mean.
# Scores use variant prefixes and the native vocabulary, without WT subtraction.
# Scoring batch size follows the full reference length: <=300: 16, <=600: 8,
# <=1000: 4, <=2000: 2, >2000: 1. Assays sharing a batch size reuse the model.
# Full-sequence scoring requires a single backbone covering the entire reference.
# The default recipe has no sequence-length cap. An optional max_length can
# limit full assay references. Long inputs require sufficient memory/runtime.
# No cropping/windowing is applied.
# Submit: sbatch sbatch_run/run_prepare_rep/prepare_esm_if1.sbatch
# Without an array, processes every configured catalog assay sequentially.
# For one assay per task, pass --array=0-(N-1) to sbatch. Obtain N with:
#   python sbatch_run/run_prepare_rep/list_assays.py --config configs/extraction/esm_if1_142m.json --count
# Indices use the full catalog order, including assays exceeding an optional limit.
# Existing verified downloads are reused; --resume validates and skips matching
# completed caches. Failed assays are reported while remaining assays/stages run.
# Submit from the repository or a subdirectory; Slurm spools the script, so its
# own path cannot locate the repository. Logs go to the submission directory.
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
while [[ ! -f "$project_root/src/feature_generation/cli.py" || ! -f "$project_root/configs/extraction/esm_if1_142m.json" ]]; do
  if [[ "$project_root" == "/" ]]; then
    echo "error: Submit this job from the nkn_protein repository or a subdirectory." >&2
    exit 1
  fi
  project_root="$(dirname -- "$project_root")"
done
cd -- "$project_root"

# Prevent ~/.local packages from shadowing the Conda environment.
export PYTHONNOUSERSITE=1
if ! command -v conda >/dev/null 2>&1; then
  module load anaconda3
fi
conda_base="$(conda info --base)"
source "$conda_base/etc/profile.d/conda.sh"
conda activate nkn_protein
# Conda/module initialization can reference unset shell variables.
set -u

export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

feature_config="configs/extraction/esm_if1_142m.json"
# Keep the standalone config's full-variant root separate from preparation caches.
output_root="data/features/pretrained/singles"
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

# Check the recipe and GPU allocation before starting downloads.
python -u - "$feature_config" "$output_root" <<'PY'
from pathlib import Path
import sys

from feature_generation.inverse_config import InverseFoldingConfig

config = InverseFoldingConfig.load(sys.argv[1])
if config.model_family != "esm_if1" or config.conditioning_mode != "reference_prefix":
    raise SystemExit("error: Expected the ESM-IF1 reference_prefix extraction recipe.")
print(f"Assay catalog: {config.assay_index}")
print(f"Output root: {Path(sys.argv[2]).resolve()}")
print(f"Cache: {config.cache_id}")
print(f"Structure source: {config.structure_source}")
print(f"Structure manifests: {config.structure_manifest_dir}")
print(f"Conditioning: {config.conditioning_mode}; seed: {config.seed}")
print(f"Precision: inference={config.inference_dtype}; output={config.output_dtype}")
if config.max_length is None:
    print("Length policy: no sequence-length cap; full inputs require sufficient memory/runtime")
else:
    print(f"Length policy: at most {config.max_length} residues in the full assay reference")

import torch

print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("error: CUDA is unavailable; check the GPU allocation and Conda environment.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

printf '[1/5] Downloading/verifying ESM-IF1 checkpoint and source\n'
# Serialize array tasks while staging the shared source directory; inventory
# verification must not see another task's unfinished download files.
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
printf '[2/5] Verifying reference structures and residue mappings\n'
if python -u - "$feature_config" "${assays[@]}" <<'PY'
import sys

from data_loading.catalog import AssayCatalog
from feature_generation.inverse_config import InverseFoldingConfig
from feature_generation.structure_inputs import ensure_structure_manifest
from feature_generation.structure_fragments import load_inverse_structure

config = InverseFoldingConfig.load(sys.argv[1])
catalog = AssayCatalog(config.assay_index)
failures = []
for assay_id in sys.argv[2:]:
    try:
        reference = catalog.get(assay_id).reference_sequence
        if config.max_length is not None and len(reference) > config.max_length:
            raise ValueError(
                f"Full reference length {len(reference)} exceeds ESM-IF1 max_length={config.max_length}"
            )
        manifest = config.structure_manifest_dir / f"{assay_id}.json"
        if not manifest.exists():
            # Ambiguous chains require a separately prepared explicit manifest.
            manifest = ensure_structure_manifest(
                config.assay_index, assay_id,
                structure_source=config.structure_source,
                manifest_dir=config.structure_manifest_dir,
            )
        # Preserve registered custom structures/mappings/fragments. ESM-IF1 uses
        # N, CA, C coordinates; oxygen is not required by this model.
        load_inverse_structure(manifest, reference, require_oxygen=False)
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

# One reference distribution is shared by the selected single-mutant variants.
# At position i, the decoder sees the backbone and preceding WT residues.
# The existing adapter writes canonical-AA-normalized probs and log_probs.
printf '[3/5] Extracting ESM-IF1 reference-prefix amino-acid distributions\n'
if python -u -m feature_generation.cli extract \
  --config "$feature_config" \
  --output-root "$output_root" \
  "${assay_args[@]}" \
  --singles-only \
  --device cuda \
  "${cache_args[@]}" \
  --continue-on-error; then
  :
else
  failed=1
fi

printf '[4/5] Scoring ESM-IF1 variants with native mean log-likelihoods\n'
# Build the plan from the same full references used by scoring, then dispatch
# each batch-size group together to avoid reloading the model for every assay.
if score_plan="$(python -u - "$feature_config" "${assays[@]}" <<'PY'
import sys

from data_loading.catalog import AssayCatalog
from feature_generation.inverse_config import InverseFoldingConfig

config = InverseFoldingConfig.load(sys.argv[1])
catalog = AssayCatalog(config.assay_index)
for assay_id in sys.argv[2:]:
    length = len(catalog.get(assay_id).reference_sequence)
    if length <= 300:
        batch_size = 16
    elif length <= 600:
        batch_size = 8
    elif length <= 1000:
        batch_size = 4
    elif length <= 2000:
        batch_size = 2
    else:
        batch_size = 1
    print(f"{assay_id}\t{length}\t{batch_size}")
PY
)"; then
  for batch_size in 16 8 4 2 1; do
    score_args=()
    while IFS=$'\t' read -r assay reference_length assay_batch_size; do
      if [[ "$assay_batch_size" == "$batch_size" ]]; then
        score_args+=(--assay "$assay")
        printf '%s: reference length=%s; scoring batch size=%s\n' "$assay" "$reference_length" "$batch_size"
      fi
    done <<< "$score_plan"
    if (( ${#score_args[@]} == 0 )); then
      continue
    fi
    if python -u -m feature_generation.cli score \
      --config "$feature_config" \
      --output-root "$output_root" \
      "${score_args[@]}" \
      --singles-only \
      --batch-size "$batch_size" \
      --device cuda \
      "${cache_args[@]}" \
      --continue-on-error; then
      :
    else
      failed=1
    fi
  done
else
  failed=1
fi

printf '[5/5] Validating ESM-IF1 feature and score caches\n'
if python -u - "$feature_config" "$output_root" "${assays[@]}" <<'PY'
from pathlib import Path
import sys

from feature_generation.inverse_config import InverseFoldingConfig
from feature_generation.inverse_storage import validate_inverse_cache
from feature_generation.score_runner import default_cache_id
from feature_generation.score_storage import validate_score_cache

config = InverseFoldingConfig.load(sys.argv[1])
output_root = Path(sys.argv[2]).resolve()
failures = []
score_cache_id = default_cache_id(config)
for assay_id in sys.argv[3:]:
    for kind, validator, cache_id in (
        ("feature", validate_inverse_cache, config.cache_id),
        ("score", validate_score_cache, score_cache_id),
    ):
        try:
            validator(output_root / assay_id, model_family=config.model_family, cache_id=cache_id)
            print(f"{assay_id}: {kind} cache verified")
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
            failures.append((assay_id, kind))
            print(f"{assay_id}: cache validation failed ({kind}): {error}", file=sys.stderr)
print(f"Caches: {2 * len(sys.argv[3:]) - len(failures)} verified; {len(failures)} failed")
raise SystemExit(bool(failures))
PY
then
  :
else
  failed=1
fi

if (( failed )); then
  echo "error: One or more ESM-IF1 stages failed; see assay errors above. Completed caches are retained for --resume." >&2
fi
exit "$failed"
