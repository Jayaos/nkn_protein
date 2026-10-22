#!/bin/bash
#SBATCH --job-name=esm_if1_rep
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

# Prerequisites: the nkn_protein Conda environment, the ProteinGym assay catalog
# and assay CSVs in data/DMS_ProteinGym_substitutions/raw_substitutions/, and outbound HTTPS access
# from the allocated compute node. Checkpoint/source and structures are prepared here.
# Uses the pinned esm_if1_142m.json recipe: WT reference-prefix conditioning,
# float32 inference, and normalized probabilities over the 20 canonical residues.
# Selects single mutants only, saved under data/features/pretrained/singles/.
# Prepares inverse-folding kernel features; ESM-IF1 mean scoring is not implemented.
# The current recipe rejects full assay references longer than 1,022 residues,
# including references with fragment manifests. No cropping/windowing is applied.
# Submit: sbatch sbatch_run/run_prepare_rep/prepare_esm_if1.sbatch
# Without an array, processes every configured catalog assay sequentially.
# For one assay per task, pass --array=0-(N-1) to sbatch. Obtain N with:
#   python sbatch_run/run_prepare_rep/list_assays.py --config configs/extraction/esm_if1_142m.json --count
# Indices use the full catalog order, including assays exceeding the length limit.
# Existing verified downloads are reused; --resume validates and skips matching
# completed caches. Failed assays are reported while remaining assays/stages run.
# Submit from the repository or a subdirectory; Slurm spools the script, so its
# own path cannot locate the repository. Logs go to the submission directory.
set -eo pipefail

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
print(f"Conditioning: {config.conditioning_mode}; seed: {config.seed}")
print(f"Precision: inference={config.inference_dtype}; output={config.output_dtype}")
print(f"Length policy: at most {config.max_length} residues in the full assay reference")

import torch

print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("error: CUDA is unavailable; check the GPU allocation and Conda environment.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

printf '[1/4] Downloading/verifying ESM-IF1 checkpoint and source\n'
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
printf '[2/4] Downloading/verifying reference structures and residue mappings\n'
if python -u - "$feature_config" "${assays[@]}" <<'PY'
import sys

from data_loading.catalog import AssayCatalog
from feature_generation.inverse_config import InverseFoldingConfig
from feature_generation.structure_inputs import download_proteingym_structure
from feature_generation.structure_fragments import load_inverse_structure

config = InverseFoldingConfig.load(sys.argv[1])
catalog = AssayCatalog(config.assay_index)
failures = []
for assay_id in sys.argv[2:]:
    try:
        reference = catalog.get(assay_id).reference_sequence
        if len(reference) > config.max_length:
            raise ValueError(
                f"Full reference length {len(reference)} exceeds ESM-IF1 max_length={config.max_length}"
            )
        manifest = config.structure_manifest_dir / f"{assay_id}.json"
        if not manifest.exists():
            # Ambiguous chains require a separately prepared explicit manifest.
            manifest = download_proteingym_structure(
                config.assay_index, assay_id,
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
printf '[3/4] Extracting ESM-IF1 reference-prefix amino-acid distributions\n'
if python -u -m feature_generation.cli extract \
  --config "$feature_config" \
  --output-root "$output_root" \
  "${assay_args[@]}" \
  --singles-only \
  --device cuda \
  --resume \
  --continue-on-error; then
  :
else
  failed=1
fi

printf '[4/4] Validating ESM-IF1 feature caches\n'
if python -u - "$feature_config" "$output_root" "${assays[@]}" <<'PY'
from pathlib import Path
import sys

from feature_generation.inverse_config import InverseFoldingConfig
from feature_generation.inverse_storage import validate_inverse_cache

config = InverseFoldingConfig.load(sys.argv[1])
output_root = Path(sys.argv[2]).resolve()
failures = []
for assay_id in sys.argv[3:]:
    try:
        validate_inverse_cache(
            output_root / assay_id, model_family=config.model_family, cache_id=config.cache_id,
        )
        print(f"{assay_id}: feature cache verified")
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        failures.append(assay_id)
        print(f"{assay_id}: cache validation failed: {error}", file=sys.stderr)
print(f"Caches: {len(sys.argv[3:]) - len(failures)} verified; {len(failures)} failed")
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
