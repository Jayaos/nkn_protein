#!/bin/bash
#SBATCH --job-name=prosst_2048_rep
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

# Prepare ProSST K=2048 single-substitution embeddings and GP mean scores for every assay in the
# configured DMS_substitutions.csv, in catalog order, without assay-count limits.
# Requires the nkn_protein environment and catalog/variant CSV files.
# Existing structures and completed matching caches are reused. Missing structures
# come from the pinned ProteinGym archive; first downloads require HTTPS.
# From the repository root or a subdirectory:
#   sbatch sbatch_run/run_prepare_rep/prepare_prosst.sbatch
# Config precedence: --config PATH, PROSST_CONFIG, then the singles preset below.
# Configuration paths are relative to the repository root.
# Without an array, all assays run sequentially. Optional --array=0-(N-1) runs
# one assay per task. Obtain N with list_assays.py --config <preset> --count.
# Failed assays are reported and later assays are attempted; any failure makes
# the job exit nonzero. Length, token, and output-size guards still apply.
set -eo pipefail

project_root="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
while [[ ! -f "$project_root/src/feature_generation/cli.py" || ! -f "$project_root/configs/extraction/prosst_2048_singles.json" ]]; do
  if [[ "$project_root" == "/" ]]; then
    echo "error: Submit this job from the nkn_protein repository or a subdirectory." >&2
    exit 1
  fi
  project_root="$(dirname -- "$project_root")"
done
cd -- "$project_root"

config="${PROSST_CONFIG:-configs/extraction/prosst_2048_singles.json}"
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
      printf 'Usage: sbatch sbatch_run/run_prepare_rep/prepare_prosst.sbatch [--config PATH]\n'
      printf 'Target: all assays in the configured DMS_substitutions.csv (single substitutions).\n'
      printf 'Optional Slurm array: one assay per zero-based catalog index.\n'
      printf 'Config precedence: --config, PROSST_CONFIG, configs/extraction/prosst_2048_singles.json\n'
      exit 0 ;;
    *)
      echo "error: Unknown argument: $1" >&2
      exit 1 ;;
  esac
done
if [[ ! -f "$config" ]]; then
  echo "error: ProSST JSON configuration file does not exist: $config" >&2
  exit 1
fi
export PYTHONNOUSERSITE=1
if ! command -v conda >/dev/null 2>&1; then
  module load anaconda3
fi
conda_base="$(conda info --base)"
source "$conda_base/etc/profile.d/conda.sh"
conda activate nkn_protein
set -u

export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
# Use the full config loader so omitted assay_index retains its default.
assay_list="$(python -u - "$config" "${SLURM_ARRAY_TASK_ID:-}" <<'PY'
import csv
import re
import sys
from feature_generation.prosst_config import ProSSTConfig

config = ProSSTConfig.load(sys.argv[1])
with config.assay_index.open(encoding="utf-8-sig", newline="") as handle:
    reader = csv.DictReader(handle)
    if "DMS_id" not in (reader.fieldnames or []):
        raise SystemExit("error: Assay catalog requires a DMS_id column.")
    assays = [row["DMS_id"] for row in reader]
if not assays or len(set(assays)) != len(assays):
    raise SystemExit("error: Assay catalog is empty or has duplicate assay IDs.")
if any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", assay) is None for assay in assays):
    raise SystemExit("error: Assay catalog contains an invalid assay ID.")
if sys.argv[2]:
    if not sys.argv[2].isdigit() or not 0 <= int(sys.argv[2]) < len(assays):
        raise SystemExit(f"error: Array task index must be within 0..{len(assays) - 1}.")
    assays = [assays[int(sys.argv[2])]]
sys.stdout.reconfigure(newline="\n")
print("\n".join(assays))
PY
)"
mapfile -t assays <<< "$assay_list"
printf 'Job: %s\nRepository: %s\nConfig: %s\nSelected assays: %s\n' \
  "${SLURM_JOB_ID:-manual}" "$project_root" "$config" "${#assays[@]}"

model_ready=0
prepare_one() {
local assay="$1"
printf '\nAssay: %s\n' "$assay"
# Validate the config, selected catalog inputs, CUDA, and quantizer dependencies
# before downloads. The full ProSST config loader also accepts omitted defaults.
python -u - "$config" "$assay" <<'PY' || return $?
import sys
import torch
from feature_generation.inputs import prepare_assay
from feature_generation.prosst_config import PRESET, ProSSTConfig
from feature_generation.prosst_structure import _runtime_versions

config = ProSSTConfig.load(sys.argv[1])
inputs = prepare_assay(config.assay_index, sys.argv[2], singles_only=True)
if PRESET["structure_vocab_size"] != 2048:
    raise SystemExit("error: This job requires the pinned K=2048 ProSST recipe.")
if inputs.length > config.max_length and (config.truncation == "reject" or config.scoring_window == "full"):
    raise SystemExit("error: Full-reference extraction or scoring exceeds the configured length budget.")
if min(inputs.length, config.max_length) + 2 > config.max_tokens_per_batch:
    raise SystemExit("error: One sequence window plus BOS/EOS exceeds the configured token budget.")
output_bytes = len(inputs.variant_ids) * PRESET["hidden_size"] * 4
if config.pooling == "none":
    output_bytes *= inputs.length
if config.site_embeddings == "centered":
    # Singles have at most one site each; WT rows make this a conservative bound.
    output_bytes += len(inputs.variant_ids) * (PRESET["hidden_size"] * 4 + 4 * 8)
output_gib = output_bytes / 1024**3
if output_gib > config.max_output_gib:
    raise SystemExit("error: Full singles cache exceeds max_output_gib; check the job memory allocation before increasing it.")
print(f"Model: {config.model_id}; K=2048; layer: {config.layer}")
print(f"Reference length: {inputs.length}; single variants: {len(inputs.variant_ids)}")
print(f"Batch token budget: {config.max_tokens_per_batch}; cache estimate: {output_gib:.3f} GiB")
print(f"Output: {config.output_root}; cache: prosst/{config.cache_id}")
print(f"Structure manifests: {config.structure_manifest_dir}")
print(f"Structure tokens: {config.structure_token_dir}")
print(f"Python: {sys.executable}; PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("error: CUDA is unavailable; check the GPU allocation and Conda environment.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Quantizer runtime: {_runtime_versions()}")
PY

if [[ "$model_ready" == 0 ]]; then
printf '[1/6] Downloading/verifying ProSST model, tokenizer, and K=2048 quantizer\n'
python -u - "$config" <<'PY' || return $?
import fcntl
import sys
from feature_generation.prosst_checkpoints import download
from feature_generation.prosst_config import ProSSTConfig

config = ProSSTConfig.load(sys.argv[1])
config.model_dir.mkdir(parents=True, exist_ok=True)
with (config.model_dir / ".download.lock").open("a+b") as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    print(download(config))
PY
model_ready=1
fi

# Serialize duplicate submissions through structure preparation and extraction,
# including structure-token publication. All paths follow the selected config.
python -u - "$config" "$assay" <<'PY' || return $?
import fcntl
import sys
from data_loading.catalog import AssayCatalog
from feature_generation.prosst_config import ProSSTConfig
from feature_generation.prosst_runner import extract
from feature_generation.score_runner import score, default_cache_id
from feature_generation.score_storage import validate_score_cache
from feature_generation.storage import validate_sequence_cache
from feature_generation.structure_inputs import download_proteingym_structure, load_structure

config = ProSSTConfig.load(sys.argv[1])
assay = sys.argv[2]
config.device = "cuda"
config.structure_manifest_dir.mkdir(parents=True, exist_ok=True)
with (config.structure_manifest_dir / f".{assay}.prosst.lock").open("a+b") as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    print("[2/6] Preparing/verifying the full reference structure", flush=True)
    reference = AssayCatalog(config.assay_index).get(assay).reference_sequence
    manifest = config.structure_manifest_dir / f"{assay}.json"
    if not manifest.exists():
        manifest = download_proteingym_structure(
            config.assay_index, assay, manifest_dir=config.structure_manifest_dir,
        )
    load_structure(manifest, reference, require_oxygen=True)
    print(f"{assay}: structure verified: {manifest}", flush=True)

    print("[3/6] Preparing structure tokens and extracting ProSST embeddings", flush=True)
    from feature_generation.cli import _progress

    results = extract(config, [assay], singles_only=True, resume=True, progress=_progress)
    if len(results) != 1 or results[0].get("status") not in ("completed", "skipped"):
        raise SystemExit(f"error: ProSST extraction did not complete: {results}")
    print(f"Extraction: {results[0]['status']}", flush=True)

    print("[4/6] Validating the ProSST embedding cache", flush=True)
    report = validate_sequence_cache(config.output_root / assay, config.cache_id, model_family="prosst")
    print(f"Validated {report['num_variants']} variants; embedding dimension: {report['embedding_dim']}")
    print(f"Cache: {report['path']}")

    print("[5/6] Preparing ProSST masked-marginal GP mean scores", flush=True)
    results = score(config, [assay], singles_only=True, resume=True, progress=_progress)
    if len(results) != 1 or results[0].get("status") not in ("completed", "skipped"):
        raise SystemExit(f"error: ProSST scoring did not complete: {results}")
    print("[6/6] Validating the ProSST mean-score cache", flush=True)
    report = validate_score_cache(config.output_root / assay, "prosst", default_cache_id(config))
    print(f"Validated {report['num_variants']} scores; cache: {report['path']}")
PY
}

status=0
failed_assays=()
for assay in "${assays[@]}"; do
  if prepare_one "$assay"; then
    printf 'Completed assay: %s\n' "$assay"
  else
    printf 'error: Failed assay: %s; continuing with remaining assays.\n' "$assay" >&2
    failed_assays+=("$assay")
    status=1
  fi
done
if [[ "$status" != 0 ]]; then
  printf 'Failed assays (%s):\n' "${#failed_assays[@]}" >&2
  printf '  %s\n' "${failed_assays[@]}" >&2
fi
exit "$status"
