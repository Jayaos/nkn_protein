#!/bin/bash
#SBATCH --job-name=kermut_esm2_proteinmpnn
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

# Independent baseline under baselines/kermut; no imports from top-level kermut.
# Requires the nkn_protein Conda environment, cached scalar scores, a registered
# structure manifest, and completed outputs from both run_prepare_rep jobs.
# ESM2 embeddings and ProteinMPNN distributions feed the
# local/global covariance with reference CA coordinates. The default mean uses
# the ESM2 score; select other scores or combinations in JSON. CPU only.
#
# From the repository root (or a subdirectory):
#   sbatch sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn.sbatch [singles|all|SETTING] [--config PATH] [--parallel]
# To run all 15 singles folds concurrently, with one CPU thread per fold:
#   sbatch sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn.sbatch singles --parallel
# Without --parallel, settings and folds run sequentially using all allocated CPUs.
# --parallel divides CPUs between settings, folds, and threads per fold. Memory
# is shared by all workers: size --mem for concurrent fits, not for one fit.
# Hyperparameters come from configs/kermut_esm2_proteinmpnn.json by default.
# Copy/edit that JSON and select it with --config PATH or KERMUT_CONFIG=PATH.
# --config takes precedence over KERMUT_CONFIG; paths use the repository root.
# Default: singles, the three published singles CV protocols (15 fold fits).
# all: three singles CV runs, one shared mixed CV run, one extrapolation fit
# (21 fits). The mixed run reports both singles and doubles evaluation settings.
# The default ASSAY has only singles; all requires another assay with doubles.
# Set ASSAY and optionally FEATURES, SINGLES_FOLDS_CSV, MULTIPLES_FOLDS_CSV,
# SAMPLE_RATIO (default 0.8), or SPLIT_SEED (default 2024) before sbatch.
# SAMPLE_RATIO=1.0 trains on all singles. Relative paths use the repository root.
# Logs go to the submission directory. Every job creates a new results directory.
set -eo pipefail

project_root="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
while [[ ! -f "$project_root/baselines/kermut/cli.py" || ! -f "$project_root/configs/kermut_esm2_proteinmpnn.json" ]]; do
  if [[ "$project_root" == "/" ]]; then
    echo "error: Submit this job from the nkn_protein repository or a subdirectory." >&2
    exit 1
  fi
  project_root="$(dirname -- "$project_root")"
done
cd -- "$project_root"

selection="singles"
selection_given=false
parallel=false
config_template="${KERMUT_CONFIG:-configs/kermut_esm2_proteinmpnn.json}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel)
      parallel=true
      shift ;;
    --config)
      if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
        echo "error: --config requires a JSON file path." >&2
        exit 1
      fi
      config_template="$2"
      shift 2 ;;
    --config=*)
      config_template="${1#--config=}"
      if [[ -z "$config_template" ]]; then
        echo "error: --config requires a JSON file path." >&2
        exit 1
      fi
      shift ;;
    -h|--help)
      printf 'Usage: sbatch %s [singles|all|SETTING] [--config PATH] [--parallel]\n' "sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn.sbatch"
      printf 'Config precedence: --config, KERMUT_CONFIG, configs/kermut_esm2_proteinmpnn.json\n'
      printf 'Parallel example: sbatch %s singles --parallel\n' "sbatch_run/run_kermut/run_kermut_esm2_proteinmpnn.sbatch"
      exit 0 ;;
    -*)
      echo "error: Unknown option: $1" >&2
      exit 1 ;;
    *)
      if [[ "$selection_given" == true ]]; then
        echo "error: Specify only one setting: singles, all, or one of the six setting names." >&2
        exit 1
      fi
      selection="$1"
      selection_given=true
      shift ;;
  esac
done
case "$selection" in
  singles) settings=(singles_random singles_modulo singles_contiguous) ;;
  all) settings=(singles_random singles_modulo singles_contiguous mixed_random_singles single_to_double) ;;
  singles_random|singles_modulo|singles_contiguous|mixed_random_singles|mixed_random_doubles|single_to_double)
    settings=("$selection") ;;
  *)
    echo "error: Use singles, all, singles_random, singles_modulo, singles_contiguous, mixed_random_singles, mixed_random_doubles, or single_to_double." >&2
    exit 1 ;;
esac
cpu_count="${SLURM_CPUS_PER_TASK:-15}"
if [[ ! "$cpu_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: SLURM_CPUS_PER_TASK must be a positive integer." >&2
  exit 1
fi
setting_workers=1
fold_workers=1
if [[ "$parallel" == true ]]; then
  setting_workers="${#settings[@]}"
  if (( setting_workers > cpu_count )); then
    setting_workers="$cpu_count"
  fi
  fold_workers=$((cpu_count / setting_workers))
  if (( fold_workers > 5 )); then
    fold_workers=5
  fi
  if [[ "$selection" == single_to_double ]]; then
    fold_workers=1
  fi
fi
threads_per_worker=$((cpu_count / (setting_workers * fold_workers)))
if [[ ! -f "$config_template" ]]; then
  echo "error: Kermut JSON configuration file does not exist: $config_template" >&2
  exit 1
fi
assay="${ASSAY:-A0A1I9GEU1_NEIME_Kennouche_2019}"
if [[ -z "$assay" || "$assay" == *[/\\]* || "$assay" == "." || "$assay" == ".." ]]; then
  echo "error: ASSAY must be a ProteinGym assay ID, not a path." >&2
  exit 1
fi
features="${FEATURES:-data/features/pretrained/$assay}"
singles_folds="${SINGLES_FOLDS_CSV:-data/DMS_ProteinGym_substitutions/cv_folds_singles_substitutions/$assay.csv}"
multiples_folds="${MULTIPLES_FOLDS_CSV:-data/DMS_ProteinGym_substitutions/cv_folds_multiples_substitutions/$assay.csv}"
split_seed="${SPLIT_SEED:-2024}"
sample_ratio="${SAMPLE_RATIO:-0.8}"
if [[ -n "${FOLDS_CSV:-}" ]]; then
  echo "error: Replace FOLDS_CSV with SINGLES_FOLDS_CSV or MULTIPLES_FOLDS_CSV." >&2
  exit 1
fi
for setting in "${settings[@]}"; do
  case "$setting" in
    singles_*) fold_source="$singles_folds" ;;
    mixed_*) fold_source="$multiples_folds" ;;
    single_to_double) continue ;;
  esac
  if [[ ! -f "$fold_source" ]]; then
    echo "error: Published ProteinGym fold CSV is required for $setting: $fold_source" >&2
    exit 1
  fi
done

# Match the environment setup in run_prepare_rep, without requesting a GPU.
export PYTHONNOUSERSITE=1
if ! command -v conda >/dev/null 2>&1; then
  module load anaconda3
fi
conda_base="$(conda info --base)"
source "$conda_base/etc/profile.d/conda.sh"
conda activate nkn_protein
set -u

export PYTHONPATH="$project_root:$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$threads_per_worker"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export CUDA_VISIBLE_DEVICES=""

assay_index="data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv"
run_id="${SLURM_JOB_ID:-manual_$(date -u +%Y%m%dT%H%M%SZ)}"
run_root="src/runs/$assay/kermut_esm2_proteinmpnn_${selection}_${run_id}"
runtime_config="$run_root/config.json"

# Refuse to reuse a previous run's directory or overwrite its results.
mkdir -p -- "$(dirname -- "$run_root")"
mkdir -- "$run_root"
printf 'Job: %s\nRepository: %s\nAssay: %s\nSettings: %s\nConfig: %s\nResults: %s\n' \
  "${SLURM_JOB_ID:-manual}" "$project_root" "$assay" "$selection" "$config_template" "$run_root"

# Record the worker thread budget and execution plan for sbatch CPU overrides.
python -u - "$config_template" "$runtime_config" "$features" "$assay" \
  "$parallel" "$cpu_count" "$setting_workers" "$fold_workers" "${settings[@]}" <<'PY'
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import torch
from baselines.kermut.config import KermutConfig

try:
    config = replace(
        KermutConfig.load(sys.argv[1]),
        device="cpu",
        num_threads=int(os.environ["OMP_NUM_THREADS"]),
    )
except (OSError, ValueError, TypeError) as error:
    raise SystemExit(f"error: Invalid Kermut JSON configuration {sys.argv[1]}: {error}") from error
if config.evaluation.n_folds != 5:
    raise SystemExit("error: Published ProteinGym protocols require five folds.")
metadata = json.loads((Path(sys.argv[3]) / "assay.json").read_text(encoding="utf-8"))
if metadata.get("assay_id") != sys.argv[4] or metadata.get("synthetic") is not False:
    raise SystemExit("error: Expected real pretrained features for the selected assay.")
Path(sys.argv[2]).write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
execution_plan = {
    "parallel": sys.argv[5] == "true",
    "allocated_cpus": int(sys.argv[6]),
    "setting_workers": int(sys.argv[7]),
    "fold_workers": int(sys.argv[8]),
    "threads_per_worker": config.num_threads,
    "settings": {
        setting: {"fits": 1 if setting == "single_to_double" else 5,
                  "fold_workers": 1 if setting == "single_to_double" else int(sys.argv[8])}
        for setting in sys.argv[9:]
    },
}
Path(sys.argv[2]).with_name("execution_plan.json").write_text(
    json.dumps(execution_plan, indent=2) + "\n", encoding="utf-8",
)
print(f"CPU allocation: {execution_plan['allocated_cpus']}; "
      f"concurrent settings: {execution_plan['setting_workers']}; "
      f"fold workers per CV setting: {execution_plan['fold_workers']}; "
      f"threads per worker: {config.num_threads}")
print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}; device: cpu; threads: {config.num_threads}")
print(f"Kernel branches: {', '.join(config.model.enabled_branches)}")
print(f"GP mean score sources: {len(config.score_sources) if config.score_sources is not None else 'cache defaults'}")
print(f"Training: {config.training.n_steps} AdamW updates; one initialization; seed {config.training.seed}; final iterate")
PY

printf '[1/3] Validating shared features, scores, and full assay coverage\n'
python -u -m baselines.kermut.cli validate \
  --features "$features" \
  --config "$runtime_config" \
  --assay-index "$assay_index" \
  --full-assay \
  --device cpu \
  --output "$run_root/validation"

# Check every requested setting before the first model fit. Published assignments
# are snapshotted unchanged; only known variants outside the setting are excluded.
printf '[2/3] Checking settings and snapshotting published fold assignments\n'
python -u - "$runtime_config" "$features" "$singles_folds" "$multiples_folds" "$split_seed" "$sample_ratio" "${settings[@]}" <<'PY'
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from baselines.kermut.config import KermutConfig
from data_loading import load_folds
from baselines.kermut.data import load_feature_bundle
from experiments.runner import _validate_fold_assignments
from experiments.settings import SETTING_SCHEMES, select_setting_batch

config = KermutConfig.load(sys.argv[1])
torch.set_num_threads(config.num_threads)
batch = load_feature_bundle(sys.argv[2], config)
output = Path(sys.argv[1]).parent
sources = {"singles": Path(sys.argv[3]), "multiples": Path(sys.argv[4])}
settings = sys.argv[7:]
manifest = {"assay_id": batch.assay_id, "requested_settings": settings, "sources": {}}
if "single_to_double" in settings:
    seed = int(sys.argv[5])
    np.random.RandomState(seed)  # Validate the same seed range used for extrapolation.
    sample_ratio = float(sys.argv[6])
    if not math.isfinite(sample_ratio) or not 0 < sample_ratio <= 1:
        raise SystemExit("error: SAMPLE_RATIO must be finite and greater than 0 and at most 1.")
    manifest.update(sample_ratio=sample_ratio, split_seed=seed)
for setting in settings:
    selected = select_setting_batch(batch, setting)
    if setting == "single_to_double":
        counts = torch.bincount(selected.owners, minlength=selected.num_variants)
        n_singles = int((counts == 1).sum())
        n_train = round(sample_ratio * n_singles)
        if n_train < 2:
            raise SystemExit("error: Extrapolation requires at least two training singles after sampling.")
        print(f"{setting}: {n_train}/{n_singles} training singles; {int((counts == 2).sum())} test doubles")
        continue
    scheme = SETTING_SCHEMES[setting]
    kind = "multiples" if setting.startswith("mixed_") else "singles"
    source = sources[kind]
    if kind not in manifest["sources"]:
        fold_bytes = source.read_bytes()
        snapshot = output / "folds" / f"{kind}.csv"
        snapshot.parent.mkdir(exist_ok=True)
        snapshot.write_bytes(fold_bytes)
        manifest["sources"][kind] = {
            "source_file": str(source.resolve()),
            "sha256": hashlib.sha256(fold_bytes).hexdigest(),
            "snapshot": str(snapshot.relative_to(output)),
            "settings": {},
        }
    snapshot = output / manifest["sources"][kind]["snapshot"]
    with snapshot.open(newline="", encoding="utf-8-sig") as handle:
        columns = csv.DictReader(handle).fieldnames or []
    column = "fold_rand_multiples" if kind == "multiples" else f"fold_{scheme}_5"
    if column not in columns:
        raise SystemExit(f"error: Published fold column is missing: {column}")
    if scheme in columns:
        raise SystemExit(f"error: Ambiguous fold columns: keep {column}, remove the {scheme} alias.")
    folds = load_folds(
        snapshot, selected.variant_ids, scheme, config.evaluation.n_folds,
        allowed_extra_ids=set(batch.variant_ids) - set(selected.variant_ids),
    )
    _validate_fold_assignments(selected, folds, config, scheme)
    counts = torch.bincount(folds, minlength=config.evaluation.n_folds).tolist()
    manifest["sources"][kind]["settings"][setting] = {
        "num_variants": selected.num_variants, "test_counts": counts,
    }
    print(f"Published {setting} folds: held-out counts {counts}")
(output / "fold_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

run_setting() {
  local setting="$1"
  local fold_snapshot
  if [[ "$setting" == "single_to_double" ]]; then
    printf '[3/3] Running single-to-double extrapolation on CPU\n'
    python -u -m baselines.kermut.cli fit \
      --features "$features" \
      --config "$runtime_config" \
      --assay-index "$assay_index" \
      --full-assay \
      --device cpu \
      --setting "$setting" \
      --sample-ratio "$sample_ratio" \
      --split-seed "$split_seed" \
      --output "$run_root/fit/$setting"
  else
    case "$setting" in
      singles_*) fold_snapshot="$run_root/folds/singles.csv" ;;
      mixed_*) fold_snapshot="$run_root/folds/multiples.csv" ;;
    esac
    printf '[3/3] Running five-fold Kermut/GP CV on CPU: %s\n' "$setting"
    python -u -m baselines.kermut.cli cv \
      --features "$features" \
      --config "$runtime_config" \
      --assay-index "$assay_index" \
      --full-assay \
      --device cpu \
      --folds "$fold_snapshot" \
      --fold-workers "$fold_workers" \
      --setting "$setting" \
      --output "$run_root/cv/$setting"
  fi
}

if [[ "$parallel" == true ]]; then
  mkdir -- "$run_root/logs"
  # Separate process groups let cancellation stop each setting and its fold
  # subprocesses together. All fit processes remain in this Slurm allocation.
  set -m
  active_pids=()
  active_settings=()
  cleanup_workers() {
    local status=$?
    local pid
    trap - EXIT INT TERM
    for pid in "${active_pids[@]}"; do
      kill -TERM -- "-$pid" 2>/dev/null || true
    done
    for pid in "${active_pids[@]}"; do
      wait "$pid" 2>/dev/null || true
    done
    return "$status"
  }
  trap cleanup_workers EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  wait_for_settings() {
    local index status
    local failed=0
    for index in "${!active_pids[@]}"; do
      if wait "${active_pids[$index]}"; then
        printf 'Finished setting: %s\n' "${active_settings[$index]}"
      else
        status=$?
        printf 'error: Setting %s failed (exit %s); see %s/logs/%s.log\n' \
          "${active_settings[$index]}" "$status" "$run_root" "${active_settings[$index]}" >&2
        failed=1
      fi
    done
    active_pids=()
    active_settings=()
    return "$failed"
  }
  for setting in "${settings[@]}"; do
    printf '[3/3] Launching %s; log: %s/logs/%s.log\n' "$setting" "$run_root" "$setting"
    run_setting "$setting" >"$run_root/logs/$setting.log" 2>&1 &
    active_pids+=("$!")
    active_settings+=("$setting")
    if (( ${#active_pids[@]} == setting_workers )); then
      wait_for_settings
    fi
  done
  wait_for_settings
  trap - EXIT INT TERM
else
  for setting in "${settings[@]}"; do
    run_setting "$setting"
  done
fi

printf 'Completed. Results: %s\n' "$run_root"

