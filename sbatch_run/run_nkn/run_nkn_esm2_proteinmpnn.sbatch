#!/bin/bash
#SBATCH --job-name=nkn_esm2_proteinmpnn
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

# Requires the nkn_protein Conda environment and completed outputs from both
# run_prepare_rep jobs. Defaults select ESM2 650M layer-33 centered-window vectors
# for sequence_site (mutation positions) and fixed non-overlapping windows for
# sequence_global (a residue-weighted full-sequence mean),
# plus sequence-conditioned ProteinMPNN features (10 orders, seed 37), preserved
# without renormalization (distribution and mutant log-probability branches),
# and physical-distance kernels on reference CA atoms
# loaded from structure_manifest_dir. Only ESM2 650M masked-marginal scores feed the
# learned GP mean in the default preset. CPU only.
# BRCA2_HUMAN_Erwood_2022_HEK293T disables structure_distance because its reference
# joins separate structure fragments; the other selected branches remain active.
# It also drops ProteinMPNN mean scores from two-score presets, retaining ESM2.
#
# From the repository root (or a subdirectory):
#   sbatch sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch [singles|all|SETTING] [--config PATH] [--parallel] [--held-out-interval N]
# To run the ProSST/ProteinMPNN preset on all 57 catalog assays with 1-1000
# single mutants (one assay per task):
#   sbatch sbatch_run/run_nkn/run_nkn_proteinmpnn_prosst_small_assays.sbatch
# The wrapper selects configs/proteinmpnn_prosst_nkn.json and forwards all arguments.
# Add --held-out-interval 100 to record held-out Spearman every 100 updates.
# Diagnostics are off by default; 0 disables them. They never select checkpoints.
# Add --stopping-criterion validation_spearman for three train / one validation /
# one test folds (singles CV only). Default remains training_nll.
# --validation-interval 100 --patience 100 --min-steps 1000 are the defaults.
# Optional noise prior: --noise-prior half_cauchy --noise-prior-scale 0.1.
# Use --noise-prior none for the original unregularized likelihood objective.
# To run all 15 singles folds concurrently, with one CPU thread per fold:
#   sbatch sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch singles --parallel
# Without --parallel, settings and folds run sequentially using all allocated CPUs.
# --parallel divides CPUs between settings, folds, and threads per fold. Memory
# is shared by all workers: size --mem for concurrent fits, not for one fit.
# Hyperparameters come from configs/esm2_proteinmpnn_nkn.json by default.
# Copy/edit that JSON and select it with --config PATH or NKN_CONFIG=PATH.
# --config takes precedence over NKN_CONFIG; paths use the repository root.
# For additive branches [1], final NKN [8, 4, 4, 2, 1], and both mean scores:
#   --config configs/esm2_proteinmpnn_nkn_two_scores_additive_branches.json
# Default: singles, the three published singles CV protocols (up to 15 fold fits).
# Use only populated published fold IDs; preserve assignments and original IDs.
# Singles use data/features/pretrained/singles/<ASSAY>; other selections use
# data/features/pretrained/<ASSAY>. FEATURES overrides either default.
# all: three singles CV runs, one shared mixed CV run, one extrapolation fit
# (21 fits). The mixed run reports both singles and doubles evaluation settings.
# The default ASSAY has only singles; all requires another assay with doubles.
# Set ASSAY and optionally FEATURES, SINGLES_FOLDS_CSV, MULTIPLES_FOLDS_CSV,
# SAMPLE_RATIO (default 0.8), or SPLIT_SEED (default 2024) before sbatch.
# SAMPLE_RATIO=1.0 trains on all singles. Relative paths use the repository root.
# Catalog: ASSAY_INDEX overrides auto-detection of data/DMS_substitutions.csv,
# then data/DMS_ProteinGym_substitutions/{,raw_substitutions/}DMS_substitutions.csv.
# Logs go to the submission directory. Every job creates a new results directory.
# --experiment-suffix NAME (or EXPERIMENT_SUFFIX=NAME) groups results under
# <output-root>/NAME/<assay>/<model-run>/. Without it, NAME is omitted.
# --output-root PATH overrides NKN_OUTPUT_ROOT (default results/).
# Relative output roots are resolved from the repository root.
set -eo pipefail

project_root="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
while [[ ! -f "$project_root/src/experiments/cli.py" || ! -f "$project_root/configs/esm2_proteinmpnn_nkn.json" ]]; do
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
experiment_suffix="${EXPERIMENT_SUFFIX:-}"
output_root="${NKN_OUTPUT_ROOT:-results}"
held_out_interval=""
stopping_criterion=""
validation_interval=""
patience=""
min_steps=""
noise_prior=""
noise_prior_scale=""
config_template="${NKN_CONFIG:-configs/esm2_proteinmpnn_nkn.json}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --noise-prior|--noise-prior-scale)
      if [[ $# -lt 2 ]]; then
        echo "error: $1 requires a value." >&2
        exit 1
      fi
      if [[ "$1" == --noise-prior ]]; then
        if [[ "$2" != none && "$2" != half_cauchy ]]; then
          echo "error: --noise-prior must be none or half_cauchy." >&2
          exit 1
        fi
        noise_prior="$2"
      else
        if [[ ! "$2" =~ ^[+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
          echo "error: --noise-prior-scale requires a positive finite number." >&2
          exit 1
        fi
        noise_prior_scale="$2"
      fi
      shift 2 ;;
    --noise-prior=*|--noise-prior-scale=*)
      set -- "${1%%=*}" "${1#*=}" "${@:2}" ;;
    --output-root)
      if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
        echo "error: --output-root requires a directory path." >&2
        exit 1
      fi
      output_root="$2"
      shift 2 ;;
    --output-root=*)
      set -- --output-root "${1#*=}" "${@:2}" ;;
    --experiment-suffix)
      if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
        echo "error: --experiment-suffix requires a folder name." >&2
        exit 1
      fi
      experiment_suffix="$2"
      shift 2 ;;
    --experiment-suffix=*)
      set -- --experiment-suffix "${1#*=}" "${@:2}" ;;
    --stopping-criterion|--validation-interval|--patience|--min-steps)
      if [[ $# -lt 2 ]]; then
        echo "error: $1 requires a value." >&2
        exit 1
      fi
      case "$1" in
        --stopping-criterion)
          if [[ "$2" != training_nll && "$2" != validation_spearman ]]; then
            echo "error: --stopping-criterion must be training_nll or validation_spearman." >&2
            exit 1
          fi
          stopping_criterion="$2" ;;
        *)
          if [[ ! "$2" =~ ^(0|[1-9][0-9]*)$ || ( "$1" != --min-steps && "$2" == 0 ) ]]; then
            echo "error: $1 requires a positive integer (--min-steps also accepts 0)." >&2
            exit 1
          fi
          case "$1" in
            --validation-interval) validation_interval="$2" ;;
            --patience) patience="$2" ;;
            --min-steps) min_steps="$2" ;;
          esac ;;
      esac
      shift 2 ;;
    --stopping-criterion=*|--validation-interval=*|--patience=*|--min-steps=*)
      set -- "${1%%=*}" "${1#*=}" "${@:2}" ;;
    --parallel)
      parallel=true
      shift ;;
    --held-out-interval)
      if [[ $# -lt 2 || ! "$2" =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo "error: --held-out-interval requires a nonnegative integer (0 disables diagnostics)." >&2
        exit 1
      fi
      held_out_interval="$2"
      shift 2 ;;
    --held-out-interval=*)
      held_out_interval="${1#--held-out-interval=}"
      if [[ ! "$held_out_interval" =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo "error: --held-out-interval requires a nonnegative integer (0 disables diagnostics)." >&2
        exit 1
      fi
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
      printf 'Catalog: ASSAY_INDEX overrides auto-detection (data/DMS_substitutions.csv first).\n'
      printf 'Output root: --output-root PATH, then NKN_OUTPUT_ROOT, then results/. Relative paths use the repository root.\n'
      printf 'Results grouping: --experiment-suffix NAME (overrides EXPERIMENT_SUFFIX); omitted keeps existing paths.\n'
      printf 'Usage: sbatch %s [singles|all|SETTING] [--config PATH] [--parallel] [--held-out-interval N]\n' "sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch"
      printf 'Diagnostic example: --held-out-interval 100 (0 disables; overrides training.held_out_interval in JSON)\n'
      printf 'Stopping: --stopping-criterion training_nll|validation_spearman (default training_nll)\n'
      printf 'NKN overrides: --validation-interval 100 --patience 100 --min-steps 1000 (optimizer steps)\n'
      printf 'Noise prior: --noise-prior none|half_cauchy; --noise-prior-scale 0.1 (overrides JSON)\n'
      printf 'Config precedence: --config, NKN_CONFIG, configs/esm2_proteinmpnn_nkn.json\n'
      printf 'Parallel example: sbatch %s singles --parallel\n' "sbatch_run/run_nkn/run_nkn_esm2_proteinmpnn.sbatch"
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
if [[ "$experiment_suffix" == *[/\\]* || "$experiment_suffix" == "." || "$experiment_suffix" == ".." ]]; then
  echo "error: Experiment suffix must be a single folder name, not a path." >&2
  exit 1
fi
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
  echo "error: NKN JSON configuration file does not exist: $config_template" >&2
  exit 1
fi
assay="${ASSAY:-A0A1I9GEU1_NEIME_Kennouche_2019}"
if [[ -z "$assay" || "$assay" == *[/\\]* || "$assay" == "." || "$assay" == ".." ]]; then
  echo "error: ASSAY must be a ProteinGym assay ID, not a path." >&2
  exit 1
fi
singles_only=true
for setting in "${settings[@]}"; do
  if [[ "$setting" != singles_* ]]; then
    singles_only=false
    break
  fi
done
feature_root="data/features/pretrained"
scope_args=()
coverage="full assay"
if [[ "$singles_only" == true ]]; then
  feature_root="$feature_root/singles"
  scope_args=(--singles-only)
  coverage="all single mutants"
fi
features="${FEATURES:-$feature_root/$assay}"
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

# Locate the catalog before activating Conda or creating a results directory.
assay_index="${ASSAY_INDEX:-}"
if [[ -z "$assay_index" ]]; then
  for candidate in \
    data/DMS_substitutions.csv \
    data/DMS_ProteinGym_substitutions/DMS_substitutions.csv \
    data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv; do
    if [[ -f "$candidate" ]]; then
      assay_index="$candidate"
      break
    fi
  done
fi
if [[ -z "$assay_index" ]]; then
  echo 'error: ProteinGym catalog not found. Set ASSAY_INDEX, or put DMS_substitutions.csv in data/, data/DMS_ProteinGym_substitutions/, or its raw_substitutions/ directory.' >&2
  exit 1
fi
if [[ ! -f "$assay_index" ]]; then
  printf 'error: ProteinGym catalog does not exist: %s\n' "$assay_index" >&2
  exit 1
fi

# Match the environment setup in run_prepare_rep, without requesting a GPU.
export PYTHONNOUSERSITE=1
if ! command -v conda >/dev/null 2>&1; then
  module load anaconda3
fi
conda_base="$(conda info --base)"
source "$conda_base/etc/profile.d/conda.sh"
conda activate nkn_protein
set -u

export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$threads_per_worker"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export CUDA_VISIBLE_DEVICES=""

run_id="${SLURM_JOB_ID:-manual_$(date -u +%Y%m%dT%H%M%SZ)}"
run_root="$output_root/${experiment_suffix:+$experiment_suffix/}$assay/nkn_esm2_proteinmpnn_${selection}_${run_id}"
runtime_config="$run_root/config.json"

# Refuse to reuse a previous run's directory or overwrite its results.
mkdir -p -- "$(dirname -- "$run_root")"
mkdir -- "$run_root"
printf 'Job: %s\nRepository: %s\nAssay: %s\nSettings: %s\nConfig: %s\nResults: %s\n' \
  "${SLURM_JOB_ID:-manual}" "$project_root" "$assay" "$selection" "$config_template" "$run_root"
printf 'Assay catalog: %s\n' "$assay_index"

# Record the worker thread budget and execution plan for sbatch CPU overrides.
python -u - "$config_template" "$runtime_config" "$features" "$assay" \
  "$parallel" "$cpu_count" "$setting_workers" "$fold_workers" "$held_out_interval" \
  "$stopping_criterion" "$validation_interval" "$patience" "$min_steps" \
  "$noise_prior" "$noise_prior_scale" "${settings[@]}" <<'PY'
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import torch
from config import ExperimentConfig

try:
    config = replace(
        ExperimentConfig.load(sys.argv[1]),
        device="cpu",
        num_threads=int(os.environ["OMP_NUM_THREADS"]),
    )
    overrides = {}
    for name, value in zip(("held_out_interval", "stopping_criterion", "validation_interval", "patience", "min_steps"), sys.argv[9:14]):
        if value:
            overrides[name] = value if name == "stopping_criterion" else int(value)
    if sys.argv[14]:
        overrides["noise_prior"] = sys.argv[14]
    if sys.argv[15]:
        overrides["noise_prior_scale"] = float(sys.argv[15])
    config = replace(config, training=replace(config.training, **overrides))
except (OSError, ValueError, TypeError) as error:
    raise SystemExit(f"error: Invalid NKN JSON configuration {sys.argv[1]}: {error}") from error
if config.evaluation.n_folds != 5:
    raise SystemExit("error: Published ProteinGym fold columns require nominal n_folds=5.")
if config.training.stopping_criterion == "validation_spearman" and any(not s.startswith("singles_") for s in sys.argv[16:]):
    raise SystemExit("error: validation_spearman supports singles CV only; choose singles or a singles_* setting.")
metadata = json.loads((Path(sys.argv[3]) / "assay.json").read_text(encoding="utf-8"))
if metadata.get("assay_id") != sys.argv[4] or metadata.get("synthetic") is not False:
    raise SystemExit("error: Expected real pretrained features for the selected assay.")
requested_branches = config.model.enabled_branches
requested_scores = config.score_sources
config = config.for_assay(metadata["assay_id"])
if config.model.enabled_branches != requested_branches:
    print(f"Assay {metadata['assay_id']}: structure_distance disabled for separate reference fragments.")
if config.score_sources != requested_scores:
    print(f"Assay {metadata['assay_id']}: ProteinMPNN mean scores disabled for incomplete reference structures; retained score sources: {config.score_sources}.")
Path(sys.argv[2]).write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
execution_plan = {
    "parallel": sys.argv[5] == "true",
    "allocated_cpus": int(sys.argv[6]),
    "setting_workers": int(sys.argv[7]),
    "fold_workers": int(sys.argv[8]),
    "threads_per_worker": config.num_threads,
    "settings": {
        setting: {"fits": 1 if setting == "single_to_double" else None,
                  "fold_workers": 1 if setting == "single_to_double" else int(sys.argv[8])}
        for setting in sys.argv[16:]
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
print(f"Sequence cache: {config.feature_selection.get('sequence')}")
print(f"ProteinMPNN cache: {config.feature_selection.get('inverse_folding')}")
print(f"ProteinMPNN input policy: {config.inverse_folding_policy}")
print(f"Structure consistency: {config.structure_consistency}; distance manifests: {config.structure_manifest_dir}")
print(f"Mean score caches: {config.score_sources}")
print(f"GP mean score sources: {len(config.score_sources) if config.score_sources is not None else 'cache defaults'}")
print(f"Training: up to {config.training.max_steps} steps per initialization; seeds {config.training.seeds}")
print(f"Noise prior: {config.training.noise_prior}; scale {config.training.noise_prior_scale}")
print(f"Stopping: {config.training.stopping_criterion}; patience {config.training.patience} optimizer steps; "
      f"min_steps {config.training.min_steps}; validation interval {config.training.validation_interval}")
print(f"Held-out Spearman diagnostic interval: {config.training.held_out_interval} (0 = disabled)")
PY

printf '[1/3] Validating shared features, scores, and %s coverage\n' "$coverage"
python -u -m experiments.cli validate \
  --features "$features" \
  --config "$runtime_config" \
  --assay-index "$assay_index" \
  --full-assay \
  "${scope_args[@]}" \
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
from config import ExperimentConfig
from data_loading import load_feature_bundle, load_folds
from experiments.runner import _validate_fold_assignments
from experiments.settings import SETTING_SCHEMES, select_setting_batch

config = ExperimentConfig.load(sys.argv[1])
torch.set_num_threads(config.num_threads)
batch = load_feature_bundle(
    sys.argv[2], feature_selection=config.feature_selection,
    score_sources=config.score_sources, pooling=config.pooling,
    inverse_folding_policy=config.inverse_folding_policy,
    structure_manifest_dir=config.structure_manifest_dir,
    structure_consistency=config.structure_consistency,
    device=config.device, enabled_branches=config.model.enabled_branches,
)
output = Path(sys.argv[1]).parent
execution_plan = json.loads((output / "execution_plan.json").read_text(encoding="utf-8"))
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
    fold_ids = folds.unique(sorted=True).tolist()
    execution_plan["settings"][setting].update(
        fits=len(fold_ids), fold_ids=fold_ids,
        fold_workers=min(execution_plan["settings"][setting]["fold_workers"], len(fold_ids)),
    )
    manifest["sources"][kind]["settings"][setting] = {
        "num_variants": selected.num_variants, "test_counts": counts,
        "fold_ids": fold_ids, "n_folds": len(fold_ids),
    }
    print(f"Published {setting} folds: {fold_ids}; held-out counts {counts}")
(output / "fold_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
(output / "execution_plan.json").write_text(json.dumps(execution_plan, indent=2) + "\n", encoding="utf-8")
PY

run_setting() {
  local setting="$1"
  local fold_snapshot
  if [[ "$setting" == "single_to_double" ]]; then
    printf '[3/3] Running single-to-double extrapolation on CPU\n'
    python -u -m experiments.cli fit \
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
    printf '[3/3] Running NKN/GP CV using populated published folds on CPU: %s\n' "$setting"
    python -u -m experiments.cli cv \
      --features "$features" \
      --config "$runtime_config" \
      --assay-index "$assay_index" \
      --full-assay \
      "${scope_args[@]}" \
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
