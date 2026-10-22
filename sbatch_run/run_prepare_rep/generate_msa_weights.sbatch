#!/bin/bash
#SBATCH --job-name=generate_msa_weights
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%j.out

# Default: snapshot the catalog/config, submit CPU generation, then summarize.
# Usage: sbatch .../generate_msa_weights.sbatch [--reference-csv PATH] [--max-concurrent N]
# Optional destination: --output-dir PATH (must be empty for generation).
# Internal modes: --worker --output-dir PATH, --summarize --output-dir PATH.
set -eo pipefail
submission_dir="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
project_root="$submission_dir"
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
export CONDA_NUMBER_CHANNEL_NOTICES=0
unset PYTHONPATH
if ! command -v conda >/dev/null 2>&1; then module load anaconda3; fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nkn_msa_transformer
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"
mode=submit
output_dir="$project_root/data/msa/DMS_msa_weights_generated"
reference="$project_root/data/DMS_ProteinGym_substitutions/DMS_substitutions.csv"
max_concurrent=4
while [[ $# -gt 0 ]]; do
  case "$1" in
    --worker|--summarize) mode="${1#--}"; shift ;;
    --output-dir|--batch-dir|--reference-csv|--max-concurrent)
      if [[ $# -lt 2 ]]; then echo "error: $1 needs a value" >&2; exit 1; fi
      case "$1" in
        --output-dir|--batch-dir) output_dir="$2" ;;
        --reference-csv) reference="$2" ;;
        --max-concurrent) max_concurrent="$2" ;;
      esac
      shift 2 ;;
    *) echo "error: Unknown option $1" >&2; exit 1 ;;
  esac
done
if [[ "$mode" != submit ]]; then
  if [[ ! -f "$output_dir/config.json" ]]; then echo "error: Invalid output directory" >&2; exit 1; fi
  if [[ "$mode" == worker ]]; then
    : "${SLURM_ARRAY_TASK_ID:?Generation worker must run as an array task}"
    python -u "$project_root/sbatch_run/run_prepare_rep/verify_msa_preprocessing.py" \
      --config "$output_dir/config.json" --generate-only \
      --assay-index "$SLURM_ARRAY_TASK_ID" --cpus "$OMP_NUM_THREADS"
  else
    python -u "$project_root/sbatch_run/run_prepare_rep/generated_msa.py" --output-dir "$output_dir"
  fi
  exit 0
fi
if [[ ! "$max_concurrent" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --max-concurrent must be positive" >&2; exit 1
fi
output_info="$(python - "$reference" "$project_root/configs/setup/msa_generation.json" "$output_dir" <<'PY'
import csv
import io
import json
from pathlib import Path
import sys

data = Path(sys.argv[1]).read_bytes()
rows = list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
ids = [row['DMS_id'] for row in rows]
if not ids or any(not name for name in ids) or len(set(ids)) != len(ids):
    raise ValueError('Catalog must have unique, nonempty assay IDs')
config = json.loads(Path(sys.argv[2]).read_text())
root = Path(sys.argv[3]).resolve()
root.mkdir(parents=True, exist_ok=True)
if any(root.iterdir()):
    raise FileExistsError(
        f'Output directory is not empty: {root}. '
        'Use --summarize to inspect existing results, or --output-dir PATH with an empty directory for a new run.'
    )
config['reference_csv'] = str(root / 'DMS_substitutions.csv')
config['output_folder'] = str(root / 'assays')
# Exclusive creation also prevents simultaneous submissions claiming this root.
with (root / 'config.json').open('x', encoding='utf-8') as handle:
    handle.write(json.dumps(config, indent=2) + '\n')
(root / 'DMS_substitutions.csv').write_bytes(data)
print(f'{len(rows)}\t{root}')
PY
)"
IFS=$'\t' read -r count output_dir <<< "$output_info"
launcher="$project_root/sbatch_run/run_prepare_rep/generate_msa_weights.sbatch"
array_result="$(sbatch --parsable --chdir="$submission_dir" \
  --job-name=generate_msa_all --output='generate_msa_all-%A_%a.out' \
  --cpus-per-task=8 --mem=64G --time=12:00:00 \
  --array="0-$((count - 1))%$max_concurrent" \
  "$launcher" --worker --output-dir "$output_dir")"
array_id="${array_result%%;*}"
if [[ ! "$array_id" =~ ^[0-9]+$ ]]; then echo "error: Unexpected sbatch response: $array_result" >&2; exit 1; fi
echo "Generation array: $array_id ($count assays, $max_concurrent concurrent tasks)"
echo "Output directory: $output_dir"
summary_result="$(sbatch --parsable --chdir="$submission_dir" \
  --job-name=summary_msa_generated --dependency="afterany:$array_id" \
  "$launcher" --summarize --output-dir "$output_dir")"
echo "Summary job: ${summary_result%%;*} (runs after all generation tasks finish)"
