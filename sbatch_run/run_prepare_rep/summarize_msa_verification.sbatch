#!/bin/bash
#SBATCH --job-name=summary_msa_verification
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --qos=inferno
#SBATCH --output=%x-%j.out

# Usage: sbatch sbatch_run/run_prepare_rep/summarize_msa_verification.sbatch
# Default: submit all verification tasks, then a summary with afterany dependency.
# Options: --reference-csv PATH --max-concurrent N (default 4).
# Existing results only: --summary-only [summary helper options].
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
if ! command -v conda >/dev/null 2>&1; then
  module load anaconda3
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nkn_msa_transformer
set -u
if [[ "${1:-}" == --summary-only ]]; then
  shift
  python -u "$project_root/sbatch_run/run_prepare_rep/summarize_msa_verification.py" \
    --logs-dir "$submission_dir" "$@"
  exit 0
fi

reference="$project_root/data/DMS_ProteinGym_substitutions/DMS_substitutions.csv"
max_concurrent=4
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reference-csv|--max-concurrent)
      if [[ $# -lt 2 ]]; then
        echo "error: $1 requires a value." >&2
        exit 1
      fi
      if [[ "$1" == --reference-csv ]]; then reference="$2"; else max_concurrent="$2"; fi
      shift 2
      ;;
    *)
      echo "error: Unknown option $1. Use --summary-only first to summarize existing results." >&2
      exit 1
      ;;
  esac
done
if [[ ! "$max_concurrent" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --max-concurrent must be a positive integer." >&2
  exit 1
fi

# Freeze the catalog so row indices and reference hashes agree across the array.
catalog_info="$(python - "$reference" "$project_root/data/msa/verification/batches" <<'PY'
import csv
import io
from pathlib import Path
import sys
import tempfile

data = Path(sys.argv[1]).read_bytes()
rows = list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
ids = [row['DMS_id'] for row in rows]
if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
    raise ValueError('Reference CSV must contain unique, nonempty assay IDs')
root = Path(sys.argv[2])
root.mkdir(parents=True, exist_ok=True)
snapshot = Path(tempfile.mkdtemp(prefix='catalog_', dir=root)) / 'DMS_substitutions.csv'
snapshot.write_bytes(data)
print(f'{len(rows)}\t{snapshot}')
PY
)"
IFS=$'\t' read -r count reference <<< "$catalog_info"
array_result="$(sbatch --parsable --chdir="$submission_dir" \
  --array="0-$((count - 1))%$max_concurrent" \
  "$project_root/sbatch_run/run_prepare_rep/verify_msa_all.sbatch" \
  --reference-csv "$reference")"
array_id="${array_result%%;*}"
if [[ ! "$array_id" =~ ^[0-9]+$ ]]; then
  echo "error: Unexpected sbatch response: $array_result" >&2
  exit 1
fi
echo "Verification array: $array_id ($count assays, $max_concurrent concurrent tasks)"
echo "Reference snapshot: $reference"
echo "Summary output: $project_root/data/msa/verification/summary_$array_id/"
summary_result="$(sbatch --parsable --chdir="$submission_dir" \
  --dependency="afterany:$array_id" \
  "$project_root/sbatch_run/run_prepare_rep/summarize_msa_verification.sbatch" \
  --summary-only --job-id "$array_id" --reference-csv "$reference" \
  --logs-dir "$submission_dir")"
echo "Summary job: ${summary_result%%;*} (runs after the verification array finishes)"
