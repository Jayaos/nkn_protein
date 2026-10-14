#!/bin/bash
#SBATCH --job-name=proteinmpnn_rep
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
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
# The default presets have no sequence-length cap. Registered BRCA2 fragments
# support inverse-folding features; use ESM2 mean scores for that assay.
# Other assays require complete structures; long inputs still need sufficient GPU memory.
# Variant scoring uses one sequence per batch to reduce peak GPU memory.
# Submit: sbatch sbatch_run/run_prepare_rep/prepare_proteinmpnn.sbatch
# Process the 57 assays with DMS_number_single_mutants < 1000 in
# raw_substitutions/DMS_substitutions.csv, in catalog order (listed below).
# Select single mutants only, saved under data/features/pretrained/singles/.
# An optional --array=0-56 selects one assay per task in the order below.
# Without an array, all 57 assays run in this job.
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
selection_args=(
  --config "$feature_config"
  --assay A0A1I9GEU1_NEIME_Kennouche_2019
  --assay A4_HUMAN_Seuma_2022
  --assay AICDA_HUMAN_Gajula_2014_3cycles
  --assay AMFR_HUMAN_Tsuboyama_2023_4G3O
  --assay B2L11_HUMAN_Dutta_2010_binding-Mcl-1
  --assay BCHB_CHLTE_Tsuboyama_2023_2KRU
  --assay BLAT_ECOLX_Jacquier_2013
  --assay BRCA2_HUMAN_Erwood_2022_HEK293T
  --assay CAPSD_AAV2S_Sinai_2021
  --assay CBX4_HUMAN_Tsuboyama_2023_2K28
  --assay CUE1_YEAST_Tsuboyama_2023_2MYX
  --assay ENV_HV1B9_DuenasDecamp_2016
  --assay ERBB2_HUMAN_Elazar_2016
  --assay F7YBW8_MESOW_Ding_2023
  --assay F7YBW8_MESOW_Aakre_2015
  --assay GCN4_YEAST_Staller_2018
  --assay GLPA_HUMAN_Elazar_2016
  --assay HIS7_YEAST_Pokusaeva_2019
  --assay ISDH_STAAW_Tsuboyama_2023_2LHR
  --assay KCNH2_HUMAN_Kozek_2020
  --assay LYAM1_HUMAN_Elazar_2016
  --assay MAFG_MOUSE_Tsuboyama_2023_1K1V
  --assay MYO3_YEAST_Tsuboyama_2023_2BTT
  --assay NPC1_HUMAN_Erwood_2022_HEK293T
  --assay NPC1_HUMAN_Erwood_2022_RPE1
  --assay NRAM_I33A0_Jiang_2016
  --assay ODP2_GEOSE_Tsuboyama_2023_1W4G
  --assay OPSD_HUMAN_Wan_2019
  --assay OTU7A_HUMAN_Tsuboyama_2023_2L2D
  --assay PIN1_HUMAN_Tsuboyama_2023_1I6C
  --assay PITX2_HUMAN_Tsuboyama_2023_2L7M
  --assay POLG_PESV_Tsuboyama_2023_2MXD
  --assay Q837P4_ENTFA_Meier_2023
  --assay Q837P5_ENTFA_Meier_2023
  --assay RAD_ANTMA_Tsuboyama_2023_2CJJ
  --assay RAF1_HUMAN_Zinkus-Boltz_2019
  --assay RBP1_HUMAN_Tsuboyama_2023_2KWH
  --assay RCD1_ARATH_Tsuboyama_2023_5OAO
  --assay RD23A_HUMAN_Tsuboyama_2023_1IFY
  --assay RFAH_ECOLI_Tsuboyama_2023_2LCL
  --assay RPC1_LAMBD_Li_2019_high-expression
  --assay RPC1_LAMBD_Li_2019_low-expression
  --assay SAV1_MOUSE_Tsuboyama_2023_2YSB
  --assay SCN5A_HUMAN_Glazer_2019
  --assay SDA_BACSU_Tsuboyama_2023_1PV0
  --assay SPG1_STRSG_Wu_2016
  --assay SQSTM_MOUSE_Tsuboyama_2023_2RRU
  --assay SR43C_ARATH_Tsuboyama_2023_2N88
  --assay TCRG1_MOUSE_Tsuboyama_2023_1E0L
  --assay THO1_YEAST_Tsuboyama_2023_2WQG
  --assay TPOR_HUMAN_Bridgford_2020
  --assay UBE4B_MOUSE_Starita_2013
  --assay VG08_BPP22_Tsuboyama_2023_2GP8
  --assay VKOR1_HUMAN_Chiasson_2020_activity
  --assay YAIA_ECOLI_Tsuboyama_2023_2KVT
  --assay YAP1_HUMAN_Araya_2012
  --assay YNZC_BACSU_Tsuboyama_2023_2JVD
)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  selection_args+=(--task-index "$SLURM_ARRAY_TASK_ID")
fi
assay_list="$(python sbatch_run/run_prepare_rep/list_assays.py "${selection_args[@]}")"
mapfile -t assays <<< "$assay_list"
assay_args=()
score_assay_args=()
for assay in "${assays[@]}"; do
  assay_args+=(--assay "$assay")
  if [[ "$assay" != BRCA2_HUMAN_Erwood_2022_HEK293T ]]; then
    score_assay_args+=(--assay "$assay")
  fi
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
if feature_config.output_root != score_config.output_root:
    raise SystemExit("error: Feature and score configs must use the same output root.")
if feature_config.structure_manifest_dir != score_config.structure_manifest_dir:
    raise SystemExit("error: Feature and score configs must use the same structure manifests.")
print(f"Assay catalog: {feature_config.assay_index}")
print(f"Kermut extraction: {feature_config.conditioning_mode}; "
      f"orders: {feature_config.num_orders}; seed: {feature_config.seed}")
print(f"Variant scoring seed: {score_config.seed}")
for stage, config in (("Feature extraction", feature_config), ("Variant scoring", score_config)):
    length_policy = "no sequence-length cap" if config.max_length is None else f"at most {config.max_length} residues"
    structure_policy = "registered BRCA2 fragments allowed" if stage == "Feature extraction" else "complete reference required"
    print(f"{stage} length policy: {length_policy}; {structure_policy}")
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
        manifest = config.structure_manifest_dir / f"{assay_id}.json"
        if not manifest.exists():
            # A sole polymer chain is selected automatically. Ambiguous chains
            # require a separately prepared manifest with an explicit chain.
            manifest = download_proteingym_structure(
                config.assay_index, assay_id,
                manifest_dir=config.structure_manifest_dir,
            )
        # Preserve valid registered structures, including custom source files.
        load_inverse_structure(manifest, reference)
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
  --singles-only \
  --device cuda \
  --resume \
  --continue-on-error; then
  :
else
  failed=1
fi

# Score every single mutant using the full sequence and the fixed reference backbone.
# Store negative mean NLL (higher likelihood is larger) in a separate mean_score.csv.
# These scores use native autoregressive probabilities, independent of the [L,20] cache.
# Retain the original score recipe/seed for existing NKN configurations.
# One variant per batch reduces GPU memory for long complete sequences.
printf '[4/4] Preparing variant mean_score.csv\n'
for assay in "${assays[@]}"; do
  if [[ "$assay" == BRCA2_HUMAN_Erwood_2022_HEK293T ]]; then
    printf '%s: fragment inverse-folding features use ESM2 mean scores; skipping full-sequence ProteinMPNN scoring.\n' "$assay"
  fi
done
if (( ${#score_assay_args[@]} )); then
  if python -u -m feature_generation.cli score \
  --config "$score_config" \
  "${score_assay_args[@]}" \
  --singles-only \
  --device cuda \
  --batch-size 1 \
  --num-orders 1 \
  --resume \
  --continue-on-error; then
    :
  else
    failed=1
  fi
fi

if (( failed )); then
  echo "error: One or more ProteinMPNN stages failed; see assay errors above. Completed caches are retained for --resume." >&2
fi
exit "$failed"
