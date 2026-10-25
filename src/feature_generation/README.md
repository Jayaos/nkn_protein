# Frozen protein representations

The CLI downloads pinned ESM-2, ProSST, ProteinMPNN, and ESM-IF1 checkpoints and writes
representations into the project's schema-v2/layout-v1 cache. Models run
locally with frozen parameters in evaluation/inference mode. DMS labels are
copied into the assay table but are never supplied to the representation models.
MSA Transformer uses the dedicated `prepare_msa_transformer.sbatch` workflow
below and writes the same cache schema from validated alignment/weight pairs.

The ESM2 and ProteinMPNN presets write to
`data/features/pretrained/singles/<assay>/`. Both preparation jobs pass
`--singles-only` to extraction and scoring; pass it explicitly when using these
presets through the CLI. Each selected single mutant is prepared once and reused
across folds and protocols. The raw catalog remains the source of assay metadata
and variants. Existing full-assay caches are not moved or modified. To prepare
all variants through the CLI, override `--output-root data/features/pretrained`
and omit `--singles-only` consistently for extraction and scoring. ProSST,
ESM-IF1, and notebook defaults retain their existing full-assay locations.

## Replacing saved assay caches

`cleanup_and_save` defaults to `False`. Enable it for the ProSST or ESM-IF1
preparation job with:

```bash
CLEANUP_AND_SAVE=True sbatch sbatch_run/run_prepare_rep/prepare_prosst.sbatch
CLEANUP_AND_SAVE=True sbatch sbatch_run/run_prepare_rep/prepare_esm_if1.sbatch
```

Both scripts also expose a `cleanup_and_save` variable near the top. The Python
extraction/scoring functions accept `cleanup_and_save=True`; the CLI equivalent
is `--cleanup-and-save` on `extract` or `score`. Cleanup overrides resume and
always recomputes. After a new cache is staged and validated, all previous cache
directories for that assay, model family, and output type are removed and the
new recipe becomes the sole cache. For example, ProSST extraction replaces all
recipes under `<output_root>/<assay>/sequence/prosst/`; scoring independently
replaces `<output_root>/<assay>/scores/prosst/`. ESM-IF1 uses
`inverse_folding/esm_if1/` and `scores/esm_if1/` respectively. An existing default
feature descriptor for the replaced family is updated to the new recipe.

Other assays, model families, shared assay files, model downloads, structures,
and structure tokens are preserved. Failed generation or staging validation
leaves the old caches intact. Feature and score stages commit independently, so
a failed score stage can leave old scores alongside newly generated features.
Changed assay inputs or variant coverage still require a separate output root.
When cleanup is disabled, the existing resume/immutable-cache behavior is unchanged.

## Data storage on Phoenix

Data stays in the repository's existing `data/` directory by default. Choose
between the current Xie directory and Wilson storage:

| Choice | Action |
| --- | --- |
| Current repository directory (Xie) | Keep `data/` as its existing directory and run the usual preparation commands. |
| Wilson | Run the optional migration helper below to redirect `data/` to Wilson. |
| Return from Wilson to Xie | Run `bash sbatch_run/migrate_data_to_xie.sh` after stopping data-using jobs. |

The optional Wilson destination is:

```text
/storage/home/hcoda1/0/jlee3541/r-cwilson323-0/projects/nkn_protein/data
```

The migration helper replaces the repository's existing `data` path with a
symbolic link to this directory. This redirects model checkpoints, feature and score caches, structures, assay
CSVs, and folds together, preserving the `data/...` paths used by extraction,
NKN, Kermut, and notebooks. Local Windows data paths do not need to change.
This is a repository-wide choice; after migration, default `data/...` paths
resolve to Wilson. Check the current physical location with `readlink -f data`.
For individual CLI extraction/scoring runs, `--output-root` can select a feature
destination separately; checkpoint storage is controlled by the extraction
configuration's `model_dir` field.

After syncing the migration helper to Phoenix, stop jobs and other processes
using this project's data, then run from the existing cluster repository:

```bash
cd /storage/home/hcoda1/0/jlee3541/r-yxie77-0/projects/nkn_protein
bash sbatch_run/migrate_data_to_wilson.sh
readlink -f data
```

The helper verifies that the Wilson home-directory alias points to the Wilson
project allocation, copies the old data with `rsync`, and checks file contents
before replacing the old path with a symlink. Existing destination files are
not overwritten; conflicting contents stop the migration. It retains the source
as a timestamped backup and prints its location. That backup still consumes
Xie storage until you remove it after validating the migrated data. A completed
migration can be checked by running the helper again.

The helper requires Linux, Bash, `rsync`, and `flock`. It refuses nested symlinks
so their targets cannot silently remain on another allocation or break after
the move. Review custom structure manifests that reference files outside `data`
before migration; moving this tree does not relocate those external inputs.
Keep cached JSON files unchanged: their checksums are part of cache validation.
Restart data-using processes after migration so they resolve the new location.

To return to Xie, sync the reverse helper to Phoenix, stop data-using jobs and
processes, then run from the same repository:

```bash
bash sbatch_run/migrate_data_to_xie.sh
readlink -f data
```

The reverse helper copies the **current Wilson data**, including files created
or modified since the first migration, into a new staging directory on Xie.
After checksum verification, it replaces the `data` symlink with that directory.
The resolved path will then be under `/storage/project/r-yxie77-0/`.
This needs enough free Xie quota for a complete copy, in addition to any old
`data.before-wilson-*` backups still present. Those backups and the Wilson copy
remain untouched. The helper retains the previous Wilson symlink under a new
name and restores it if promotion fails while the active `data` path is absent.
If copying fails, it leaves the active Wilson link in place and prints the
staging location for inspection. Running it again after a completed return
reports that Xie is already active. Restart jobs after the switch.

## Installation

From the repository root:

```powershell
conda env create -f env.yaml
conda activate kernel_protein_fitness
$env:PYTHONPATH = (Resolve-Path ./src).Path + [IO.Path]::PathSeparator + $env:PYTHONPATH
```

If this Conda environment already exists, use `conda env update -f env.yaml`
instead of creating it again. All dependencies are declared in `env.yaml`,
including Transformers 4.48.1, huggingface-hub 0.29.3, and safetensors. Run the
source directly with `src/` on `PYTHONPATH`; set it in each new PowerShell session.
This implementation was tested with Python 3.10 and PyTorch 2.5.1.

### MSA Transformer: Phoenix runtime setup (step 1)

From the synced repository on Phoenix, install the environment manually first
(no GPU is needed for installation):

```bash
# If conda is not already available:
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create --file env_msa_transformer.yaml
conda activate nkn_msa_transformer
```

Then submit the download and GPU validation job:

```bash
sbatch sbatch_run/run_prepare_rep/setup_msa_transformer.sbatch
```

The job requires the separate `nkn_msa_transformer` environment installed from
`env_msa_transformer.yaml`; it exits with installation instructions if the
environment is missing or cannot run Python. It uses Python 3.10, with direct inference package
versions from ProteinGym's published environment: PyTorch 1.13.1 (CUDA 11.7),
NumPy 1.24.4, Pandas 1.5.1, Numba 0.58.1, and Biopython 1.82. HH-suite 3.3.0
provides `hhfilter`. The existing NKN environment is not updated. An existing
MSA environment is reused only if validation finds the required versions;
the job does not automatically repair a conflicting environment.

ProteinGym's package initialization imports its download utilities during model
loading, so `pooch==1.8.0` is also required and checked by runtime validation.
For an environment created before this dependency was added, install it manually
and resubmit the job:

```bash
conda activate nkn_msa_transformer
python -m pip install pooch==1.8.0
sbatch sbatch_run/run_prepare_rep/setup_msa_transformer.sbatch
```

The completed checkpoint and its manifest are verified and reused on resubmission.

`configs/setup/msa_transformer_runtime.json` pins the ProteinGym source commit.
The job checks out that source under `data/models/msa_transformer/`, uses its
bundled ESM implementation, and downloads the official
`esm_msa1b_t12_100M_UR50S.pt` checkpoint (1,387,625,262 bytes). The first complete
HTTPS download establishes the SHA-256 in `checkpoint_manifest.json`; later
runs require that checksum. This is a recorded download fingerprint, not an
independently published upstream checksum. Incomplete or inconsistent existing
checkpoint/manifest pairs are rejected instead of overwritten.

The GPU check uses a tiny artificial MSA and checks finite layer-12 embeddings
and language-model logits. Success writes `runtime_report.json`,
`conda-explicit.txt`, and `pip-freeze.txt` under the runtime directory. A CPU
check can be run manually with the helper's `validate --device cpu`, but does
not establish GPU readiness. The upstream local loader may warn about missing
contact-regression weights; this setup tests embeddings/logits and does not
request contact predictions.

This is runtime setup only: it does not filter assay MSAs, establish the
historical row-to-weight mapping, generate embeddings/scores, or add an MSA
backend to the feature CLI. Source and checkpoint downloads require outbound
HTTPS, and GPU validation must run inside an allocation. Record the successful
job report before moving to MSA preprocessing. The pinned source is a reference
implementation; it is not evidence that the historical filtering recipe has
already been reproduced.

### MSA Transformer: BRCA1 preprocessing verification (step 2)

After step 1 passes, sync the repository to Phoenix and submit this CPU-only job:

```bash
sbatch sbatch_run/run_prepare_rep/verify_msa_preprocessing.sbatch
```

The pilot uses `configs/setup/msa_preprocessing_brca1.json` and the
`BRCA1_HUMAN_Findlay_2018` row in
`data/DMS_ProteinGym_substitutions/DMS_substitutions.csv`.
If the configured reference is missing, it checks this location, the older
`data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv`, and
`data/DMS_substitutions.csv`; conflicting fallback copies require an explicit
selection. To use a different location, submit:

```bash
sbatch sbatch_run/run_prepare_rep/verify_msa_preprocessing.sbatch \
  --reference-csv /path/to/DMS_substitutions.csv
```

An explicit path must exist; the report records the resolved path and file hash.
The pilot checks that
the alignment query matches the reference interval, normalizes dots to gaps
and uppercases the input (including headers, following the upstream procedure),
then runs HHfilter 3.3.0 with coverage 75, maximum pairwise identity 90, and
minimum query identity 0. Lowercase residues are retained, not stripped as
A3M insertions. ProteinGym's pinned `MSA_processing` implementation performs
the remaining cleaning and computes new EVE-style weights with theta 0.2.
One-hot encodings are skipped to reduce memory use; sequence weighting still
uses the upstream implementation.

Two comparisons are made: the normalized alignment after basic cleaning against
the general weights (a control), and the HHfiltered alignment after cleaning
against the specialized MSA Transformer weights. Each comparison checks counts,
finite positive values, and agreement in row order (rtol 1e-10, atol 1e-12).
No filtering parameters are adjusted to force a matching count. Original inputs
are never overwritten; every submission creates a fresh run directory under
`data/msa/verification/BRCA1_HUMAN_Findlay_2018_*`.

The job prints the report path. Its artifacts include:

- `verification_report.json`: recipe, source/input hashes, counts, comparisons,
  and overall result; `hhfilter.log` contains the external tool's output.
- `normalized.a2m`, `hhfiltered.a2m`, and `specialized_processed.a2m`: the
  alignment at each stage, with the last file containing the final retained rows.
- `specialized_recomputed_weights.npy`: weights freshly calculated for those rows.
- `specialized_rows.csv`: original row/header, final row/header, sequence hashes,
  recomputed weight, and the supplied weight at that index when counts match.
- `specialized_row_disposition.csv`: retained/removed status for every raw row;
  `columns.csv` maps final columns to raw columns and reference positions.
- Corresponding `general_*` files record the control comparison.

`weights_reproduced` means both ordered weight vectors agree with the downloads;
`weight_mismatch` exits with code 2 and preserves diagnostics. An execution or
identity-check error is reported as `failed`. Numerical agreement establishes
consistency with this recorded recipe, not unique historical row identity:
multiple sequences can have identical weights. Accordingly,
`historical_row_identity_verified` remains false. This pilot neither generates
embeddings/scores nor establishes preprocessing for the remaining assays.

### MSA Transformer: full-catalog verification

From the repository root on Phoenix, activate the setup environment and submit
one CPU task per reference CSV row, with at most four concurrent tasks:

```bash
conda activate nkn_msa_transformer
reference=data/DMS_ProteinGym_substitutions/DMS_substitutions.csv
count=$(python -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))))' "$reference")
sbatch --array="0-$((count - 1))%4" \
  sbatch_run/run_prepare_rep/verify_msa_all.sbatch --reference-csv "$reference"
```

Keep the reference CSV unchanged while the array runs. The array uses its actual
row count, not a hard-coded assay count. Each task writes a fresh assay-specific
directory under `data/msa/verification/` and logs to
`verify_msa_all-<job>_<index>.out`. Assays sharing an alignment are tested
independently. The job requests 8 CPUs, 64 GB RAM, and 12 hours per task; large
alignments may require more time or memory, and generated alignments require
additional disk space.

This run uses `--specialized-only`: it omits the expensive general-weight
recomputation on raw alignments. The fixed recipe is HHfilter 75/90/0 followed
by the pinned scoring code's default theta 0.2. The reference CSV's theta is
recorded separately, since it may describe the general weights. Settings are
not adjusted per assay to force agreement. A match therefore establishes
numerical reproduction for this recipe; mismatches need investigation. For
these tasks, `weights_reproduced` refers only to the specialized comparison.
Identity/preprocessing failures also stop their individual task. Check the
Slurm logs for tasks that fail before a report is created or exceed resources;
absence of a report is not a successful verification.

#### Overall verification summary

To run verification for every catalog assay and then summarize automatically,
submit from the repository root:

```bash
sbatch sbatch_run/run_prepare_rep/summarize_msa_verification.sbatch
```

The launcher snapshots the reference CSV, counts its rows, submits the full
verification array (four concurrent tasks by default), and schedules a summary
with an `afterany` dependency. This includes failed tasks in the final accounting
and selects only reports from the new array. It prints both job IDs and the
summary output directory. Use `--reference-csv PATH` or `--max-concurrent N`
to override the catalog or concurrency. Each submission starts new verification
work, even if older reports exist.

To summarize existing results without launching verification, use
`--summary-only`. For example, wait for an already submitted array:

```bash
sbatch --dependency=afterany:YOUR_ARRAY_JOB_ID \
  sbatch_run/run_prepare_rep/summarize_msa_verification.sbatch \
  --summary-only --job-id YOUR_ARRAY_JOB_ID
```

The summary job requests one CPU, 2 GB RAM, and ten minutes. It reuses
`nkn_msa_transformer` and logs totals to `summary_msa_verification-<job>.out`.
For job-specific summaries, task logs are searched in the submission directory;
override this with `--logs-dir PATH` if the array was submitted elsewhere.

Summarize all expected assays in one command after the array finishes (or rerun
it while jobs are in progress). This uses only Python's standard library:

```bash
python sbatch_run/run_prepare_rep/summarize_msa_verification.py --job-id YOUR_ARRAY_JOB_ID
```

Replace `YOUR_ARRAY_JOB_ID` with the numeric ID printed by `sbatch`. The tool
uses `verify_msa_all-<job>_<index>.out` to locate the reports from that specific
array. Use `--logs-dir /submission/directory` if the array was submitted outside
the repository root. `--reference-csv PATH` overrides the reference CSV.

The terminal prints overall counts; `data/msa/verification/summary_<job>/`
contains `verification_summary.csv` (one row per expected assay with counts,
differences, report/log paths, and status) and `verification_summary.json`
(totals and reference hash). Missing reports stay in the denominator and are
never treated as matches. They can indicate pending/running jobs, early failures,
or jobs terminated by Slurm; this tool does not query the scheduler.

Without `--job-id`, the tool selects the latest available run directory for each
assay and writes `summary_latest/`. This can mix arrays and cannot detect a newer
attempt that failed before creating a directory; use job-specific summaries for
full-catalog verification. Duplicate run counts are reported. Malformed reports,
wrong assay IDs, and reports from a different reference-file hash do not pass.
Re-running the summary refreshes its CSV/JSON without modifying individual runs.

### MSA Transformer: generate paired alignments and weights

To calculate our own weights for every assay in the reference CSV, submit from
the synced Phoenix repository:

```bash
sbatch sbatch_run/run_prepare_rep/generate_msa_weights.sbatch
```

This uses the existing `nkn_msa_transformer` environment and pinned ProteinGym
checkout. No GPU is required. The coordinator snapshots the catalog and
`configs/setup/msa_generation.json`, submits one CPU task per assay (eight CPUs,
64 GB, twelve hours per task; four concurrent tasks), then schedules an overall
summary after all tasks finish, including failed tasks. Pass `--reference-csv PATH`
or `--max-concurrent N` to override the catalog or concurrency. The default catalog
is `data/DMS_ProteinGym_substitutions/DMS_substitutions.csv`.

Generation applies the same normalization, HHfilter settings (`-cov 75 -id 90
-qid 0`), and pinned ProteinGym postprocessing used in the specialized-weight
verification. It calculates fresh EVE weights with theta 0.2 directly from the
retained rows. Downloaded weights are neither read nor required, so a previous
count mismatch does not block generation. Outputs are saved directly in
`data/msa/DMS_msa_weights_generated/`, without a `batch_<id>` directory.
The output directory must be empty when starting generation; a nonempty directory
is rejected to prevent mixing runs or overwriting results. Use `--output-dir PATH`
to select another empty destination for a new run.

```text
data/msa/DMS_msa_weights_generated/
  DMS_substitutions.csv            # exact catalog snapshot
  config.json                     # recipe and output paths
  generation_summary.csv          # status and directory for every expected assay
  generation_summary.json         # overall counts, completeness and reference hash
  assays/<DMS_id>_<run_id>/
    alignment.a2m                 # final ordered alignment, query first
    weights.npy                   # weights[i] belongs to alignment row i
    rows.csv                      # original row/header, sequence hashes, index, weight
    columns.csv                   # final column -> raw column -> target position
    manifest.json                 # completed pair, checksums and provenance
    generation_report.json        # success/failure diagnostics
```

Each assay directory also retains the normalized and HHfilter alignments, the
HHfilter log, and a table showing which original rows were removed. A completion
manifest is published only after reading the saved pair back and validating its
dimensions, weights, ordered sequence identities, row indices and checksums.
`validate_pair` in `sbatch_run/run_prepare_rep/generated_msa.py` provides the same
checks for subsequent readers and returns the alignment and weight vector.

The query has weight index zero and is included in the saved vector and Neff.
For later MSA Transformer sampling, retain the query separately and normalize
the non-query weights over eligible homologs. Query-only outputs are counted in
the summary so an inference policy can handle them explicitly. These files are
preprocessing inputs for the preparation job below.

The summary validates every completed pair again and counts missing, failed,
incomplete, corrupt or ambiguous runs as needing attention. To refresh it without
launching generation again:

```bash
sbatch sbatch_run/run_prepare_rep/generate_msa_weights.sbatch --summarize
```

### MSA Transformer: prepare sequence features and mean scores

After runtime setup and pair generation succeed, submit from the repository:

```bash
sbatch sbatch_run/run_prepare_rep/prepare_msa_transformer.sbatch
```

This processes every assay in the extraction catalog, in catalog order, and
selects the same single mutants as `prepare_esm2.sbatch`. For one GPU job per
assay, with at most four concurrent jobs:

```bash
n=$(python sbatch_run/run_prepare_rep/prepare_msa_transformer.py --count)
sbatch --array="0-$((n-1))%4" sbatch_run/run_prepare_rep/prepare_msa_transformer.sbatch
```

The count command needs only the standard library. The batch job activates
`nkn_msa_transformer` and verifies the existing pinned runtime and checkpoint.
It performs no downloads. To check the example first:

```bash
sbatch sbatch_run/run_prepare_rep/prepare_msa_transformer.sbatch --assay YAP1_HUMAN_Araya_2012
```

Configuration precedence is `--config PATH`, `MSA_TRANSFORMER_CONFIG`, then
`configs/extraction/msa_transformer.json`. The preset uses layer 12 (768
dimensions), 400 MSA rows including the query, and sampling seeds 1, 2, 3, 4, 5.
For each seed, Python's `random.choices` draws 399 homolog rows **with
replacement** using normalized non-query weights. Duplicate rows and draw order
are preserved, including when the source alignment has fewer than 400 rows.
Each sampled context is reused across variants and windows. Query-only
alignments use depth 1. Scores are averaged arithmetically over the five samples.
These defaults match the pinned ProteinGym
[scoring launcher](https://github.com/OATML-Markslab/ProteinGym/blob/144fe22b07dfaeec2b366f2346203a9838a55b4c/scripts/scoring_DMS_zero_shot/scoring_MSA_transformer_substitutions.sh)
and [sampler/scorer](https://github.com/OATML-Markslab/ProteinGym/blob/144fe22b07dfaeec2b366f2346203a9838a55b4c/proteingym/baselines/esm/compute_fitness.py).

Each assay must have exactly one `assays/<DMS_id>_<run_id>/` directory under
`msa_root`. Preparation verifies the manifest, saved alignment/weight pair and
column mapping. The generation catalog snapshot is checked against the manifest;
if it was not copied, `msa_reference_csv` must match the recorded checksum.
Missing, ambiguous, corrupt or mismatched pairs fail that assay. For a CPU
input check, without loading a model or writing caches:

```bash
conda activate nkn_msa_transformer
python sbatch_run/run_prepare_rep/prepare_msa_transformer.py --assay YAP1_HUMAN_Araya_2012 --validate-inputs-only
```

Mutation scoring crops the WT query and homologs to the catalog's MSA interval,
then tokenizes the complete interval. It masks one query residue at a time,
keeps homologs unchanged, and subtracts WT from mutant log probabilities.
Each required position is evaluated once per seed and reused across variants.
For long alignments, scoring slices existing tokens into ProteinGym's optimal
1,024-token windows. It preserves upstream's `MSA length + 2` window calculation,
including clipped tail windows; interior slices do not insert a new BOS token.
The output averages float32 per-seed log-odds in float64, as ProteinGym does.
Mutation sites outside the MSA interval fail score preparation rather than
producing a complete score cache. Input preflight checks this coverage too.

These per-variant zero-shot scores are saved in `mean_score.csv` and selected
through `score_sources` in both MSA Transformer NKN presets. They supply the GP
mean: `m(x) = intercept + coefficient * standardized_MSA_score(x)`. Score
standardization is fitted on the training fold only; the intercept and
coefficient are learned during GP training. The saved CSV contains the raw
ProteinGym-style seed-averaged log-odds. Length-weighted embedding pooling does
not alter these scores.

The v3 global/site embeddings are an NKN extension, averaged over the same five
sampled contexts. Both use unmasked mutant queries and layer-12 query-row
representations from MSA Transformer:

- **Global:** cover only the MSA interval, using fixed consecutive nonoverlapping
  windows of at most 1,022 residues. The same boundaries apply to every variant,
  including variants mutated inside the interval. One window covers a short
  interval. Combine window means as `sum(window_length * window_mean) /
  MSA_interval_length`, accumulating in float64 and saving float32. Thus a short
  final window contributes in proportion to its residue count. Mutations outside
  coverage leave this input WT and share the WT global embedding.
- **Site:** use a centered, clamped window in the **full mutated query**, then
  extract the mutated residue's vector. Homologs are mapped through `columns.csv`
  and gap-padded outside MSA coverage. Remove homolog rows that are all gaps in
  each embedding window, preserving sampled order and duplicates among remaining
  rows. If none remain, run the same MSA Transformer with the query as its only
  row. This also applies to internal windows lacking homolog residue evidence.

Special tokens and homolog rows are excluded from pooling. The cache records
zero-based half-open global interval bounds and its represented length; site
coordinates and sequence identities still refer to the full protein. These
embedding policies do not change ProteinGym scoring windows, rows or coverage.

Preparation consumes the project's generated MSA/weight pairs. Matching scoring
settings alone does not reproduce published ProteinGym numbers: that also
requires the same alignments, preprocessing, sequence weights and checkpoint.

Outputs share the existing immutable single-mutant assay table:

```text
data/features/pretrained/singles/<assay>/
  sequence/msa_transformer/msa1b-layer12-depth400-seeds1-5-msaregion1022-sites-v3/features.pt
  scores/msa_transformer/msa1b-depth400-seeds1-5-proteingym-masked-marginals-v2/mean_score.csv
```

Both leaves include configuration, provenance and completion checksums.
`--resume` is enabled by the sbatch script and validates completed leaves before
skipping them. Feature and score stages resume independently; failed assays do
not stop later assays, but any failure makes the job exit nonzero. Changing the
checkpoint, pair, sampled rows or recipe requires new cache IDs. Existing v1/v2
embedding caches remain readable under their original IDs. The shipped NKN
presets select new v3 features; existing valid v2 score caches are reused. The
16,384-token setting controls upstream axial-attention chunking; it does not
truncate the alignment. Reducing depth creates a custom recipe that differs
from ProteinGym's default and requires new cache IDs.

Use `configs/msa_transformer_proteinmpnn_nkn.json` to replace ESM-2 in the existing
NKN sequence branches and GP mean, keeping the ProteinMPNN and structure branches.
Those branches still require their usual prepared inputs. For sequence-only
experiments use `configs/msa_transformer_nkn.json`. Neither preset changes the
existing ESM-2 selection. Check the NKN inputs in the usual NKN environment:

```bash
python -m experiments.cli validate --features data/features/pretrained/singles/YAP1_HUMAN_Araya_2012 --config configs/msa_transformer_nkn.json --no-output
```

Use the same `--config` with the normal NKN `cv` command and published folds.
The MSA environment is for feature preparation; run NKN in its usual environment.
In a custom experiment config, select:

```json
{
  "feature_selection": {
    "sequence": "msa_transformer/msa1b-layer12-depth400-seeds1-5-msaregion1022-sites-v3"
  },
  "score_sources": [
    "scores/msa_transformer/msa1b-depth400-seeds1-5-proteingym-masked-marginals-v2"
  ]
}
```

Validate saved leaves in either the MSA environment or the NKN environment:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m feature_generation.cli validate --features data/features/pretrained/singles/YAP1_HUMAN_Araya_2012 --model-family msa_transformer --cache-id msa1b-layer12-depth400-seeds1-5-msaregion1022-sites-v3
python -m feature_generation.cli validate --features data/features/pretrained/singles/YAP1_HUMAN_Araya_2012 --feature-type scores --model-family msa_transformer --cache-id msa1b-depth400-seeds1-5-proteingym-masked-marginals-v2
```

## Inverse folding: ProteinMPNN and ESM-IF1

The CLI is the main workflow. Both models write one `[L, 20]` distribution for
the reference structure (or `[C, 20]` for explicitly covered fragment positions), shared by all assay variants; DMS labels are not model
inputs. The supported recipes have different conditioning:

| Config | Checkpoint | Conditioning |
| --- | --- | --- |
| `proteinmpnn_v48_020_kermut.json` | vanilla `v_48_020` | `sequence_conditional`: structure and all other wild-type residues, ten orders, seed 37 |
| `proteinmpnn_v48_020.json` | vanilla `v_48_020` | Optional legacy `backbone_only`: structure without sequence context |
| `esm_if1_142m.json` | `esm_if1_gvp4_t16_142M_UR50` | `reference_prefix`: structure and preceding wild-type residues |

ESM-IF1 output is an autoregressive reference-context distribution, not an exact
backbone-only marginal or a conditional on the complete remaining WT sequence.
ProteinMPNN's Kermut recipe uses the native `conditional_probs` routine: the
target position is decoded after all other wild-type residues. It averages the
log probabilities from ten randomized orders, exponentiates, and removes `X`
without renormalizing, matching the authors'
[released preprocessing](https://github.com/petergroth/kermut/blob/reproduce/src/data/extract_ProteinMPNN_probs.py).
These are geometric-mean probabilities and their rows can sum to less than one.
The checkpoint is shared with the backbone-only recipe; its output cache is
separate. Sequence generation is not implemented. ProteinMPNN variant sequence
likelihoods are prepared separately with the `score` command below.

The ProteinMPNN and ESM-IF1 presets set `max_length: null`, so inverse-folding
extraction and ProteinMPNN scoring have no sequence-length cap. A positive integer
can impose an optional operational limit, including values above 1,022. Long
inputs require sufficient memory and runtime. These workflows never truncate
model inputs. Explicit fragment manifests supply each registered piece separately;
an explicit `max_length` still limits the full assay reference.

Install/update from `env.yaml`, or use `python -m pip install -r requirements.txt`
in a Python 3.10 venv. Added inference dependencies are Biopython, Biotite, SciPy,
and PyTorch Geometric. Official inference source is downloaded at pinned commits
alongside the verified weights; no `fair-esm` installation is needed. If a notebook
has already imported a different top-level `esm` package, restart its kernel before
loading ESM-IF1. Fresh CLI processes avoid that import conflict.

From the project root, with `src` on `PYTHONPATH`, prepare the pilot structure:

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path
$assay = "A0A1I9GEU1_NEIME_Kennouche_2019"
python -m feature_generation.cli download-structure --assay $assay
```

This caches the checksum-pinned ProteinGym v1.1 structure archive and extracts
only the selected catalog PDB. To use a local PDB/mmCIF instead:

```powershell
python -m feature_generation.cli prepare-structure --assay $assay --structure path/to/reference.pdb --chain A
```

A direct structure URL can be supplied with `download-structure --url ...`.
Raw downloads live in `data/structures/proteingym/raw/`, and validated manifests in
`data/structures/proteingym/manifests/<assay>.json`. The manifest records the structure hash,
coordinate model, target chain, and reference-position-to-author-residue mapping.

Structure collections live under `data/structures/proteingym/` and
`data/structures/protssn_proteingym/`. Each collection has `source.json`,
`downloads/`, `raw/`, `manifests/`, and `tokens/prosst/`. Source metadata records
upstream versions and per-file checksums; raw filenames and PDB contents are
preserved. Manifest filenames use canonical assay IDs; known upstream filename
aliases are recorded in the collection metadata.

Structure-conditioned extraction configs select `structure_source` (default:
`proteingym`). Omitted manifest/token paths resolve inside that source. Explicit
paths and source-prefixed cache IDs belonging to another collection are rejected.
Automatic feature and score cache names use `<source>--<model-recipe>`; shipped
presets and experiment selections follow this convention. Deliberately supplied
custom cache IDs remain supported and retain all existing identity checks.
ProSST token caches remain keyed by quantizer recipe and structure signature
inside the selected collection. The structure-registration CLI also accepts
`--structure-source`. Automatic archive downloads remain restricted to the
official collection; jobs require an existing manifest for other sources.

When migrating an existing workspace, relocate `raw/`, `manifests/`, and
`tokens/` together under `proteingym/` to preserve relative manifest references.
Legacy feature caches retain their original IDs. Source-qualified copies need
their checkpoint/config signatures and completion checksums updated consistently;
renaming a feature directory alone is insufficient. Existing legacy cache IDs
can still be selected explicitly for historical runs.

For a single-structure manifest, the selected chain must match the entire reference sequence with N/CA/C atoms;
ProteinMPNN additionally requires O. Partial coverage, sequence mismatches,
noncanonical residues, and ambiguous backbone alternate conformations are rejected.
Other chains are not used as structural context in this first version.
Choose `--chain` explicitly for multichain structures and `--model-index` for
multi-model structures. No sequence offset or cropping is inferred from `pdb_range`.

The ESM-IF1 preset selects `protssn_proteingym`. For direct offline CLI extraction,
register its assay manifest in `data/structures/protssn_proteingym/manifests/`; the official
ProteinGym manifest prepared above belongs to the ProteinMPNN preset.
Download both checkpoints explicitly, then extract offline:

```powershell
python -m feature_generation.cli download --config configs/extraction/proteinmpnn_v48_020_kermut.json
python -m feature_generation.cli download --config configs/extraction/esm_if1_142m.json
python -m feature_generation.cli extract --config configs/extraction/proteinmpnn_v48_020_kermut.json --assay $assay --singles-only --resume
python -m feature_generation.cli extract --config configs/extraction/esm_if1_142m.json --assay $assay --resume
python -m feature_generation.cli validate --features "data/features/pretrained/singles/$assay" --feature-type inverse_folding --model-family proteinmpnn --cache-id proteingym--v_48_020-sequence_conditional-orders10-seed37-8907e6671bfb
python -m feature_generation.cli validate --features "data/features/pretrained/$assay" --feature-type inverse_folding --model-family esm_if1
```

For Slurm preparation, [`prepare_esm_if1.sbatch`](../../sbatch_run/run_prepare_rep/prepare_esm_if1.sbatch)
uses `esm_if1_142m.json` and explicitly saves single-mutant caches under
`data/features/pretrained/singles/`. It selects `protssn_proteingym`, downloads/verifies
the pinned model, automatically registers missing manifests from that collection's local
raw PDBs, verifies reference structures, extracts WT-prefix distributions, scores complete variants,
and validates both caches. It preserves registered structure manifests and requires
N/CA/C coordinates. Mean scoring uses the ProteinGym recipe described below, with
batch sizes selected from each assay's full reference length: 16 for <=300 residues,
8 for 301-600, 4 for 601-1,000, 2 for 1,001-2,000, and 1 for >2,000. Assays sharing
a batch size are scored together to reuse the loaded model; the log records each
assay's length and batch size. Full-sequence scores require one backbone covering the entire
reference; partial structures and independent fragments cannot supply that score.
Their kernel features remain available even if the separate scoring stage fails.
Automatic registration checks the PDB against `source.json`, resolves its filename
aliases, and validates the reference mapping. Missing raw files or invalid structures
fail the assay. Feature and score cache IDs start with `protssn_proteingym--`.

```bash
sbatch sbatch_run/run_prepare_rep/prepare_esm_if1.sbatch
# Alternatively, one assay per array task in the configured catalog order:
n=$(python sbatch_run/run_prepare_rep/list_assays.py --config configs/extraction/esm_if1_142m.json --count)
sbatch --array="0-$((n - 1))" sbatch_run/run_prepare_rep/prepare_esm_if1.sbatch
```

Without an array, the job processes the full catalog sequentially, with no default
reference-length cap. Array indices are never filtered or renumbered. Other assays
continue after a failure, and any failed stage gives the job a nonzero exit status.
Resubmission reuses verified downloads and resumes matching completed caches,
including those saved under the previous 1,022-residue guard. Changing only this
operational guard preserves cache compatibility and leaves saved provenance intact.
These normalized ESM-IF1 probabilities
require `inverse_folding_policy="normalized"` when selecting them for NKN kernels.

The optional legacy backbone-only recipe uses the same downloaded checkpoint.
To prepare it for experiments that still select that cache, run it separately:

```powershell
python -m feature_generation.cli extract --config configs/extraction/proteinmpnn_v48_020.json --assay $assay --singles-only --resume
python -m feature_generation.cli validate --features "data/features/pretrained/singles/$assay" --feature-type inverse_folding --model-family proteinmpnn --cache-id proteingym--v_48_020-backbone_only-8907e6671bfb
```

Conditional inference performs one decoder pass per reference position per
order, so it costs substantially more than backbone-only extraction. This
matches the extraction recipe for supported single-chain inputs; exact published
outputs additionally depend on the same structure and runtime. Longer references
can exceed GPU memory or the job's time allocation despite the removed length
guard. Seed 37 and the order count are recorded in cache provenance.
`--resume` rejects a changed
recipe under an existing cache ID.

The original ESM-IF1 checkpoint download is approximately 1.7 GB. Model/source files
are verified against the pinned artifact manifest before loading. Their provenance
includes exact source URLs, upstream commits, and file SHA-256 checksums.

Use `--output-root data/features/inverse_smoke --limit 4` on every extraction
command in an isolated smoke test, with the same `--singles-only` selection for
all models sharing that root. `--limit` changes only the shared variant
table; inverse-folding inference still covers every registered structure residue. A full-assay
cache and a subset cannot occupy the same assay directory. `--resume` checks the
checkpoint, structure, mapping, extraction settings, and all saved artifact hashes.

Each model saves its own leaf under its configured output root (the ProteinMPNN
presets use `data/features/pretrained/singles`; ESM-IF1 uses
`data/features/pretrained`):

```text
<output_root>/<assay>/inverse_folding/<family>/<recipe>/
  features.pt
  config.json
  provenance.json
  completion.json
```

`features.pt` stores `probabilities[C,20]`, `log_probs[C,20]`, one-based `positions`,
reference identity, and alphabet `ACDEFGHIKLMNPQRSTVWY`. Vocabulary entries are
selected by amino-acid identity. The backbone-only ProteinMPNN and ESM-IF1 recipes
normalize over these 20 candidates at T=1; their probability rows sum to one.
Kermut's sequence-conditioned ProteinMPNN recipe preserves the values after
log averaging and removing `X`, without a second normalization. Both tensors
are finite float32, and log probabilities agree with their probabilities.
Special tokens and ProteinMPNN's `X` are excluded from the saved tensors.
For a full structure, `C=L` and positions are `1..L`. Fragment caches store only
covered positions; probability row indices are resolved through `positions`.

All recipe leaves can coexist; the first completed leaf becomes the inverse-folding
default. Select one using `feature_selection["inverse_folding"] = "<family>/<recipe>"`
in the experiment config. GP experiments load only the feature types required by
`model.enabled_branches`; structure embeddings are needed only when `structure` is
enabled. At least one mean score source is still required for the GP mean;
standalone inverse cache validation does not require scores.

The NKN preset `configs/esm2_proteinmpnn_nkn.json` selects the same
sequence-conditioned ProteinMPNN cache as Kermut, together with ESM2 650M.
It declares `inverse_folding_policy: "sequence_conditional"` so the NKN loader
and kernels preserve the saved values without renormalization. Configurations
using normalized backbone-only or ESM-IF1 distributions retain the default
`"normalized"` policy. The current NKN preset uses ESM2 mean scores. Optional
ProteinMPNN mean scores for full structures use one autoregressive order with
seed 0; feature conditioning does not change that scoring recipe.

### BRCA2 fragment preparation

`BRCA2_HUMAN_Erwood_2022_HEK293T` uses three ProteinGym PDB files with separate
coordinate frames. The standard download command recognizes this assay and
registers a schema-v2 fragment manifest:

```bash
python -m feature_generation.cli download-structure --assay BRCA2_HUMAN_Erwood_2022_HEK293T
python -m feature_generation.cli extract --config configs/extraction/proteinmpnn_v48_020_kermut.json --assay BRCA2_HUMAN_Erwood_2022_HEK293T --singles-only --device cuda --resume
python -m feature_generation.cli validate --features data/features/pretrained/singles/BRCA2_HUMAN_Erwood_2022_HEK293T --feature-type inverse_folding --model-family proteinmpnn --cache-id proteingym--v_48_020-sequence_conditional-orders10-seed37-8907e6671bfb
```

ProteinMPNN processes reference intervals 1Ã¢â‚¬â€œ1000, 1001Ã¢â‚¬â€œ2085, and 2086Ã¢â‚¬â€œ2832 in
three independent calls, each using the configured seed and order count. Its
sequence context is local to each fragment. Output rows are concatenated in
reference-position order without changing probabilities or normalizing them
again. The cache retains the full 3418-residue reference and contains `[2832,20]`
probabilities plus their explicit positions. All 265 variants in the current
assay mutate covered sites. Positions 2833Ã¢â‚¬â€œ3418 have no structure-derived
probabilities, and extraction/loading rejects mutations there.

Each fragment must exactly match its declared reference slice. Overlapping or
out-of-order intervals, changed files, ambiguous chains, and missing required
backbone atoms are rejected. The manifest records each fragment's checksum,
local author numbering, reference interval, and a combined identity. Coordinates
are never concatenated into a global structure. Single-structure consumers,
including distance kernels and full-sequence ProteinMPNN scoring, reject these
manifests. The ProteinMPNN preparation job skips its optional mean-score stage
for this assay; prepare ESM2 scores for the default NKN mean instead.

NKN automatically disables `structure_distance` for this assay before feature
loading and records the effective branches in the saved run configuration.
The other configured branches retain their usual inputs. ESM2 sequence features
and mean scores must also be prepared with the existing ESM2 workflow.

Local fragments can be registered explicitly using repeated file/range pairs:

```bash
python -m feature_generation.cli prepare-structure --assay BRCA2_HUMAN_Erwood_2022_HEK293T --structure data/structures/proteingym/raw/BRCA2_HUMAN_1-1000.pdb --reference-range 1-1000 --structure data/structures/proteingym/raw/BRCA2_HUMAN_1001-2085.pdb --reference-range 1001-2085 --structure data/structures/proteingym/raw/BRCA2_HUMAN_2086-2832.pdb --reference-range 2086-2832
```

### Kermut reference mappings

Six assays require the same post-inference mapping used by Kermut's
`extract_ProteinMPNN_probs.py` and `extract_3d_coords.py`. Structure download and
registration apply these explicit exceptions automatically:

| Assay | Treatment |
| --- | --- |
| `CAS9_STRP1_Spencer_2017_positive` | Infer features on all 1390 PDB residues, then retain output rows 1-1368. |
| Three `P53_HUMAN_Giacomelli_2018_*` assays | Allow exactly PDB P72 versus assay R72 when matching; feature inference still uses the original PDB P72 sequence. |
| `A0A140D2T1_ZIKV_Sourisseau_2019` | Map the 504 PDB residues to reference positions 291-794. |
| `POLG_HCVJF_Qi_2014` | Map the 244 PDB residues to reference positions 1982-2225. |

Schema-v3 manifests preserve the source file, checksum, original PDB sequence,
and explicit source-to-reference mapping. Other mismatches remain errors.
For domain structures, caches store only covered rows with explicit reference
positions rather than Kermut's NaN-padded probability arrays. They retain the
full assay reference and reject mutations outside structural coverage. Domain
coordinates share one physical frame, so the distance kernel remains available.

ZIKV and HCV do not have full-reference ProteinMPNN likelihood scores. Preparation
skips those scores, readiness reports SKIP, and NKN drops the ProteinMPNN mean
source while retaining ESM2. CAS9 and P53 can use the existing full-sequence
scorer with their mapped reference backbones and assay variant sequences; this
score is separate from Kermut's reference probability features. For CAS9, the
scoring backbone has 1368 residues, while feature inference uses all 1390.

The ProteinMPNN script now defaults to retrying these six failed assays as
array tasks 0,31,116-118,130 in the current 217-row catalog:

```bash
sbatch sbatch_run/run_prepare_rep/prepare_proteinmpnn.sbatch
```

To prepare the whole current catalog, override with
`sbatch --array=0-216 sbatch_run/run_prepare_rep/prepare_proteinmpnn.sbatch`. Array indices follow
catalog order; confirm them if the catalog changes. Existing
valid manifests and matching completed feature/score caches are reused. ESM2
does not need rerunning for this mapping change. Run `validate_prepared.sbatch`
after preparation to refresh the readiness log and CSV.

Upstream implementations: [ProteinMPNN](https://github.com/dauparas/ProteinMPNN),
[ESM-IF1](https://github.com/facebookresearch/esm/tree/main/examples/inverse_folding).

## ESM2 scores for the GP mean

ESM2 scoring uses the pretrained Hugging Face `EsmForMaskedLM` head and WT
masked marginals. For each position mutated by any selected variant, it masks
exactly that residue in its WT context and computes model log-probabilities.
The score for variant `v` is:

```text
score(v) = sum over mutated positions i:
    log p(mutant_i | WT context with position i masked)
  - log p(WT_i     | WT context with position i masked)
```

WT has score zero. Multiple substitutions add contributions from the same WT
context for each position. Higher values indicate that the model favors the mutant residues over
WT. The score does not use experimental labels or a structure. Each unique
mutation position is evaluated once per assay and reused across variants. This
is the masked-marginal scoring recipe; the existing embedding cache alone cannot
supply these scores, so a separate inference stage uses the checkpoint's MLM head.

`scoring_window="centered"` uses the full WT reference when it fits in
`max_length`, otherwise a window centered on the masked position and clamped to
the sequence ends. The default NKN preparation preset enables this mode with
`max_length=1022`. For one-based position `p`, set `W=min(L,max_length)` and
`start=max(0,min(p-1-W//2,L-W))`; score the slice `WT[start:start+W]`, masking
its residue at zero-based offset `p-1-start`. BOS/EOS tokens are added afterward.
For a 3,000-residue reference, positions 1,000 and 2,300 use residues 489Ã¢â‚¬â€œ1,510
and 1,789Ã¢â‚¬â€œ2,810 respectively. Other positions in each window stay wild type,
including positions mutated elsewhere in the variant. Distant context outside
the window does not affect the score.

All ESM2 extraction presets work for scoring (8M and 650M, pooled, residue, or
centered-site). Configs omitting `scoring_window` retain `"full"`: full-reference
scoring with the existing length check and cache identity. This choice is
independent of embedding pooling and `site_embeddings`.
Scoring always uses the final pretrained MLM head; embedding `layer` and
representation `cache_id` do not
select its scoring layer or output directory. The main NKN and Kermut presets
select the 650M checkpoint. The 8M preset remains available for smoke tests.

```bash
# Required only if weights/tokenizer are not downloaded yet; valid assets are reused.
python -m feature_generation.cli download --config configs/extraction/esm2_650m_centered.json

python -m feature_generation.cli score --config configs/extraction/esm2_650m_centered.json --assay A0A1I9GEU1_NEIME_Kennouche_2019 --singles-only --device cuda --batch-size 8 --resume

python -m feature_generation.cli validate --features data/features/pretrained/singles/A0A1I9GEU1_NEIME_Kennouche_2019 --feature-type scores --model-family esm2 --cache-id esm2_t33_650M_UR50D-masked_marginals-float32-08e4846e5371-centered1022
```

Use `--device cpu` without a GPU. `--batch-size` caps the number of masked WT
copies per forward pass; `max_tokens_per_batch` additionally limits total tokens,
including BOS/EOS; the budget applies to each context window. `--num-orders` and structure-manifest
options apply to ProteinMPNN and are rejected for ESM2. New dependencies are not
needed: the verified safetensors checkpoint already contains the MLM head.

`prepare_esm2.sbatch` downloads/verifies ESM2 650M and then runs both layer-33
embedding extraction and scoring with `--singles-only` for only
`PIN1_HUMAN_Tsuboyama_2023_1I6C`, `VKOR1_HUMAN_Chiasson_2020_activity`, and
`BLAT_ECOLX_Jacquier_2013`, in that order, using the configured
`data/DMS_ProteinGym_substitutions/DMS_substitutions.csv` catalog.
Its default
`configs/extraction/esm2_650m_centered.json` saves centered mutation-site vectors
and a full-sequence mean from fixed non-overlapping windows for NKN's sequence
branches, plus centered-window masked-marginal scores. Kermut's default reuses
the global mean and score caches; it does not use the saved site vectors. The Conda environment and
ProteinGym assay CSVs must already exist. First-time downloads need outbound
HTTPS from the compute node; verified completed caches are reused on resubmission.

```text
data/features/pretrained/singles/<assay>/
  scores/esm2/esm2_t33_650M_UR50D-masked_marginals-float32-08e4846e5371-centered1022/
    mean_score.csv
    config.json
    provenance.json
    completion.json
```

The CSV has `variant_id,score` with exact coverage of the shared `variants.csv`.
The filename describes its role in the GP mean; the ESM2 value is a sum of mutation
log-odds. Its metadata records checkpoint hashes, masking recipe, precision and
score direction. The recipe ID includes checkpoint, inference precision and
revision; centered scoring adds `-centered<max_length>` and recipe version 2.
Its metadata declares the window policy and coordinate convention. Full-reference
version-1 scores remain readable under their original IDs. Switching scoring
mode does not change the embedding cache identity, so completed embeddings can
be reused. NKN and Kermut both explicitly select the centered scores.
ProteinMPNN score caches are preserved.

Select just ESM2 scores for the mean function:

```json
{
  "score_sources": [
    "scores/esm2/esm2_t33_650M_UR50D-masked_marginals-float32-08e4846e5371-centered1022"
  ]
}
```

The generic NKN batch preset selects only ESM2 650M scores. The five-assay array
launcher uses `configs/esm2_proteinmpnn_nkn_two_scores.json`, which also selects
ProteinMPNN sequence-likelihood scores. Each source creates a separate score
column; each is standardized
using training variants only, and the GP learns a coefficient for each.
This choice is independent of `model.enabled_branches`.

Recipe references: [official ESM variant scoring](https://github.com/facebookresearch/esm/blob/main/examples/variant-prediction/predict.py),
[Kermut ESM2 scoring](https://github.com/petergroth/kermut/blob/main/kermut/cmdline/preprocess_data/extract_esm2_zero_shots.py).

## ESM-IF1 scores for the GP mean

The `score` command implements ProteinGym's `esmif1_ll` recipe on the fixed
reference backbone. For each complete variant sequence `s`, it computes
`mean_i log p(s_i | backbone, s_<i)` using that variant's causal prefix, in N-to-C
order. Every residue contributes, including unchanged residues. Cross entropy
uses the full native output vocabulary; padding is excluded from the mean.
There is no WT subtraction or additional normalization over the 20 canonical
amino acids. Higher scores indicate greater model likelihood.

These scores require a separate forward pass for each variant (or variant batch).
The shared WT-prefix, 20-way kernel distributions cannot reproduce them. Model
weights stay frozen, and experimental labels are never passed to the scorer.

```bash
python -m feature_generation.cli score --config configs/extraction/esm_if1_142m.json --output-root data/features/pretrained/singles --assay YAP1_HUMAN_Araya_2012 --singles-only --device cuda --batch-size 1 --resume
python -m feature_generation.cli validate --features data/features/pretrained/singles/YAP1_HUMAN_Araya_2012 --feature-type scores --model-family esm_if1 --cache-id protssn_proteingym--esm_if1_gvp4_t16_142M_UR50-sequence_log_likelihood-native-v1-2b369911bb5b
```

The existing extraction config supplies the checkpoint, paths and optional
`max_length` guard; its `reference_prefix` conditioning identifies kernel features,
while the score cache records `sequence_autoregressive` and `prefix_sequence=variant`.
No sequence-length cap is applied by default. Scoring uses float32 and defaults to
one variant per batch; `--batch-size` can increase this when memory permits.
`--num-orders` is specific to ProteinMPNN and does not apply to ESM-IF1.
The N/CA/C backbone must cover the entire reference in one coordinate frame.
Unsupported fragments or partial mappings fail explicitly without cropping or
aggregating fragment scores.

Scores are saved as `variant_id,score` in a separate `mean_score.csv`, with recipe,
checkpoint, structure and checksum metadata. `--resume` validates and reuses
completed caches independently of kernel-feature caches. The same output root
and variant selection must be used for scores and features. The preparation job
above runs both stages and validates both even when another stage fails.

To use this score in NKN, select the following source in the experiment config,
either alone or alongside existing ESM2/ProteinMPNN sources:

```json
"score_sources": [
  "scores/esm_if1/protssn_proteingym--esm_if1_gvp4_t16_142M_UR50-sequence_log_likelihood-native-v1-2b369911bb5b"
]
```

The cached values are raw log-likelihoods. Existing GP preprocessing standardizes
each score column using the training fold only, then applies those statistics to
validation/test variants. The GP learns its intercept and each score coefficient.
This choice is independent of the kernel-feature selection.

Recipe reference: [ProteinGym ESM-IF1 scoring](https://github.com/OATML-Markslab/ProteinGym/blob/main/proteingym/baselines/esm/compute_fitness_esm_if1.py).

## ProteinMPNN scores for the GP mean

The `score` command runs the pinned official ProteinMPNN forward pass on every
complete variant sequence and the fixed reference backbone. It stores the
negative of the official mean negative log-likelihood: higher values indicate
higher sequence likelihood. All reference residues contribute, including unchanged
positions. Probabilities retain the native 21-token vocabulary (including X in
the normalization); the score is not derived from the canonical 20-way extraction
cache. There is no WT subtraction, DMS-label input, or finetuning.

```bash
python -m feature_generation.cli score --config configs/extraction/proteinmpnn_v48_020.json --assay A0A1I9GEU1_NEIME_Kennouche_2019 --singles-only --device cuda --batch-size 1 --resume
python -m feature_generation.cli validate --features data/features/pretrained/singles/A0A1I9GEU1_NEIME_Kennouche_2019 --feature-type scores --model-family proteinmpnn --cache-id proteingym--v_48_020-sequence_log_likelihood-orders1-seed0-8907e6671bfb
```

Use `--device cpu` when no GPU is available. This reuses the existing ProteinMPNN
checkpoint and structure manifest; no new dependencies or model downloads are
needed if extraction already ran. Scoring can also run before extraction.
`prepare_proteinmpnn.sbatch` downloads and verifies the pinned checkpoint/source
and ProteinGym reference structures for all catalog assays, then
extracts the Kermut sequence-conditioned features and prepares variant scores,
using `--singles-only` for both stages. The reference distribution still covers
every residue; the shared variant table and scores contain only single mutants.
Existing structure manifests, including custom sources, are validated and
reused. New structures use their sole polymer chain; multichain structures need
a separately prepared manifest with an explicit chain. These are four
stages: model assets, structure, features, and scores. The job retains seed 0
and one order for variant scoring, independently of the Kermut feature recipe.
Scoring uses one complete variant per batch to reduce peak GPU memory for long
proteins. Both ProteinMPNN presets disable the optional sequence-length guard;
the job still requires full-reference structures and sufficient memory/runtime.
It uses `proteinmpnn_v48_020.json` only for that separate scoring stage; the
job's feature extraction uses `proteinmpnn_v48_020_kermut.json`.
No checkpoint or structure needs to be downloaded before submission. The Conda
environment and ProteinGym assay CSVs must already exist, and the allocated
compute node needs outbound HTTPS access. Valid assets and completed caches are
verified and reused on subsequent submissions.

`--num-orders` defaults to 1. Larger values average the mean log-likelihoods over
that many random decoding orders. The config's `seed` drives a local CPU random
generator; the same orders are shared across variants and batch sizes. This uses
the official sequence-scoring formula with reproducible common decoding orders,
not a claim of identical random draws to ProteinGym's scoring script. Batch size
changes memory use, not the recipe; device/batching can introduce floating-point
roundoff. The default score cache ID includes the checkpoint revision, number of
orders and seed. `--cache-id` can select an explicit name, but `--resume` rejects
any changed recipe under that name. The extraction config's `cache_id` and
`conditioning_mode` describe only the representation cache and are not used as
the score recipe identity.

Scores use the same immutable assay/variant table as representations. A subset
requires its own `--output-root`, consistently with extraction. Repeat `--assay`
or use `--all-assays`; `--continue-on-error` records failures with a nonzero exit.
No model inference runs for a matching validated completed score cache.

```text
data/features/pretrained/singles/<assay>/
  assay.json
  variants.csv
  scores/proteinmpnn/proteingym--v_48_020-sequence_log_likelihood-orders1-seed0-8907e6671bfb/
    mean_score.csv
    config.json
    provenance.json
    completion.json
```

`mean_score.csv` contains `variant_id,score`, exactly one finite value per shared
variant ID. Its `config.json` declares `mean_score` with `file`, `id_column`, and
`score_column`, as well as the conditioning, score direction, normalization,
checkpoint, structure, seed and decoding-order settings. Completion hashes protect
the score file and metadata. Score generation never rewrites representation
configs or changes the default feature selections.

Set experiment `score_sources` explicitly to the generated score source:

```json
{
  "model": {"enabled_branches": ["sequence_global", "inverse_folding"]},
  "score_sources": [
    "scores/proteinmpnn/proteingym--v_48_020-sequence_log_likelihood-orders1-seed0-8907e6671bfb"
  ]
}
```

The GP loader validates dedicated score caches, joins scores by variant ID and
standardizes each score using training variants only. Legacy `zero_shot`
descriptors remain readable; all new score artifacts use `mean_score.csv`.

Official scoring sources: [ProteinMPNN sequence scoring](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/protein_mpnn_run.py),
[ProteinMPNN NLL reduction](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/protein_mpnn_utils.py).

## Preparing the full substitutions catalog on Slurm

Both preparation scripts support every assay in their configured
`DMS_substitutions.csv`, in catalog order (217 assays in the current local catalog).
ProteinMPNN defaults to the six failed-assay retry tasks; the explicit array
below overrides that default to process the full catalog. Neither script applies
a single-mutant-count threshold.
All shipped preparation presets use the catalog at
`data/DMS_ProteinGym_substitutions/DMS_substitutions.csv`. Each entry must have its
corresponding variant CSV beside the catalog or in its `raw_substitutions/`
subdirectory. Custom configs may set `assay_index` to another catalog; MSA
Transformer also has `msa_reference_csv` for validating generated-pair metadata
when the generation catalog snapshot is absent. Both scripts still filter variants to single
mutants and save under `data/features/pretrained/singles/<assay>/`; this does not
add indel assays or multi-mutant variants.

Use one assay per array task. From the repository root:

```bash
count=$(python sbatch_run/run_prepare_rep/list_assays.py \
  --config configs/extraction/esm2_650m_centered.json --count)
esm_job=$(sbatch --parsable --array="0-$((count - 1))%4" \
  sbatch_run/run_prepare_rep/prepare_esm2.sbatch)
count=$(python sbatch_run/run_prepare_rep/list_assays.py \
  --config configs/extraction/proteinmpnn_v48_020_kermut.json --count)
sbatch --array="0-$((count - 1))%4" \
  --dependency="afterany:${esm_job%%;*}" \
  sbatch_run/run_prepare_rep/prepare_proteinmpnn.sbatch
```

`%4` limits each array to four simultaneous tasks. The dependency waits for ESM2
before starting ProteinMPNN to avoid simultaneous writes to shared assay metadata.
Array indices now refer to the full catalog, **not the previous 57-assay subset**.
Use `list_assays.py --config <extraction-config> --task-index N` to inspect an
index; omit `--task-index` to list the entire catalog. When overriding the ESM2
config, use that same config to compute its array size.

Without `--array`, ESM2 processes the whole catalog sequentially in one GPU
allocation. ProteinMPNN uses its built-in six-assay retry array unless overridden.
The current time limits are four hours for ESM2 and one hour per ProteinMPNN
task; increase time or memory with `sbatch` overrides when needed.

Reruns reuse verified checkpoint files and registered structures. Both extraction
and scoring use `--resume`: a matching, complete, validated cache is skipped
before model loading/inference. Existing files alone do not qualify: corrupt,
incomplete published caches or incompatible recipes are reported as errors;
missing outputs are computed. Slurm jobs still start and perform validation.
Both scripts use `--continue-on-error`, retaining completed outputs while reporting
assay failures and returning a nonzero status. Shared setup failures stop the job.
ProteinMPNN still needs compatible structures. BRCA2 uses fragment features;
ZIKV and HCV use domain features. These three assays explicitly skip full-sequence
ProteinMPNN scoring (use ESM2 scores).

After preparation finishes, submit the CPU-only readiness check:

```bash
sbatch sbatch_run/run_prepare_rep/validate_prepared.sbatch
```

This validates the assets selected by `prepare_esm2.sbatch` and
`prepare_proteinmpnn.sbatch`, without an NKN experiment config: ESM2 embeddings
and masked-marginal scores, ProteinMPNN sequence-conditioned features and
seed-0 single-order scores, model/checkpoint files, and registered structures.
It verifies cache completion markers, checksums, saved recipes, and complete
single-mutant coverage against the current catalog and assay CSVs. BRCA2, ZIKV,
and HCV ProteinMPNN scores are explicitly skipped, matching preparation.
Use `--esm2-config PATH` (or `ESM2_CONFIG`) if ESM2 was prepared with a different
extraction preset; the CLI option takes precedence. ProteinMPNN uses the same
two fixed extraction configs as its preparation job. This job does not perform
inference, training, downloads, or modify feature caches.
It requests four CPUs, 64 GiB RAM, and four hours; resource limits can be
overridden with `sbatch` options.

All validation details, errors, catalog array indices, READY/FAILED statuses,
and final totals print directly to `validate_prepared-<job-id>.out` in the
submission directory. A single CSV summary is also saved to
`results/preparation-check-<job-id>-<timestamp>.csv`; its full path prints in the
log. Use `--summary-csv PATH` to choose a new destination (existing files are
never overwritten). Each row contains the catalog array index, assay ID,
per-asset READY/FAILED/SKIP statuses, output readiness, overall readiness,
errors, and notes. Shared model statuses are repeated for each assay; `status`
is FAILED if a shared asset fails even when `outputs_status` is READY. Intentional
score omissions are SKIP, not failures. Rows are flushed after each
assay, so an interrupted job retains a partial summary; absent rows have not
been checked yet. No per-assay report directories are created. Failed assays
do not stop subsequent checks. Any assay failure makes the job exit nonzero;
a catalog/setup error stops before the final summary is printed.
This checks preparation assets, not kernel branches or CV fold files. Wait for
both preparation jobs to finish before running it, or use a Slurm `afterany`
dependency on their job IDs.

The default ESM2 compact preset uses at most 1,022 residues in each model input
and supports mutation sites anywhere in the full reference. Its output estimate
is `(number_of_variants + mutation_occurrences) * 1280 * 4` bytes plus four int64
arrays per mutation occurrence. It has a 32 GiB output guard; the batch job
requests 16 GiB host RAM. Peak process memory includes input tables, model
weights, and validation. Completed assays can resume; partial assays restart.
The default preset also scores long references with centered WT windows, so
both ESM2 stages cover mutation positions beyond residue 1,022. ProteinMPNN's
default presets also have no reference-length cap, but still require a valid
full-reference structure. Its job requests 16 GiB host RAM and one hour, and
scores one variant per batch; long references may need more memory or time.
ESM-IF1's default preset also has no sequence-length cap; long references may
require more memory or time than its preparation job requests.

To prepare optional legacy pooled prefix embeddings, pass the pooled preset:

```bash
sbatch sbatch_run/run_prepare_rep/prepare_esm2.sbatch --config configs/extraction/esm2_650m.json
```

`ESM2_CONFIG` can also select the extraction config; `--config` takes precedence.
Using this optional recipe with Kermut also requires a copied experiment config
selecting the `-mean-first1022` embedding cache and full-reference ESM2 score
cache. The default Kermut config selects the centered preparation outputs.
Use the same extraction config when listing array indices. The
pooled preset averages the first 1,022 residues (the full sequence when shorter)
and retains its 4 GiB output limit. It can extract embeddings for longer
references, but its scoring stage still fails the full-reference length check.
These pooled caches cannot supply NKN's `sequence_site` branch.

## ProSST: sequence embeddings conditioned on a reference structure

The `prosst_2048.json` preset uses [AI4Protein/ProSST-2048](https://huggingface.co/AI4Protein/ProSST-2048)
at revision `e94ffee7846d7f55c1bf5efa8ec7372a336ac4b8`. It saves the final
layer (12) as float32 global `embeddings[N, 768]` and packed mutation-site
vectors `[number_of_mutations, 768]`, excluding special tokens and padding.
Each unmasked variant sequence is paired with the **same reference backbone**.
These are joint sequence/structure representations for substitution variants;
this recipe does not calculate candidate probabilities or mean scores.

The official GVP encoder and 2048-cluster codebook first convert the reference
structure to one discrete token per residue. This pipeline uses the pinned
[official graph, subgraph and encoder implementations](https://github.com/ai4protein/ProSST),
with serial graph construction and deterministic CPU quantization. Complete
N/CA/C/O coordinates and an exact full-reference sequence match are required.
Raw token IDs are 0..2047; the adapter adds 3 and aligns the structure and amino-acid
special tokens before inference. The quantizer removes amino-acid node features,
as in the official preprocessing. Variants reuse the tokens without predicting
variant-specific structures.

Update dependencies using `conda env update -f env.yaml` or
`python -m pip install -r requirements.txt`. ProSST adds `joblib==1.3.2`,
`scikit-learn==1.2.2` (matching the serialized codebook), and the PyTorch 2.5/CUDA 12.1
`torch-scatter` wheel. The dependency files include the official PyG wheel index.
The CUDA runtime also supports `--device cpu`. If replacing PyTorch with a CPU-only
build, select the corresponding CPU `torch-scatter` wheel and index too.

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path
$assay = "A0A1I9GEU1_NEIME_Kennouche_2019"
python -m feature_generation.cli download-structure --assay $assay
python -m feature_generation.cli download --config configs/extraction/prosst_2048.json
python -m feature_generation.cli extract --config configs/extraction/prosst_2048.json --assay $assay --resume
python -m feature_generation.cli validate --features "data/features/pretrained/$assay" --model-family prosst
```

For a small CLI smoke test, append `--limit 4 --output-root data/features/prosst_smoke`
to extraction and validate that output root. The expected pilot shape is
`[4, 768]` for globals, plus one site vector per annotated mutation.
Use a separate output root when changing the variant subset.

```text
data/models/prosst/ProSST-2048/<revision>/        # weights, codebook, verified source
data/structures/proteingym/tokens/prosst/<recipe>/<identity>/  # reusable reference tokens
data/features/pretrained/<assay>/sequence/prosst/<recipe>/
  features.pt
  config.json
  provenance.json
  completion.json
```

The download command verifies the pinned HF model, custom model code, quantizer
encoder, codebook, and graph code. Extraction loads only these local assets.
Token caches record the structure/mapping, preprocessing, runtime, artifact hashes,
and token checksums. Feature caches record those identities alongside the LM
checkpoint, layer, and sequence coverage; `--resume` rejects mismatches or corruption.
The quantizer cache directory can be overridden with `--structure-token-dir`.

ESM-2 and ProSST can coexist within one assay when their output root and variant
selection match. The first sequence recipe remains
the default. To use ProSST in a GP experiment, select
`feature_selection["sequence"] = "prosst/proteingym--ProSST-2048-layer12-fixed-backbone-e94ffee7846d-mean-all-nonoverlap2046-centered-sites"`.
The loader supplies both global and mutation-site sequence branches from this compact cache. A full GP run still
requires the other configured modalities and scores. This recipe supplies the
variant sequence feature branch; it does not supply a separate WT structure cache.

Both ProSST presets use `max_length=2046` residues per model window:
2046 residues plus BOS/EOS gives 2048 total tokens per window. The singles
preparation preset uses `max_tokens_per_batch=8192`, allowing up to four
maximum-length windows per batch; the separate `prosst_2048.json` preset retains
`max_tokens_per_batch=2048`. The [paper, Appendix A.2](https://proceedings.neurips.cc/paper_files/paper/2024/file/3ed57b293db0aab7cc30c44f45262348-Paper-Conference.pdf)
reports removing proteins longer than 2048 residues during pre-training. This
recipe uses a 2048-token per-window cap including special tokens, within that training range. The model name's `2048`
separately denotes the structure codebook size.

Global embeddings use non-overlapping windows covering every residue and a
length-weighted mean, so a shorter final window gets its correct weight.
Mutation-site embeddings use the same centered, clamped windows as ESM2,
with up to 2046 residues of the actual unmasked variant. Short proteins use
one full-reference window. The full backbone is quantized once; both amino-acid
and structure-token streams then use identical slices. This preserves the
quantizer's full-structure neighborhoods, but attention cannot cross window
boundaries. Long-protein outputs are therefore an approximation to full-context
inference. Inference and output remain float32; peak GPU memory still depends
on window length, and the token budget is not a GPU-memory guarantee.

New embedding cache IDs end in `-mean-all-nonoverlap2046-centered-sites`; centered
score IDs end in `-centered2046`. Existing full-residue and full-reference score
caches remain readable and are separate recipes. To retain the old layout in a
custom config, use `pooling="none"`, `global_window="prefix"`,
`truncation="reject"`, `site_embeddings="none"`, and `scoring_window="full"`.

### ProSST Slurm preparation and GP mean scores

[`prepare_prosst.sbatch`](../../sbatch_run/run_prepare_rep/prepare_prosst.sbatch)
prepares single-substitution embeddings and GP mean scores for every assay in the
configured `DMS_substitutions.csv`, in catalog order. It uses ProSST's K=2048
structure vocabulary. Without an array it runs sequentially; `--array=0-(N-1)`
selects one assay per task. Obtain N with `list_assays.py --config <preset> --count`.

```bash
sbatch sbatch_run/run_prepare_rep/prepare_prosst.sbatch
```

The job uses the `nkn_protein` Conda environment and a V100/16 GiB allocation.
The [`prosst_2048_singles.json`](../../configs/extraction/prosst_2048_singles.json)
preset writes to `data/features/pretrained/singles`. It uses a 2,048-token
batching budget, 2,046-residue windows, and a 4 GiB compact-output guard. Failed
assays are reported and later assays are attempted; any failure makes the job
exit nonzero. The job verifies both embedding and score caches. Shared downloads
and duplicate assay preparations use filesystem locks.

The default singles preset selects `structure_source="protssn_proteingym"`:
it reads `data/structures/protssn_proteingym/manifests/<assay>.json` and stores
tokens under `data/structures/protssn_proteingym/tokens/prosst/`. The launcher
automatically creates missing manifests from the collection's local raw PDBs,
using the checksums and filename aliases in `source.json` before validating the
reference mapping. Missing raw files or invalid structures fail the assay.
Embedding and score cache IDs start with `protssn_proteingym--`.
The separate `prosst_2048.json` CLI preset retains the official `proteingym` source.
Structures must cover the full reference with finite N/CA/C/O coordinates.
Use `--config PATH` or `PROSST_CONFIG=PATH` to override paths, layer, batching, or
resource guards; `--config` takes precedence. Relative paths use the repository
root. Model assets and structure tokens are cached separately from embeddings.

ProSST scoring uses the same masked-marginal formula as ESM2: for each mutated
position, mask that WT amino acid and retain the aligned, unmasked structure tokens,
and subtract the WT log probability from the mutant log probability. Sum these
contributions for multi-substitution variants; WT scores are zero. This is a
model log-odds score, not a calibrated fitness prediction. Scoring uses centered
WT windows for long references, full context for short references, float32
log-softmax over the model vocabulary, and no DMS labels. Each window slices
the cached full-reference structure tokens at the same bounds as the sequence.

After preparing structures, scores can also be generated independently:

```bash
python -m feature_generation.cli score \
  --config configs/extraction/prosst_2048_singles.json \
  --all-assays --singles-only --device cuda --resume --continue-on-error
```

Each assay gets a separate score leaf:

```text
data/features/pretrained/singles/<assay>/scores/prosst/
  protssn_proteingym--ProSST-2048-masked_marginals-fixed-backbone-float32-e94ffee7846d-centered2046/
    mean_score.csv
    config.json
    provenance.json
    completion.json
```

`mean_score.csv` contains `variant_id,score`. To use it for the GP mean, set
`score_sources` to
`["scores/prosst/protssn_proteingym--ProSST-2048-masked_marginals-fixed-backbone-float32-e94ffee7846d-centered2046"]`.
Embedding selection remains independent. The score recipe does not depend on the
embedding layer; resume verifies checkpoint, variant coverage, structure and
structure-token identities, and file checksums.

## Notebook extraction and inspection

The CLI remains the default workflow. These notebooks call the same extraction
and validation functions, retaining their existing full-assay feature caches:

| Notebook | Expected pilot output | Conditioning |
| --- | --- | --- |
| [ESM-2](../../notebooks/esm2_smoke_test.ipynb) | `[922, 161, 320]` embeddings | Sequence |
| [ProSST](../../notebooks/prosst_smoke_test.ipynb) | `[922, 161, 768]` embeddings | Sequence and fixed reference backbone |
| [ProteinMPNN](../../notebooks/proteinmpnn_smoke_test.ipynb) | `[161, 20]` probabilities | Backbone only |
| [ESM-IF1](../../notebooks/esm_if1_smoke_test.ipynb) | `[161, 20]` probabilities | Backbone and WT prefix |

Select a kernel with the current `env.yaml` or `requirements.txt` dependencies
and run the cells in order. `ipykernel` is included; each notebook has setup
instructions and configures imports from the project location automatically.

All notebooks save under `data/features/pretrained/<ASSAY_ID>/`, separate from
the new ESM2/ProteinMPNN singles CLI outputs. The assay ID includes the protein identifier
and experiment so distinct variant tables remain separate. Model outputs coexist
inside that directory:

```text
data/features/pretrained/<ASSAY_ID>/
  assay.json
  variants.csv
  sequence/esm2/<recipe>/
  sequence/prosst/<recipe>/
  inverse_folding/proteinmpnn/<recipe>/
  inverse_folding/esm_if1/<recipe>/
```

The notebooks use `LIMIT = None` for the complete assay (922 variants in the
default pilot). Rerunning configuration or extraction keeps the same destination;
`resume=True` verifies and reuses matching results. Keep full variant coverage
when using this shared location: an initialized assay's coverage is immutable.
The inverse-folding notebooks reuse verified local structures/checkpoints,
inspect per-position candidates, validate distributions, and test cache reuse.
Each inverse-folding model computes one full-reference `[L, 20]` array.
Restart the kernel before ESM-IF1 if it has already imported a different `esm`
package. Notebooks contain no separate model implementation.

## Download and extract

```powershell
python -m feature_generation.cli download --config configs/extraction/esm2_8m.json
python -m feature_generation.cli extract --config configs/extraction/esm2_8m.json --assay A0A1I9GEU1_NEIME_Kennouche_2019 --singles-only
python -m feature_generation.cli validate --features data/features/pretrained/singles/A0A1I9GEU1_NEIME_Kennouche_2019
```

Use `python -m feature_generation.cli` for all extraction commands. For the
larger checkpoint, choose `esm2_650m_centered.json` for both the NKN and Kermut
presets. Kermut uses only the global vectors. `esm2_650m.json` retains the optional
legacy prefix recipe. Download is a separate
explicit step; extraction uses local files only and works with
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

The presets are:

| Configuration | Model | Revision | Layer | Dimensions |
| --- | --- | --- | --- | --- |
| `esm2_8m.json` | `facebook/esm2_t6_8M_UR50D` | `c731040fcd8d73dceaa04b0a8e6329b345b0f5df` | 6 | 320 |
| `esm2_650m.json` | `facebook/esm2_t33_650M_UR50D` | `08e4846e537177426273712802403f7ba8261b6c` | 33 | 1280 |
| `esm2_650m_residue.json` | `facebook/esm2_t33_650M_UR50D` | `08e4846e537177426273712802403f7ba8261b6c` | 33 | 1280 |
| `esm2_650m_centered.json` | `facebook/esm2_t33_650M_UR50D` | `08e4846e537177426273712802403f7ba8261b6c` | 33 | 1280 |

The `esm2_8m.json` and `esm2_650m.json` presets use `pooling="mean"` and
`truncation="prefix"`: tokenize only the first `min(sequence_length, 1022)`
residues and average their embeddings,
excluding special tokens and padding. This retains the legacy Kermut prefix-mean
preprocessing. Mutations after residue 1,022 do not affect these ESM2 vectors.
Cache IDs end in `-mean-first1022`; rerun preparation to create these caches.
Using these caches requires explicitly selecting their pooled IDs in the
experiment config; the default Kermut preset selects the compact cache below.

The optional `esm2_650m_residue.json` preset
uses `pooling="none"` and `truncation="reject"`. It saves `[N,L,1280]` embeddings
for the full sequence and rejects references longer than 1,022 residues.
Its cache ID is `esm2_t33_650M_UR50D-layer33-08e4846e5371`, so compatible existing
residue caches can be reused with that selection. The loader takes mutation
vectors and averages all residues when this legacy recipe is selected.

The NKN and Kermut presets select `esm2_650m_centered.json`, with
`site_embeddings="centered"`. For a one-based mutation
position `p`, reference length `L`, and window length `W=min(L,1022)`, it uses
zero-based bounds `start=max(0,min(p-1-W//2,L-W))`, `end=start+W`.
It processes the actual variant sequence without masking and retains the vector
at local index `p-1-start`. Other mutations inside the window provide context;
mutations outside it cannot influence that site's vector. Identical window
bounds within a variant reuse one model evaluation. Short variants use their
full sequence for both sites and the mean. Kermut reads only the global mean.

For the global vector, this preset uses `pooling="mean"`, `truncation="none"`,
and `global_window="non_overlapping"`. Split the full variant at fixed offsets
`0,1022,2044,...`, letting the final window be shorter. Combine window means
using each window's residue count as its weight; exclude BOS, EOS, and padding.
For `L=3000`, the mean is `(1022*g1 + 1022*g2 + 956*g3)/3000`. Every residue
contributes exactly once, and the same boundaries apply to WT and all variants.
Mutations after residue 1,022 can therefore affect the global vector. These are
window-context embeddings, so they do not reproduce full-sequence attention
across window boundaries. Centered site windows contribute to the global vector
only when their bounds exactly match a global window; otherwise they supply
site vectors alone. Model evaluations with matching bounds are reused.

The cache ID ends in `-mean-all-nonoverlap1022-centered-sites`. The compact leaf stores
float32 `embeddings[N,D]` (global means), `site_embeddings[M,D]` (one row per
mutation occurrence), and int64 `site_variant_indices`, `site_positions`,
`site_window_starts`, `site_window_ends`. Site positions are one-based; window
bounds are zero-based and half-open. WT variants have zero site rows but still
have a global vector. Full reference positions, sequences, hashes, and variant
IDs remain intact. Validation checks exact mutation coverage and window bounds,
and finite-value checks use bounded chunks. Loading uses memory mapping for the
compact file. Extraction accumulates the compact tensors in CPU RAM and commits
one atomic `features.pt` per assay; it does not persist intermediate batches.
Global-only pooled, full-residue, and compact caches use separate recipe IDs.
The previous `-mean-first1022-centered-sites` caches remain readable with their
prefix recipe, but cannot be resumed as full-sequence globals. Prepare the new
embedding cache before running the NKN or Kermut preset. Masked-marginal score
metadata and cache IDs do not change, so completed centered score caches are
reused. Full-window pooling adds model passes for long proteins, without
increasing the number of saved global or site vectors.

Weights, tokenizer files and a checksum manifest live under
`data/models/esm2/<model>/<revision>/`. Only safetensors weights are downloaded.
The preset weight hashes are checked against their published values; subsequent
loads also verify all recorded model/tokenizer hashes. Downloads reuse verified
checkpoints. A complete manually staged preset can be verified and registered by
rerunning `download`.

If Python's Hub client cannot connect but Windows HTTPS works, use the included
transport helper. It fetches the same files from the exact configured revision,
then runs the Python verifier:

```powershell
python scripts/download_esm2_windows.py --config configs/extraction/esm2_8m.json
```

The helper writes each new download to a temporary filename before renaming it.
It preserves existing files; if verification reports a corrupt file, replace
that specific file from the pinned revision and rerun verification.

## Assay selection and resumption

```powershell
# A small real-model development run, separate from the complete assay cache.
python -m feature_generation.cli extract --config configs/extraction/esm2_8m.json --assay A0A1I9GEU1_NEIME_Kennouche_2019 --singles-only --limit 4 --output-root data/features/esm2_smoke

# Reuse only fully verified caches matching the inputs and extraction settings.
python -m feature_generation.cli extract --config configs/extraction/esm2_8m.json --assay A0A1I9GEU1_NEIME_Kennouche_2019 --singles-only --resume

# Repeat --assay for selected assays, or process the whole catalog explicitly.
python -m feature_generation.cli extract --config configs/extraction/esm2_8m.json --all-assays --singles-only --resume --continue-on-error
```

`--limit` selects the first N eligible source rows without using labels.
`--singles-only` selects only single substitutions; otherwise both single and
multiple substitutions are accepted. Variant IDs, source-row indices, complete
sequences and targets are preserved. Reordering and batching never change the
saved row order. Once an assay cache is initialized, its variant coverage and
reference are immutable. Use a separate output root to change a subset.

Without `--resume`, an existing leaf is an error. With `--resume`, a completed
leaf is reused only after checking hashes, tensors, input identity and extraction
configuration. The model loads only if at least one assay requires inference.
Interrupted extraction restarts that assay; earlier completed assays are reused.
Incomplete staging directories are ignored and retained for inspection.

Progress goes to stderr and the JSON result report goes to stdout. Without
`--continue-on-error` the first failed assay stops execution. With that option,
other assays continue and the process still returns a nonzero exit status when
any assay failed; the JSON report identifies every completed, skipped or failed
assay.

## Configuration and memory

Paths in configuration JSON are resolved relative to the project root. CLI path
overrides are resolved relative to the working directory. Available overrides
include `--assay-index`, `--output-root`, `--device cpu|cuda|auto`, and
`--max-tokens-per-batch`.

`max_tokens_per_batch` includes the two special tokens per sequence. Increase
it when a single sequence exceeds the budget, or reduce it to fit GPU memory.
With `pooling="mean", truncation="prefix", site_embeddings="none"`, ESM2 processes
the first `max_length` residues (at most 1022) and budgets tokens for that prefix.
With `site_embeddings="centered"`, it also processes the windows needed for
mutation-site vectors, using the same per-window token budget.
With `global_window="non_overlapping"`, `pooling="mean"`, and `truncation="none"`,
all residues contribute through fixed windows no larger than `max_length`.
The final shorter window receives proportionally less weight. This mode also
supports global-only caches with `site_embeddings="none"`.
The residue preset and legacy configs omitting these settings use
`pooling="none", truncation="reject"`: they save per-residue embeddings and
reject sequences longer than `max_length`.
Prefix truncation requires mean pooling. Masked-marginal scoring independently
uses `scoring_window="full"` (the default) or `"centered"`; the centered preset
selects the latter so `max_length` limits each scoring window, not the reference.

Saved embeddings always use float32. ESM-2 inference defaults to float32 and supports
explicit float16/bfloat16 configurations on suitable devices. Select a new
`cache_id` when changing numerical precision, checkpoint, layer, or preprocessing.
Configuration mismatches under an existing cache ID are rejected. `device=auto`
uses CUDA when available, otherwise CPU; a CUDA failure does not silently switch
the execution device. The 650M preset has not been benchmarked on a 4-GB GPU;
adjust batch size/precision or select CPU as needed.

The runner holds one assay's output tensors in CPU memory. Estimated tensor
storage is `N * D * 4` bytes for pooled embeddings, `N * L * D * 4` for full
residue embeddings, or `(N + M) * D * 4 + M * 32` for compact caches with `M`
mutation occurrences. `max_output_gib` defaults to 4; the centered preset sets
it to 32. This guards output tensor size, not total process memory. Model
weights, batches, and validation require additional memory. There is no sharded
output format in this version.

## Saved contract and GP integration

```text
data/features/pretrained/singles/<assay>/
  assay.json
  variants.csv
  sequence/esm2/<checkpoint_recipe>/
    features.pt
    config.json
    provenance.json
    completion.json
```

`features.pt` contains `schema_version=2`, `synthetic=false`, `assay_id`,
`reference_sequence`, one-based int64 `positions`, ordered `variant_ids`,
`sequence_hashes`, and float32 `embeddings[N,D]` for the pooled presets. The
reference, positions, variant sequences and hashes retain full assay identity;
they are not shortened to the prefix. Payload fields `pooling`, `truncation`,
`max_length` and `represented_length` record the embedding coverage. Prefix
recipes use `represented_length=min(reference_length,max_length)`. Full-window
recipes use the complete reference length and additionally record
`global_window="non_overlapping"` and
`global_window_aggregation="length_weighted_mean"`. Beginning/end tokens and padding are excluded from
the average. The model layer is configurable; the default is the final
normalized layer. Unpooled caches use `embeddings[N,L,D]`.

`config.json` records stable model/preprocessing identity, pinned revision,
checkpoint hashes, layer, dtype and backend/library versions. `provenance.json`
records runtime, source hashes and resolved device. `completion.json` records
artifact hashes. A leaf is committed only after writing all artifacts; readers
never see a partially written completed leaf. The first completed sequence
recipe becomes the assay default. Select another recipe explicitly with
`validate --cache-id ...` or the GP configuration's `feature_selection`.
If an assay already defaults to a residue cache, creating the pooled cache does
not change that default; use the updated GP configs or select the new ID.

The existing synthetic fixture is preserved. The new sequence cache is
compatible with the sequence portion of `data_loading.load_feature_bundle`;
standalone extraction validation works before other modalities exist. GP runs
require the modalities selected by `model.enabled_branches` (all four branches by
default) and at least one score source. ESM2 masked-marginal scores and ProteinMPNN
sequence likelihoods are generated with `score` and selected through `score_sources`.
Structure embeddings are not generated by the current extraction commands.
Global-only pooled caches support `sequence_global`; `sequence_site` requires
either a full-residue cache or the explicit compact centered-site recipe.

## Validation

```powershell
python -m unittest discover -s src/tests -v
```

Offline tests exercise residue/special-token alignment, layer selection, frozen
inference, reordered batches, checkpoint/cache integrity, interrupted writes,
resume conflicts and input limits. They do not download models. Use the small
extraction command above for an actual pretrained-model integration run.

Upstream references: [Hugging Face ESM](https://huggingface.co/docs/transformers/model_doc/esm),
[8M checkpoint](https://huggingface.co/facebook/esm2_t6_8M_UR50D),
[650M checkpoint](https://huggingface.co/facebook/esm2_t33_650M_UR50D).
