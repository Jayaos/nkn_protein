# Frozen protein representations

The CLI downloads pinned ESM-2, ProSST, ProteinMPNN, and ESM-IF1 checkpoints and writes
representations into the project's schema-v2/layout-v1 cache. Models run
locally with frozen parameters in evaluation/inference mode. DMS labels are
copied into the assay table but are never supplied to the representation models.

The ESM2 and ProteinMPNN presets write to
`data/features/pretrained/singles/<assay>/`. Both preparation jobs pass
`--singles-only` to extraction and scoring; pass it explicitly when using these
presets through the CLI. Each selected single mutant is prepared once and reused
across folds and protocols. The raw catalog remains the source of assay metadata
and variants. Existing full-assay caches are not moved or modified. To prepare
all variants through the CLI, override `--output-root data/features/pretrained`
and omit `--singles-only` consistently for extraction and scoring. ProSST,
ESM-IF1, and notebook defaults retain their existing full-assay locations.

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

The ProteinMPNN presets set `max_length: null`, so extraction and scoring have
no sequence-length cap. A positive integer can impose an optional operational
limit. ESM-IF1 retains its maximum of 1,022 residues. These inverse-folding
workflows never truncate model inputs. Explicit fragment manifests supply each
registered piece separately; `max_length` still limits the full assay reference.

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
Raw downloads live in `data/structures/raw/`, and validated manifests in
`data/structures/manifests/<assay>.json`. The manifest records the structure hash,
coordinate model, target chain, and reference-position-to-author-residue mapping.
For a single-structure manifest, the selected chain must match the entire reference sequence with N/CA/C atoms;
ProteinMPNN additionally requires O. Partial coverage, sequence mismatches,
noncanonical residues, and ambiguous backbone alternate conformations are rejected.
Other chains are not used as structural context in this first version.
Choose `--chain` explicitly for multichain structures and `--model-index` for
multi-model structures. No sequence offset or cropping is inferred from `pdb_range`.

Download both checkpoints explicitly, then extract offline:

```powershell
python -m feature_generation.cli download --config configs/extraction/proteinmpnn_v48_020_kermut.json
python -m feature_generation.cli download --config configs/extraction/esm_if1_142m.json
python -m feature_generation.cli extract --config configs/extraction/proteinmpnn_v48_020_kermut.json --assay $assay --singles-only --resume
python -m feature_generation.cli extract --config configs/extraction/esm_if1_142m.json --assay $assay --resume
python -m feature_generation.cli validate --features "data/features/pretrained/singles/$assay" --feature-type inverse_folding --model-family proteinmpnn --cache-id v_48_020-sequence_conditional-orders10-seed37-8907e6671bfb
python -m feature_generation.cli validate --features "data/features/pretrained/$assay" --feature-type inverse_folding --model-family esm_if1
```

The optional legacy backbone-only recipe uses the same downloaded checkpoint.
To prepare it for experiments that still select that cache, run it separately:

```powershell
python -m feature_generation.cli extract --config configs/extraction/proteinmpnn_v48_020.json --assay $assay --singles-only --resume
python -m feature_generation.cli validate --features "data/features/pretrained/singles/$assay" --feature-type inverse_folding --model-family proteinmpnn --cache-id v_48_020-backbone_only-8907e6671bfb
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
python -m feature_generation.cli validate --features data/features/pretrained/singles/BRCA2_HUMAN_Erwood_2022_HEK293T --feature-type inverse_folding --model-family proteinmpnn --cache-id v_48_020-sequence_conditional-orders10-seed37-8907e6671bfb
```

ProteinMPNN processes reference intervals 1â€“1000, 1001â€“2085, and 2086â€“2832 in
three independent calls, each using the configured seed and order count. Its
sequence context is local to each fragment. Output rows are concatenated in
reference-position order without changing probabilities or normalizing them
again. The cache retains the full 3418-residue reference and contains `[2832,20]`
probabilities plus their explicit positions. All 265 variants in the current
assay mutate covered sites. Positions 2833â€“3418 have no structure-derived
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
python -m feature_generation.cli prepare-structure --assay BRCA2_HUMAN_Erwood_2022_HEK293T --structure data/structures/raw/BRCA2_HUMAN_1-1000.pdb --reference-range 1-1000 --structure data/structures/raw/BRCA2_HUMAN_1001-2085.pdb --reference-range 1001-2085 --structure data/structures/raw/BRCA2_HUMAN_2086-2832.pdb --reference-range 2086-2832
```

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
For a 3,000-residue reference, positions 1,000 and 2,300 use residues 489â€“1,510
and 1,789â€“2,810 respectively. Other positions in each window stay wild type,
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
`raw_substitutions/DMS_substitutions.csv` catalog.
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
python -m feature_generation.cli validate --features data/features/pretrained/singles/A0A1I9GEU1_NEIME_Kennouche_2019 --feature-type scores --model-family proteinmpnn --cache-id v_48_020-sequence_log_likelihood-orders1-seed0-8907e6671bfb
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
  scores/proteinmpnn/v_48_020-sequence_log_likelihood-orders1-seed0-8907e6671bfb/
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
    "scores/proteinmpnn/v_48_020-sequence_log_likelihood-orders1-seed0-8907e6671bfb"
  ]
}
```

The GP loader validates dedicated score caches, joins scores by variant ID and
standardizes each score using training variants only. Legacy `zero_shot`
descriptors remain readable; all new score artifacts use `mean_score.csv`.

Official scoring sources: [ProteinMPNN sequence scoring](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/protein_mpnn_run.py),
[ProteinMPNN NLL reduction](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/protein_mpnn_utils.py).

## Preparing the full substitutions catalog on Slurm

Both preparation scripts select every assay in their configured
`DMS_substitutions.csv`, in catalog order (217 assays in the current local catalog).
There is no hard-coded assay list or single-mutant-count threshold.
`DMS_substitutions.csv` supplies metadata; each entry must have its corresponding
variant CSV beside the catalog. Both scripts still filter variants to single
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

Without `--array`, each script processes the whole catalog sequentially in one
GPU allocation. The current limits are four hours for ESM2 and one hour for
ProteinMPNN; increase time or memory with `sbatch` overrides when needed.

Reruns reuse verified checkpoint files and registered structures. Both extraction
and scoring use `--resume`: a matching, complete, validated cache is skipped
before model loading/inference. Existing files alone do not qualify: corrupt,
incomplete published caches or incompatible recipes are reported as errors;
missing outputs are computed. Slurm jobs still start and perform validation.
Both scripts use `--continue-on-error`, retaining completed outputs while reporting
assay failures and returning a nonzero status. Shared setup failures stop the job.
ProteinMPNN still needs compatible structures; BRCA2 uses fragment features and
explicitly skips full-sequence ProteinMPNN scoring (use ESM2 scores).

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
ESM-IF1 retains its 1,022-residue limit.

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
layer (12) as float32 `embeddings[N, L, 768]`, excluding special tokens and padding.
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
`[4, 161, 768]`. Use a separate output root when changing the variant subset.

```text
data/models/prosst/ProSST-2048/<revision>/        # weights, codebook, verified source
data/structures/tokens/prosst/<recipe>/<identity>/  # reusable reference tokens
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
`feature_selection["sequence"] = "prosst/ProSST-2048-layer12-fixed-backbone-e94ffee7846d"`.
The existing loader accepts the same `[N,L,D]` sequence contract. A full GP run still
requires the other configured modalities and scores. This recipe supplies the
variant sequence feature branch; it does not supply a separate WT structure cache.

The default `max_length=1022` is an operational guard, not a ProSST architecture
limit. Extraction never crops structures or sequences. Increase the batching budget
if one sequence plus its two special tokens exceeds `max_tokens_per_batch`.
This initial ProSST recipe uses float32 inference and output only.

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
