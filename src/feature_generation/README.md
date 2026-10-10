# Frozen protein representations

The CLI downloads pinned ESM-2, ProSST, ProteinMPNN, and ESM-IF1 checkpoints and writes
representations into the project's schema-v2/layout-v1 cache. Models run
locally with frozen parameters in evaluation/inference mode. DMS labels are
copied into the assay table but are never supplied to the representation models.

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
the reference structure, shared by all assay variants; DMS labels are not model
inputs. The supported initial recipes have different conditioning:

| Config | Checkpoint | Conditioning |
| --- | --- | --- |
| `proteinmpnn_v48_020.json` | vanilla `v_48_020` | `backbone_only`: structure without sequence context |
| `esm_if1_142m.json` | `esm_if1_gvp4_t16_142M_UR50` | `reference_prefix`: structure and preceding wild-type residues |

ESM-IF1 output is an autoregressive reference-context distribution, not an exact
backbone-only marginal or a conditional on the complete remaining WT sequence.
The extraction recipe does not implement Kermut's averaged WT-prefix distributions
or sequence generation. ProteinMPNN variant sequence likelihoods are prepared
separately with the `score` command below.

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
The selected chain must match the entire reference sequence with N/CA/C atoms;
ProteinMPNN additionally requires O. Partial coverage, sequence mismatches,
noncanonical residues, and ambiguous backbone alternate conformations are rejected.
Other chains are not used as structural context in this first version.
Choose `--chain` explicitly for multichain structures and `--model-index` for
multi-model structures. No sequence offset or cropping is inferred from `pdb_range`.

Download both checkpoints explicitly, then extract offline:

```powershell
python -m feature_generation.cli download --config src/configs/extraction/proteinmpnn_v48_020.json
python -m feature_generation.cli download --config src/configs/extraction/esm_if1_142m.json
python -m feature_generation.cli extract --config src/configs/extraction/proteinmpnn_v48_020.json --assay $assay --resume
python -m feature_generation.cli extract --config src/configs/extraction/esm_if1_142m.json --assay $assay --resume
python -m feature_generation.cli validate --features "data/features/pretrained/$assay" --feature-type inverse_folding --model-family proteinmpnn
python -m feature_generation.cli validate --features "data/features/pretrained/$assay" --feature-type inverse_folding --model-family esm_if1
```

The original ESM-IF1 checkpoint download is approximately 1.7 GB. Model/source files
are verified against the pinned artifact manifest before loading. Their provenance
includes exact source URLs, upstream commits, and file SHA-256 checksums.

Use `--output-root data/features/inverse_smoke --limit 4` on **both** extraction
commands for an isolated smoke test. `--limit` changes only the shared variant
table; inverse-folding inference still covers the complete reference. A full-assay
cache and a subset cannot occupy the same assay directory. `--resume` checks the
checkpoint, structure, mapping, extraction settings, and all saved artifact hashes.

Each model saves its own leaf:

```text
data/features/pretrained/<assay>/inverse_folding/<family>/<recipe>/
  features.pt
  config.json
  provenance.json
  completion.json
```

`features.pt` stores `probabilities[L,20]`, `log_probs[L,20]`, one-based `positions`,
reference identity, and alphabet `ACDEFGHIKLMNPQRSTVWY`. Vocabulary entries are
selected by amino-acid identity and normalized over these 20 candidates at T=1.
Both tensors are finite float32, probability rows sum to one, and log probabilities
agree with their probabilities. Special tokens and ProteinMPNN's X are excluded.

Both model leaves can coexist; the first completed leaf becomes the inverse-folding
default. Select either using `feature_selection["inverse_folding"] = "<family>/<recipe>"`
in the experiment config. GP experiments load only the feature types required by
`model.enabled_branches`; structure embeddings are needed only when `structure` is
enabled. At least one mean score source is still required for the GP mean;
standalone inverse cache validation does not require scores.

Upstream implementations: [ProteinMPNN](https://github.com/dauparas/ProteinMPNN),
[ESM-IF1](https://github.com/facebookresearch/esm/tree/main/examples/inverse_folding).

## ESM2 scores for the GP mean

ESM2 scoring uses the pretrained Hugging Face `EsmForMaskedLM` head and WT
masked marginals. For each position mutated by any selected variant, it masks
exactly that residue in the WT sequence and computes model log-probabilities.
The score for variant `v` is:

```text
score(v) = sum over mutated positions i:
    log p(mutant_i | WT with position i masked)
  - log p(WT_i     | WT with position i masked)
```

WT has score zero. Multiple substitutions add contributions from the same WT
context. Higher values indicate that the model favors the mutant residues over
WT. The score does not use experimental labels or a structure. Each unique
mutation position is evaluated once per assay and reused across variants. This
is the masked-marginal scoring recipe; the existing embedding cache alone cannot
supply these scores, so a separate inference stage uses the checkpoint's MLM head.

Both existing ESM2 checkpoint configs work (8M and 650M). Scoring always uses the
final pretrained MLM head; embedding `layer` and representation `cache_id` do not
select its scoring layer or output directory. The 8M preset is appropriate for
smoke tests; Kermut's published main ESM2 scores use the 650M checkpoint.

```bash
# Required only if weights/tokenizer are not downloaded yet; valid assets are reused.
python -m feature_generation.cli download --config src/configs/extraction/esm2_8m.json

python -m feature_generation.cli score --config src/configs/extraction/esm2_8m.json --assay A0A1I9GEU1_NEIME_Kennouche_2019 --device cuda --batch-size 8 --max-tokens-per-batch 2048 --resume

python -m feature_generation.cli validate --features data/features/pretrained/A0A1I9GEU1_NEIME_Kennouche_2019 --feature-type scores --model-family esm2 --cache-id esm2_t6_8M_UR50D-masked_marginals-float32-c731040fcd8d
```

Use `--device cpu` without a GPU. `--batch-size` caps the number of masked WT
copies per forward pass; `max_tokens_per_batch` additionally limits total tokens,
including BOS/EOS. Scoring never truncates. `--num-orders` and structure-manifest
options apply to ProteinMPNN and are rejected for ESM2. New dependencies are not
needed: the verified safetensors checkpoint already contains the MLM head.

`prepare_esm2.sbatch` downloads/verifies the checkpoint and then runs both embedding
extraction and scoring for the pilot assay. The Conda environment and ProteinGym
assay CSVs must already exist. First-time downloads need outbound HTTPS from the
compute node; verified completed caches are reused on resubmission.

```text
data/features/pretrained/<assay>/
  scores/esm2/esm2_t6_8M_UR50D-masked_marginals-float32-c731040fcd8d/
    mean_score.csv
    config.json
    provenance.json
    completion.json
```

The CSV has `variant_id,score` with exact coverage of the shared `variants.csv`.
The filename describes its role in the GP mean; the ESM2 value is a sum of mutation
log-odds. Its metadata records checkpoint hashes, masking recipe, precision and
score direction. The recipe ID includes checkpoint, inference precision and
revision. Representation caches and ProteinMPNN score caches are preserved.

Select just ESM2 scores for the mean function:

```json
{
  "score_sources": [
    "scores/esm2/esm2_t6_8M_UR50D-masked_marginals-float32-c731040fcd8d"
  ]
}
```

To use ESM2 and ProteinMPNN together, list both sources. They become separate
score columns, each standardized using training variants only; the GP learns a
coefficient for each. This choice is independent of `model.enabled_branches`.

`src/configs/esm2_mean_score_smoke.json` selects ESM2 representations and ESM2
scores. `src/configs/esm2_proteinmpnn_combined_mean_smoke.json` selects both kernel
modalities and both mean-score sources. Both configs run only five training steps
for integration testing:

```bash
python -m experiments.cli cv --features data/features/pretrained/A0A1I9GEU1_NEIME_Kennouche_2019 --config src/configs/esm2_mean_score_smoke.json --folds data/DMS_ProteinGym_substitutions/cv_folds_singles_substitutions/A0A1I9GEU1_NEIME_Kennouche_2019.csv --setting singles_random --output src/runs/esm2_mean_smoke
```

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
python -m feature_generation.cli score --config src/configs/extraction/proteinmpnn_v48_020.json --assay A0A1I9GEU1_NEIME_Kennouche_2019 --device cuda --batch-size 8 --resume
python -m feature_generation.cli validate --features data/features/pretrained/A0A1I9GEU1_NEIME_Kennouche_2019 --feature-type scores --model-family proteinmpnn --cache-id v_48_020-sequence_log_likelihood-orders1-seed0-8907e6671bfb
```

Use `--device cpu` when no GPU is available. This reuses the existing ProteinMPNN
checkpoint and structure manifest; no new dependencies or model downloads are
needed if extraction already ran. Scoring can also run before extraction.
`prepare_proteinmpnn.sbatch` downloads and verifies the pinned checkpoint/source
and chain-A ProteinGym reference structure, then runs extraction and scoring.
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
data/features/pretrained/<assay>/
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

`src/configs/esm2_proteinmpnn_mean_score_smoke.json` selects the existing ESM2-8M
and ProteinMPNN representations with these scores and five GP training steps.
This is an integration smoke test, not a converged performance experiment:

```bash
python -m experiments.cli cv --features data/features/pretrained/A0A1I9GEU1_NEIME_Kennouche_2019 --config src/configs/esm2_proteinmpnn_mean_score_smoke.json --folds data/DMS_ProteinGym_substitutions/cv_folds_singles_substitutions/A0A1I9GEU1_NEIME_Kennouche_2019.csv --setting singles_random --output src/runs/esm2_proteinmpnn_mean_smoke
```

The GP loader validates dedicated score caches, joins scores by variant ID and
standardizes each score using training variants only. Legacy `zero_shot`
descriptors remain readable; all new score artifacts use `mean_score.csv`.

Official scoring sources: [ProteinMPNN sequence scoring](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/protein_mpnn_run.py),
[ProteinMPNN NLL reduction](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/protein_mpnn_utils.py).

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
python -m feature_generation.cli download --config src/configs/extraction/prosst_2048.json
python -m feature_generation.cli extract --config src/configs/extraction/prosst_2048.json --assay $assay --resume
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

ESM-2 and ProSST can coexist within one assay. The first sequence recipe remains
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
and validation functions and share its full-assay feature caches:

| Notebook | Expected pilot output | Conditioning |
| --- | --- | --- |
| [ESM-2](../../notebooks/esm2_smoke_test.ipynb) | `[922, 161, 320]` embeddings | Sequence |
| [ProSST](../../notebooks/prosst_smoke_test.ipynb) | `[922, 161, 768]` embeddings | Sequence and fixed reference backbone |
| [ProteinMPNN](../../notebooks/proteinmpnn_smoke_test.ipynb) | `[161, 20]` probabilities | Backbone only |
| [ESM-IF1](../../notebooks/esm_if1_smoke_test.ipynb) | `[161, 20]` probabilities | Backbone and WT prefix |

Select a kernel with the current `env.yaml` or `requirements.txt` dependencies
and run the cells in order. `ipykernel` is included; each notebook has setup
instructions and configures imports from the project location automatically.

All notebooks save under `data/features/pretrained/<ASSAY_ID>/`, using the same
protein-assay directory as the CLI. The assay ID includes the protein identifier
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
python -m feature_generation.cli download --config src/configs/extraction/esm2_8m.json
python -m feature_generation.cli extract --config src/configs/extraction/esm2_8m.json --assay A0A1I9GEU1_NEIME_Kennouche_2019
python -m feature_generation.cli validate --features data/features/pretrained/A0A1I9GEU1_NEIME_Kennouche_2019
```

Use `python -m feature_generation.cli` for all extraction commands, and the
`esm2_650m.json` configuration for the larger checkpoint. Download is a separate
explicit step; extraction uses local files only and works with
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

The presets are:

| Configuration | Model | Revision | Layer | Dimensions |
| --- | --- | --- | --- | --- |
| `esm2_8m.json` | `facebook/esm2_t6_8M_UR50D` | `c731040fcd8d73dceaa04b0a8e6329b345b0f5df` | 6 | 320 |
| `esm2_650m.json` | `facebook/esm2_t33_650M_UR50D` | `08e4846e537177426273712802403f7ba8261b6c` | 33 | 1280 |

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
python scripts/download_esm2_windows.py --config src/configs/extraction/esm2_8m.json
```

The helper writes each new download to a temporary filename before renaming it.
It preserves existing files; if verification reports a corrupt file, replace
that specific file from the pinned revision and rerun verification.

## Assay selection and resumption

```powershell
# A small real-model development run, separate from the complete assay cache.
python -m feature_generation.cli extract --config src/configs/extraction/esm2_8m.json --assay A0A1I9GEU1_NEIME_Kennouche_2019 --limit 4 --output-root data/features/esm2_smoke

# Reuse only fully verified caches matching the inputs and extraction settings.
python -m feature_generation.cli extract --config src/configs/extraction/esm2_8m.json --assay A0A1I9GEU1_NEIME_Kennouche_2019 --resume

# Repeat --assay for selected assays, or process the whole catalog explicitly.
python -m feature_generation.cli extract --config src/configs/extraction/esm2_8m.json --all-assays --resume --continue-on-error
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
Batching never truncates sequences. ESM-2 rejects sequences longer
than `max_length` (at most 1022 residues); windowing is not implemented.

Saved embeddings always use float32. ESM-2 inference defaults to float32 and supports
explicit float16/bfloat16 configurations on suitable devices. Select a new
`cache_id` when changing numerical precision, checkpoint, layer, or preprocessing.
Configuration mismatches under an existing cache ID are rejected. `device=auto`
uses CUDA when available, otherwise CPU; a CUDA failure does not silently switch
the execution device. The 650M preset has not been benchmarked on a 4-GB GPU;
adjust batch size/precision or select CPU as needed.

The runner holds one assay tensor in CPU memory. Estimated tensor storage is
`N * L * D * 4` bytes. `max_output_gib` defaults to 4 and guards that tensor's
size, not total process memory. Model weights, batches, and validation require
additional memory. There is no sharded output format in this version.

## Saved contract and GP integration

```text
data/features/pretrained/<assay>/
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
`sequence_hashes`, and float32 `embeddings[N,L,D]`. Beginning/end tokens and
padding are excluded. The model layer is configurable; the default is the final
normalized layer. Pooling is left to the existing GP loader.

`config.json` records stable model/preprocessing identity, pinned revision,
checkpoint hashes, layer, dtype and backend/library versions. `provenance.json`
records runtime, source hashes and resolved device. `completion.json` records
artifact hashes. A leaf is committed only after writing all artifacts; readers
never see a partially written completed leaf. The first completed sequence
recipe becomes the assay default. Select another recipe explicitly with
`validate --cache-id ...` or the GP configuration's `feature_selection`.

The existing synthetic fixture is preserved. The new sequence cache is
compatible with the sequence portion of `data_loading.load_feature_bundle`;
standalone extraction validation works before other modalities exist. GP runs
require the modalities selected by `model.enabled_branches` (all four branches by
default) and at least one score source. ESM2 masked-marginal scores and ProteinMPNN
sequence likelihoods are generated with `score` and selected through `score_sources`.
Structure embeddings are not generated by the current extraction commands.

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
