# NKN GPU array for selected human assays

`run_nkn_gpu.sbatch` loads real prepared ProteinGym features, scores, and
published folds for each ID in its editable `assays=(...)` list. It supports
these four human assays:

| Assay ID | Default task |
| --- | --- |
| A4_HUMAN_Seuma_2022 | 0 |
| BRCA2_HUMAN_Erwood_2022_HEK293T | 1 |
| FKBP3_HUMAN_Tsuboyama_2023_2KFV | 2 |
| S22A1_HUMAN_Yee_2023_activity | Optional |

Edit the list directly in the sbatch file to select or reorder these IDs.
For example, to run just the 10,094-single-variant OCT1 assay:

~~~bash
assays=(
  "S22A1_HUMAN_Yee_2023_activity"
)
~~~

Also change the directive to `#SBATCH --array=0-0`, or submit with
`sbatch --array=0-0 sbatch_run/run_nkn/run_nkn_gpu.sbatch singles_random`.
For N entries, the array range is `0-(N-1)`. Slurm reads its directives
before the shell starts, so the range must be updated manually. Empty lists,
duplicate IDs, unsupported IDs, and out-of-range task indices fail before
starting the runner. This launcher supports the four IDs above; it does not
dispatch the earlier full 58-assay list.

Show the configured list and required range without starting training:

~~~bash
bash sbatch_run/run_nkn/run_nkn_gpu.sbatch --help
~~~

Each selected ID resolves the same prepared singles data paths used by the
CPU framework:

- Features and configured score caches: `data/features/pretrained/singles/<assay>/`.
- Published folds: `data/DMS_ProteinGym_substitutions/cv_folds_singles_substitutions/<assay>.csv`.
- Coverage reference: `data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv`.

The runner checks the prepared assay identity, validates complete single-variant
coverage, and snapshots the published folds before fitting. Missing assets or
invalid folds fail explicitly. `--features`, `--folds`, and `--assay-index`
can override their paths; the selected assay identity and coverage checks still
apply. Paths passed in an array job apply to every task, so overrides are usually
appropriate for a single-assay submission.

The configuration is resolved with `ExperimentConfig.for_assay`, preserving
existing assay-specific handling. For BRCA2, this disables the structure-distance
branch and excludes the ProteinMPNN mean score because its structures represent
separate reference fragments. Prepared ProteinMPNN features remain in use.

Submit the same five-fold random setting for each configured assay:

~~~bash
sbatch sbatch_run/run_nkn/run_nkn_gpu.sbatch singles_random
sbatch sbatch_run/run_nkn/run_nkn_gpu.sbatch singles_random --inference-backend bbmm
~~~

For a short check, append `--max-steps 2 --min-steps 0`. Omitting
`singles_random` runs random, modulo, and contiguous settings, sequentially
within each task. Folds also run sequentially. The array parallelizes assays;
no `--parallel` argument is needed. Limit concurrent GPU jobs with
`sbatch --array=0-2%1 ...` for the default three-assay list.

Each task requests one A100, 8 CPUs, 64 GB host memory, and two hours. The
existing account, QoS, conda environment, four-branch preset, initial noise 1.0,
float64 precision, and training limits are retained, subject to assay-specific
rules. The preset defaults to 1,000 minimum and 10,000 maximum optimizer steps.
Resource overrides go before the script path. `NKN_PYTHON` can select an existing
Python interpreter; `NKN_CONFIG` or `--config` can select a configuration.

Install dependencies in the job's Python environment:

~~~bash
python -m pip install -r requirements.txt
~~~

Cholesky is the default. BBMM uses `linear_operator` directly; the environment
pins `gpytorch==1.11` and `linear_operator==0.5.2`. BBMM controls include
`--bbmm-cg-tolerance`, `--bbmm-max-cg-iterations`,
`--bbmm-num-trace-samples`, `--bbmm-max-lanczos-iterations`,
`--bbmm-preconditioner-size`, and `--bbmm-probe-seed`.

BBMM approximates the training likelihood, gradients, and posterior solves.
It reuses a fixed base probe seed and uses the initial jitter throughout.
Numerical failures are explicit. The kernel is still dense, with quadratic
memory use. `--bbmm-num-trace-samples 0` is an expensive deterministic-trace
diagnostic for tiny inputs. After prepared-asset and fold validation, the
GPU runner starts training directly on the selected assay.

Logs: `nkn_gpu-<array-job>_<task>.out`.

Results:
`results/<assay>/nkn_esm2_proteinmpnn_<selection>_gpu_<job-id>_<task>/`.

Use `--experiment-suffix NAME` to name a results group, matching the CPU
launcher's argument. For example, appending `--experiment-suffix comparison_01`
to your submission arguments saves runs under
`results/comparison_01/<assay>/nkn_esm2_proteinmpnn_<selection>_gpu_<job-id>_<task>/`.
Both `--experiment-suffix NAME` and `--experiment-suffix=NAME` are accepted.
You can also put the group name first, before the optional setting:

~~~bash
sbatch sbatch_run/run_nkn/run_nkn_gpu.sbatch gpu_bbmm_test --inference-backend bbmm
sbatch sbatch_run/run_nkn/run_nkn_gpu.sbatch gpu_bbmm_test singles_random --inference-backend bbmm
~~~

The group name is consumed by the launcher instead of being passed to Python
as a selection. A group name alone does not enable BBMM; use
`--inference-backend bbmm` as shown. `--experiment-suffix` overrides a leading
group name, which overrides `EXPERIMENT_SUFFIX`; omitting all three preserves
the existing output path. The name must be a single folder, without `/` or `\`,
and cannot be `.` or `..`.

Set `NKN_OUTPUT_ROOT` to change the parent results directory. Combined with a
suffix, the path is `<NKN_OUTPUT_ROOT>/<suffix>/<assay>/<run>/`. The launcher
manages the assay ID, CUDA selection, and per-task output path. Existing output
directories are never overwritten. The chosen result path is printed in the log.

Each run writes `config.json`, `execution_plan.json`, `fold_manifest.json`,
the exact source fold CSV under `folds/`, progress logs, and normal CV
checkpoints, predictions, and metrics under `cv/<setting>/`.
`benchmark.json` records the selected assay, inference backend, elapsed CV time
(including evaluation/output), and peak CUDA memory.
