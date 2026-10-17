# NKN GPU timing run

`run_nkn_gpu.sbatch` targets **S22A1_HUMAN_Yee_2023_activity**,
the human OCT1 assay with **10,094 single mutants**. This job uses real
prepared assets and published folds. Cholesky remains the default inference
backend; BBMM is an optional backend selected with `--inference-backend bbmm`.

Submit one five-fold setting for timing:

~~~bash
sbatch sbatch_run/run_nkn/run_nkn_gpu.sbatch singles_random
~~~

Omit `singles_random` to run all three singles settings. For a short full-size
run with two optimizer updates per fold:

~~~bash
sbatch sbatch_run/run_nkn/run_nkn_gpu.sbatch singles_random --max-steps 2 --min-steps 0
~~~

For BBMM, install the project dependencies in the same Python environment
used by the job, then select the backend:

~~~bash
python -m pip install -r requirements.txt
sbatch sbatch_run/run_nkn/run_nkn_gpu.sbatch singles_random --inference-backend bbmm
~~~

Append `--max-steps 2 --min-steps 0` for an initial full-size BBMM timing run.
The same launcher, GPU resources, model, variants, and folds are used for
both backends. No additional Slurm script or parallel option is needed.
`env.yaml` and `requirements.txt` pin
`gpytorch==1.11` and `linear_operator==0.5.2`, matching the validation environment.
The current BBMM adapter calls `linear_operator` directly.

The job requests one A100 GPU, 8 CPUs, 64 GB of host memory, and two hours,
using the existing account, QoS, and conda environment. Resource options can
be overridden before the script path. `NKN_PYTHON` can select an existing
interpreter instead of activating conda. GPU memory is separate from host RAM;
the full assay's peak VRAM and runtime still need measurement on the cluster.

The model/training/evaluation preset matches `run_nkn_assays.sbatch`:
four branches, initial noise 1.0, float64, and five folds. Training limits are
1,000 minimum and 10,000 maximum steps unless explicitly overridden. All
settings and folds run sequentially on the allocated GPU. Variant subsampling
is rejected.

Inputs:

- Prepared assets: `data/features/pretrained/singles/S22A1_HUMAN_Yee_2023_activity/`
- Published folds: `data/DMS_ProteinGym_substitutions/cv_folds_singles_substitutions/S22A1_HUMAN_Yee_2023_activity.csv`
- The processed assay index under `data/DMS_ProteinGym_substitutions/raw_substitutions/`

BBMM replaces dense Cholesky inference with iterative conjugate-gradient
solves and stochastic Lanczos estimates of the log determinant. It preserves
the NKN kernel and float64 precision, but introduces numerical approximation:
gradients, stopping times, and fitted predictions can differ from Cholesky.
Compare held-out metrics and predictive uncertainty as well as runtime.
The NKN kernel is still constructed densely, so its quadratic memory cost
remains. Full-assay BBMM speed and peak VRAM have not been measured.

Optional BBMM controls are `--bbmm-cg-tolerance`, `--bbmm-max-cg-iterations`,
`--bbmm-num-trace-samples`, `--bbmm-max-lanczos-iterations`,
`--bbmm-preconditioner-size`, and `--bbmm-probe-seed`. Tighter tolerances,
more trace samples, or more iterations can improve accuracy at additional
cost. Use separate output directories when comparing backends.

BBMM reuses the configured base random probe seed for every objective
evaluation and preserves the caller's random-number state. This avoids fresh
probe-sampling noise between training steps; the fixed finite-sample
approximation can still differ from Cholesky. BBMM uses the configured
initial jitter throughout and fails explicitly if the solve is invalid;
Cholesky retains its existing adaptive jitter retries. Setting
`--bbmm-num-trace-samples 0` enables a deterministic full-trace diagnostic
with high memory and computation costs; use it only for tiny numerical checks.

The runner validates real-assay identity, complete single-variant coverage,
and every requested fold assignment before fitting. It snapshots the fold CSV
unchanged and records its SHA256. A small synthetic CPU/CUDA numerical parity
check precedes real training. BBMM additionally runs a tiny deterministic-trace
comparison with Cholesky; this checks the integration without quantifying
full-assay stochastic approximation error. Missing CUDA, invalid inputs, or a missing BBMM
dependency stop the job.
Existing output directories are never overwritten.

Logs: `nkn_s22a1_gpu-<array-job>_0.out`.
Results: `results/S22A1_HUMAN_Yee_2023_activity/nkn_esm2_proteinmpnn_<selection>_gpu_<job-id>/`.
The normal CV checkpoints, predictions, and metrics are under `cv/<setting>/`.
Console progress defaults to every 10 steps; each update is flushed to a
progress JSONL file. `benchmark.json` records elapsed CV time (including
evaluation/output) and peak CUDA allocated/reserved bytes for each setting.
The resolved inference backend and its settings are recorded in the log,
`config.json`, `execution_plan.json`, and `benchmark.json`.

Other options include `--config`, `--log-every`, `--num-threads`, `--output`,
`--features`, `--folds`, and `--assay-index`.

The synthetic numerical entry point also supports both backends. For a small
BBMM CUDA check without real assay assets:

~~~bash
PYTHONPATH=src python -m experiments.synthetic_gpu singles_random --device cuda --variants 40 --max-steps 2 --min-steps 0 --inference-backend bbmm --output results/synthetic_bbmm_gpu_check
~~~

Use `--inference-backend cholesky` and a different `--output` directory for
the corresponding Cholesky run.
