# NKN GPU timing run

`run_nkn_synthetic_gpu.sbatch` now targets **S22A1_HUMAN_Yee_2023_activity**,
the human OCT1 assay with **10,094 single mutants**. The historical filename
is retained, but this job uses real prepared assets and published folds.

Submit one five-fold setting for timing:

~~~bash
sbatch sbatch_run/run_nkn/run_nkn_synthetic_gpu.sbatch singles_random
~~~

Omit `singles_random` to run all three singles settings. For a short full-size
run with two optimizer updates per fold:

~~~bash
sbatch sbatch_run/run_nkn/run_nkn_synthetic_gpu.sbatch singles_random --max-steps 2 --min-steps 0
~~~

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

The runner validates real-assay identity, complete single-variant coverage,
and every requested fold assignment before fitting. It snapshots the fold CSV
unchanged and records its SHA256. A small synthetic CPU/CUDA numerical parity
check precedes real training. Missing CUDA or invalid inputs stop the job.
Existing output directories are never overwritten.

Logs: `nkn_s22a1_gpu-<array-job>_0.out`.
Results: `results/S22A1_HUMAN_Yee_2023_activity/nkn_esm2_proteinmpnn_<selection>_gpu_<job-id>/`.
The normal CV checkpoints, predictions, and metrics are under `cv/<setting>/`.
Console progress defaults to every 10 steps; each update is flushed to a
progress JSONL file. `benchmark.json` records elapsed CV time (including
evaluation/output) and peak CUDA allocated/reserved bytes for each setting.

Other options include `--config`, `--log-every`, `--num-threads`, `--output`,
`--features`, `--folds`, and `--assay-index`.

The original synthetic-only numerical entry point remains available separately:

~~~bash
PYTHONPATH=src python -m experiments.synthetic_gpu singles_random --device cuda --variants 40 --max-steps 2 --min-steps 0 --output results/synthetic_gpu_check
~~~
