"""Exercise the existing NKN/GP framework using generated numerical inputs only."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import traceback

import torch

from config import ExperimentConfig
from data_loading import FeatureBatch
from experiments.runner import run_cv
from gp.model import ExactProteinGP
from gp.preprocessing import FoldTransform
from gp.inference import add_inference_arguments, inference_metadata, resolve_training_config


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs"
    / "esm2_proteinmpnn_nkn_two_scores_additive_four_branches_noise_initial_1p0.json"
)
SETTINGS = ("singles_random", "singles_modulo", "singles_contiguous")


def make_synthetic_batch(n_variants=256, feature_dim=32, seed=2024):
    """Generate reproducible CPU tensors, without sequences or prepared assets."""
    if n_variants < 4 or feature_dim < 1 or seed < 0:
        raise ValueError("variants must be >= 4, feature-dim >= 1, and seed >= 0")
    generator = torch.Generator().manual_seed(seed)

    def normal(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    n_positions = min(128, n_variants // 2)
    positions = torch.arange(n_variants) % n_positions
    coordinates = normal(n_positions, 3)
    probabilities = normal(n_positions, 20).softmax(dim=-1)
    global_features = normal(n_variants, feature_dim)
    local_features = normal(n_variants, feature_dim)
    scores = normal(n_variants, 2)
    targets = (
        0.6 * scores[:, 0] - 0.2 * scores[:, 1]
        + 0.3 * global_features[:, 0].sin() + 0.1 * normal(n_variants)
    )
    return FeatureBatch(
        assay_id="synthetic_nkn_gpu",
        variant_ids=tuple(f"synthetic_{index}" for index in range(n_variants)),
        targets=targets,
        scores=scores,
        global_features=global_features,
        local_features={
            "sequence_site": local_features,
            "inverse_folding": probabilities[positions],
            "inverse_folding_site": probabilities[positions, :1].log(),
            "structure_distance": coordinates[positions],
            "structure": coordinates[positions],
        },
        owners=torch.arange(n_variants),
        mutation_positions=positions + 1,
        score_names=("synthetic_first", "synthetic_second"),
        feature_signature={
            "source": "synthetic_gpu_v1", "seed": str(seed),
            "variants": str(n_variants), "feature_dim": str(feature_dim),
        },
    )


def make_folds(batch, scheme, n_folds, seed):
    """Use balanced random folds or position-grouped modulo/contiguous folds."""
    if n_folds < 2 or batch.num_variants < 2 * n_folds:
        raise ValueError("At least two synthetic variants per fold are required")
    if scheme == "random":
        order = torch.randperm(batch.num_variants, generator=torch.Generator().manual_seed(seed))
        folds = torch.empty(batch.num_variants, dtype=torch.long)
        folds[order] = torch.arange(batch.num_variants) % n_folds
        return folds
    if scheme not in ("modulo", "contiguous"):
        raise ValueError(f"Unknown synthetic split scheme: {scheme}")
    positions = torch.empty(batch.num_variants, dtype=torch.long)
    positions[batch.owners.cpu()] = batch.mutation_positions.cpu()
    unique, inverse = positions.unique(sorted=True, return_inverse=True)
    if len(unique) < n_folds:
        raise ValueError("At least one distinct synthetic position per fold is required")
    if scheme == "modulo":
        position_folds = (unique - 1) % n_folds
    else:
        position_folds = torch.empty(len(unique), dtype=torch.long)
        for fold, indices in enumerate(torch.tensor_split(torch.arange(len(unique)), n_folds)):
            position_folds[indices] = fold
    return position_folds[inverse]


def _numerics(model, train, test, targets):
    model.kernel.prepare_training_geometry(train)
    result = {"kernel": model.kernel(train).detach().cpu()}
    loss, _ = model.objective(train, targets)
    loss.backward()
    result["objective"] = loss.detach().cpu()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                raise RuntimeError(f"Missing synthetic gradient: {name}")
            result[f"gradient:{name}"] = parameter.grad.detach().cpu()
    with torch.no_grad():
        mean, latent, observation, _ = model.posterior(train, targets, test)
    result.update(
        posterior_mean=mean.cpu(), posterior_latent=latent.cpu(),
        posterior_observation=observation.cpu(),
    )
    return result


def validate_device_parity(config, device="cuda"):
    """Compare Cholesky reference numerics across CPU and device, without sampling."""
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; GPU validation cannot fall back to CPU")
    batch = make_synthetic_batch(16, 4, seed=37)
    train, test = batch.select(range(12)), batch.select(range(12, 16))
    transform = FoldTransform.fit(train)
    train, test = transform.apply(train), transform.apply(test)
    targets = transform.standardize_targets(train.targets)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    training = replace(config.training, inference_backend="cholesky")
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        cpu_model = ExactProteinGP(train, config.model, training)
        other_model = ExactProteinGP(train.to(device), config.model, training)
    other_model.load_state_dict(cpu_model.state_dict())
    reference = _numerics(cpu_model, train, test, targets)
    actual = _numerics(other_model, train.to(device), test.to(device), targets.to(device))
    errors = {}
    for name, expected in reference.items():
        torch.testing.assert_close(actual[name], expected, rtol=1e-6, atol=1e-8)
        group = "gradients" if name.startswith("gradient:") else name
        errors[group] = max(errors.get(group, 0.0), float((actual[name] - expected).abs().max()))
    return {"status": "passed", "backend": "cholesky", "rtol": 1e-6, "atol": 1e-8,
            "max_abs_error": errors}


def validate_backend_parity(config, device="cuda"):
    """Check BBMM's deterministic trace limit against Cholesky on tiny synthetic inputs.

    This validates the integration, not the stochastic error on a full assay.
    Production solver/probe settings remain unchanged in the experiment config.
    """
    device = torch.device(device)
    batch = make_synthetic_batch(16, 4, seed=37).to(device)
    train, test = batch.select(range(12)), batch.select(range(12, 16))
    transform = FoldTransform.fit(train)
    targets = transform.standardize_targets(train.targets)
    train, test = transform.apply(train), transform.apply(test)
    exact_training = replace(config.training, inference_backend="cholesky")
    iterative_training = replace(
        config.training, inference_backend="bbmm", bbmm_num_trace_samples=0,
        bbmm_preconditioner_size=0, bbmm_cg_tolerance=1e-9,
        bbmm_max_cg_iterations=100, bbmm_max_lanczos_iterations=12,
    )
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        exact = ExactProteinGP(train, config.model, exact_training)
        iterative = ExactProteinGP(train, config.model, iterative_training)
    iterative.load_state_dict(exact.state_dict())
    reference = _numerics(exact, train, test, targets)
    actual = _numerics(iterative, train, test, targets)
    errors = {}
    for name, expected in reference.items():
        torch.testing.assert_close(actual[name], expected, rtol=1e-5, atol=1e-7)
        group = "gradients" if name.startswith("gradient:") else name
        errors[group] = max(errors.get(group, 0.0), float((actual[name] - expected).abs().max()))
    return {"status": "passed", "device": str(device), "rtol": 1e-5, "atol": 1e-7,
            "max_abs_error": errors, "validation_settings": inference_metadata(iterative_training),
            "scope": "tiny synthetic deterministic-trace limit; not full-assay stochastic accuracy"}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", nargs="?", default="singles", choices=("singles", *SETTINGS))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--variants", type=int, default=256)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--min-steps", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    add_inference_arguments(parser)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    output_created = False
    try:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use CUDA-enabled PyTorch in a GPU allocation")
        if args.log_every < 1:
            raise ValueError("log-every must be positive")
        base = ExperimentConfig.load(args.config)
        training = resolve_training_config(base.training, args)
        if training.min_steps > training.max_steps:
            raise ValueError("min-steps exceeds max-steps; supply both for a shorter smoke run")
        config = replace(
            base, training=training, device=args.device, num_threads=args.num_threads,
            feature_selection={}, score_sources=["synthetic_first", "synthetic_second"],
            inverse_folding_policy="normalized",
        )
        torch.set_num_threads(config.num_threads)
        batch = make_synthetic_batch(args.variants, args.feature_dim, args.seed)
        settings = SETTINGS if args.selection == "singles" else (args.selection,)
        splits = {
            setting: make_folds(batch, setting.removeprefix("singles_"),
                                config.evaluation.n_folds, args.seed)
            for setting in settings
        }
        args.output.mkdir(parents=True, exist_ok=False)
        output_created = True
        _write_json(args.output / "resolved_config.json", config.to_dict())
        _write_json(args.output / "synthetic_manifest.json", {
            "source": "generated numerical tensors; no biological assets",
            "assay_id": batch.assay_id, "variants": args.variants,
            "feature_dim": args.feature_dim, "seed": args.seed,
            "score_names": batch.score_names, "settings": settings,
            "base_config": str(args.config.resolve()), "fold_workers": 1,
        })
        print(f"Synthetic NKN: {args.variants} variants; feature dimension {args.feature_dim}", flush=True)
        print(f"PyTorch: {torch.__version__}; device: {args.device}; dtype: float64; "
              f"threads: {config.num_threads}; sequential folds", flush=True)
        print(f"Training: min_steps={training.min_steps}; max_steps={training.max_steps}; "
              f"stopping={training.stopping_criterion}; seeds={training.seeds}", flush=True)
        inference = inference_metadata(training)
        print(f"Inference: {json.dumps(inference)}", flush=True)
        summary = {"device": args.device, "torch_version": torch.__version__, "settings": {},
                   "inference_backend": training.inference_backend, "inference": inference}
        if args.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
            summary["gpu"] = torch.cuda.get_device_name()
            summary["parity"] = validate_device_parity(config)
            _write_json(args.output / "device_parity.json", summary["parity"])
            print("CPU/CUDA parity: passed (kernel, objective, gradients, posterior)", flush=True)
        if training.inference_backend == "bbmm":
            summary["backend_parity"] = validate_backend_parity(config, args.device)
            _write_json(args.output / "backend_parity.json", summary["backend_parity"])
            print("BBMM/Cholesky synthetic deterministic-trace preflight passed", flush=True)
        for setting in settings:
            if args.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            log_path = args.output / f"{setting}.progress.jsonl"
            with log_path.open("x", encoding="utf-8") as handle:
                def progress(event):
                    elapsed = time.perf_counter() - started
                    handle.write(json.dumps({**event, "elapsed_seconds": elapsed}) + "\n")
                    handle.flush()
                    if event.get("event") == "fold_start":
                        print(f"{setting} fold {event['fold']}: {event['n_train']} training, "
                              f"{event['n_test']} held-out", flush=True)
                    elif "step" in event and event["step"] % args.log_every == 0:
                        print(f"{setting} fold {event['fold']} seed {event['seed']} "
                              f"step {event['step']}: NLL/variant={event['nll_per_variant']:.6g}; "
                              f"setting elapsed={elapsed:.2f}s", flush=True)

                run_cv(
                    batch, splits[setting], config, args.output / "cv" / setting,
                    scheme=setting.removeprefix("singles_"), setting=setting,
                    split_source="synthetic", fold_workers=1, progress=progress,
                )
            if args.device == "cuda":
                torch.cuda.synchronize()
            timing = {"cv_seconds": time.perf_counter() - started}
            if args.device == "cuda":
                timing.update(
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                )
            summary["settings"][setting] = timing
            _write_json(args.output / "benchmark.json", summary)
            print(f"Completed {setting}: {timing} (includes fitting, evaluation, and output)", flush=True)
        print(f"Completed synthetic run. Results: {args.output}", flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, AssertionError) as error:
        if output_created:
            _write_json(args.output / "failure.json", {
                "error": str(error), "traceback": traceback.format_exc(),
            })
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
