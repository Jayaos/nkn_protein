"""Shared inference options for GPU and synthetic experiment entry points."""

from dataclasses import replace


BBMM_FIELDS = (
    "bbmm_cg_tolerance", "bbmm_max_cg_iterations", "bbmm_num_trace_samples",
    "bbmm_max_lanczos_iterations", "bbmm_preconditioner_size", "bbmm_probe_seed",
)


def add_inference_arguments(parser):
    parser.add_argument("--gp-computation-backend", choices=("cholesky", "bbmm"),
                        help="GP computations for training and prediction: override "
                             "training.gp_computation_backend (default: cholesky)")
    for name in BBMM_FIELDS:
        parser.add_argument("--" + name.replace("_", "-"),
                            type=float if name == "bbmm_cg_tolerance" else int,
                            help=f"Override training.{name}")


def resolve_training_config(training, args):
    names = ("max_steps", "min_steps", "gp_computation_backend", *BBMM_FIELDS)
    updates = {name: getattr(args, name) for name in names if getattr(args, name, None) is not None}
    resolved = replace(training, **updates)
    if resolved.gp_computation_backend == "bbmm":
        from .bbmm import require_bbmm
        require_bbmm()
    return resolved


def inference_metadata(training):
    result = {"backend": training.gp_computation_backend, "dtype": "float64"}
    if training.gp_computation_backend == "bbmm":
        import linear_operator
        result.update({name: getattr(training, name) for name in BBMM_FIELDS})
        result.update(linear_operator_version=linear_operator.__version__,
                      jitter=training.jitter_initial, fixed_base_probes=True,
                      dense_kernel=True)
    return result
