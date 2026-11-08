"""Compare Kermut/NKN predictive variances in the style of Kermut Figure 2.

One observation per assay is the mean variance across its held-out variants,
not an individual variant or an equally weighted mean of unequal CV folds.
Only the three 1M -> 1M settings are used. No model or tensor library is loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

from plotting.calibration import _expected_fold_ids, _read, discover_runs


SETTINGS = ("singles_random", "singles_modulo", "singles_contiguous")
MODELS = ("kermut", "nkn")
COLORS = {"kermut": "#4575B4", "nkn": "#D95F02"}
LABELS = {"kermut": "Kermut", "nkn": "PHKN"}


def _count(value, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _variance(value, name: str, kind: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (kind == "observation" and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if kind == 'observation' else 'nonnegative'}")
    return value


def select_runs(results_dir: Path, model: str, run_dirs=None, assays=None) -> dict[str, Path]:
    """Select exactly one run per assay, refusing to combine configurations."""
    runs = ([Path(path).resolve() for path in run_dirs] if run_dirs
            else discover_runs(results_dir, assays, model))
    selected = {}
    for run in sorted(set(runs)):
        if not run.is_dir() or not run.name.startswith(f"{model}_"):
            raise ValueError(f"Expected a {model}_* run directory: {run}")
        assay = run.parent.name
        if assays and assay not in assays:
            continue
        if assay in selected:
            raise ValueError(
                f"Multiple {model} runs for {assay}: {selected[assay]} and {run}. "
                f"Select one per assay with --{model}-run-dir (repeatable)."
            )
        selected[assay] = run
    if not selected:
        raise ValueError(
            f"No {model} runs found. Use --{model}-results-dir COLLECTION with "
            f"COLLECTION/<assay>/{model}_*/cv/, or explicit --{model}-run-dir paths."
        )
    return selected


def _single_mean(metrics: dict, kind: str) -> tuple[int, float]:
    summary = metrics.get("variance_summary", {}).get("single_mutants")
    if not isinstance(summary, dict):
        raise ValueError("Missing variance_summary.single_mutants in saved metrics")
    count = _count(summary.get("n_variants"), "single-mutant count", 1)
    if count != _count(metrics.get("n_variants"), "n_variants", 1):
        raise ValueError("1M -> 1M results must contain only single mutants")
    key = f"mean_{kind}_variance_standardized"
    return count, _variance(summary.get(key), key, kind)


def load_setting(run: Path, setting: str, kind: str = "observation") -> dict:
    """Read a complete CV snapshot and its saved standardized variance mean."""
    if setting not in SETTINGS or kind not in ("observation", "latent"):
        raise ValueError("Unknown single-mutant setting or variance kind")
    directory = run / "cv" / setting
    path = directory / "metrics.json"
    if (directory / "failure.json").exists():
        raise ValueError(f"Setting has failure.json: {directory}")
    metrics = _read(path)
    if metrics.get("setting", setting) != setting:
        raise ValueError(f"Setting identity disagrees with directory: {path}")
    if metrics.get("assay_id", run.parent.name) != run.parent.name:
        raise ValueError(f"Assay identity disagrees with directory: {path}")
    expected = _expected_fold_ids(directory, run, metrics)
    folds = metrics.get("fold_metrics")
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"Missing aggregate fold_metrics: {path}")
    ids = [_count(fold.get("fold", index), "fold ID") for index, fold in enumerate(folds)]
    if sorted(ids) != expected or metrics.get("n_folds") != len(expected):
        raise ValueError(f"Incomplete CV: expected folds {expected}, found {ids}")
    counts = [_count(fold.get("n_variants"), "fold n_variants", 1) for fold in folds]
    total = _count(metrics.get("n_variants"), "n_variants", 1)
    if sum(counts) != total:
        raise ValueError("Aggregate and fold variant counts disagree")
    sources = [str(path)]
    # Recover older aggregate summaries only when all completed folds have the
    # required standardized single-mutant mean. Never average folds equally.
    if "variance_summary" in metrics:
        count, mean = _single_mean(metrics, kind)
        source = "aggregate_single_mutant_summary"
    else:
        values = [_single_mean(fold, kind) for fold in folds]
        count = sum(n for n, _ in values)
        mean = math.fsum(n * value for n, value in values) / count
        source = "variant_weighted_fold_summaries"
    assignments = None
    assignment_path = directory / "fold_assignments.csv"
    if assignment_path.is_file():
        assignments = {}
        with assignment_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not {"variant_id", "fold"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Fold assignments lack variant_id/fold: {assignment_path}")
            for row in reader:
                variant = row["variant_id"]
                fold = int(row["fold"])
                if not variant or variant in assignments or fold not in expected:
                    raise ValueError(f"Invalid or duplicate fold assignment: {assignment_path}")
                assignments[variant] = fold
        actual_counts = {fold: 0 for fold in expected}
        for fold in assignments.values():
            actual_counts[fold] += 1
        if actual_counts != dict(zip(ids, counts)):
            raise ValueError(f"Assignment and metric variant counts disagree: {assignment_path}")
        sources.append(str(assignment_path))
    return {"assay_id": run.parent.name, "model": run.name.split("_", 1)[0],
            "setting": setting, "run_dir": str(run), "n_variants": count,
            "n_folds": len(expected), "mean_variance": mean, "variance_kind": kind,
            "units": "saved_training_standardized", "aggregation_source": source,
            "source_files": sources, "fold_counts": dict(zip(ids, counts)),
            "assignments": assignments}


def build_comparison(kermut_runs: dict[str, Path], nkn_runs: dict[str, Path],
                     kind: str = "observation") -> dict:
    """Use the same complete, matched assay cohort in all six distributions."""
    if kind not in ("observation", "latent"):
        raise ValueError("variance kind must be observation or latent")
    selections = dict(zip(MODELS, (kermut_runs, nkn_runs)))
    rows, issues, included = [], [], []
    for assay in sorted(set(kermut_runs) | set(nkn_runs)):
        missing = [model for model in MODELS if assay not in selections[model]]
        if missing:
            issues.append({"assay_id": assay, "status": "excluded",
                           "reason": f"No matching {'/'.join(missing)} run"})
            continue
        assay_rows = []
        try:
            for setting in SETTINGS:
                pair = [load_setting(selections[model][assay], setting, kind) for model in MODELS]
                first, second = pair
                if first["fold_counts"] != second["fold_counts"]:
                    raise ValueError(f"{setting}: model fold IDs or held-out counts differ")
                verified = all(row["assignments"] is not None for row in pair)
                if verified and first["assignments"] != second["assignments"]:
                    raise ValueError(f"{setting}: models have different variant/fold assignments")
                for row in pair:
                    row["pairing_verification"] = "variant_fold_assignments" if verified else "fold_counts_only"
                    assay_rows.append(row)
            if len({row["n_variants"] for row in assay_rows}) != 1:
                raise ValueError("Single-mutant counts differ between schemes")
            variant_sets = [set(row["assignments"]) for row in assay_rows if row["assignments"] is not None]
            if variant_sets and any(variants != variant_sets[0] for variants in variant_sets):
                raise ValueError("Single-mutant variant IDs differ between schemes")
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            issues.append({"assay_id": assay, "status": "excluded", "reason": str(error)})
            continue
        if any(row["pairing_verification"] == "fold_counts_only" for row in assay_rows):
            issues.append({"assay_id": assay, "status": "warning",
                           "reason": "Missing fold_assignments.csv; pairing checked using fold counts only"})
        for row in assay_rows:
            row.pop("assignments")
            row.pop("fold_counts")
        rows.extend(assay_rows)
        included.append(assay)
    return {
        "reference": "https://arxiv.org/pdf/2407.00002 (Figure 2)",
        "variance_kind": kind, "units": "saved_training_standardized",
        "method": "Mean over held-out single-mutant variants within each assay; equal assay weight in distributions.",
        "cohort": "Shared assays complete for both models and all three single-mutant schemes; no double-mutant requirement.",
        "settings": list(SETTINGS), "models": list(MODELS), "n_assays": len(included),
        "assays": included, "selected_runs": {
            model: {assay: str(run) for assay, run in runs.items()} for model, runs in selections.items()},
        "rows": rows, "issues": issues,
    }


def render_comparison(summary: dict, output_dir: Path, formats=("png", "pdf"), dpi=200) -> list[str]:
    """Draw adjacent violins with quartile bars, medians, and assay points."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    if not summary["n_assays"]:
        raise ValueError("No matched complete assays to plot")
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
        try:
            jitter = (np.random.default_rng(0).uniform(-0.065, 0.065, summary["n_assays"])
                      if summary["n_assays"] > 1 else np.zeros(1))
            for index, setting in enumerate(SETTINGS):
                for model, offset in zip(MODELS, (-0.21, 0.21)):
                    values = np.array([row["mean_variance"] for row in summary["rows"]
                                       if row["setting"] == setting and row["model"] == model])
                    position, color = index + offset, COLORS[model]
                    # KDE is undefined for a singleton or a constant sample.
                    if len(values) > 1 and np.ptp(values) > 0:
                        violin = ax.violinplot([values], positions=[position], widths=0.36,
                                              showextrema=False)
                        for body in violin["bodies"]:
                            body.set_facecolor(color)
                            body.set_edgecolor(color)
                            body.set_alpha(0.28)
                    q1, median, q3 = np.percentile(values, [25, 50, 75])
                    lo = min(value for value in values if value >= q1 - 1.5 * (q3 - q1))
                    hi = max(value for value in values if value <= q3 + 1.5 * (q3 - q1))
                    ax.vlines(position, lo, hi, color=color, linewidth=1.2, zorder=3)
                    ax.vlines(position, q1, q3, color=color, linewidth=7, zorder=3)
                    ax.scatter(position + jitter, values, color=color, s=13, alpha=0.45, zorder=4)
                    ax.plot(position, median, marker="_", markersize=10,
                            markeredgewidth=2, color="black", zorder=5)
            ax.set_xticks(range(3), ["Random", "Modulo", "Contiguous"])
            ax.set_xlim(-0.6, 2.6)
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Avg. predictive variance $\\hat{\\sigma}^2$")
            ax.legend(handles=[Patch(facecolor=COLORS[m], label=LABELS[m], alpha=0.7) for m in MODELS],
                      loc="upper left", frameon=False)
            ax.grid(axis="y", alpha=0.22)
            ax.set_axisbelow(True)
            files = []
            for extension in dict.fromkeys(formats):
                destination = output_dir / f"predictive_variance.{extension}"
                fig.savefig(destination, dpi=dpi, bbox_inches="tight")
                files.append(str(destination))
            return files
        finally:
            plt.close(fig)


def write_comparison(summary: dict, output_dir: Path, formats=("png", "pdf"), dpi=200) -> list[str]:
    """Export numeric values/provenance, including exclusion reasons on failure."""
    output_dir = Path(output_dir)
    if dpi < 1 or not formats or any(fmt not in ("png", "pdf", "svg") for fmt in formats):
        raise ValueError("Use a positive dpi and png/pdf/svg formats")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "predictive_variance_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    csv_path = output_dir / "predictive_variance_per_assay.csv"
    fields = ("assay_id", "model", "setting", "mean_variance", "variance_kind", "units", "n_variants",
              "n_folds", "run_dir", "aggregation_source", "pairing_verification", "source_files")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary["rows"]:
            writer.writerow({**row, "source_files": json.dumps(row["source_files"])})
    if not summary["n_assays"]:
        raise ValueError(f"No matched complete assays across all three schemes; see {summary_path} for exclusions")
    return [str(summary_path), str(csv_path), *render_comparison(summary, output_dir, formats, dpi)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for model in MODELS:
        parser.add_argument(f"--{model}-results-dir", type=Path, default=Path("results"),
                            help=f"Collection containing <assay>/{model}_* runs (default: results)")
        parser.add_argument(f"--{model}-run-dir", type=Path, action="append",
                            help="Explicit run directory (repeatable); bypasses discovery for this model")
    parser.add_argument("--assay", action="append", help="Restrict to assay ID (repeatable)")
    parser.add_argument("--output-dir", type=Path, default=Path("plots/predictive_variance"))
    parser.add_argument("--variance-kind", choices=("observation", "latent"), default="observation",
                        help="Observation includes noise, matching the paper (default); latent excludes it")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    try:
        selections = [select_runs(getattr(args, f"{model}_results_dir"), model,
                                  getattr(args, f"{model}_run_dir"), args.assay) for model in MODELS]
        summary = build_comparison(*selections, kind=args.variance_kind)
        for issue in summary["issues"]:
            print(f"{issue['status']}: {issue['assay_id']}: {issue['reason']}", file=sys.stderr)
        files = write_comparison(summary, args.output_dir.resolve(), args.formats, args.dpi)
    except (OSError, ValueError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    count = summary["n_assays"]
    print(f"Compared Kermut and NKN on {count} matched assay{'s' if count != 1 else ''} across all three schemes.")
    for path in files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
