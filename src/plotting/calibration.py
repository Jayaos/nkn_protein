"""Plot interval and pairwise calibration from saved NKN fold metrics.

Run ``python -m plotting.calibration --help`` with src on PYTHONPATH.
This module reads saved metrics and fold assignments; it never loads checkpoints
or recomputes pairs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

from reporting.aggregate import _expected_fold_ids as _saved_fold_ids


SETTING_ORDER = (
    "singles_random", "singles_modulo", "singles_contiguous",
    "mixed_random_singles", "mixed_random_doubles", "single_to_double",
)
TITLES = {
    "singles_random": "Random", "singles_modulo": "Modulo",
    "singles_contiguous": "Contiguous", "mixed_random_singles": "Mixed CV: singles",
    "mixed_random_doubles": "Mixed CV: doubles", "single_to_double": "Single to double",
}


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _number(value, name: str, lower=0.0, upper=1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
    return value


def _count(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _stats(values: list[float]) -> dict:
    return {"mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "n_folds": len(values)}


def summarize_setting(folds: list[dict]) -> dict:
    """Equal-fold summaries, retaining undefined bins and metric values.

    ECE is averaged from individual fold ECEs, never reconstructed from the
    mean curve. Pair points average only folds that populate that bin; counts
    are summed solely for the accompanying occupancy histogram.
    """
    coverage_grid = bin_grid = None
    coverages, pair_bins = {}, {}
    interval_values, pair_values = [], []
    variants = pairs = 0
    seen = set()
    for index, fold in enumerate(folds):
        fold_id = fold.get("fold", index)
        if fold_id in seen:
            raise ValueError(f"Duplicate fold ID: {fold_id}")
        seen.add(fold_id)
        n_variants = _count(fold["n_variants"], "n_variants")
        n_pairs = _count(fold["n_pairs"], "n_pairs")
        variants += n_variants
        pairs += n_pairs
        coverage = sorted(fold.get("coverage", []), key=lambda row: _number(row["nominal"], "nominal"))
        bins = fold.get("bin_stats", [])
        if n_variants:
            if not coverage or fold.get("interval_ece") is None:
                raise ValueError(f"Fold {fold_id} is missing interval calibration data")
            grid = tuple(_number(row["nominal"], "nominal") for row in coverage)
            if list(grid) != sorted(set(grid)):
                raise ValueError("Coverage levels must be unique")
            if coverage_grid is not None and grid != coverage_grid:
                raise ValueError("Coverage grids differ across folds")
            coverage_grid = grid
            for row in coverage:
                coverages.setdefault(row["nominal"], []).append(_number(row["empirical"], "empirical"))
            interval_values.append(_number(fold["interval_ece"], "interval_ece"))
        if bins:
            grid = tuple((row["bin"], _number(row["lower"], "lower", .5),
                          _number(row["upper"], "upper", .5)) for row in bins)
            if len({row[0] for row in grid}) != len(grid) or any(lo >= hi for _, lo, hi in grid):
                raise ValueError("Invalid or duplicate pair bins")
            if bin_grid is not None and grid != bin_grid:
                raise ValueError("Pair bin edges differ across folds")
            bin_grid = grid
            total = 0
            for row in bins:
                count = _count(row["count"], "bin count")
                total += count
                dest = pair_bins.setdefault(row["bin"], {"bin": row["bin"], "lower": row["lower"],
                    "upper": row["upper"], "count": 0, "confidences": [], "accuracies": []})
                dest["count"] += count
                if count:
                    dest["confidences"].append(_number(row["confidence"], "confidence", .5))
                    dest["accuracies"].append(_number(row["accuracy"], "accuracy"))
                elif row.get("confidence") is not None or row.get("accuracy") is not None:
                    raise ValueError("Empty pair bins must have null confidence and accuracy")
            if total != n_pairs:
                raise ValueError("Pair bin counts do not match n_pairs")
        elif n_pairs:
            raise ValueError(f"Fold {fold_id} is missing pair calibration bins")
        if n_pairs:
            pair_values.append(_number(fold.get("pair_ece"), "pair_ece"))
        elif fold.get("pair_ece") is not None:
            raise ValueError("pair_ece must be null when there are no eligible pairs")
    averaged_bins = []
    for row in pair_bins.values():
        conf, acc = _stats(row["confidences"]), _stats(row["accuracies"])
        averaged_bins.append({k: row[k] for k in ("bin", "lower", "upper", "count")} | {
            "confidence": conf["mean"], "accuracy": acc["mean"], "confidence_std": conf["std"],
            "accuracy_std": acc["std"], "n_folds": conf["n_folds"],
        })
    return {"n_folds": len(folds), "n_variants": variants, "n_pairs": pairs,
            "interval_ece": _stats(interval_values), "pair_ece": _stats(pair_values),
            "coverage": [{"nominal": q, "empirical": _stats(values)["mean"],
                          "std": _stats(values)["std"], "n_folds": len(values)}
                         for q, values in coverages.items()],
            "bin_stats": averaged_bins}


def discover_runs(results_dir: Path, assays: list[str] | None = None) -> list[Path]:
    """Find NKN run directories, never individual fold/validation directories."""
    root = Path(results_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Results directory does not exist: {root}")
    runs = sorted(path for path in root.glob("*/nkn_*") if path.is_dir()
                  and ((path / "cv").is_dir() or (path / "fit").is_dir()))
    if assays:
        runs = [path for path in runs if path.parent.name in assays]
    return runs


def _expected_fold_ids(path: Path, run_dir: Path, aggregate: dict) -> list[int]:
    if path.parent.name == "fit":
        return [0]
    nominal = None
    for config_path in (path / "resolved_config.json", run_dir / "config.json"):
        if config_path.is_file():
            value = _read(config_path).get("evaluation", {}).get("n_folds")
            if value is not None:
                nominal = _count(value, "expected folds")
                break
    manifest_path = run_dir / "fold_manifest.json"
    manifest = _read(manifest_path) if manifest_path.is_file() else {}
    return _saved_fold_ids(path, {"n_folds": 5, **aggregate}, nominal, manifest, path.name)


def _load_setting(path: Path, run_dir: Path, aggregate: dict | None = None,
                  name: str | None = None) -> dict:
    source_files = []
    aggregate_path = path / "metrics.json"
    if aggregate is None:
        aggregate = _read(aggregate_path) if aggregate_path.is_file() else {}
    expected_ids = _expected_fold_ids(path, run_dir, aggregate)
    expected = len(expected_ids)
    folds = []
    if "fold_metrics" in aggregate:
        folds = aggregate["fold_metrics"]
        source_files.append(str(aggregate_path))
    elif "coverage" in aggregate:  # Single fit evaluation.
        folds = [aggregate]
        source_files.append(str(aggregate_path))
    else:
        for fold_path in sorted(path.glob("fold_*/metrics.json")):
            fold = _read(fold_path)
            if name and fold.get("setting_metrics"):
                fold = fold["setting_metrics"].get(name)
                if fold is None:
                    continue
            fold.setdefault("fold", int(fold_path.parent.name.removeprefix("fold_")))
            folds.append(fold)
            source_files.append(str(fold_path))
    if not isinstance(folds, list):
        raise ValueError(f"fold_metrics must be a list: {aggregate_path}")
    if path.parent.name != "fit":
        observed = {_count(fold.get("fold", index), "fold ID") for index, fold in enumerate(folds)}
        if not observed.issubset(expected_ids):
            raise ValueError(f"Fold metrics contain IDs absent from the saved assignments: {path}")
    summary = summarize_setting(folds)
    failure_path = path / "failure.json"
    failed = failure_path.is_file()
    reason = None
    if failed:
        failure = _read(failure_path)
        source_files.append(str(failure_path))
        reason = str(failure.get("error", "Run failed before evaluation completed"))
    if not folds:
        status = "unavailable"
        reason = reason or "No completed fold metrics"
    elif (len(folds) < expected or failed
          or (path.parent.name != "fit" and aggregate.get("n_folds", expected) != expected)):
        status = "partial"
        reason = reason or (f"Only {len(folds)} of {expected} expected folds have metrics"
                            if len(folds) < expected else "Aggregate fold count disagrees with saved assignments")
    else:
        status = "complete"
    if len(folds) > expected:
        raise ValueError(f"More folds than expected in {path}")
    return {"name": name or path.name, "status": status, "reason": reason,
            "expected_folds": expected, "folds": folds, "summary": summary, "source_files": source_files}


def load_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    paths = sorted(p for kind in ("cv", "fit") for p in (run_dir / kind).glob("*") if p.is_dir())
    plan = run_dir / "execution_plan.json"
    selected, priorities = {}, {}
    for path in paths:
        aggregate_path = path / "metrics.json"
        aggregate = _read(aggregate_path) if aggregate_path.is_file() else {}
        # Mixed runs contain separate singles/doubles summaries; avoid repeating
        # the selected top-level summary as a third copy.
        children = aggregate.get("setting_metrics")
        if children:
            candidates = [_load_setting(path, run_dir, child, name) for name, child in children.items()]
        else:
            # A failed mixed run may have completed per-fold subgroup metrics
            # without a final aggregate. Retain both mutation-depth panels.
            child_names = set()
            if not aggregate:
                for fold_path in path.glob("fold_*/metrics.json"):
                    child_names.update(_read(fold_path).get("setting_metrics", {}))
            candidates = ([_load_setting(path, run_dir, aggregate, name) for name in sorted(child_names)]
                          if child_names else [_load_setting(path, run_dir, aggregate)])
        for setting in candidates:
            name = setting["name"]
            # Mixed directories can each include both subgroup summaries.
            # Prefer available data, then the subgroup's own run directory.
            priority = (bool(setting["folds"]), path.name == name, len(setting["folds"]))
            if name not in selected or priority > priorities[name]:
                selected[name], priorities[name] = setting, priority
    # Add planned missing settings only after mixed subgroup expansion.
    if plan.is_file():
        for name in _read(plan).get("settings", []):
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError(f"Invalid setting name in {plan}")
            if name not in selected:
                path = run_dir / ("fit" if name == "single_to_double" else "cv") / name
                selected[name] = _load_setting(path, run_dir)
    settings = list(selected.values())
    if not settings:
        raise ValueError(f"No cv/fit settings found in {run_dir}")
    settings.sort(key=lambda s: (SETTING_ORDER.index(s["name"]) if s["name"] in SETTING_ORDER else 99, s["name"]))
    return {"assay": run_dir.parent.name, "model": run_dir.name, "run_dir": str(run_dir), "settings": settings}


def _metric_label(name: str, stats: dict) -> str:
    if stats["mean"] is None:
        return f"{name}: undefined"
    label = f"{name}: {stats['mean']:.4f}"
    if stats["std"] is not None:
        label += f" ± {stats['std']:.4f}"
    return label


def _render(run: dict, kind: str, destination: Path, formats, dpi: int) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import FuncFormatter
    except ImportError as error:
        raise RuntimeError("Plotting requires Matplotlib. Install matplotlib>=3.8,<3.11 in the plotting environment.") from error

    settings = run["settings"]
    pair = kind == "pair_ece"
    columns = min(3, len(settings))
    rows = math.ceil(len(settings) / columns)
    color = "#C65D24" if pair else "#2466A0"
    with plt.rc_context({"font.size": 10, "axes.titlesize": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(rows * (2 if pair else 1), columns, squeeze=False,
            figsize=(4.2 * max(2, columns), (5.8 if pair else 4.6) * rows),
            gridspec_kw={"height_ratios": [3, 1] * rows} if pair else None)
        title = "Pairwise calibration" if pair else "Prediction interval calibration"
        fig.suptitle(f"{run['assay']}\n{title} · {run['model']}", fontsize=12, y=.98)
        for index, setting in enumerate(settings):
            row, col = divmod(index, columns)
            ax = axes[row * (2 if pair else 1), col]
            hist = axes[row * 2 + 1, col] if pair else None
            summary = setting["summary"]
            ax.set_title(f"{TITLES.get(setting['name'], setting['name'])}\n"
                         f"{summary['n_folds']}/{setting['expected_folds']} folds"
                         + (" · partial" if setting["status"] == "partial" else ""))
            if setting["status"] == "unavailable":
                ax.text(.5, .5, "Unavailable\nNo completed fold metrics", ha="center", va="center",
                        transform=ax.transAxes, color="#666666", fontsize=11)
                ax.set_axis_off()
                if hist is not None:
                    hist.set_axis_off()
                continue
            ax.set_xlim(.49 if pair else 0, 1.01 if pair else 1)
            ax.set_ylim(0, 1)
            ax.plot([.5 if pair else 0, 1], [.5 if pair else 0, 1], "--", color="#333333", lw=1, zorder=1)
            ax.grid(alpha=.16)
            ax.set_xlabel("Predicted ordering confidence" if pair else "Nominal coverage")
            ax.set_ylabel("Observed ordering accuracy" if pair else "Observed coverage")
            for fold in setting["folds"]:
                if pair:
                    values = fold.get("bin_stats", [])
                    x = [b["confidence"] if b["count"] else math.nan for b in values]
                    y = [b["accuracy"] if b["count"] else math.nan for b in values]
                else:
                    values = sorted(fold.get("coverage", []), key=lambda value: value["nominal"])
                    x, y = [v["nominal"] for v in values], [v["empirical"] for v in values]
                ax.plot(x, y, color="#87939B", alpha=.5, lw=.8, zorder=2)
            values = summary["bin_stats" if pair else "coverage"]
            if pair:
                x = [v["confidence"] if v["count"] else math.nan for v in values]
                y = [v["accuracy"] if v["count"] else math.nan for v in values]
                xerr = [v["confidence_std"] or 0 for v in values]
                yerr = [v["accuracy_std"] or 0 for v in values]
            else:
                x, y = [v["nominal"] for v in values], [v["empirical"] for v in values]
                xerr, yerr = None, [v["std"] or 0 for v in values]
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, color=color, marker="o", ms=4, lw=1.6,
                        capsize=2.5, elinewidth=.9, zorder=3)
            annotation = _metric_label("Pair ECE" if pair else "Interval ECE", summary[kind])
            if pair:
                annotation += f"\n{summary['pair_ece']['n_folds']} valid folds · {summary['n_pairs']:,} pairs"
                if not summary["n_pairs"]:
                    ax.text(.5, .5, "No eligible variant pairs", ha="center", transform=ax.transAxes)
                hist.bar([(v["lower"] + v["upper"]) / 2 for v in values],
                         [v["count"] for v in values], width=[.85 * (v["upper"] - v["lower"]) for v in values],
                         color=color, alpha=.65)
                hist.set_xlim(.49, 1.01)
                hist.set_xlabel("Confidence bin")
                hist.set_ylabel("Pairs")
                hist.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:g}k" if x >= 1000 else f"{x:g}"))
                hist.grid(axis="y", alpha=.15)
            ax.text(.97, .035, annotation, transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                    bbox={"facecolor": "white", "alpha": .87, "edgecolor": "none", "pad": 3})
        for index in range(len(settings), rows * columns):
            row, col = divmod(index, columns)
            axes[row * (2 if pair else 1), col].set_axis_off()
            if pair:
                axes[row * 2 + 1, col].set_axis_off()
        fig.legend(handles=[Line2D([], [], color="#333333", ls="--", label="Ideal calibration"),
                            Line2D([], [], color="#87939B", lw=.8, label="Individual folds"),
                            Line2D([], [], color=color, marker="o", label="Fold mean ± 1 SD")],
                   loc="lower center", bbox_to_anchor=(.5, .037), ncol=3, frameon=False)
        note = ("Bin means give equal weight to nonempty folds; bars sum within-fold pair counts. "
                "ECE averages fold ECEs." if pair else "Coverage and ECE average folds equally. Error bars show fold SD, not confidence intervals.")
        fig.text(.5, .012, note, ha="center", fontsize=8, color="#555555")
        fig.tight_layout(rect=(0, .09, 1, 1))
        files = []
        try:
            for extension in formats:
                path = destination / f"{kind}.{extension}"
                fig.savefig(path, dpi=dpi, facecolor="white")
                files.append(str(path))
        finally:
            plt.close(fig)
    return files


def plot_run(run_dir: Path, output_dir: Path | None = None,
             formats=("png", "pdf"), dpi: int = 200) -> dict:
    """Save under run/plots or custom-root/assay/model-run; return provenance."""
    if not formats or any(ext not in ("png", "pdf", "svg") for ext in formats):
        raise ValueError("Select at least one output format: png, pdf, svg")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
        raise ValueError("dpi must be a positive integer")
    run = load_run(run_dir)
    destination = (Path(output_dir).resolve() / run["assay"] / run["model"]
                   if output_dir is not None else Path(run["run_dir"]) / "plots")
    destination.mkdir(parents=True, exist_ok=True)
    run["output_dir"] = str(destination)
    run["aggregation"] = {
        "metrics": "Arithmetic mean of defined fold ECEs; never ECE of the averaged curve.",
        "coverage": "Equal weight per nonempty fold at each common coverage level.",
        "pair_curve": "Equal weight per fold with pairs in the bin; undefined bins stay empty.",
        "pair_counts": "Sum of within-fold pairs; no pairs across folds.",
        "error_bars": "Sample standard deviation across contributing folds (ddof=1); not a confidence interval.",
    }
    run["files"] = [file for kind in ("interval_ece", "pair_ece")
                    for file in _render(run, kind, destination, tuple(dict.fromkeys(formats)), dpi)]
    report = destination / "calibration_summary.json"
    run["files"].append(str(report))
    report.write_text(json.dumps(run, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Root containing assay/model-run directories")
    parser.add_argument("--assay", action="append", help="Only this assay (repeatable)")
    parser.add_argument("--run-dir", type=Path, action="append", help="Explicit model-run directory (repeatable); bypasses discovery")
    parser.add_argument("--output-dir", type=Path, help="Custom output root; saves under ROOT/assay/model-run (default: each run/plots)")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)
    if args.run_dir and args.assay:
        parser.error("Use --run-dir or --assay, not both")
    try:
        runs = ([path.resolve() for path in args.run_dir] if args.run_dir
                else discover_runs(args.results_dir, args.assay))
        if not runs:
            raise ValueError("No matching NKN runs found")
        plotted = 0
        for path in dict.fromkeys(runs):
            result = plot_run(path, args.output_dir, args.formats, args.dpi)
            for setting in result["settings"]:
                if setting["status"] != "complete":
                    print(f"{result['assay']} / {setting['name']}: {setting['status']}: {setting['reason']}", file=sys.stderr)
                if setting["summary"]["n_variants"]:
                    plotted += 1
            print(f"Saved calibration plots: {result['output_dir']}", flush=True)
        if not plotted:
            raise ValueError("No completed evaluation data were available; see the generated summaries")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
