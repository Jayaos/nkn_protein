"""Collect saved NKN/Kermut metrics without loading checkpoints or predictions."""

import csv
import json
import re
from pathlib import Path
from typing import Any


IDENTITY = ("assay_id", "model", "run_id", "stage", "setting")
METRICS = (
    "avg_spearman", "std_spearman", "spearman_oof", "spearman_oof_standardized",
    "spearman", "interval_ece", "pair_ece", "n_variants", "n_pairs", "n_folds",
    "n_valid_spearman_folds", "n_valid_pair_folds", "n_valid_interval_folds",
)


def _read(path: Path, issues: list[dict]) -> dict | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        issues.append({"path": str(path), "error": str(error)})
        return None


def _scalars(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items()
            if key not in IDENTITY and (value is None or isinstance(value, (str, int, float, bool)))}


def _write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    fields = list(dict.fromkeys((*columns, *(key for row in rows for key in row))))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_settings(run: Path, manifest: dict, plan: dict) -> set[tuple[str, str]]:
    """Include settings that were requested but never started."""
    settings = {(stage, directory.name)
                for stage in ("cv", "fit") for directory in (run / stage).glob("*")
                if directory.is_dir()}
    requested = manifest.get("requested_settings", [])
    if not isinstance(requested, list):
        raise ValueError("Manifest requested_settings must be a list")
    names = list(requested)
    mappings = [manifest.get("fold_ids", {}), plan.get("settings", {})]
    sources = manifest.get("sources", {})
    if not isinstance(sources, dict):
        raise ValueError("Manifest sources must be an object")
    for source in sources.values():
        if not isinstance(source, dict):
            raise ValueError("Manifest source must be an object")
        mappings.append(source.get("settings", {}))
    for mapping in mappings:
        if not isinstance(mapping, dict):
            raise ValueError("Saved settings and fold_ids must be objects")
        names.extend(mapping)
    for name in names:
        if (not isinstance(name, str) or not name or name in {".", ".."}
                or "/" in name or "\\" in name):
            raise ValueError("Requested settings must be nonempty directory names")
        settings.add(("fit" if name == "single_to_double" else "cv", name))
    return settings


def _job_id(run: Path) -> int | None:
    # GPU names end in <job>_<task>; CPU names end in <job>. Manual GPU
    # runs also end in a task index, which must not be treated as a job ID.
    if "_manual_" in run.name:
        return None
    match = re.search(r"_(\d+)(?:_\d+)?$", run.name)
    return int(match.group(1)) if match else None


def _select_runs(root: Path, issues: list[dict]) -> list[tuple[Path, dict, dict]]:
    """Select one run with all aggregate metric files per assay and model."""
    candidates: dict[tuple[str, str], list[tuple[Path, dict, dict]]] = {}
    for run in sorted(root.glob("*/*")):
        model = run.name.split("_", 1)[0]
        if not run.is_dir() or model not in {"nkn", "kermut"}:
            continue
        manifest_path, plan_path = run / "fold_manifest.json", run / "execution_plan.json"
        manifest = _read(manifest_path, issues)
        plan = _read(plan_path, issues)
        if ((manifest_path.exists() and manifest is None)
                or (plan_path.exists() and plan is None)):
            continue
        manifest = manifest or {}
        try:
            settings = _run_settings(run, manifest, plan or {})
        except ValueError as error:
            issues.append({"path": str(run), "error": str(error)})
            continue
        if not settings:
            issues.append({"path": str(run), "error": "No setting directories or requested settings found"})
            continue
        saved_metrics = {}
        for stage, setting in sorted(settings):
            path = run / stage / setting / "metrics.json"
            data = _read(path, issues)
            if not path.exists():
                issues.append({"path": str(path), "error": "Skipping run: missing aggregate metrics.json"})
            elif data == {}:
                issues.append({"path": str(path), "error": "Skipping run: empty aggregate metrics.json"})
            if data:
                saved_metrics[stage, setting] = data
        if len(saved_metrics) == len(settings):
            candidates.setdefault((run.parent.name, model), []).append((run, manifest, saved_metrics))

    selected = []
    for key in sorted(candidates):
        runs = candidates[key]
        # Job IDs order Slurm submissions numerically, independent of task
        # index or copied-directory timestamps. Legacy/manual groups without
        # comparable job IDs fall back to directory modification time.
        job_ids = {run: _job_id(run) for run, _, _ in runs}
        if all(job_id is not None for job_id in job_ids.values()):
            latest = max(runs, key=lambda item: (job_ids[item[0]], item[0].stat().st_mtime_ns, item[0].name))
        else:
            latest = max(runs, key=lambda item: (item[0].stat().st_mtime_ns, item[0].name))
        selected.append(latest)
    return selected


def _expected_fold_ids(
    directory: Path, metrics: dict, nominal_n_folds: int | None,
    manifest: dict | None = None, setting: str | None = None,
) -> list[int] | None:
    """Resolve populated folds from saved assignments, retaining legacy runs."""
    def validate(values, source):
        if (not isinstance(values, list) or not values
                or any(type(value) is not int or value < 0 for value in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"{source} must contain unique nonnegative integer fold IDs")
        if nominal_n_folds is not None and (
            type(nominal_n_folds) is not int or nominal_n_folds < 1
            or any(value >= nominal_n_folds for value in values)
        ):
            raise ValueError(f"{source} contains IDs outside the configured fold range")
        return sorted(values)

    saved = None
    assignments = directory / "fold_assignments.csv"
    if assignments.is_file():
        with assignments.open(newline="", encoding="utf-8-sig") as handle:
            rows = csv.DictReader(handle)
            if "fold" not in (rows.fieldnames or []):
                raise ValueError(f"Fold assignments require a fold column: {assignments}")
            try:
                values = sorted({int(row["fold"]) for row in rows})
            except (TypeError, ValueError) as error:
                raise ValueError(f"Fold assignments require integer IDs: {assignments}") from error
        saved = validate(values, str(assignments))
    if manifest:
        published = manifest.get("fold_ids", {}).get(setting)
        for source in manifest.get("sources", {}).values():
            value = source.get("settings", {}).get(setting, {}).get("fold_ids")
            if value is not None:
                published = value
        if published is not None:
            published = validate(published, "Manifest fold_ids")
            if saved is not None and saved != published:
                raise ValueError("Manifest fold_ids disagree with saved fold assignments")
            saved = published
    declared = metrics.get("fold_ids")
    if declared is not None:
        declared = validate(declared, "Metric fold_ids")
        if saved is not None and saved != declared:
            raise ValueError("Metric fold_ids disagree with saved fold assignments")
        saved = declared
    if saved is not None:
        return saved
    count = nominal_n_folds if nominal_n_folds is not None else metrics.get("n_folds")
    if count is None:
        return None
    if type(count) is not int or count < 1:
        raise ValueError("Expected fold count must be a positive integer")
    return list(range(count))


def aggregate_results(
    results_dir: str | Path = "results", output_dir: str | Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return tables for the latest eligible run per assay and model.

    Scan results/<assay>/{nkn,kermut}_*/{cv,fit}/<setting>/metrics.json.
    Every discovered or requested setting must have a readable, nonempty JSON
    metrics object. Missing or malformed files exclude the entire run and are
    listed in issues. Manifests and execution plans identify unstarted settings;
    legacy runs without this metadata use their existing setting directories.
    Select the highest numeric Slurm job ID, ignoring the array task index. If
    any eligible run in a group lacks a job ID, use directory modification time.
    NKN and Kermut are selected independently within each assay.

    CV metrics are copied from the selected runner aggregate, not recomputed
    from partial folds. Existing fold and identity checks still annotate its
    rows with their status. Nested mixed singles/doubles metrics become separate
    rows. Null/undefined metrics remain None (empty in CSV).

    If output_dir is supplied, write summary.csv, folds.csv, and issues.csv there.
    Existing report CSVs at that destination are replaced; experiment files are
    never modified. Uses only the Python standard library.
    """
    root = Path(results_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {root}")
    summary, folds, issues = [], [], []
    for run, manifest, saved_metrics in _select_runs(root, issues):
        model = run.name.split("_", 1)[0]
        config = _read(run / "config.json", issues) or {}
        expected = config.get("evaluation", {}).get("n_folds")
        for (stage, setting), data in sorted(saved_metrics.items()):
            directory = run / stage / setting
            path = directory / "metrics.json"
            # The embedded fold records belong to the same snapshot as the
            # aggregate; individual fold files cannot complete an unfinished run.
            fold_data = data.get("fold_metrics", []) if data else []
            variants = data.get("setting_metrics", {setting: data}) if data else {setting: None}
            for label, metrics in sorted(variants.items()):
                identity = dict(zip(IDENTITY, (run.parent.name, model, run.name, stage, label)))
                selected_folds = metrics.get("fold_metrics", []) if metrics else fold_data
                if not metrics and label != setting:
                    selected_folds = [f.get("setting_metrics", {}).get(label, f) for f in fold_data]
                status = "complete" if metrics else ("invalid" if path.exists() else "incomplete")
                expected_ids = None
                if stage == "cv":
                    try:
                        expected_ids = _expected_fold_ids(directory, metrics or {}, expected, manifest, setting)
                    except (OSError, ValueError) as error:
                        status = "invalid"
                        issues.append({"path": str(directory), "error": str(error)})
                if metrics and stage == "cv":
                    ids = [f.get("fold") for f in selected_folds]
                    if status != "invalid" and (
                        expected_ids is None or metrics.get("n_folds") != len(expected_ids)
                        or len(ids) != len(expected_ids) or set(ids) != set(expected_ids)
                    ):
                        status = "incomplete"
                if metrics and (metrics.get("assay_id", run.parent.name) != run.parent.name
                                or metrics.get("setting", label) != label):
                    status = "invalid"
                    issues.append({"path": str(path), "error": "Metric identity disagrees with directory"})
                row = {
                    **identity, "status": status,
                    "expected_folds": len(expected_ids) if expected_ids is not None else None,
                    "completed_folds": len(selected_folds) if stage == "cv" else None,
                    "stopping_criterion": config.get("training", {}).get("stopping_criterion"),
                    "config_path": str(run / "config.json"), "metrics_path": str(path),
                }
                if metrics:
                    row.update(_scalars(metrics))
                summary.append(row)
                for fold in selected_folds:
                    folds.append({**identity, **_scalars(fold), "metrics_path": str(path) if data
                                  else str(directory / f"fold_{fold.get('fold')}" / "metrics.json")})
    tables = {"summary": summary, "folds": folds, "issues": issues}
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        _write_csv(output / "summary.csv", summary, (*IDENTITY, "status", *METRICS))
        _write_csv(output / "folds.csv", folds, (*IDENTITY, "fold", "spearman", "interval_ece", "pair_ece"))
        _write_csv(output / "issues.csv", issues, ("path", "error"))
    return tables
