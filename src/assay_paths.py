"""Resolve ProteinGym assay tables across supported extracted-data layouts."""

from pathlib import Path


def resolve_assay_csv(index_path: str | Path, filename: str) -> Path:
    """Find a catalog entry beside the index or in known extraction folders."""
    root = Path(index_path).resolve().parent
    if not isinstance(filename, str) or not filename:
        raise ValueError("Assay filename must be nonempty")
    searched = []
    for directory in (root, root / "raw_substitutions",
                      root / "DMS_ProteinGym_substitutions",
                      root / "DMS_ProteinGym_substitutions" / "raw_substitutions"):
        candidate = (directory / filename).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("Assay filename must stay inside the index directory")
        if candidate.is_file():
            return candidate
        searched.append(str(candidate))
    raise FileNotFoundError(f"Assay CSV not found; searched: {', '.join(searched)}")
