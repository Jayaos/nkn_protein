"""Collection names and paths shared by structure-conditioned extractors."""

from pathlib import Path

from .config import PROJECT_ROOT


STRUCTURE_SOURCES = ("proteingym", "protssn_proteingym")


def structure_root(source: str) -> Path:
    if source not in STRUCTURE_SOURCES:
        raise ValueError(f"structure_source must be one of {STRUCTURE_SOURCES}")
    return PROJECT_ROOT / "data/structures" / source


def structure_path(source: str, value: str | Path | None, relative: str) -> Path:
    """Resolve a source default or a custom path; reject crossed source roots."""
    root = structure_root(source).resolve()
    if value is None:
        return root / relative
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"Structure {relative} directory must be a nonempty path")
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    for other in STRUCTURE_SOURCES:
        if other != source and path.is_relative_to(structure_root(other).resolve()):
            raise ValueError(f"Structure directory belongs to {other}, not structure_source={source}")
    return path


def structure_cache_id(source: str, recipe: str) -> str:
    """Name caches by model recipe; structure identity stays in their metadata."""
    structure_root(source)
    return recipe


def validate_structure_cache_id(source: str, cache_id: str) -> None:
    """Do not let a copied preset label another collection's output as its own."""
    for other in STRUCTURE_SOURCES:
        if other != source and cache_id.startswith(other + "--"):
            raise ValueError(f"Cache ID belongs to {other}, not structure_source={source}")
