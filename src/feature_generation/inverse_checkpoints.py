"""Pinned, independently verified native source code and model checkpoints."""

import hashlib
import json
from pathlib import Path
from uuid import uuid4

from .checkpoints import sha256
from .downloads import download_verified
from .inverse_config import InverseFoldingConfig, PRESETS

MANIFEST_NAME = "checkpoint_manifest.json"


def checkpoint_path(config: InverseFoldingConfig) -> Path:
    return config.model_dir / config.model_id.split("/")[-1] / config.revision


def _expected_manifest(config: InverseFoldingConfig) -> dict:
    preset = PRESETS[config.model_family]
    return {
        "schema_version": 1, "model_family": config.model_family,
        "model_id": config.model_id, "revision": config.revision,
        "upstream_repository": preset["upstream_repository"],
        "weights_filename": preset["weights_filename"],
        "files_sha256": {name: artifact["sha256"] for name, artifact in preset["artifacts"].items()},
        "artifact_urls": {name: artifact["url"] for name, artifact in preset["artifacts"].items()},
    }


def _verify_artifacts(config: InverseFoldingConfig):
    path = checkpoint_path(config)
    artifacts = PRESETS[config.model_family]["artifacts"]
    for name, artifact in artifacts.items():
        item = path / name
        if not item.resolve().is_relative_to(path.resolve()):
            raise ValueError("Checkpoint artifact escapes its directory")
        if not item.is_file() or item.stat().st_size != artifact["size_bytes"] or sha256(item) != artifact["sha256"]:
            raise ValueError(f"Local inverse folding checkpoint checksum mismatch: {name}")
    # Source is executed only after verification. Reject extra modules/bytecode that
    # could alter Python's resolution of the pinned upstream imports.
    source_names = {item.relative_to(path).as_posix() for item in (path / "source").rglob("*") if item.is_file()}
    if source_names != {name for name in artifacts if name.startswith("source/")}:
        raise ValueError("Local checkpoint source files do not exactly match the pinned source inventory")


def checkpoint_manifest(config: InverseFoldingConfig) -> dict:
    manifest_path = checkpoint_path(config) / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No completed local checkpoint at {manifest_path.parent}; run download first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest != _expected_manifest(config):
        raise ValueError("Local checkpoint manifest does not match the pinned model/source identity")
    _verify_artifacts(config)
    return manifest


def local_checkpoint(config: InverseFoldingConfig) -> Path:
    checkpoint_manifest(config)
    return checkpoint_path(config)


def download(config: InverseFoldingConfig) -> Path:
    path = checkpoint_path(config)
    if (path / MANIFEST_NAME).exists():
        return local_checkpoint(config)
    path.mkdir(parents=True, exist_ok=True)
    for name, artifact in PRESETS[config.model_family]["artifacts"].items():
        download_verified(artifact["url"], path / name, artifact["sha256"])
    _verify_artifacts(config)
    temporary = path / f".{MANIFEST_NAME}.{uuid4().hex}.tmp"
    temporary.write_text(json.dumps(_expected_manifest(config), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path / MANIFEST_NAME)
    return path
