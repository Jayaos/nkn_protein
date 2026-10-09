"""Download and verify pinned ProSST weights, custom model code, and quantizer."""

import json
from pathlib import Path
from uuid import uuid4

from .checkpoints import sha256
from .downloads import download_verified
from .prosst_config import PRESET, ProSSTConfig

MANIFEST_NAME = "checkpoint_manifest.json"


def checkpoint_path(config: ProSSTConfig) -> Path:
    return config.model_dir / config.model_id.split("/")[-1] / config.revision


def quantizer_preset() -> dict:
    return json.loads(Path(__file__).with_name("prosst_quantizer_artifacts.json").read_text(encoding="utf-8"))


def artifacts() -> dict:
    quantizer = quantizer_preset()["artifacts"]
    if set(PRESET["artifacts"]) & set(quantizer):
        raise ValueError("Model and quantizer artifact paths overlap")
    return dict(PRESET["artifacts"], **quantizer)


def _expected_manifest(config: ProSSTConfig) -> dict:
    inventory = artifacts()
    quantizer = quantizer_preset()
    return {
        "schema_version": 1, "model_family": "prosst", "model_id": config.model_id,
        "revision": config.revision, "upstream_repository": PRESET["upstream_repository"],
        "quantizer_revision": quantizer["revision"],
        "quantizer_repository": quantizer["upstream_repository"],
        "files_sha256": {name: item["sha256"] for name, item in inventory.items()},
        "artifact_urls": {name: item["url"] for name, item in inventory.items()},
    }


def _verify_artifacts(config: ProSSTConfig) -> None:
    path = checkpoint_path(config)
    inventory = artifacts()
    for name, item in inventory.items():
        local = path / name
        if not local.resolve().is_relative_to(path.resolve()):
            raise ValueError("ProSST artifact escapes the checkpoint directory")
        if not local.is_file() or local.stat().st_size != item["size_bytes"] or sha256(local) != item["sha256"]:
            raise ValueError(f"Local ProSST checkpoint checksum mismatch: {name}")
    # Execute only our locked upstream source inventory, without shadow modules.
    source_files = {
        item.relative_to(path).as_posix() for item in path.rglob("*")
        if item.is_file() and item.suffix in (".py", ".pyc", ".pyo", ".pyd")
    }
    expected_source = {name for name in inventory if Path(name).suffix == ".py"}
    if source_files != expected_source:
        raise ValueError("ProSST source files do not exactly match the pinned source inventory")


def checkpoint_manifest(config: ProSSTConfig) -> dict:
    manifest_path = checkpoint_path(config) / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No completed ProSST checkpoint at {manifest_path.parent}; run download first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest != _expected_manifest(config):
        raise ValueError("ProSST checkpoint manifest does not match the pinned model/source identity")
    _verify_artifacts(config)
    return manifest


def local_checkpoint(config: ProSSTConfig) -> Path:
    checkpoint_manifest(config)
    return checkpoint_path(config)


def download(config: ProSSTConfig) -> Path:
    path = checkpoint_path(config)
    if (path / MANIFEST_NAME).exists():
        return local_checkpoint(config)
    path.mkdir(parents=True, exist_ok=True)
    for name, item in artifacts().items():
        download_verified(item["url"], path / name, item["sha256"])
    _verify_artifacts(config)
    temporary = path / f".{MANIFEST_NAME}.{uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(_expected_manifest(config), indent=2) + "\n", encoding="utf-8")
        temporary.replace(path / MANIFEST_NAME)
    finally:
        temporary.unlink(missing_ok=True)
    return path
