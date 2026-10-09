"""Explicit pinned downloads and independently verified offline checkpoints."""

import hashlib
import json
from pathlib import Path
import re

from .config import ExtractionConfig, MODEL_PRESETS


MANIFEST_NAME = "checkpoint_manifest.json"
ALLOW_PATTERNS = (
    "config.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
    "tokenizer.json", "model.safetensors", "model.safetensors.index.json", "model-*.safetensors",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_path(config: ExtractionConfig) -> Path:
    return config.model_dir / config.model_id.split("/")[-1] / config.revision


def _required_files(path: Path, names: set[str]):
    required = {"config.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"}
    if not required.issubset(names):
        raise ValueError(f"Checkpoint missing model/tokenizer files: {sorted(required - names)}")
    if "model.safetensors" not in names:
        if "model.safetensors.index.json" not in names:
            raise ValueError("Checkpoint requires safetensors weights")
        index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
        shards = set(index.get("weight_map", {}).values())
        if not shards or not shards.issubset(names) or any(not name.endswith(".safetensors") for name in shards):
            raise ValueError("Checkpoint safetensors shards are incomplete")


def checkpoint_manifest(config: ExtractionConfig) -> dict:
    """Read and verify the completion manifest, hashes, and pinned identity offline."""
    path = checkpoint_path(config)
    manifest_path = path / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No completed local checkpoint at {path}; run the explicit download command first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version"), manifest.get("model_id"), manifest.get("revision")) != (1, config.model_id, config.revision):
        raise ValueError("Local checkpoint identity does not match extraction configuration")
    files = manifest.get("files_sha256")
    if not isinstance(files, dict) or not files:
        raise ValueError("Local checkpoint manifest has no file hashes")
    local_names = {item.name for pattern in ALLOW_PATTERNS for item in path.glob(pattern) if item.is_file()}
    if set(files) != local_names:
        raise ValueError("Local checkpoint files do not exactly match the verified manifest")
    for name, expected in files.items():
        if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name or ":" in name:
            raise ValueError("Checkpoint manifest filenames must be local basenames")
        item = path / name
        if not item.resolve().is_relative_to(path.resolve()):
            raise ValueError("Checkpoint file escapes the local checkpoint directory")
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(f"Invalid checkpoint SHA256: {name}")
        if not item.is_file() or sha256(item) != expected:
            raise ValueError(f"Local checkpoint checksum mismatch: {name}")
    _required_files(path, set(files))
    preset = MODEL_PRESETS[config.model_id]
    if config.revision == preset["revision"] and files.get("model.safetensors") != preset["weights_sha256"]:
        raise ValueError("Local weights do not match the published pinned checkpoint SHA256")
    return manifest


def local_checkpoint(config: ExtractionConfig) -> Path:
    checkpoint_manifest(config)
    return checkpoint_path(config)


def download(config: ExtractionConfig) -> Path:
    """Download only explicit safetensors/model/tokenizer artifacts, then mark complete."""
    path = checkpoint_path(config)
    if (path / MANIFEST_NAME).exists():
        return local_checkpoint(config)
    path.mkdir(parents=True, exist_ok=True)
    names = {item.name for pattern in ALLOW_PATTERNS for item in path.glob(pattern) if item.is_file()}
    preset = MODEL_PRESETS[config.model_id]
    try:
        _required_files(path, names)
        staged = config.revision == preset["revision"]
    except ValueError:
        staged = False
    if not staged:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError("Install the dependencies in env.yaml before downloading checkpoints") from exc
        snapshot_download(
            repo_id=config.model_id, revision=config.revision, local_dir=str(path),
            allow_patterns=list(ALLOW_PATTERNS),
        )
        names = {item.name for pattern in ALLOW_PATTERNS for item in path.glob(pattern) if item.is_file()}
        try:
            _required_files(path, names)
        except ValueError as exc:
            raise RuntimeError(
                f"Checkpoint download is incomplete at {path}. The Hub client may have returned "
                "an incomplete local directory after a network/TLS failure. Retry the explicit "
                "download with working Hugging Face access, or stage the model/tokenizer files "
                "from the configured exact revision here and rerun download to verify them. "
                f"Details: {exc}"
            ) from exc
    manifest = {
        "schema_version": 1, "model_id": config.model_id, "revision": config.revision,
        "files_sha256": {name: sha256(path / name) for name in sorted(names)},
    }
    if config.revision == preset["revision"] and manifest["files_sha256"].get("model.safetensors") != preset["weights_sha256"]:
        raise ValueError(
            f"Weights at {path / 'model.safetensors'} do not match the published checkpoint SHA256. "
            "Replace this incomplete or incorrect file with model.safetensors from the exact "
            "configured Hugging Face revision, then rerun download; no completion manifest was written."
        )
    temporary = path / f"{MANIFEST_NAME}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path / MANIFEST_NAME)
    return path
