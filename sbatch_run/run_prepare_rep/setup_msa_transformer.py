"""Step-1 checkpoint download and runtime validation; no assay preparation."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/setup/msa_transformer_runtime.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_checkpoint(runtime_dir, spec):
    path = runtime_dir / spec["filename"]
    manifest_path = runtime_dir / "checkpoint_manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError("No complete checkpoint and manifest; run the download stage first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {key: spec[key] for key in ("model_id", "filename", "url", "size_bytes")}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint manifest does not match the configured artifact")
    if path.stat().st_size != spec["size_bytes"] or sha256(path) != manifest.get("sha256"):
        raise ValueError("Checkpoint checksum or size mismatch; existing files were not replaced")
    return manifest


def download_checkpoint(runtime_dir, spec):
    runtime_dir.mkdir(parents=True, exist_ok=True)
    destination = runtime_dir / spec["filename"]
    manifest_path = runtime_dir / "checkpoint_manifest.json"
    if destination.exists() or manifest_path.exists():
        return verify_checkpoint(runtime_dir, spec)
    print(f"Downloading {spec['model_id']} ({spec['size_bytes'] / 1e9:.2f} GB)", flush=True)
    digest = hashlib.sha256()
    size = 0
    with tempfile.NamedTemporaryFile(dir=runtime_dir, prefix=".checkpoint-", suffix=".part", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with urllib.request.urlopen(spec["url"], timeout=120) as response:
                if not response.geturl().startswith("https://"):
                    raise ValueError("Checkpoint download redirected away from HTTPS")
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    size += len(block)
                    if size > spec["size_bytes"]:
                        raise ValueError("Checkpoint exceeds the expected size")
                    handle.write(block)
                    digest.update(block)
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        if size != spec["size_bytes"]:
            raise ValueError("Incomplete checkpoint download")
        manifest = {key: spec[key] for key in ("model_id", "filename", "url", "size_bytes")}
        manifest.update(sha256=digest.hexdigest(), downloaded_at=datetime.now(timezone.utc).isoformat(),
                        checksum_origin="computed_from_first_official_https_download")
        temporary.replace(destination)
        write_json(manifest_path, manifest)
        return manifest
    finally:
        temporary.unlink(missing_ok=True)


def validate_runtime(runtime_dir, config, device, *, return_model=False):
    if f"{sys.version_info.major}.{sys.version_info.minor}" != config["python_version"]:
        raise ValueError(f"Activate {config['environment_name']}: Python {config['python_version']} is required")
    installed = {name: importlib.metadata.version(name) for name in config["packages"]}
    if installed != config["packages"]:
        mismatches = {name: {"expected": version, "actual": installed[name]}
                      for name, version in config["packages"].items() if installed[name] != version}
        raise ValueError(f"Runtime package versions differ from the pinned environment: {mismatches}")
    source = runtime_dir / f"ProteinGym-{config['proteingym_commit']}"
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    if revision != config["proteingym_commit"] or dirty:
        raise ValueError("ProteinGym source must be clean and at the pinned commit")
    hhfilter = shutil.which("hhfilter")
    if hhfilter is None:
        raise ValueError("hhfilter is missing from PATH")
    help_result = subprocess.run([hhfilter, "-h"], text=True, capture_output=True, check=True)
    help_text = help_result.stdout + help_result.stderr
    if config["hhfilter_version"] not in help_text:
        raise ValueError("hhfilter does not report the pinned version")
    manifest = verify_checkpoint(runtime_dir, config["checkpoint"])

    # ProteinGym's bundled code uses all three import roots. Avoid this
    # project's top-level baselines package and any installed esm package.
    sys.path[:0] = [str(source / "proteingym/baselines/esm"), str(source / "proteingym"), str(source)]
    import esm
    import torch

    if not Path(esm.__file__).resolve().is_relative_to(source.resolve()):
        raise ValueError("A different esm package is already imported; use a fresh process")
    if torch.version.cuda != config["cuda_runtime"]:
        raise ValueError("PyTorch CUDA runtime differs from the pinned environment")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; run validation inside a Phoenix GPU allocation")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.manual_seed(0)
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(runtime_dir / config["checkpoint"]["filename"])
    if model.num_layers != config["checkpoint"]["num_layers"] or model.args.embed_dim != config["checkpoint"]["embedding_dim"]:
        raise ValueError("Loaded model architecture differs from the configured MSA Transformer")
    model.requires_grad_(False).eval().to(device)
    # Artificial alphabet-only rows: this check does not read any assay/MSA.
    msa = [("query", "ACDEFGHIKLMNPQRSTVWY"), ("context", "ACDEFGHIKLMNPQRSTVWY")]
    _, _, tokens = alphabet.get_batch_converter()([msa])
    with torch.inference_mode():
        output = model(tokens.to(device), repr_layers=[model.num_layers])
    embedding = output["representations"][model.num_layers]
    logits = output["logits"]
    if tuple(embedding.shape) != (*tokens.shape, 768) or tuple(logits.shape) != (*tokens.shape, len(alphabet)):
        raise ValueError("Unexpected representation/logit shapes in the runtime test")
    if not torch.isfinite(embedding).all() or not torch.isfinite(logits).all():
        raise ValueError("Nonfinite model output in the runtime test")
    report = {
        "status": "runtime_validated", "validated_at": datetime.now(timezone.utc).isoformat(),
        "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "python": sys.version, "packages": installed, "torch_cuda": torch.version.cuda,
        "proteingym_commit": revision, "esm_source": str(Path(esm.__file__).resolve()),
        "hhfilter": hhfilter, "hhfilter_version": config["hhfilter_version"], "hhfilter_sha256": sha256(hhfilter),
        "checkpoint": manifest, "representation_shape": list(embedding.shape), "logit_shape": list(logits.shape),
        "runtime_config_sha256": sha256(CONFIG_PATH),
        "environment_file_sha256": sha256(PROJECT_ROOT / "env_msa_transformer.yaml"),
        "historical_msa_filtering_verified": False,
    }
    if return_model:
        # Preparation arrays reuse the validated model without racing to rewrite
        # the shared setup report or loading a second checkpoint into GPU memory.
        return report, model, alphabet
    write_json(runtime_dir / "runtime_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "validate"))
    parser.add_argument("--runtime-dir", type=Path, default=PROJECT_ROOT / "data/models/msa_transformer")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    runtime_dir = args.runtime_dir.resolve()
    result = (download_checkpoint(runtime_dir, config["checkpoint"]) if args.command == "download"
              else validate_runtime(runtime_dir, config, args.device))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
