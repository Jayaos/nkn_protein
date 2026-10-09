"""Atomic HTTPS downloads with SHA256 verification and native Windows transport."""

import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from .checkpoints import sha256


def download_verified(url: str, destination: Path, expected_sha256: str) -> Path:
    """Download an explicit HTTPS artifact; reuse only a checksum-matching local file."""
    if not isinstance(url, str) or urlparse(url).scheme != "https" or not urlparse(url).hostname:
        raise ValueError("Artifact downloads require an explicit HTTPS URL")
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Artifact downloads require a lowercase SHA256 checksum")
    destination = Path(destination)
    if destination.is_file() and sha256(destination) == expected_sha256:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.partial")
    try:
        try:
            request = Request(url, headers={"User-Agent": "nkn-protein-feature-generation/1"})
            with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        except (OSError, ValueError) as exc:
            if os.name != "nt":
                raise RuntimeError(f"Could not download {url}: {exc}") from exc
            # No interpolated shell input and no execution-policy changes. URLs and
            # destination are data passed through separate environment variables.
            command = (
                "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; "
                "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; "
                "$client=New-Object Net.WebClient; "
                "try { $client.DownloadFile($env:NKN_ARTIFACT_URL, $env:NKN_ARTIFACT_PATH) } "
                "finally { $client.Dispose() }"
            )
            environment = dict(os.environ, NKN_ARTIFACT_URL=url, NKN_ARTIFACT_PATH=str(temporary.resolve()))
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                env=environment, capture_output=True, text=True, check=False, timeout=3600,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode:
                raise RuntimeError(f"Could not download {url}: {result.stderr.strip()}") from exc
        if sha256(temporary) != expected_sha256:
            raise ValueError(f"Downloaded artifact SHA256 mismatch: {url}; no checkpoint file replaced")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
