"""Shared validation and provenance for frozen native inverse-folding adapters."""

from importlib.metadata import PackageNotFoundError, version
import hashlib
import json

import torch

from ..inverse_checkpoints import checkpoint_manifest, checkpoint_path
from ..inverse_config import ALPHABET, InverseFoldingConfig, PRESETS


class InverseFoldingAdapter:
    dependency_names = ("torch", "numpy")

    def __init__(self, config: InverseFoldingConfig):
        self.config = config
        self._model = None
        self._manifest = None
        self._device = None

    @property
    def device(self) -> str:
        if self._device is not None:
            return str(self._device)
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    def _checkpoint(self):
        if self._manifest is None:
            self._manifest = checkpoint_manifest(self.config)
        return checkpoint_path(self.config)

    def _select_device(self):
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        self._device = torch.device(self.device)

    def metadata(self) -> dict:
        self._checkpoint()
        try:
            versions = {name: version(name) for name in self.dependency_names}
        except PackageNotFoundError as exc:
            raise ImportError("Install inverse folding dependencies from env.yaml or requirements.txt") from exc
        return {
            "feature_type": "inverse_folding", "model_family": self.config.model_family,
            "checkpoint": self.config.cache_id, "model_id": self.config.model_id,
            "revision": self.config.revision, "synthetic": False,
            "generation_method": "pretrained_frozen", "backend": "pinned_upstream_source",
            "upstream_repository": PRESETS[self.config.model_family]["upstream_repository"],
            "source_revision": self.config.revision,
            "checkpoint_sha256": hashlib.sha256(json.dumps(self._manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "checkpoint_files_sha256": dict(self._manifest["files_sha256"]),
            "library_versions": versions, "conditioning_mode": self.config.conditioning_mode,
            "alphabet": ALPHABET, "temperature": 1.0,
            "normalization": "canonical_amino_acids_only",
            "noncanonical_tokens": "excluded_then_renormalized",
            "precision_policy": "float32_log_softmax_reject_probability_underflow",
            "inference_dtype": self.config.inference_dtype, "dtype": self.config.output_dtype,
            "residue_indexing": "one_based_reference_position", "structure_context": "selected_single_chain",
            "max_length": self.config.max_length, "truncation": "reject", "seed": self.config.seed,
            "backbone_noise": 0.0,
        }

    def _coords(self, structure, required_atoms: int):
        sequence = structure.reference_sequence
        if not isinstance(sequence, str) or not sequence or set(sequence) - set(ALPHABET):
            raise ValueError("Inverse folding requires a nonempty canonical reference sequence")
        if self.config.max_length is not None and len(sequence) > self.config.max_length:
            raise ValueError(f"Structure length exceeds max_length={self.config.max_length}; truncation is disabled")
        coords = torch.as_tensor(structure.coords, dtype=torch.float32)
        if tuple(coords.shape) != (len(sequence), 4, 3):
            raise ValueError("Structure coordinates must have shape [reference_length, 4, 3] in N/CA/C/O order")
        coords = coords[:, :required_atoms, :]
        if not torch.isfinite(coords).all():
            atoms = "N, CA, C, O" if required_atoms == 4 else "N, CA, C"
            raise ValueError(f"Every reference residue requires finite {atoms} backbone coordinates")
        return coords

    @staticmethod
    def _probabilities(values, indices, length):
        if values.ndim != 2 or values.shape[0] != length:
            raise ValueError("Native inverse folding output does not align with reference residues")
        selected = values[:, indices].detach().to(device="cpu", dtype=torch.float32)
        if tuple(selected.shape) != (length, 20) or not torch.isfinite(selected).all():
            raise ValueError("Native inverse folding output must contain 20 finite canonical values per position")
        log_probs = torch.log_softmax(selected, dim=-1)
        probabilities = log_probs.exp()
        if not torch.isfinite(log_probs).all() or (probabilities <= 0).any():
            raise ValueError("Canonical amino-acid probabilities underflowed float32; no probability floor is applied")
        return {"alphabet": ALPHABET, "probabilities": probabilities.contiguous(), "log_probs": log_probs.contiguous()}
