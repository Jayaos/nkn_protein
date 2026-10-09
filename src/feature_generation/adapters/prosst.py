"""Frozen ProSST hidden states from unmasked sequences and fixed structure tokens."""

from collections.abc import Iterator, Sequence
from importlib.metadata import PackageNotFoundError, version
from importlib.util import spec_from_file_location
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import torch
from torch import Tensor

from ..prosst_checkpoints import checkpoint_manifest, checkpoint_path
from ..prosst_config import PRESET, ProSSTConfig

ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")


def import_pinned_model(path: Path, revision: str):
    """Import already-verified source without remote execution or bytecode caches."""
    package_name = f"_nkn_prosst_{revision}"
    package = sys.modules.get(package_name)
    if package is None:
        package = ModuleType(package_name)
        package.__path__ = []
        package.__package__ = package_name
        sys.modules[package_name] = package
    for basename in ("configuration_prosst", "modeling_prosst"):
        name = f"{package_name}.{basename}"
        source = (path / f"{basename}.py").resolve()
        module = sys.modules.get(name)
        if module is not None:
            if Path(module.__file__).resolve() != source:
                raise RuntimeError("A different local ProSST source directory is already imported; use a fresh process")
            continue
        module = ModuleType(name)
        module.__file__ = str(source)
        module.__package__ = package_name
        module.__spec__ = spec_from_file_location(name, source)
        sys.modules[name] = module
        try:
            exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        setattr(package, basename, module)
    return sys.modules[f"{package_name}.configuration_prosst"], sys.modules[f"{package_name}.modeling_prosst"]


class ProSSTAdapter:
    def __init__(self, config: ProSSTConfig):
        if config.model_family != "prosst":
            raise ValueError("ProSSTAdapter requires model_family=prosst")
        self.config = config
        self._model = None
        self._tokenizer = None
        self._device = None
        self._manifest = None
        self._structure_tokens = None

    @property
    def hidden_size(self) -> int:
        return PRESET["hidden_size"]

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

    def metadata(self) -> dict:
        self._checkpoint()
        try:
            versions = {name: version(name) for name in ("transformers", "torch", "safetensors")}
        except PackageNotFoundError as exc:
            raise ImportError("Install ProSST dependencies from env.yaml or requirements.txt") from exc
        manifest_digest = hashlib.sha256(
            json.dumps(self._manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "feature_type": "sequence", "model_family": "prosst", "checkpoint": self.config.cache_id,
            "synthetic": False, "generation_method": "pretrained_frozen",
            "model_id": self.config.model_id, "revision": self.config.revision,
            "checkpoint_sha256": manifest_digest,
            "checkpoint_files_sha256": dict(self._manifest["files_sha256"]),
            "backend": "transformers_pinned_upstream_source", "library_versions": versions,
            "model_class": "ProSSTForMaskedLM", "tokenizer_class": "EsmTokenizer",
            "upstream_repository": PRESET["upstream_repository"], "source_revision": self.config.revision,
            "quantizer_revision": self._manifest["quantizer_revision"],
            "layer": self.config.layer, "embedding_dim": self.hidden_size, "hidden_size": self.hidden_size,
            "inference_dtype": self.config.inference_dtype, "dtype": self.config.output_dtype,
            "pooling": "none", "includes_special_tokens": False,
            "residue_indexing": "one_based_reference_position", "truncation": "reject",
            "max_length": self.config.max_length, "max_length_policy": "configured_memory_guard",
            "conditioning_mode": "sequence_and_fixed_reference_structure",
            "inference_method": "unmasked_hidden_states", "amino_acid_masking": "none",
            "structure_context": "selected_single_chain_fixed_reference_backbone",
            "structure_vocab_size": PRESET["structure_vocab_size"],
            "structure_token_offset": 3, "structure_special_tokens": {"pad": 0, "bos": 1, "eos": 2},
        }

    def load(self) -> None:
        if self._model is not None:
            return
        path = self._checkpoint() / "model"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        try:
            from transformers import EsmTokenizer
            config_module, model_module = import_pinned_model(path, self.config.revision)
        except ImportError as exc:
            raise ImportError("Install ProSST dependencies from env.yaml or requirements.txt") from exc
        tokenizer = EsmTokenizer.from_pretrained(str(path), local_files_only=True)
        architecture = config_module.ProSSTConfig.from_pretrained(str(path), local_files_only=True)
        if (architecture.hidden_size, architecture.num_hidden_layers, architecture.ss_vocab_size) != (768, 12, 2051):
            raise ValueError("ProSST checkpoint architecture does not match the pinned recipe")
        # Upstream construction initializes modules on CPU; restore the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            model, loading_info = model_module.ProSSTForMaskedLM.from_pretrained(
                str(path), config=architecture, local_files_only=True, use_safetensors=True,
                torch_dtype=torch.float32, output_loading_info=True,
            )
        if any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"ProSST checkpoint did not load exactly: {loading_info}")
        if (tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.eos_token_id) != (0, 1, 2):
            raise ValueError("ProSST tokenizer special tokens do not match the pinned recipe")
        self._device = torch.device(self.device)
        model.requires_grad_(False).eval().to(self._device)
        self._model, self._tokenizer = model, tokenizer

    def set_structure_tokens(self, tokens: Tensor) -> None:
        if not isinstance(tokens, Tensor) or tokens.dtype != torch.long or tokens.ndim != 1 or not tokens.numel():
            raise ValueError("Structure tokens must be a nonempty torch.long tensor with shape [L]")
        if tokens.min().item() < 0 or tokens.max().item() >= PRESET["structure_vocab_size"]:
            raise ValueError("Structure tokens must be raw codebook indices between 0 and 2047")
        self._structure_tokens = tokens.detach().to(device="cpu").clone()

    def _batches(self, sequences: Sequence[str]) -> Iterator[list[int]]:
        if not sequences:
            raise ValueError("No sequences supplied for ProSST extraction")
        if self._structure_tokens is None:
            raise ValueError("Bind validated structure tokens before ProSST extraction")
        length = self._structure_tokens.numel()
        for sequence in sequences:
            if not isinstance(sequence, str) or not sequence or set(sequence) - ALPHABET:
                raise ValueError("ProSST requires nonempty canonical amino-acid sequences")
            if len(sequence) != length:
                raise ValueError("Every substitution sequence must align exactly with the fixed structure tokens")
            if length > self.config.max_length:
                raise ValueError(f"Sequence exceeds max_length={self.config.max_length}; truncation is disabled")
            if length + 2 > self.config.max_tokens_per_batch:
                raise ValueError("One sequence exceeds max_tokens_per_batch, including BOS/EOS tokens")
        batch_size = self.config.max_tokens_per_batch // (length + 2)
        for start in range(0, len(sequences), batch_size):
            yield list(range(start, min(start + batch_size, len(sequences))))

    def iter_embeddings(self, sequences: Sequence[str]) -> Iterator[tuple[list[int], Tensor]]:
        batches = list(self._batches(sequences))
        self.load()
        length = self._structure_tokens.numel()
        structure_ids = torch.cat((torch.tensor([1]), self._structure_tokens + 3, torch.tensor([2])))
        for indices in batches:
            strings = [sequences[index] for index in indices]
            tokens = self._tokenizer(
                strings, add_special_tokens=True, padding=True, truncation=False,
                return_attention_mask=True, return_special_tokens_mask=True, return_tensors="pt",
            )
            input_ids, attention_mask = tokens["input_ids"], tokens["attention_mask"]
            residue_mask = attention_mask.bool() & ~tokens["special_tokens_mask"].bool()
            if tuple(input_ids.shape) != (len(indices), length + 2) or input_ids.shape != attention_mask.shape or input_ids.shape != residue_mask.shape:
                raise ValueError("ProSST tokenizer must yield exactly BOS, L residues, and EOS")
            if not attention_mask.bool().all() or not torch.all(residue_mask[:, 1:-1]) or residue_mask[:, (0, -1)].any():
                raise ValueError("ProSST amino-acid and structure special tokens are not aligned")
            if not torch.all(input_ids[:, 0] == self._tokenizer.cls_token_id) or not torch.all(input_ids[:, -1] == self._tokenizer.eos_token_id):
                raise ValueError("ProSST amino-acid BOS/EOS tokens are not aligned")
            for row, sequence in enumerate(strings):
                if self._tokenizer.convert_ids_to_tokens(input_ids[row, 1:-1].tolist()) != list(sequence):
                    raise ValueError("ProSST tokenizer residues do not match the complete unmasked sequence")
            with torch.inference_mode():
                output = self._model.prosst(
                    input_ids=input_ids.to(self._device), attention_mask=attention_mask.to(self._device),
                    ss_input_ids=structure_ids.unsqueeze(0).expand(len(indices), -1).to(self._device),
                    output_hidden_states=self.config.layer != PRESET["num_layers"], return_dict=True,
                )
                hidden = output.last_hidden_state if self.config.layer == PRESET["num_layers"] else output.hidden_states[self.config.layer]
                if hidden.ndim != 3 or tuple(hidden.shape) != (len(indices), length + 2, self.hidden_size):
                    raise ValueError("ProSST returned an unexpected representation shape")
                values = hidden[:, 1:-1].detach().to(device="cpu", dtype=torch.float32).clone().contiguous()
                del hidden, output
            if not torch.isfinite(values).all():
                raise ValueError("ProSST embeddings contain nonfinite values")
            yield indices, values
