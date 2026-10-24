"""Frozen Hugging Face ESM2 embeddings with strictly offline inference."""

from collections.abc import Iterator, Sequence
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json

import torch
from torch import Tensor

from .window_embeddings import WindowedEmbeddings

from ..checkpoints import checkpoint_manifest, checkpoint_path
from ..config import ExtractionConfig, MODEL_PRESETS


ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")


class ESM2Adapter(WindowedEmbeddings):
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._device = None
        self._manifest = None

    @property
    def hidden_size(self) -> int:
        return MODEL_PRESETS[self.config.model_id]["hidden_size"]

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
            raise ImportError("Install the dependencies in env.yaml before extracting features") from exc
        manifest_digest = hashlib.sha256(
            json.dumps(self._manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        metadata = {
            "feature_type": "sequence", "model_family": "esm2", "checkpoint": self.config.cache_id,
            "synthetic": False, "generation_method": "pretrained_frozen",
            "model_id": self.config.model_id, "revision": self.config.revision,
            "checkpoint_sha256": manifest_digest,
            "checkpoint_files_sha256": dict(self._manifest["files_sha256"]),
            "backend": "transformers", "library_versions": versions,
            "model_class": "EsmModel", "tokenizer_class": "EsmTokenizer",
            "layer": self.config.layer, "embedding_dim": self.hidden_size, "hidden_size": self.hidden_size,
            "inference_dtype": self.config.inference_dtype, "dtype": self.config.output_dtype,
            "pooling": self.config.pooling, "includes_special_tokens": False,
            "residue_indexing": "one_based_reference_position", "truncation": self.config.truncation,
            "max_length": self.config.max_length,
        }
        if self.config.site_embeddings == "centered":
            metadata.update(
                site_embeddings="centered", cache_layout="mutation_sites_v1",
                site_context="mutated_sequence", window_policy="centered_clamped",
                window_indexing="zero_based_half_open",
            )
        if self.config.global_window == "non_overlapping":
            metadata.update(
                global_window="non_overlapping", global_window_aggregation="length_weighted_mean",
            )
        return metadata

    def load(self) -> None:
        if self._model is not None:
            return
        path = self._checkpoint()
        try:
            from transformers import EsmModel, EsmTokenizer
        except ImportError as exc:
            raise ImportError("Install the dependencies in env.yaml before extracting features") from exc
        device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if device == "cpu" and self.config.inference_dtype == "float16":
            raise ValueError("Use float32 or bfloat16 for CPU extraction")
        if device == "cuda" and self.config.inference_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA device does not support bfloat16; choose float32 or float16")
        self._device = torch.device(device)
        tokenizer = EsmTokenizer.from_pretrained(str(path), local_files_only=True)
        model = EsmModel.from_pretrained(
            str(path), local_files_only=True, use_safetensors=True, add_pooling_layer=False,
            torch_dtype=getattr(torch, self.config.inference_dtype),
        )
        preset = MODEL_PRESETS[self.config.model_id]
        if model.config.hidden_size != self.hidden_size or model.config.num_hidden_layers != preset["num_layers"]:
            raise ValueError("Local model architecture does not match the selected ESM2 checkpoint")
        model.requires_grad_(False)
        model.eval()
        model.to(self._device)
        self._tokenizer, self._model = tokenizer, model

    def _validate_sequences(self, sequences: Sequence[str]) -> int:
        if isinstance(sequences, (str, bytes)) or not sequences:
            raise ValueError("No sequences supplied for extraction")
        full_length = None
        for sequence in sequences:
            if not isinstance(sequence, str) or not sequence or set(sequence) - ALPHABET:
                raise ValueError("ESM2 extraction requires nonempty canonical amino-acid sequences")
            if len(sequence) > self.config.max_length and self.config.truncation == "reject":
                raise ValueError(f"Sequence has {len(sequence)} residues, exceeding max_length={self.config.max_length}; truncation is disabled")
            represented_length = min(len(sequence), self.config.max_length)
            if represented_length + 2 > self.config.max_tokens_per_batch:
                raise ValueError("One sequence exceeds max_tokens_per_batch, including BOS/EOS tokens")
            if full_length is None:
                full_length = len(sequence)
            elif len(sequence) != full_length:
                raise ValueError("An assay cache requires same-length substitution sequences")
        return full_length

    def _batches(self, sequences: Sequence[str]) -> Iterator[list[int]]:
        full_length = self._validate_sequences(sequences)
        batch_size = self.config.max_tokens_per_batch // (min(full_length, self.config.max_length) + 2)
        for start in range(0, len(sequences), batch_size):
            yield list(range(start, min(start + batch_size, len(sequences))))

    def _embed_strings(self, strings: Sequence[str], *, pooling: str) -> Tensor:
        """Evaluate one bounded batch of equally sized windows or sequences."""
        final_layer = MODEL_PRESETS[self.config.model_id]["num_layers"]
        tokens = self._tokenizer(
            strings, add_special_tokens=True, padding=True, truncation=False,
            return_attention_mask=True, return_special_tokens_mask=True, return_tensors="pt",
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        residue_mask = attention_mask.bool() & ~tokens["special_tokens_mask"].bool()
        if input_ids.shape != attention_mask.shape or input_ids.shape != residue_mask.shape:
            raise ValueError("Tokenizer masks do not align with input tokens")
        if input_ids.numel() > self.config.max_tokens_per_batch:
            raise ValueError("Tokenized batch exceeds max_tokens_per_batch")
        for row, sequence in enumerate(strings):
            ids = input_ids[row, residue_mask[row]].tolist()
            if self._tokenizer.convert_ids_to_tokens(ids) != list(sequence):
                raise ValueError("Tokenizer residues do not match the represented input sequence")
        with torch.inference_mode():
            output = self._model(
                input_ids=input_ids.to(self._device), attention_mask=attention_mask.to(self._device),
                output_hidden_states=self.config.layer != final_layer, return_dict=True,
            )
            hidden = output.last_hidden_state if self.config.layer == final_layer else output.hidden_states[self.config.layer]
            if hidden.ndim != 3 or tuple(hidden.shape[:2]) != tuple(input_ids.shape) or hidden.shape[-1] != self.hidden_size:
                raise ValueError("ESM2 returned an unexpected representation shape")
            residue_values = [
                hidden[row, residue_mask[row].to(hidden.device)] for row in range(len(strings))
            ]
            if pooling == "mean":
                residue_values = [values.to(dtype=torch.float32).mean(dim=0) for values in residue_values]
            values = torch.stack(residue_values).to(device="cpu", dtype=torch.float32).clone()
            del residue_values
            del hidden, output
        if not torch.isfinite(values).all():
            raise ValueError("ESM2 embeddings contain nonfinite values")
        return values

    def iter_embeddings(self, sequences: Sequence[str]) -> Iterator[tuple[list[int], Tensor]]:
        if self.config.global_window == "non_overlapping":
            for index, global_vector, _ in self._iter_window_embeddings(sequences):
                yield [index], global_vector.unsqueeze(0)
            return
        # Validate all sequences before loading weights or beginning inference.
        batches = list(self._batches(sequences))
        self.load()
        for indices in batches:
            strings = [
                sequences[index][:self.config.max_length]
                if self.config.truncation == "prefix" else sequences[index]
                for index in indices
            ]
            yield indices, self._embed_strings(strings, pooling=self.config.pooling)

    def _embed_windows(self, strings, bounds):
        return self._embed_strings(strings, pooling="none")
