"""Frozen Hugging Face ESM2 embeddings with strictly offline inference."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json

import torch
from torch import Tensor

from data_loading.sequence_windows import centered_window_bounds

from ..checkpoints import checkpoint_manifest, checkpoint_path
from ..config import ExtractionConfig, MODEL_PRESETS


ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class _SiteResult:
    variant_index: int
    global_vector: Tensor
    site_vectors: Tensor
    remaining_windows: int
    global_sum: Tensor | None = None


class ESM2Adapter:
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

    def iter_site_embeddings(
        self, sequences: Sequence[str], mutation_positions: Sequence[Sequence[int]],
    ) -> Iterator[tuple[int, Tensor, Tensor]]:
        """Yield a global mean and centered mutation-site vectors per variant.

        Each input window is taken from the actual variant, without masking.
        Original positions are one based; the shared bounds helper returns
        zero-based, half-open slices. Identical windows within a variant,
        including global windows, are evaluated only once. Tasks stream
        into token-limited batches without allocating an assay residue tensor.
        """
        if self.config.site_embeddings != "centered":
            raise ValueError("iter_site_embeddings requires site_embeddings=centered")
        yield from self._iter_window_embeddings(sequences, mutation_positions)

    def _iter_window_embeddings(
        self, sequences: Sequence[str], mutation_positions: Sequence[Sequence[int]] | None = None,
    ) -> Iterator[tuple[int, Tensor, Tensor]]:
        """Stream global and site tasks without retaining complete residue caches."""
        length = self._validate_sequences(sequences)
        if mutation_positions is not None:
            if isinstance(mutation_positions, (str, bytes)) or len(mutation_positions) != len(sequences):
                raise ValueError("mutation_positions must contain one position collection per variant")
            for positions in mutation_positions:
                if isinstance(positions, (str, bytes)) or not isinstance(positions, Sequence):
                    raise ValueError("Mutation positions must be collections of one-based integers")
                if any(type(position) is not int or not 1 <= position <= length for position in positions):
                    raise ValueError("Mutation positions must be one-based integers within the full sequence")
                if len(set(positions)) != len(positions):
                    raise ValueError("Mutation positions must not contain duplicates")
        self.load()
        window_length = min(length, self.config.max_length)
        if self.config.global_window == "non_overlapping":
            global_windows = tuple(
                (start, min(start + window_length, length)) for start in range(0, length, window_length)
            )
        else:
            global_windows = ((0, window_length),)
        global_bounds = frozenset(global_windows)

        def tasks():
            for index, sequence in enumerate(sequences):
                positions = () if mutation_positions is None else mutation_positions[index]
                windows = {bounds: [] for bounds in global_windows}
                for row, position in enumerate(positions):
                    bounds = centered_window_bounds(position, length, self.config.max_length)
                    windows.setdefault(bounds, []).append((row, position - bounds[0] - 1))
                result = _SiteResult(
                    index, torch.empty(self.hidden_size, dtype=torch.float32),
                    torch.empty((len(positions), self.hidden_size), dtype=torch.float32), len(windows),
                    torch.zeros(self.hidden_size, dtype=torch.float64) if len(global_windows) > 1 else None,
                )
                for (start, end), sites in windows.items():
                    yield sequence[start:end], (start, end) in global_bounds, sites, result

        def batches():
            # A final global chunk can be shorter than the centered site
            # windows. Flush on length changes so neither padding nor stacking
            # introduces extra residues or exceeds the padded token budget.
            batch, batch_length = [], None
            for task in tasks():
                task_length = len(task[0])
                if batch and (
                    task_length != batch_length
                    or (len(batch) + 1) * (task_length + 2) > self.config.max_tokens_per_batch
                ):
                    yield batch
                    batch = []
                batch.append(task)
                batch_length = task_length
            if batch:
                yield batch

        for batch in batches():
            embeddings = self._embed_strings([task[0] for task in batch], pooling="none")
            for (_, contributes_global, sites, result), values in zip(batch, embeddings):
                if contributes_global:
                    if result.global_sum is None:
                        # Preserve the existing float32 mean for one-window
                        # inputs, including all short proteins.
                        result.global_vector.copy_(values.mean(dim=0))
                    else:
                        result.global_sum.add_(values.sum(dim=0, dtype=torch.float64))
                for row, local_position in sites:
                    result.site_vectors[row].copy_(values[local_position])
                result.remaining_windows -= 1
                if result.remaining_windows == 0:
                    if result.global_sum is not None:
                        result.global_vector.copy_(result.global_sum / length)
                    if not torch.isfinite(result.global_vector).all():
                        raise ValueError("ESM2 global embeddings contain nonfinite values")
                    yield result.variant_index, result.global_vector, result.site_vectors
