"""Run frozen ProSST on variants sharing one validated reference backbone."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import time
from typing import TYPE_CHECKING, Callable, Sequence

import torch

from .inputs import prepare_assay, safe_component
from .storage import check_existing, ensure_assay, make_leaf_config, write_sequence_cache
from .structure_inputs import load_structure

if TYPE_CHECKING:
    from .prosst_config import ProSSTConfig


def extract(
    config: ProSSTConfig,
    assay_ids: Sequence[str],
    *,
    singles_only: bool = False,
    limit: int | None = None,
    resume: bool = False,
    continue_on_error: bool = False,
    progress: Callable[[dict], None] | None = None,
    adapter_factory=None,
    token_factory=None,
) -> list[dict]:
    """Extract sequence-and-structure embeddings into independent ProSST leaves.

    Each assay quantizes its registered reference once and shares these tokens
    among its substitution variants. The model loads only when inference is
    needed. Resume verifies current structure and token identities against the
    complete feature cache; a different backbone needs a new cache ID.
    """
    if not assay_ids or len(set(assay_ids)) != len(assay_ids):
        raise ValueError("Select at least one unique assay")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    if config.model_family != "prosst":
        raise ValueError("ProSST runner requires model_family='prosst'")
    if adapter_factory is None:
        from .adapters.prosst import ProSSTAdapter

        adapter_factory = ProSSTAdapter
    if token_factory is None:
        from .prosst_structure import prepare_structure_tokens

        token_factory = prepare_structure_tokens
    adapter = adapter_factory(config)
    model_metadata = None
    loaded = False
    results = []

    def emit(event: dict) -> None:
        if progress is not None:
            progress(event)

    for assay_id in assay_ids:
        started = time.monotonic()
        embeddings = None
        try:
            safe_component(assay_id, "assay ID")
            inputs = prepare_assay(config.assay_index, assay_id, singles_only=singles_only, limit=limit)
            n = len(inputs.variant_ids)
            if inputs.length > config.max_length:
                raise ValueError(
                    f"Reference length {inputs.length} exceeds max_length={config.max_length}; "
                    "structures and sequences are never truncated"
                )
            if inputs.length + 2 > config.max_tokens_per_batch:
                raise ValueError("A sequence plus its two special tokens exceeds max_tokens_per_batch")
            output_bytes = n * inputs.length * adapter.hidden_size * 4
            if output_bytes > config.max_output_gib * 1024**3:
                raise ValueError(
                    f"Output requires {output_bytes / 1024**3:.3f} GiB, above "
                    f"max_output_gib={config.max_output_gib}; increase the explicit memory limit "
                    "or select a smaller development subset in a separate output root"
                )
            manifest_root = config.structure_manifest_dir.resolve()
            manifest_path = (manifest_root / f"{assay_id}.json").resolve()
            if not manifest_path.is_relative_to(manifest_root):
                raise ValueError("Structure manifest path escapes structure_manifest_dir")
            structure = load_structure(manifest_path, inputs.reference_sequence, require_oxygen=True)
            root = (config.output_root / assay_id).resolve()
            if not root.is_relative_to(config.output_root.resolve()):
                raise ValueError("Assay path escapes output_root")
            # Reject changed variant coverage before costly quantization.
            ensure_assay(root, inputs)
            emit({"assay_id": assay_id, "status": "structure_tokens", "model_family": "prosst"})
            structure_tokens = token_factory(config, structure)
            if (
                not isinstance(structure_tokens.tokens, torch.Tensor)
                or structure_tokens.tokens.device.type != "cpu"
                or structure_tokens.tokens.dtype != torch.long
                or tuple(structure_tokens.tokens.shape) != (inputs.length,)
                or (structure_tokens.tokens < 0).any()
                or (structure_tokens.tokens >= 2048).any()
            ):
                raise ValueError("ProSST structure tokens must be CPU int64 [L] with IDs in 0..2047")
            if structure_tokens.input_signature != structure_tokens.metadata.get("input_signature"):
                raise ValueError("ProSST structure-token object identity differs from its metadata")
            from .prosst_structure import token_ids_sha256

            if token_ids_sha256(structure_tokens.tokens) != structure_tokens.metadata.get("token_ids_sha256"):
                raise ValueError("ProSST structure-token IDs differ from their metadata checksum")
            if model_metadata is None:
                model_metadata = deepcopy(adapter.metadata())
            # Never retain mutable per-assay metadata in the reused model metadata.
            leaf_config = make_leaf_config(config.cache_id, {
                **deepcopy(model_metadata),
                "structure": deepcopy(structure.metadata),
                "structure_input_signature": structure.input_signature,
                "structure_tokens": deepcopy(structure_tokens.metadata),
                "structure_token_input_signature": structure_tokens.input_signature,
            }, model_family="prosst")
            if check_existing(root, config.cache_id, inputs, leaf_config, resume=resume, model_family="prosst"):
                result = {
                    "assay_id": assay_id, "status": "skipped", "model_family": "prosst",
                    "path": str(root), "num_variants": n,
                }
                results.append(result)
                emit(result)
                continue
            emit({
                "assay_id": assay_id, "status": "extracting", "model_family": "prosst",
                "num_variants": n, "output_bytes": output_bytes,
            })
            if not loaded:
                adapter.load()
                loaded = True
            adapter.set_structure_tokens(structure_tokens.tokens)
            embeddings = torch.empty((n, inputs.length, adapter.hidden_size), dtype=torch.float32)
            seen = torch.zeros(n, dtype=torch.bool)
            for indices, values in adapter.iter_embeddings(inputs.sequences):
                if not indices or any(type(i) is not int or i < 0 or i >= n for i in indices):
                    raise ValueError("Adapter returned invalid variant indices")
                if len(set(indices)) != len(indices) or seen[indices].any():
                    raise ValueError("Adapter returned duplicate variant indices")
                if (
                    not isinstance(values, torch.Tensor)
                    or values.device.type != "cpu"
                    or values.dtype != torch.float32
                    or tuple(values.shape) != (len(indices), inputs.length, adapter.hidden_size)
                    or not torch.isfinite(values).all()
                ):
                    raise ValueError("Adapter must return finite CPU float32 residue embeddings")
                embeddings[indices] = values
                seen[indices] = True
                emit({
                    "assay_id": assay_id, "status": "batch", "model_family": "prosst",
                    "completed": int(seen.sum()), "num_variants": n,
                })
            if not seen.all():
                raise ValueError("Adapter did not cover all selected variants")
            provenance = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - started,
                "requested_device": config.device,
                "device": str(getattr(adapter, "device", config.device)),
                "max_tokens_per_batch": config.max_tokens_per_batch,
                "source_assay": str(inputs.source_assay_path),
                "source_assay_sha256": inputs.source_assay_sha256,
                "source_index_sha256": inputs.source_index_sha256,
                "structure_manifest": str(manifest_path),
                "structure_sha256": structure.structure_sha256,
                "structure_input_signature": structure.input_signature,
                "structure_token_cache": str(structure_tokens.path),
                "structure_token_input_signature": structure_tokens.input_signature,
                "uses_dms_labels": False,
            }
            leaf = write_sequence_cache(
                root, config.cache_id, inputs, leaf_config, embeddings, provenance, model_family="prosst",
            )
            result = {
                "assay_id": assay_id, "status": "completed", "model_family": "prosst",
                "path": str(leaf), "shape": [n, inputs.length, adapter.hidden_size],
                "num_variants": n, "elapsed_seconds": time.monotonic() - started,
            }
            results.append(result)
            emit(result)
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
            if not continue_on_error:
                raise
            result = {"assay_id": assay_id, "status": "failed", "model_family": "prosst", "error": str(error)}
            results.append(result)
            emit(result)
        finally:
            embeddings = None
    return results
