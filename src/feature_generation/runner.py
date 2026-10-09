"""Run one frozen checkpoint across assays and publish validated caches."""

from __future__ import annotations

from datetime import datetime, timezone
import re
import time
from typing import Callable, Sequence

import torch

from .config import ExtractionConfig
from .inputs import prepare_assay
from .storage import check_existing, ensure_assay, make_leaf_config, write_sequence_cache


def extract(
    config: ExtractionConfig,
    assay_ids: Sequence[str],
    *,
    singles_only: bool = False,
    limit: int | None = None,
    resume: bool = False,
    continue_on_error: bool = False,
    progress: Callable[[dict], None] | None = None,
    adapter_factory=None,
) -> list[dict]:
    """Extract complete assay leaves, loading the model only if work is needed.

    ``limit`` is an explicit, deterministic development subset. A different
    variant table requires a different output root. Failed assays never acquire
    a completed model leaf. With ``continue_on_error`` failures are returned in
    the report so the CLI can still return a nonzero status.
    """
    if not assay_ids or len(set(assay_ids)) != len(assay_ids):
        raise ValueError("Select at least one unique assay")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    if adapter_factory is None:
        from .adapters.esm2 import ESM2Adapter

        adapter_factory = ESM2Adapter
    adapter = adapter_factory(config)
    leaf_config = None
    loaded = False
    results = []

    def emit(event: dict) -> None:
        if progress is not None:
            progress(event)

    for assay_id in assay_ids:
        started = time.monotonic()
        embeddings = None
        try:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", assay_id):
                raise ValueError(f"Unsafe assay identifier: {assay_id!r}")
            inputs = prepare_assay(config.assay_index, assay_id, singles_only=singles_only, limit=limit)
            n = len(inputs.variant_ids)
            if inputs.length > config.max_length:
                raise ValueError(
                    f"Reference length {inputs.length} exceeds max_length={config.max_length}; "
                    "sequences are never truncated"
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
            root = (config.output_root / assay_id).resolve()
            if not root.is_relative_to(config.output_root.resolve()):
                raise ValueError("Assay path escapes output_root")
            if leaf_config is None:
                leaf_config = make_leaf_config(config.cache_id, adapter.metadata())
            ensure_assay(root, inputs)
            if check_existing(root, config.cache_id, inputs, leaf_config, resume=resume):
                result = {"assay_id": assay_id, "status": "skipped", "path": str(root), "num_variants": n}
                results.append(result)
                emit(result)
                continue
            emit({"assay_id": assay_id, "status": "extracting", "num_variants": n, "output_bytes": output_bytes})
            if not loaded:
                adapter.load()
                loaded = True
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
                emit({"assay_id": assay_id, "status": "batch", "completed": int(seen.sum()), "num_variants": n})
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
                "uses_dms_labels": False,
            }
            leaf = write_sequence_cache(root, config.cache_id, inputs, leaf_config, embeddings, provenance)
            del embeddings
            result = {
                "assay_id": assay_id,
                "status": "completed",
                "path": str(leaf),
                "shape": [n, inputs.length, adapter.hidden_size],
                "elapsed_seconds": time.monotonic() - started,
            }
            results.append(result)
            emit(result)
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
            if not continue_on_error:
                raise
            result = {"assay_id": assay_id, "status": "failed", "error": str(error)}
            results.append(result)
            emit(result)
        finally:
            # Release a partially filled assay tensor before allocating the next one.
            embeddings = None
    return results
