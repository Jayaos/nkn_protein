"""Prepare variant-level GP mean scores with frozen pretrained models."""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Callable, Sequence

import torch

from .config import ExtractionConfig
from .inputs import prepare_assay, safe_component
from .inverse_config import InverseFoldingConfig
from .prosst_config import ProSSTConfig
from .score_storage import check_existing, make_score_config, write_score_cache
from .storage import ensure_assay
from .structure_inputs import load_structure
from .structure_sources import structure_cache_id, validate_structure_cache_id


def default_cache_id(config: InverseFoldingConfig | ExtractionConfig | ProSSTConfig, num_orders: int = 1) -> str:
    """Keep score recipes independent of representation-layer/cache selections."""
    family = getattr(config, "model_family", "esm2")
    if family == "esm2":
        cache_id = (
            f"{config.model_id.split('/')[-1]}-masked_marginals-"
            f"{config.inference_dtype}-{config.revision[:12]}"
        )
        if config.scoring_window == "centered":
            cache_id += f"-centered{config.max_length}"
        return cache_id
    if family == "prosst":
        cache_id = f"ProSST-2048-masked_marginals-fixed-backbone-{config.inference_dtype}-{config.revision[:12]}"
        if config.scoring_window == "centered":
            cache_id += f"-centered{config.max_length}"
        return structure_cache_id(config.structure_source, cache_id)
    if family == "esm_if1":
        if type(num_orders) is not int or num_orders != 1:
            raise ValueError("ESM-IF1 scoring uses one fixed N-to-C decoding order")
        return structure_cache_id(config.structure_source,
                                  f"{config.model_id.split('/')[-1]}-sequence_log_likelihood-native-v1-{config.revision[:12]}")
    if family != "proteinmpnn":
        raise ValueError("Mean scoring supports ProteinMPNN, ESM-IF1, ESM2, and ProSST")
    return structure_cache_id(config.structure_source,
        f"{config.model_id.split('/')[-1]}-sequence_log_likelihood-"
        f"orders{num_orders}-seed{config.seed}-{config.revision[:12]}"
    )


def score(
    config: InverseFoldingConfig | ExtractionConfig | ProSSTConfig,
    assay_ids: Sequence[str],
    *,
    batch_size: int | None = None,
    num_orders: int = 1,
    cache_id: str | None = None,
    singles_only: bool = False,
    limit: int | None = None,
    resume: bool = False,
    continue_on_error: bool = False,
    progress: Callable[[dict], None] | None = None,
    adapter_factory=None,
    token_factory=None,
) -> list[dict]:
    """Score variants with the selected pretrained recipe, without labels.

    ProteinMPNN and ESM-IF1 evaluate full variants on the reference backbone.
    ESM-IF1 uses native-vocabulary log-likelihoods with each variant's prefix.
    ESM2 evaluates masked WT positions and sums mutation log-odds, optionally
    using centered WT windows for long references. The extraction config supplies
    checkpoint, paths and scoring_window; embedding layer selections are ignored.
    ProSST aligns fixed reference structure tokens with each masked WT window.
    Scoring has a separate cache identity and never edits representation leaves.
    """
    family = getattr(config, "model_family", "esm2")
    if family not in ("proteinmpnn", "esm_if1", "esm2", "prosst"):
        raise ValueError("Mean scoring supports ProteinMPNN, ESM-IF1, ESM2, and ProSST")
    if batch_size is None:
        batch_size = 1 if family == "esm_if1" else 8
    centered = family in ("esm2", "prosst") and config.scoring_window == "centered"
    if not assay_ids or len(set(assay_ids)) != len(assay_ids):
        raise ValueError("Select at least one unique assay")
    for name, value in (("batch_size", batch_size), ("num_orders", num_orders)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if family in ("esm2", "prosst") and num_orders != 1:
        raise ValueError("num_orders does not apply to masked-marginal scoring")
    if family == "esm_if1" and num_orders != 1:
        raise ValueError("ESM-IF1 scoring uses one fixed N-to-C decoding order")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    cache_id = default_cache_id(config, num_orders) if cache_id is None else cache_id
    safe_component(cache_id, "score cache ID")
    if family in ("proteinmpnn", "esm_if1", "prosst"):
        validate_structure_cache_id(config.structure_source, cache_id)
    if adapter_factory is None:
        if family == "proteinmpnn":
            from .adapters.proteinmpnn import ProteinMPNNAdapter

            adapter_factory = ProteinMPNNAdapter
        elif family == "esm_if1":
            from .adapters.esm_if1 import ESMIF1Adapter

            adapter_factory = ESMIF1Adapter
        elif family == "prosst":
            from .adapters.prosst_scoring import ProSSTScoreAdapter

            adapter_factory = ProSSTScoreAdapter
        else:
            from .adapters.esm2_scoring import ESM2ScoreAdapter

            adapter_factory = ESM2ScoreAdapter
    if family == "prosst" and token_factory is None:
        from .prosst_structure import prepare_structure_tokens

        token_factory = prepare_structure_tokens
    adapter = adapter_factory(config)
    loaded = False
    model_metadata = None
    results = []

    def emit(event: dict) -> None:
        if progress is not None:
            progress(event)

    for assay_id in assay_ids:
        started = time.monotonic()
        try:
            safe_component(assay_id, "assay ID")
            inputs = prepare_assay(config.assay_index, assay_id, singles_only=singles_only, limit=limit)
            if config.max_length is not None and inputs.length > config.max_length and not centered:
                raise ValueError(
                    f"Reference length {inputs.length} exceeds max_length={config.max_length}; "
                    "structures and sequences are never truncated"
                )
            structure = None
            if family in ("proteinmpnn", "esm_if1", "prosst"):
                manifest_root = config.structure_manifest_dir.resolve()
                manifest_path = (manifest_root / f"{assay_id}.json").resolve()
                if not manifest_path.is_relative_to(manifest_root):
                    raise ValueError("Structure manifest path escapes structure_manifest_dir")
                structure = load_structure(manifest_path, inputs.reference_sequence, require_oxygen=family != "esm_if1")
            if family in ("esm2", "prosst") and (min(inputs.length, config.max_length) if centered else inputs.length) + 2 > config.max_tokens_per_batch:
                raise ValueError("One sequence exceeds max_tokens_per_batch, including BOS/EOS tokens")
            if model_metadata is None:
                model_metadata = (adapter.score_metadata(num_orders=num_orders) if family == "proteinmpnn"
                                  else adapter.score_metadata())
            root = (config.output_root / assay_id).resolve()
            if not root.is_relative_to(config.output_root.resolve()):
                raise ValueError("Assay path escapes output_root")
            ensure_assay(root, inputs)
            assay_metadata = dict(model_metadata)
            if family == "prosst":
                from .prosst_structure import token_ids_sha256

                structure_tokens = token_factory(config, structure)
                tokens = structure_tokens.tokens
                if (not isinstance(tokens, torch.Tensor) or tokens.device.type != "cpu"
                    or tokens.dtype != torch.long or tuple(tokens.shape) != (inputs.length,)
                    or (tokens < 0).any() or (tokens >= 2048).any()):
                    raise ValueError("ProSST structure tokens must be CPU int64 [L] with IDs in 0..2047")
                if (structure_tokens.input_signature != structure_tokens.metadata.get("input_signature")
                    or token_ids_sha256(tokens) != structure_tokens.metadata.get("token_ids_sha256")):
                    raise ValueError("ProSST structure-token identity or checksum mismatch")
                assay_metadata.update(
                    structure_input_signature=structure.input_signature,
                    structure_tokens=structure_tokens.metadata,
                    structure_token_input_signature=structure_tokens.input_signature,
                )
            leaf_config = make_score_config(cache_id, family, assay_metadata,
                                            structure.metadata if structure is not None else None)
            n = len(inputs.variant_ids)
            source = f"scores/{family}/{cache_id}"
            if check_existing(root, family, cache_id, inputs, leaf_config, resume=resume):
                result = {
                    "assay_id": assay_id, "status": "skipped", "model_family": family,
                    "path": str(root / source), "score_source": source, "num_variants": n,
                }
                results.append(result)
                emit(result)
                continue
            event = {"assay_id": assay_id, "status": "scoring", "model_family": family, "num_variants": n}
            if family == "proteinmpnn":
                event["num_orders"] = num_orders
            elif family in ("esm2", "prosst"):
                event["num_masked_positions"] = len({
                    index for sequence in inputs.sequences
                    for index, (wt, mutant) in enumerate(zip(inputs.reference_sequence, sequence)) if wt != mutant
                })
                if centered:
                    event["scoring_window"] = "centered"
                    event["window_length"] = min(inputs.length, config.max_length)
            emit(event)
            if not loaded:
                adapter.load()
                loaded = True
            if family == "prosst":
                adapter.set_structure_tokens(structure_tokens.tokens)
            scores = torch.empty(n, dtype=torch.float32)
            seen = torch.zeros(n, dtype=torch.bool)
            with torch.inference_mode():
                if family == "proteinmpnn":
                    batches = adapter.iter_scores(structure, inputs.sequences, batch_size=batch_size, num_orders=num_orders)
                elif family == "esm_if1":
                    batches = adapter.iter_scores(structure, inputs.sequences, batch_size=batch_size)
                else:
                    batches = adapter.iter_scores(inputs.reference_sequence, inputs.sequences, batch_size=batch_size)
                for indices, values in batches:
                    if (
                        not isinstance(indices, (list, tuple)) or not indices
                        or any(type(index) is not int or not 0 <= index < n for index in indices)
                        or len(set(indices)) != len(indices)
                    ):
                        raise ValueError("Score adapter returned invalid or duplicate variant indices")
                    if seen[list(indices)].any():
                        raise ValueError("Score adapter returned duplicate variant indices")
                    if (
                        not isinstance(values, torch.Tensor) or values.device.type != "cpu"
                        or values.dtype != torch.float32 or values.shape != (len(indices),)
                        or not torch.isfinite(values).all()
                    ):
                        raise ValueError("Score adapter must return finite CPU float32 scores, one per variant index")
                    scores[list(indices)] = values
                    seen[list(indices)] = True
                    emit({"assay_id": assay_id, "status": "batch", "completed": int(seen.sum()), "num_variants": n})
            if not seen.all():
                raise ValueError("Score adapter did not cover every variant")
            provenance = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - started,
                "requested_device": config.device,
                "device": str(getattr(adapter, "device", config.device)),
                "batch_size": batch_size,
                "source_assay": str(inputs.source_assay_path),
                "source_assay_sha256": inputs.source_assay_sha256,
                "source_index_sha256": inputs.source_index_sha256,
                "uses_dms_labels": False,
            }
            if structure is not None:
                provenance.update(
                    structure_manifest=str(manifest_path), structure_sha256=structure.structure_sha256,
                    structure_input_signature=structure.input_signature,
                )
            if family == "prosst":
                provenance.update(structure_token_cache=str(structure_tokens.path),
                                  structure_token_input_signature=structure_tokens.input_signature)
            if family in ("esm2", "prosst"):
                provenance["max_tokens_per_batch"] = config.max_tokens_per_batch
                provenance["num_masked_positions"] = event["num_masked_positions"]
            leaf = write_score_cache(root, family, cache_id, inputs, leaf_config, scores, provenance)
            result = {
                "assay_id": assay_id, "status": "completed", "model_family": family,
                "path": str(leaf), "score_source": source, "num_variants": n,
                "elapsed_seconds": time.monotonic() - started,
            }
            results.append(result)
            emit(result)
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
            if not continue_on_error:
                raise
            result = {"assay_id": assay_id, "status": "failed", "model_family": family, "error": str(error)}
            results.append(result)
            emit(result)
    return results
