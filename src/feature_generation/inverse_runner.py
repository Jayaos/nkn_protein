"""Run frozen inverse-folding checkpoints once per reference structure."""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import TYPE_CHECKING, Callable, Sequence

import torch

from .inputs import prepare_assay, safe_component
from .inverse_storage import check_existing, make_leaf_config, validate_distributions, write_inverse_cache
from .storage import ensure_assay
from .structure_fragments import StructureFragments, load_inverse_structure as load_structure
from data_loading.features import parse_substitutions

if TYPE_CHECKING:
    from .inverse_config import InverseFoldingConfig


def extract(
    config: InverseFoldingConfig,
    assay_ids: Sequence[str],
    *,
    singles_only: bool = False,
    limit: int | None = None,
    resume: bool = False,
    continue_on_error: bool = False,
    progress: Callable[[dict], None] | None = None,
    adapter_factory=None,
) -> list[dict]:
    """Generate reference distributions; variant coverage affects only the assay table.

    A development subset needs its own output root, just as with ESM-2. Both
    inverse-folding families may share the same immutable assay inputs.
    """
    if not assay_ids or len(set(assay_ids)) != len(assay_ids):
        raise ValueError("Select at least one unique assay")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    if adapter_factory is None:
        if config.model_family == "proteinmpnn":
            from .adapters.proteinmpnn import ProteinMPNNAdapter

            adapter_factory = ProteinMPNNAdapter
        elif config.model_family == "esm_if1":
            from .adapters.esm_if1 import ESMIF1Adapter

            adapter_factory = ESMIF1Adapter
        else:
            raise ValueError(f"Unsupported inverse-folding model family: {config.model_family}")
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
            if config.max_length is not None and inputs.length > config.max_length:
                raise ValueError(
                    f"Reference length {inputs.length} exceeds max_length={config.max_length}; "
                    "structures and sequences are never truncated"
                )
            manifest_root = config.structure_manifest_dir.resolve()
            manifest_path = (manifest_root / f"{assay_id}.json").resolve()
            if not manifest_path.is_relative_to(manifest_root):
                raise ValueError("Structure manifest path escapes structure_manifest_dir")
            structure = load_structure(
                manifest_path,
                inputs.reference_sequence,
                require_oxygen=config.model_family == "proteinmpnn",
            )
            if structure.metadata.get("assay_id") != assay_id:
                raise ValueError("Structure manifest belongs to a different assay")
            fragments = structure.fragments if isinstance(structure, StructureFragments) else (structure,)
            covered_positions = (structure.positions if isinstance(structure, StructureFragments)
                                 else torch.arange(1, inputs.length + 1))
            covered = set(covered_positions.tolist())
            missing = sorted({position for row in inputs.rows for position in parse_substitutions(
                row["mutant"], row["mutated_sequence"], inputs.reference_sequence,
            )} - covered)
            if missing:
                raise ValueError(f"Inverse-folding structures do not cover mutation positions: {missing}")
            if model_metadata is None:
                model_metadata = {
                    **adapter.metadata(),
                    "model_id": config.model_id,
                    "revision": config.revision,
                    "conditioning_mode": config.conditioning_mode,
                    "seed": config.seed,
                    "inference_dtype": config.inference_dtype,
                }
            effective_metadata = model_metadata
            if isinstance(structure, StructureFragments):
                effective_metadata = dict(
                    model_metadata, structure_context="independent_reference_fragments",
                    residue_indexing="explicit_one_based_reference_positions",
                    fragment_aggregation="concatenate_disjoint_reference_ranges",
                    fragment_seed_policy="same_seed_per_fragment",
                )
            leaf_config = make_leaf_config(config.cache_id, config.model_family, effective_metadata, structure.metadata)
            root = (config.output_root / assay_id).resolve()
            if not root.is_relative_to(config.output_root.resolve()):
                raise ValueError("Assay path escapes output_root")
            ensure_assay(root, inputs)
            n = len(inputs.variant_ids)
            if check_existing(root, config.model_family, config.cache_id, inputs, leaf_config, resume=resume):
                result = {
                    "assay_id": assay_id, "status": "skipped", "model_family": config.model_family,
                    "path": str(root), "num_variants": n,
                }
                results.append(result)
                emit(result)
                continue
            emit({
                "assay_id": assay_id, "status": "extracting", "model_family": config.model_family,
                "reference_length": inputs.length, "num_variants": n,
            })
            if not loaded:
                adapter.load()
                loaded = True
            predictions = []
            for fragment in fragments:
                with torch.inference_mode():
                    prediction = adapter.predict(fragment)
                if not isinstance(prediction, dict):
                    raise ValueError("Inverse-folding adapter must return probabilities and log_probs")
                validate_distributions(
                    prediction.get("probabilities"), prediction.get("log_probs"),
                    len(fragment.reference_sequence), normalization=leaf_config["normalization"],
                )
                predictions.append(prediction)
            probabilities = torch.cat([prediction["probabilities"] for prediction in predictions])
            log_probs = torch.cat([prediction["log_probs"] for prediction in predictions])
            validate_distributions(
                probabilities, log_probs, len(covered_positions), normalization=leaf_config["normalization"],
            )
            provenance = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - started,
                "requested_device": config.device,
                "device": str(getattr(adapter, "device", config.device)),
                "source_assay": str(inputs.source_assay_path),
                "source_assay_sha256": inputs.source_assay_sha256,
                "source_index_sha256": inputs.source_index_sha256,
                "structure_manifest": str(manifest_path),
                "structure_sha256": structure.structure_sha256,
                "structure_input_signature": structure.input_signature,
                "uses_dms_labels": False,
            }
            leaf = write_inverse_cache(
                root, config.model_family, config.cache_id, inputs, leaf_config,
                probabilities, log_probs, provenance,
            )
            result = {
                "assay_id": assay_id,
                "status": "completed",
                "model_family": config.model_family,
                "path": str(leaf),
                "shape": [len(covered_positions), 20],
                "reference_length": inputs.length,
                "covered_residues": len(covered_positions),
                "num_structure_fragments": len(fragments),
                "num_variants": n,
                "elapsed_seconds": time.monotonic() - started,
            }
            results.append(result)
            emit(result)
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
            if not continue_on_error:
                raise
            result = {"assay_id": assay_id, "status": "failed", "model_family": config.model_family, "error": str(error)}
            results.append(result)
            emit(result)
    return results
