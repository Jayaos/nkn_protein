"""Download frozen models, extract representations, and prepare GP mean scores."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from .config import ExtractionConfig, PROJECT_ROOT


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download and verify a pinned pretrained checkpoint")
    download.add_argument("--config", required=True, type=Path)
    extraction = commands.add_parser("extract", help="Extract from locally downloaded weights")
    extraction.add_argument("--config", required=True, type=Path)
    assays = extraction.add_mutually_exclusive_group(required=True)
    assays.add_argument("--assay", action="append", help="Assay ID; repeat to select several assays")
    assays.add_argument("--all-assays", action="store_true")
    extraction.add_argument("--assay-index", type=Path)
    extraction.add_argument("--output-root", type=Path)
    extraction.add_argument("--structure-manifest-dir", type=Path, help="Directory of prepared structure manifests for structure-conditioned models")
    extraction.add_argument("--structure-token-dir", type=Path, help="ProSST: directory for verified structure-token caches")
    extraction.add_argument("--device", choices=("auto", "cpu", "cuda"))
    extraction.add_argument("--max-tokens-per-batch", type=int, help="ESM-2/ProSST sequence batching budget")
    extraction.add_argument("--limit", type=int, help="Explicit first-N-variant development subset; use a separate output root")
    extraction.add_argument("--singles-only", action="store_true")
    extraction.add_argument("--resume", action="store_true", help="Reuse only matching, fully validated completed caches")
    extraction.add_argument("--continue-on-error", action="store_true", help="Report failed assays and continue; exit status remains nonzero")
    extraction.add_argument("--quiet", action="store_true")
    scoring = commands.add_parser("score", help="Prepare ProteinMPNN, ESM2, or ProSST mean_score.csv for the GP mean")
    scoring.add_argument("--config", required=True, type=Path, help="ESM2, ProSST, or ProteinMPNN extraction config supplies checkpoint and paths")
    score_assays = scoring.add_mutually_exclusive_group(required=True)
    score_assays.add_argument("--assay", action="append", help="Assay ID; repeat to select several assays")
    score_assays.add_argument("--all-assays", action="store_true")
    scoring.add_argument("--assay-index", type=Path)
    scoring.add_argument("--output-root", type=Path)
    scoring.add_argument("--structure-manifest-dir", type=Path)
    scoring.add_argument("--structure-token-dir", type=Path)
    scoring.add_argument("--device", choices=("auto", "cpu", "cuda"))
    scoring.add_argument("--batch-size", type=int, default=8, help="Variant sequences (ProteinMPNN) or masked WT copies (ESM2/ProSST) per forward pass")
    scoring.add_argument("--num-orders", type=int, help="ProteinMPNN only: random decoding orders to average; default 1")
    scoring.add_argument("--max-tokens-per-batch", type=int, help="ESM2/ProSST: token budget including BOS/EOS")
    scoring.add_argument("--cache-id", help="Score recipe directory; default identifies checkpoint and scoring settings")
    scoring.add_argument("--limit", type=int, help="Explicit first-N-variant subset; use a separate output root")
    scoring.add_argument("--singles-only", action="store_true")
    scoring.add_argument("--resume", action="store_true", help="Reuse only matching, fully validated score caches")
    scoring.add_argument("--continue-on-error", action="store_true")
    scoring.add_argument("--quiet", action="store_true")
    validation = commands.add_parser("validate", help="Validate one cache without requiring other feature modalities")
    validation.add_argument("--features", required=True, type=Path, help="Assay directory containing assay.json")
    validation.add_argument("--feature-type", choices=("sequence", "inverse_folding", "scores"), help="Use scores for mean_score.csv; otherwise inferred from --model-family")
    validation.add_argument("--model-family", choices=("esm2", "prosst", "msa_transformer", "proteinmpnn", "esm_if1"))
    validation.add_argument("--cache-id", help="Model recipe directory; defaults to the selected feature descriptor")
    for command, help_text in (
        ("prepare-structure", "Validate a local PDB/mmCIF and register its reference-residue mapping"),
        ("download-structure", "Download a ProteinGym structure (or an explicit URL) and register its mapping"),
    ):
        structure = commands.add_parser(command, help=help_text)
        structure.add_argument("--assay", required=True)
        structure.add_argument("--assay-index", type=Path, default=PROJECT_ROOT / "data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv")
        structure.add_argument("--chain", help="Author chain ID; required when the structure has multiple polymer chains")
        structure.add_argument("--model-index", type=int, default=1, help="One-based coordinate model index")
        structure.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "data/structures/manifests")
        if command == "prepare-structure":
            structure.add_argument("--structure", type=Path, action="append", required=True,
                                   help="Local structure; repeat with matching --reference-range for fragments")
            structure.add_argument("--reference-range", action="append", metavar="START-END",
                                   help="One-based inclusive reference interval for each --structure, in order")
            structure.add_argument("--source-url", help="Optional provenance URL for the local structure")
        else:
            structure.add_argument("--url", help="Direct HTTPS PDB/mmCIF URL; omitted to use the checksum-pinned ProteinGym archive")
            structure.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/structures/raw")
    return parser


def _progress(event: dict) -> None:
    assay, status = event["assay_id"], event["status"]
    if status == "batch":
        print(f"{assay}: {event['completed']}/{event['num_variants']} variants", file=sys.stderr)
    elif status == "scoring":
        if event.get("model_family") in ("esm2", "prosst"):
            label = "ProSST" if event["model_family"] == "prosst" else "ESM2"
            detail = f"{label} masked marginals at {event['num_masked_positions']} WT positions"
            if event.get("scoring_window") == "centered":
                detail += f", centered windows of {event['window_length']} residues"
        else:
            detail = f"{event['num_orders']} decoding orders"
        print(f"{assay}: scoring {event['num_variants']} variants ({detail})", file=sys.stderr)
    elif status == "extracting":
        if "reference_length" in event:
            print(f"{assay}: extracting inverse-folding distributions for {event['reference_length']} residues", file=sys.stderr)
        elif "num_variants" in event and "output_bytes" in event:
            print(f"{assay}: extracting {event['num_variants']} variants ({event['output_bytes'] / 1024**2:.1f} MiB cache)", file=sys.stderr)
        else:
            print(f"{assay}: extracting", file=sys.stderr)
    elif status == "failed":
        print(f"{assay}: failed: {event['error']}", file=sys.stderr)
    else:
        print(f"{assay}: {status}", file=sys.stderr)


def _load_config(path: Path):
    with path.open(encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError("Extraction configuration must be a JSON object")
    if values.get("model_family") == "prosst":
        from .prosst_config import ProSSTConfig
        return ProSSTConfig.load(path)
    if values.get("feature_type") == "inverse_folding" or values.get("model_family") in ("proteinmpnn", "esm_if1"):
        from .inverse_config import InverseFoldingConfig
        return InverseFoldingConfig.load(path)
    return ExtractionConfig.load(path)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command in ("prepare-structure", "download-structure"):
            from .structure_inputs import prepare_structure, download_structure, download_proteingym_structure
            common = dict(chain_id=args.chain, model_index=args.model_index)
            if args.command == "prepare-structure":
                if args.reference_range:
                    from .structure_fragments import prepare_structure_fragments
                    import re

                    if len(args.reference_range) != len(args.structure) or any(
                        re.fullmatch(r"[1-9][0-9]*-[1-9][0-9]*", value) is None for value in args.reference_range
                    ):
                        raise ValueError("Provide one START-END --reference-range per --structure")
                    if args.source_url:
                        raise ValueError("--source-url applies to single-structure registration")
                    ranges = [tuple(map(int, value.split("-"))) for value in args.reference_range]
                    path = prepare_structure_fragments(
                        args.assay_index, args.assay, args.structure, ranges,
                        output_dir=args.manifest_dir, **common,
                    )
                else:
                    if len(args.structure) != 1:
                        raise ValueError("Multiple --structure files require a --reference-range for each")
                    path = prepare_structure(args.assay_index, args.assay, args.structure[0],
                                             output_dir=args.manifest_dir, source_url=args.source_url, **common)
            elif args.url:
                path = download_structure(args.assay_index, args.assay, args.url,
                                          raw_dir=args.raw_dir, manifest_dir=args.manifest_dir, **common)
            else:
                path = download_proteingym_structure(args.assay_index, args.assay,
                                                     raw_dir=args.raw_dir, manifest_dir=args.manifest_dir, **common)
            print(json.dumps({"assay_id": args.assay, "manifest": str(path)}, indent=2))
            return 0
        if args.command == "validate":
            if args.feature_type == "scores":
                from .score_storage import validate_score_cache
                report = validate_score_cache(args.features, model_family=args.model_family or "proteinmpnn", cache_id=args.cache_id)
                print(json.dumps(report, indent=2))
                return 0
            family_type = ("inverse_folding" if args.model_family in ("proteinmpnn", "esm_if1")
                           else "sequence" if args.model_family in ("esm2", "prosst", "msa_transformer") else None)
            feature_type = args.feature_type or family_type or "sequence"
            if family_type is not None and feature_type != family_type:
                raise ValueError("--feature-type and --model-family select different feature types")
            if feature_type == "inverse_folding":
                from .inverse_storage import validate_inverse_cache
                report = validate_inverse_cache(args.features, model_family=args.model_family, cache_id=args.cache_id)
            else:
                from .storage import validate_sequence_cache
                if args.model_family is None:
                    report = validate_sequence_cache(args.features, args.cache_id)
                else:
                    report = validate_sequence_cache(args.features, args.cache_id, model_family=args.model_family)
            print(json.dumps(report, indent=2))
            return 0
        config = _load_config(args.config)
        prosst = getattr(config, "model_family", None) == "prosst"
        inverse = getattr(config, "model_family", None) in ("proteinmpnn", "esm_if1")
        if args.command == "download":
            if prosst:
                from .prosst_checkpoints import download
            elif inverse:
                from .inverse_checkpoints import download
            else:
                from .checkpoints import download
            path = download(config)
            print(json.dumps({"model_id": config.model_id, "revision": config.revision, "path": str(path)}, indent=2))
            return 0
        overrides = {}
        for name in ("assay_index", "output_root", "device", "max_tokens_per_batch", "structure_manifest_dir", "structure_token_dir"):
            value = getattr(args, name, None)
            if value is not None:
                if not hasattr(config, name):
                    raise ValueError(f"--{name.replace('_', '-')} does not apply to this model")
                overrides[name] = value.resolve() if isinstance(value, Path) else value
        config = replace(config, **overrides)
        from data_loading import AssayCatalog
        if args.command == "score":
            from .score_runner import score
            if args.num_orders is not None and getattr(config, "model_family", "esm2") != "proteinmpnn":
                raise ValueError("--num-orders applies only to ProteinMPNN sequence scoring")
            assay_ids = [record.assay_id for record in AssayCatalog(config.assay_index)] if args.all_assays else args.assay
            results = score(
                config, assay_ids, batch_size=args.batch_size, num_orders=1 if args.num_orders is None else args.num_orders,
                cache_id=args.cache_id, singles_only=args.singles_only, limit=args.limit,
                resume=args.resume, continue_on_error=args.continue_on_error,
                progress=None if args.quiet else _progress,
            )
            print(json.dumps({"results": results}, indent=2))
            return int(any(result["status"] == "failed" for result in results))
        if prosst:
            from .prosst_runner import extract
        elif inverse:
            from .inverse_runner import extract
        else:
            from .runner import extract
        assay_ids = [record.assay_id for record in AssayCatalog(config.assay_index)] if args.all_assays else args.assay
        results = extract(
            config, assay_ids, singles_only=args.singles_only, limit=args.limit,
            resume=args.resume, continue_on_error=args.continue_on_error,
            progress=None if args.quiet else _progress,
        )
        print(json.dumps({"results": results}, indent=2))
        return int(any(result["status"] == "failed" for result in results))
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
