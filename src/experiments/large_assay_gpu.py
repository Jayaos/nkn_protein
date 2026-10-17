"""GPU runs using prepared assets and published folds for selected human assays."""

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import torch

from config import ExperimentConfig
from data_loading import load_feature_bundle, load_folds, validate_assay_coverage
from experiments.runner import _validate_fold_assignments, run_cv
from experiments.settings import select_setting_batch
from gp.inference import add_inference_arguments, inference_metadata, resolve_training_config


SUPPORTED_ASSAYS = (
  "A0A1I9GEU1_NEIME_Kennouche_2019"
  "A4_HUMAN_Seuma_2022"
  "AICDA_HUMAN_Gajula_2014_3cycles"
  "AMFR_HUMAN_Tsuboyama_2023_4G3O"
  "B2L11_HUMAN_Dutta_2010_binding-Mcl-1"
  "BCHB_CHLTE_Tsuboyama_2023_2KRU"
  "BLAT_ECOLX_Jacquier_2013"
  "BRCA2_HUMAN_Erwood_2022_HEK293T"
  "CAPSD_AAV2S_Sinai_2021"
  "CBX4_HUMAN_Tsuboyama_2023_2K28"
  "CUE1_YEAST_Tsuboyama_2023_2MYX"
  "ENV_HV1B9_DuenasDecamp_2016"
  "ERBB2_HUMAN_Elazar_2016"
  "F7YBW8_MESOW_Ding_2023"
  "F7YBW8_MESOW_Aakre_2015"
  "GCN4_YEAST_Staller_2018"
  "GLPA_HUMAN_Elazar_2016"
  "HIS7_YEAST_Pokusaeva_2019"
  "ISDH_STAAW_Tsuboyama_2023_2LHR"
  "KCNH2_HUMAN_Kozek_2020"
  "LYAM1_HUMAN_Elazar_2016"
  "MAFG_MOUSE_Tsuboyama_2023_1K1V"
  "MYO3_YEAST_Tsuboyama_2023_2BTT"
  "NPC1_HUMAN_Erwood_2022_HEK293T"
  "NPC1_HUMAN_Erwood_2022_RPE1"
  "NRAM_I33A0_Jiang_2016"
  "ODP2_GEOSE_Tsuboyama_2023_1W4G"
  "OPSD_HUMAN_Wan_2019"
  "OTU7A_HUMAN_Tsuboyama_2023_2L2D"
  "PIN1_HUMAN_Tsuboyama_2023_1I6C"
  "PITX2_HUMAN_Tsuboyama_2023_2L7M"
  "POLG_PESV_Tsuboyama_2023_2MXD"
  "Q837P4_ENTFA_Meier_2023"
  "Q837P5_ENTFA_Meier_2023"
  "RAD_ANTMA_Tsuboyama_2023_2CJJ"
  "RAF1_HUMAN_Zinkus-Boltz_2019"
  "RBP1_HUMAN_Tsuboyama_2023_2KWH"
  "RCD1_ARATH_Tsuboyama_2023_5OAO"
  "RD23A_HUMAN_Tsuboyama_2023_1IFY"
  "RFAH_ECOLI_Tsuboyama_2023_2LCL"
  "RPC1_LAMBD_Li_2019_high-expression"
  "RPC1_LAMBD_Li_2019_low-expression"
  "SAV1_MOUSE_Tsuboyama_2023_2YSB"
  "SCN5A_HUMAN_Glazer_2019"
  "SDA_BACSU_Tsuboyama_2023_1PV0"
  "SPG1_STRSG_Wu_2016"
  "SQSTM_MOUSE_Tsuboyama_2023_2RRU"
  "SR43C_ARATH_Tsuboyama_2023_2N88"
  "TCRG1_MOUSE_Tsuboyama_2023_1E0L"
  "THO1_YEAST_Tsuboyama_2023_2WQG"
  "TPOR_HUMAN_Bridgford_2020"
  "UBE4B_MOUSE_Starita_2013"
  "VG08_BPP22_Tsuboyama_2023_2GP8"
  "VKOR1_HUMAN_Chiasson_2020_activity"
  "YAIA_ECOLI_Tsuboyama_2023_2KVT"
  "YAP1_HUMAN_Araya_2012"
  "YNZC_BACSU_Tsuboyama_2023_2JVD"
  "FKBP3_HUMAN_Tsuboyama_2023_2KFV"
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (ROOT / "configs"
                  / "esm2_proteinmpnn_nkn_two_scores_additive_four_branches_noise_initial_1p0.json")
SETTINGS = ("singles_random", "singles_modulo", "singles_contiguous")


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", nargs="?", default="singles", choices=("singles", *SETTINGS))
    parser.add_argument("--assay", choices=SUPPORTED_ASSAYS, default=ASSAY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--min-steps", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--features", type=Path,
                        help="Prepared feature directory; defaults to data/features/pretrained/singles/ASSAY")
    parser.add_argument("--folds", type=Path,
                        help="Published fold CSV; defaults to cv_folds_singles_substitutions/ASSAY.csv")
    parser.add_argument("--assay-index", type=Path, default=ROOT
                        / "data/DMS_ProteinGym_substitutions/raw_substitutions/DMS_substitutions.csv")
    add_inference_arguments(parser)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    assay = args.assay
    if args.features is None:
        args.features = ROOT / "data/features/pretrained/singles" / assay
    if args.folds is None:
        args.folds = (ROOT / "data/DMS_ProteinGym_substitutions/cv_folds_singles_substitutions"
                      / f"{assay}.csv")
    output_created = False
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use CUDA-enabled PyTorch in a GPU allocation")
        if args.output.exists():
            raise ValueError(f"Output directory already exists: {args.output}")
        if args.log_every < 1:
            raise ValueError("log-every must be positive")
        base = ExperimentConfig.load(args.config).for_assay(assay)
        config = replace(base, device="cuda", num_threads=args.num_threads,
                         training=resolve_training_config(base.training, args))
        if config.evaluation.n_folds != 5:
            raise ValueError("Published ProteinGym protocols require five folds")
        if config.training.min_steps > config.training.max_steps:
            raise ValueError("min-steps exceeds max-steps; supply both for a shorter timing run")
        if config.training.max_train_variants is not None:
            raise ValueError("This run requires all single variants; max_train_variants must be null")
        metadata = json.loads((args.features / "assay.json").read_text(encoding="utf-8"))
        if metadata.get("assay_id") != assay or metadata.get("synthetic") is not False:
            raise ValueError(f"Expected real prepared assets for {assay}")
        torch.set_num_threads(config.num_threads)
        inference = inference_metadata(config.training)
        print(f"Inference: {json.dumps(inference)}", flush=True)
        print(f"Assay: {assay}; features: {args.features}", flush=True)
        print(f"Folds: {args.folds}; kernel branches: {', '.join(config.model.enabled_branches)}; "
              f"score sources: {config.score_sources}", flush=True)
        print("[1/3] Validating prepared assets and complete single-variant coverage on CPU", flush=True)
        batch = load_feature_bundle(
            args.features, feature_selection=config.feature_selection,
            score_sources=config.score_sources, pooling=config.pooling, device="cpu",
            enabled_branches=config.model.enabled_branches,
            inverse_folding_policy=config.inverse_folding_policy,
            structure_manifest_dir=config.structure_manifest_dir,
        )
        if batch.assay_id != assay:
            raise ValueError(f"Loaded feature batch does not identify {assay}")
        batch = select_setting_batch(batch, "singles_random")
        validate_assay_coverage(batch, args.assay_index, singles_only=True)
        settings = SETTINGS if args.selection == "singles" else (args.selection,)
        args.output.mkdir(parents=True, exist_ok=False)
        output_created = True
        _write_json(args.output / "config.json", config.to_dict())
        _write_json(args.output / "execution_plan.json", {
            "assay_id": assay, "device": "cuda", "parallel": False,
            "inference_backend": config.training.inference_backend, "inference": inference,
            "setting_workers": 1, "fold_workers": 1, "threads_per_worker": config.num_threads,
            "settings": {setting: {"fits": 5, "fold_workers": 1} for setting in settings},
        })
        print("[2/3] Snapshotting and validating published folds", flush=True)
        fold_dir = args.output / "folds"
        fold_dir.mkdir()
        snapshot = fold_dir / "singles.csv"
        source_bytes = args.folds.read_bytes()
        snapshot.write_bytes(source_bytes)
        with snapshot.open(newline="", encoding="utf-8-sig") as handle:
            columns = csv.DictReader(handle).fieldnames or []
        splits, fold_counts = {}, {}
        for setting in settings:
            scheme = setting.removeprefix("singles_")
            if f"fold_{scheme}_5" not in columns or scheme in columns:
                raise ValueError(f"Expected unambiguous published fold_{scheme}_5 assignments")
            splits[setting] = _validate_fold_assignments(
                batch, load_folds(snapshot, batch.variant_ids, scheme, 5), config, scheme,
            )
            fold_counts[setting] = torch.bincount(splits[setting], minlength=5).tolist()
            print(f"{setting}: {batch.num_variants} singles; held-out counts {fold_counts[setting]}", flush=True)
        _write_json(args.output / "fold_manifest.json", {
            "assay_id": assay, "source": str(args.folds.resolve()), "snapshot": "folds/singles.csv",
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "n_variants": batch.num_variants, "held_out_counts": fold_counts,
        })
        print(f"PyTorch: {torch.__version__}; GPU: {torch.cuda.get_device_name()}; "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB; "
              f"float64; sequential folds; threads: {config.num_threads}", flush=True)
        print(f"Training: min_steps={config.training.min_steps}; max_steps={config.training.max_steps}; "
              f"stopping={config.training.stopping_criterion}; seeds={config.training.seeds}", flush=True)
        summary = {"assay_id": assay, "n_variants": batch.num_variants,
                   "inference_backend": config.training.inference_backend, "inference": inference,
                   "gpu": torch.cuda.get_device_name(), "settings": {}}
        for setting in settings:
            print(f"[3/3] Running five-fold NKN/GP CV on CUDA: {setting}", flush=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with (args.output / f"{setting}.progress.jsonl").open("x", encoding="utf-8") as handle:
                def progress(event):
                    elapsed = time.perf_counter() - started
                    handle.write(json.dumps({**event, "elapsed_seconds": elapsed}) + "\n")
                    handle.flush()
                    if event.get("event") == "fold_start":
                        print(f"Fold {event['fold']}: {event['n_train']} training, "
                              f"{event.get('n_validation', 0)} validation, "
                              f"{event['n_test']} held-out variants", flush=True)
                    elif "step" in event and event["step"] % args.log_every == 0:
                        print(f"fold {event['fold']} seed {event['seed']} step {event['step']}: "
                              f"NLL/variant={event['nll_per_variant']:.6g}; "
                              f"setting elapsed={elapsed:.2f}s", flush=True)
                run_cv(
                    batch, splits[setting], config, args.output / "cv" / setting,
                    scheme=setting.removeprefix("singles_"), setting=setting,
                    split_source="provided", fold_workers=1, progress=progress,
                )
            torch.cuda.synchronize()
            timing = {
                "cv_seconds": time.perf_counter() - started,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            summary["settings"][setting] = timing
            _write_json(args.output / "benchmark.json", summary)
            print(f"Completed {setting}: {timing} (includes fitting, evaluation, and output)", flush=True)
        print(f"Completed. Results: {args.output}", flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, AssertionError, KeyError, TypeError) as error:
        if output_created:
            _write_json(args.output / "failure.json", {
                "error": str(error), "traceback": traceback.format_exc(),
            })
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
