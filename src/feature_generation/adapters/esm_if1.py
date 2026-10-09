"""Frozen official ESM-IF1 probabilities conditioned on the reference prefix."""

import argparse
import importlib
from pathlib import Path
import sys

import torch

from .inverse_base import InverseFoldingAdapter
from ..inverse_config import ALPHABET, PRESETS


def import_pinned_esm(source: Path):
    """Use verified upstream code without replacing an already imported esm package."""
    source = source.resolve()
    for name, module in list(sys.modules.items()):
        if name == "esm" or name.startswith("esm."):
            filename = getattr(module, "__file__", None)
            if filename is None or not Path(filename).resolve().is_relative_to(source):
                raise RuntimeError(
                    "A different esm package is already imported. Run ESM-IF1 through a fresh CLI "
                    "process, or restart the notebook kernel before importing the pipeline. "
                    "The pinned upstream source is separate from the installed esm package."
                )
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    try:
        esm = importlib.import_module("esm")
        importlib.import_module("esm.inverse_folding")
    except ImportError as exc:
        raise ImportError("ESM-IF1 requires scipy, biotite and torch-geometric from env.yaml or requirements.txt") from exc
    finally:
        sys.path.remove(str(source))
        sys.dont_write_bytecode = previous
    return esm


class ESMIF1Adapter(InverseFoldingAdapter):
    dependency_names = ("torch", "numpy", "scipy", "biotite", "torch-geometric")

    def __init__(self, config):
        if config.model_family != "esm_if1":
            raise ValueError("ESMIF1Adapter requires model_family=esm_if1")
        super().__init__(config)
        self._alphabet = None
        self._converter = None

    def metadata(self):
        return dict(super().metadata(), model_class="GVPTransformerModel",
                    inference_method="teacher_forced_causal_decoder",
                    decoding_order="reference_n_to_c", coordinate_confidence=1.0,
                    coordinate_confidence_policy="native_default_uniform_one",
                    prefix_sequence="assay_reference", required_backbone_atoms=["N", "CA", "C"])

    def load(self):
        if self._model is not None:
            return
        path = self._checkpoint()
        self._select_device()
        esm = import_pinned_esm(path / "source")
        # The official .pt stores argparse.Namespace and optimizer state. Permit
        # only that known metadata type; never unpickle arbitrary checkpoint code.
        with torch.serialization.safe_globals([argparse.Namespace]):
            checkpoint = torch.load(path / PRESETS["esm_if1"]["weights_filename"], map_location="cpu", weights_only=True)
        if checkpoint["args"].arch != "vt_medium_with_invariant_gvp":
            raise ValueError("ESM-IF1 checkpoint has an unexpected architecture")
        # Drop unused training state before constructing the 142M parameter model.
        checkpoint = {"args": checkpoint["args"], "model": checkpoint["model"]}
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.config.seed)
            model, alphabet, model_state = esm.pretrained._load_model_and_alphabet_core_v1(checkpoint)
        model.load_state_dict(model_state, strict=True)
        model.requires_grad_(False).eval().to(self._device)
        self._model, self._alphabet = model, alphabet
        self._converter = esm.inverse_folding.util.CoordBatchConverter(alphabet)

    def predict(self, structure):
        coords = self._coords(structure, required_atoms=3)
        sequence = structure.reference_sequence
        self.load()
        coords, confidence, sequences, tokens, padding_mask = self._converter(
            [(coords.cpu().numpy(), None, sequence)], device=self._device,
        )
        expected_tokens = [self._alphabet.get_idx(aa) for aa in sequence]
        if tokens.shape != (1, len(sequence) + 1) or tokens[0, 1:].tolist() != expected_tokens:
            raise ValueError("ESM-IF1 tokenizer does not align with the complete reference sequence")
        with torch.inference_mode():
            logits, _ = self._model(coords, padding_mask, confidence, tokens[:, :-1])
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[2] != len(sequence):
            raise ValueError("ESM-IF1 returned an unexpected output shape")
        indices = [self._alphabet.get_idx(aa) for aa in ALPHABET]
        if len(set(indices)) != 20 or any(self._alphabet.get_tok(i) != aa for aa, i in zip(ALPHABET, indices)):
            raise ValueError("ESM-IF1 vocabulary lacks distinct canonical amino-acid tokens")
        return self._probabilities(logits[0].transpose(0, 1), indices, len(sequence))
