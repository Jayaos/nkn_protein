"""Frozen official ProteinMPNN backbone-only canonical candidate probabilities."""

import importlib.util
import sys

import torch

from .inverse_base import InverseFoldingAdapter
from ..inverse_config import ALPHABET, PRESETS


class ProteinMPNNAdapter(InverseFoldingAdapter):
    def __init__(self, config):
        if config.model_family != "proteinmpnn":
            raise ValueError("ProteinMPNNAdapter requires model_family=proteinmpnn")
        super().__init__(config)

    def metadata(self):
        return dict(super().metadata(), model_class="ProteinMPNN",
                    inference_method="unconditional_probs", decoding_order="none",
                    residue_idx_policy="zero_based_contiguous_reference_order",
                    num_edges=48, hidden_size=128)

    def load(self):
        if self._model is not None:
            return
        path = self._checkpoint()
        self._select_device()
        source = path / "source" / "protein_mpnn_utils.py"
        name = f"_nkn_proteinmpnn_{self.config.revision}"
        module = sys.modules.get(name)
        if module is None:
            spec = importlib.util.spec_from_file_location(name, source)
            module = importlib.util.module_from_spec(spec)
            # Compile verified source directly, so unverified bytecode is never loaded.
            exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
            sys.modules[name] = module
        elif getattr(module, "__file__", None) != str(source):
            raise RuntimeError("Pinned ProteinMPNN module name is already occupied by another source")
        checkpoint = torch.load(path / PRESETS["proteinmpnn"]["weights_filename"], map_location="cpu", weights_only=True)
        if checkpoint.get("num_edges") != 48:
            raise ValueError("ProteinMPNN checkpoint has unexpected neighbor count")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.config.seed)
            model = module.ProteinMPNN(
                num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
                num_encoder_layers=3, num_decoder_layers=3, k_neighbors=48,
                augment_eps=0.0, dropout=0.0, ca_only=False,
            )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.requires_grad_(False).eval().to(self._device)
        self._model = model

    def predict(self, structure):
        coords = self._coords(structure, required_atoms=4)
        length = len(structure.reference_sequence)
        self.load()
        coords = coords.to(self._device).unsqueeze(0)
        mask = torch.ones((1, length), dtype=torch.float32, device=self._device)
        residue_idx = torch.arange(length, dtype=torch.long, device=self._device).unsqueeze(0)
        chain_encoding = torch.ones((1, length), dtype=torch.long, device=self._device)
        with torch.inference_mode():
            values = self._model.unconditional_probs(coords, mask, residue_idx, chain_encoding)
        if tuple(values.shape) != (1, length, 21):
            raise ValueError("ProteinMPNN returned an unexpected output shape")
        native_alphabet = "ACDEFGHIKLMNPQRSTVWYX"
        indices = [native_alphabet.index(aa) for aa in ALPHABET]
        return self._probabilities(values[0], indices, length)
