"""Frozen official ProteinMPNN distributions and autoregressive sequence scores."""

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

    def score_metadata(self, *, num_orders: int = 1) -> dict:
        """Describe full-sequence scoring, separately from backbone-only features."""
        self._validate_num_orders(num_orders)
        return dict(
            self.metadata(),
            feature_type="scores",
            conditioning_mode="sequence_autoregressive",
            inference_method="forward",
            decoding_order="shared_random_orders",
            decoding_order_scope="common_across_variants_and_batches",
            decoding_order_algorithm="torch_cpu_generator_randn_abs_argsort",
            decoding_order_rng="torch_cpu_generator",
            decoding_order_seed=self.config.seed,
            num_orders=num_orders,
            alphabet="ACDEFGHIKLMNPQRSTVWYX",
            normalization="native_21_amino_acid_tokens",
            noncanonical_tokens="retained_in_normalization",
            precision_policy="float32_native_log_softmax",
            score_definition="negative_mean_sequence_nll",
            scoring_recipe_version=1,
            output_dtype="float32",
            score_direction="higher_is_more_likely",
            residue_aggregation="mean_over_all_reference_residues",
            order_aggregation="arithmetic_mean_of_scores",
            wild_type_subtraction=False,
        )

    @staticmethod
    def _validate_num_orders(num_orders):
        if type(num_orders) is not int or num_orders < 1:
            raise ValueError("num_orders must be a positive integer")

    def iter_scores(self, structure, sequences, *, batch_size: int = 8, num_orders: int = 1):
        """Yield frozen native sequence log-likelihoods in input order.

        Each score is the negative of ProteinMPNN's official ``_scores`` NLL:
        mean native log-probability over the complete supplied sequence. The
        same random decoding orders are reused across variants and batches.
        Multiple orders average log-likelihoods, without WT subtraction.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self._validate_num_orders(num_orders)
        coords = self._coords(structure, required_atoms=4)
        length = len(structure.reference_sequence)
        if isinstance(sequences, (str, bytes)):
            raise ValueError("sequences must be a nonempty collection of full-length sequences")
        try:
            sequences = list(sequences)
        except TypeError as exc:
            raise ValueError("sequences must be a nonempty collection of full-length sequences") from exc
        if not sequences:
            raise ValueError("sequences must be a nonempty collection of full-length sequences")
        for index, sequence in enumerate(sequences):
            if not isinstance(sequence, str) or len(sequence) != length or set(sequence) - set(ALPHABET):
                raise ValueError(f"Sequence {index} must contain exactly {length} canonical amino acids; only substitutions are supported")

        # This generator never touches PyTorch's global CPU or CUDA RNG state.
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        orders = torch.randn((num_orders, length), generator=generator, dtype=torch.float32)
        self.load()
        coords = coords.to(self._device).unsqueeze(0)
        orders = orders.to(self._device)
        native_alphabet = "ACDEFGHIKLMNPQRSTVWYX"
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            count = len(batch)
            sequence_tokens = torch.tensor(
                [[native_alphabet.index(aa) for aa in sequence] for sequence in batch],
                dtype=torch.long, device=self._device,
            )
            mask = torch.ones((count, length), dtype=torch.float32, device=self._device)
            residue_idx = torch.arange(length, dtype=torch.long, device=self._device).unsqueeze(0).expand(count, -1)
            chain_encoding = torch.ones((count, length), dtype=torch.long, device=self._device)
            with torch.inference_mode():
                scores = torch.zeros(count, dtype=torch.float32, device=self._device)
                for randn in orders:
                    log_probs = self._model(
                        coords.expand(count, -1, -1, -1), sequence_tokens, mask,
                        mask, residue_idx, chain_encoding, randn.unsqueeze(0).expand(count, -1),
                    )
                    if tuple(log_probs.shape) != (count, length, 21) or not torch.isfinite(log_probs).all():
                        raise ValueError("ProteinMPNN scoring requires finite native log-probabilities with shape [batch, length, 21]")
                    # Match upstream _scores exactly; retain X probability mass.
                    losses = torch.nn.functional.nll_loss(
                        log_probs.reshape(-1, 21), sequence_tokens.reshape(-1), reduction="none",
                    ).reshape(count, length)
                    scores -= (losses * mask).sum(dim=-1) / mask.sum(dim=-1)
                scores /= num_orders
                if not torch.isfinite(scores).all():
                    raise ValueError("ProteinMPNN returned nonfinite sequence scores")
                result = scores.detach().to(device="cpu", dtype=torch.float32).contiguous()
            yield list(range(start, start + count)), result
