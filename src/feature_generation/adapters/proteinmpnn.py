"""Frozen official ProteinMPNN distributions and autoregressive sequence scores."""

import importlib.util
import sys

import numpy as np
import torch

from .inverse_base import InverseFoldingAdapter
from ..inverse_config import ALPHABET, PRESETS


class ProteinMPNNAdapter(InverseFoldingAdapter):
    def __init__(self, config):
        if config.model_family != "proteinmpnn":
            raise ValueError("ProteinMPNNAdapter requires model_family=proteinmpnn")
        super().__init__(config)
        self._initialization_rng_state = None

    def _backbone_metadata(self):
        return dict(super().metadata(), model_class="ProteinMPNN",
                    inference_method="unconditional_probs", decoding_order="none",
                    residue_idx_policy="zero_based_contiguous_reference_order",
                    num_edges=48, hidden_size=128)

    def metadata(self):
        metadata = dict(self._backbone_metadata())
        if self.config.conditioning_mode == "sequence_conditional":
            metadata.update(
                inference_method="conditional_probs",
                decoding_order="target_last_random_context_orders",
                decoding_order_scope="reset_per_structure",
                decoding_order_rng="torch_device_generator",
                decoding_order_seed=self.config.seed,
                decoding_order_rng_initialization="upstream_seed_then_cpu_model_initialization",
                extraction_batch_size=1,
                num_orders=self.config.num_orders,
                order_aggregation="geometric_mean_of_probabilities",
                normalization="native_21_amino_acid_tokens",
                noncanonical_tokens="excluded_without_renormalization",
                precision_policy="float32_native_log_probabilities_numpy_mean_exp",
                conditional_recipe_version=1,
            )
        return metadata

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
            # Upstream seeds before constructing the CPU model. Its CPU
            # decoding RNG therefore starts after parameter initialization,
            # while its CUDA decoding RNG starts directly from the seed.
            self._initialization_rng_state = torch.get_rng_state().clone()
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
        if self.config.conditioning_mode == "sequence_conditional":
            return self._conditional_probabilities(structure.reference_sequence, coords, mask, residue_idx, chain_encoding)
        with torch.inference_mode():
            values = self._model.unconditional_probs(coords, mask, residue_idx, chain_encoding)
        if tuple(values.shape) != (1, length, 21):
            raise ValueError("ProteinMPNN returned an unexpected output shape")
        native_alphabet = "ACDEFGHIKLMNPQRSTVWYX"
        indices = [native_alphabet.index(aa) for aa in ALPHABET]
        return self._probabilities(values[0], indices, length)

    def _conditional_probabilities(self, sequence, coords, mask, residue_idx, chain_encoding):
        """Match Kermut's native conditional runs and NumPy postprocessing.

        Reset the private RNG for each reference structure, as Kermut invokes
        ProteinMPNN separately for each protein. Full WT tokens are supplied;
        native conditional_probs masks the target identity via decoding order.
        """
        native_alphabet = "ACDEFGHIKLMNPQRSTVWYX"
        sequence_tokens = torch.tensor([[native_alphabet.index(aa) for aa in sequence]],
                                       dtype=torch.long, device=self._device)
        generator = torch.Generator(device=self._device).manual_seed(self.config.seed)
        if self._device.type == "cpu" and self._initialization_rng_state is not None:
            generator.set_state(self._initialization_rng_state)
        all_log_probs = []
        with torch.inference_mode():
            for _ in range(self.config.num_orders):
                # Separate [1, L] draws match the upstream batch-size-one loop.
                randn = torch.randn(mask.shape, device=self._device, generator=generator)
                values = self._model.conditional_probs(
                    coords, sequence_tokens, mask, mask, residue_idx, chain_encoding,
                    randn, backbone_only=False,
                )
                if tuple(values.shape) != (1, len(sequence), 21) or not torch.isfinite(values).all():
                    raise ValueError("ProteinMPNN conditional inference requires finite native log-probabilities with shape [1, length, 21]")
                if values.dtype != torch.float32 or (values > 0).any():
                    raise ValueError("ProteinMPNN conditional inference requires float32 nonpositive native log-probabilities")
                all_log_probs.append(values.detach().cpu().numpy())
        # The released preprocessing uses NumPy float32 mean, then exp, then
        # drops X. Do not softmax or renormalize the remaining twenty values.
        mean_log_probs = np.concatenate(all_log_probs, axis=0).mean(axis=0)
        probabilities = np.exp(mean_log_probs)
        indices = [native_alphabet.index(aa) for aa in ALPHABET]
        selected_probs = torch.from_numpy(probabilities[:, indices].copy()).contiguous()
        selected_logs = torch.from_numpy(mean_log_probs[:, indices].copy()).contiguous()
        if not torch.isfinite(selected_probs).all() or (selected_probs <= 0).any():
            raise ValueError("Canonical amino-acid probabilities underflowed float32; no probability floor is applied")
        return {"alphabet": ALPHABET, "probabilities": selected_probs, "log_probs": selected_logs}

    def score_metadata(self, *, num_orders: int = 1) -> dict:
        """Describe full-sequence scoring independently of distribution features."""
        self._validate_num_orders(num_orders)
        return dict(
            self._backbone_metadata(),
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
