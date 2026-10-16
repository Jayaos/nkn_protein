"""Independent implementation of the published Kermut covariance equations.

The reproduction configuration uses a weighted sum of a mutation kernel and
an RBF kernel on frozen sequence embeddings. All distances are label-free.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from data_loading.features import ALPHABET


@dataclass(frozen=True)
class PairDistances:
    hellinger: torch.Tensor
    spatial: torch.Tensor | None
    log_probability: torch.Tensor
    sequence_squared: torch.Tensor
    mutation_pair_owners: torch.Tensor
    shape: tuple[int, int]


def _mutant_log_probabilities(batch):
    """Map each packed occurrence to its substituted amino-acid probability."""
    if not batch.reference_sequence or len(batch.sequences) != batch.num_variants:
        raise ValueError("Kermut requires a reference and every variant sequence")
    amino_acids = []
    for owner, position in zip(batch.owners.cpu().tolist(), batch.mutation_positions.cpu().tolist()):
        sequence = batch.sequences[owner]
        if len(sequence) != len(batch.reference_sequence):
            raise ValueError("Kermut variant sequence length differs from its reference")
        amino_acid = sequence[position - 1]
        if amino_acid not in ALPHABET or amino_acid == batch.reference_sequence[position - 1]:
            raise ValueError("Packed Kermut mutation must identify a substituted canonical amino acid")
        amino_acids.append(ALPHABET.index(amino_acid))
    probabilities = batch.local_features["inverse_folding"]
    columns = torch.tensor(amino_acids, device=batch.device, dtype=torch.long)
    selected = probabilities[torch.arange(len(columns), device=batch.device), columns]
    if (selected <= 0).any():
        raise ValueError("Kermut requires positive probabilities for substituted amino acids; a zero has undefined log probability")
    return selected.log()


@torch.no_grad()
def pair_distances(left, right, *, use_structure_distance=True):
    """Precompute parameter-independent distances and packed-to-variant mapping."""
    p, q = left.local_features["inverse_folding"], right.local_features["inverse_folding"]
    hellinger = torch.cdist(p.sqrt(), q.sqrt(), compute_mode="donot_use_mm_for_euclid_dist") / math.sqrt(2.0)
    spatial = (torch.cdist(left.local_features["structure"], right.local_features["structure"],
                           compute_mode="donot_use_mm_for_euclid_dist") if use_structure_distance else None)
    log_p, log_q = _mutant_log_probabilities(left), _mutant_log_probabilities(right)
    log_probability = (log_p[:, None] - log_q[None, :]).abs()
    sequence_squared = torch.cdist(left.global_features, right.global_features,
                                   compute_mode="donot_use_mm_for_euclid_dist").square()
    owners = left.owners[:, None] * right.num_variants + right.owners[None, :]
    return PairDistances(hellinger, spatial, log_probability, sequence_squared,
                         owners.flatten(), (left.num_variants, right.num_variants))


class KermutKernel(nn.Module):
    """Published weighted local/global kernel with trainable raw parameters."""

    def __init__(self, train, model_config):
        super().__init__()
        from .data import validate_batch

        model_config = model_config.for_assay(train.assay_id)
        self.use_structure_distance = model_config.use_structure_distance
        validate_batch(train, use_structure_distance=self.use_structure_distance)
        for name in ("raw_h_scale", "raw_h_lengthscale", "raw_d_lengthscale", "raw_p_lengthscale",
                     "raw_alpha", "raw_sequence_lengthscale"):
            value = torch.tensor(getattr(model_config, name), dtype=train.scores.dtype, device=train.device)
            self.register_parameter(name, nn.Parameter(value, requires_grad=(name != "raw_d_lengthscale" or self.use_structure_distance)))
        self._train_batch = train
        self._train_distances = pair_distances(train, train, use_structure_distance=self.use_structure_distance)

    @property
    def h_scale(self):
        return F.softplus(self.raw_h_scale)

    @property
    def h_lengthscale(self):
        return F.softplus(self.raw_h_lengthscale)

    @property
    def d_lengthscale(self):
        return F.softplus(self.raw_d_lengthscale) if self.use_structure_distance else self.raw_d_lengthscale.new_zeros(())

    @property
    def p_lengthscale(self):
        return F.softplus(self.raw_p_lengthscale)

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    @property
    def sequence_lengthscale(self):
        return F.softplus(self.raw_sequence_lengthscale)

    def from_distances(self, distances):
        exponent = -self.h_lengthscale * distances.hellinger
        if self.use_structure_distance:
            exponent = exponent - self.d_lengthscale * distances.spatial
        local_pairs = torch.exp(exponent - self.p_lengthscale * distances.log_probability)
        local = local_pairs.new_zeros(math.prod(distances.shape))
        local = local.index_add(0, distances.mutation_pair_owners, local_pairs.flatten()).reshape(distances.shape)
        local = self.h_scale * local
        sequence = torch.exp(-0.5 * distances.sequence_squared / self.sequence_lengthscale.square())
        return self.alpha * local + (1.0 - self.alpha) * sequence

    def forward(self, left, right=None):
        right = left if right is None else right
        if left is self._train_batch and right is self._train_batch:
            distances = self._train_distances
        else:
            distances = pair_distances(left, right, use_structure_distance=self.use_structure_distance)
        return self.from_distances(distances)
