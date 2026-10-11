"""Small, aligned protein features and independent numerical references."""

from dataclasses import replace
import csv
import hashlib
import json
import math
from pathlib import Path

import torch

from data_loading import FeatureBatch
from data_loading.features import ALPHABET


CONDITIONAL_PROBABILITY_RECIPE = {
    "model_family": "proteinmpnn", "conditioning_mode": "sequence_conditional",
    "inference_method": "conditional_probs", "normalization": "native_21_amino_acid_tokens",
    "noncanonical_tokens": "excluded_without_renormalization",
    "order_aggregation": "geometric_mean_of_probabilities", "num_orders": 10, "seed": 37,
}


def synthetic_batch():
    generator = torch.Generator().manual_seed(302)
    reference = "ACDEF"
    substitutions = [(position, alternate) for position in range(5) for alternate in "WY"]
    mutations = [[entry] for entry in substitutions]
    mutations += [[(first, "W"), (second, "Y")] for first in range(5) for second in range(first + 1, 5)]
    ids, sequences, owners, positions = [], [], [], []
    for owner, changes in enumerate(mutations):
        sequence = list(reference)
        ids.append(":".join(f"{reference[position]}{position + 1}{amino}" for position, amino in changes))
        for position, amino in changes:
            sequence[position] = amino
            owners.append(owner)
            positions.append(position + 1)
        sequences.append("".join(sequence))
    positions = torch.tensor(positions, dtype=torch.long)
    probabilities = torch.softmax(torch.randn(5, 20, dtype=torch.float64, generator=generator), dim=-1)
    coordinates = torch.tensor([[0, 0, 0], [1, 2, 1], [2, 3, 1], [2, 4, 4], [6, 3, 5]], dtype=torch.float64)
    n_variants = len(ids)
    return FeatureBatch(
        assay_id="kermut_synthetic", variant_ids=tuple(ids),
        targets=torch.linspace(-1, 2, n_variants, dtype=torch.float64) + torch.randn(n_variants, dtype=torch.float64, generator=generator) * 0.1,
        scores=torch.randn(n_variants, 2, dtype=torch.float64, generator=generator) * 0.2,
        global_features=torch.randn(n_variants, 4, dtype=torch.float64, generator=generator) * 0.3,
        local_features={"inverse_folding": probabilities[positions - 1], "structure": coordinates[positions - 1]},
        owners=torch.tensor(owners, dtype=torch.long), mutation_positions=positions,
        reference_sequence=reference, sequences=tuple(sequences),
        score_names=("sequence/synthetic/v1", "inverse_folding/synthetic/v1"),
        feature_signature={"source": "independent_kermut_tests", "structure": "five_aligned_CA_atoms"},
    )


def small_config(n_steps=2):
    from baselines.kermut.config import KermutConfig

    config = KermutConfig()
    return replace(
        config, training=replace(config.training, n_steps=n_steps, dtype="float64"),
        evaluation=replace(config.evaluation, n_folds=5, n_bins=3, coverage_levels=(0.5, 0.9), pair_block_size=2),
        num_threads=1,
    )


def scalar_kernel(left, right, config):
    """Equation-level reference using scalar loops, without implementation helpers."""
    softplus = lambda value: math.log1p(math.exp(value))
    scale = softplus(config.raw_h_scale)
    h_decay = softplus(config.raw_h_lengthscale)
    d_decay = softplus(config.raw_d_lengthscale)
    p_decay = softplus(config.raw_p_lengthscale)
    alpha = 1 / (1 + math.exp(-config.raw_alpha))
    lengthscale = softplus(config.raw_sequence_lengthscale)
    result = torch.empty(left.num_variants, right.num_variants, dtype=torch.float64)
    for first in range(left.num_variants):
        for second in range(right.num_variants):
            mutation_sum = 0.0
            for a in torch.nonzero(left.owners == first).flatten().tolist():
                for b in torch.nonzero(right.owners == second).flatten().tolist():
                    p = left.local_features["inverse_folding"][a].tolist()
                    q = right.local_features["inverse_folding"][b].tolist()
                    hellinger = math.sqrt(sum((math.sqrt(x) - math.sqrt(y)) ** 2 for x, y in zip(p, q)) / 2)
                    distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(left.local_features["structure"][a].tolist(), right.local_features["structure"][b].tolist())))
                    aa = ALPHABET.index(left.sequences[first][left.mutation_positions[a].item() - 1])
                    bb = ALPHABET.index(right.sequences[second][right.mutation_positions[b].item() - 1])
                    log_probability_distance = abs(math.log(p[aa]) - math.log(q[bb]))
                    mutation_sum += math.exp(-h_decay * hellinger - d_decay * distance - p_decay * log_probability_distance)
            embedding_distance_squared = sum((x - y) ** 2 for x, y in zip(left.global_features[first].tolist(), right.global_features[second].tolist()))
            result[first, second] = alpha * scale * mutation_sum + (1 - alpha) * math.exp(-embedding_distance_squared / (2 * lengthscale ** 2))
    return result


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, columns, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def make_cache(root):
    """Write real project-format feature leaves, scores, and a local PDB manifest."""
    from feature_generation.structure_inputs import prepare_structure

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    batch = synthetic_batch()
    features = root / "features"
    features.mkdir()
    names = ("sequence", "inverse_folding")
    metadata = {
        "schema_version": 2, "layout_version": 1, "synthetic": True,
        "assay_id": batch.assay_id, "reference_sequence": batch.reference_sequence,
        "reference_length": len(batch.reference_sequence), "variants_file": "variants.csv",
        "features": {name: {"model_family": "synthetic", "checkpoint": "v1", "path": f"{name}/synthetic/v1"} for name in names},
    }
    (features / "assay.json").write_text(json.dumps(metadata), encoding="utf-8")
    rows = [[variant, variant, sequence, float(target), index] for index, (variant, sequence, target) in enumerate(zip(batch.variant_ids, batch.sequences, batch.targets))]
    write_csv(features / "variants.csv", ["variant_id", "mutant", "mutated_sequence", "DMS_score", "source_row_index"], rows)
    common = {"schema_version": 2, "synthetic": True, "assay_id": batch.assay_id,
              "reference_sequence": batch.reference_sequence, "positions": torch.arange(1, 6)}
    order = list(reversed(range(batch.num_variants)))
    probabilities = batch.local_features["inverse_folding"][[0, 2, 4, 6, 8]]
    payloads = {
        "sequence": {**common, "variant_ids": [batch.variant_ids[i] for i in order],
                     "sequence_hashes": [hashlib.sha256(batch.sequences[i].encode()).hexdigest() for i in order],
                     "embeddings": batch.global_features[order, None, :].expand(-1, 5, -1).clone()},
        "inverse_folding": {**common, "alphabet": ALPHABET, "probabilities": probabilities, "log_probs": probabilities.log()},
    }
    for scorer, name in enumerate(names):
        leaf = features / name / "synthetic" / "v1"
        leaf.mkdir(parents=True)
        torch.save(payloads[name], leaf / "features.pt")
        config = {"feature_type": name, "model_family": "synthetic", "checkpoint": "v1", "synthetic": True,
                  "zero_shot": {"file": "zero_shot.csv", "id_column": "variant_id", "score_column": "score"}}
        (leaf / "config.json").write_text(json.dumps(config), encoding="utf-8")
        write_csv(leaf / "zero_shot.csv", ["variant_id", "score"], [[batch.variant_ids[i], float(batch.scores[i, scorer])] for i in order])
    index = root / "index.csv"
    write_csv(index, ["DMS_id", "DMS_filename", "target_seq", "pdb_file", "pdb_range"], [[batch.assay_id, "raw.csv", batch.reference_sequence, "reference.pdb", "1-5"]])
    write_csv(root / "raw.csv", ["mutant", "mutated_sequence", "DMS_score"], [[row[0], row[2], row[3]] for row in rows])
    three = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE"}
    lines = []
    for position, amino in enumerate(batch.reference_sequence, 1):
        for atom_index, atom in enumerate(("N", "CA", "C", "O")):
            serial = len(lines) + 1
            x = position * 4 + atom_index * 0.3
            lines.append(f"ATOM  {serial:5d} {atom:^4s} {three[amino]} A{position:4d}    {x:8.3f}{1.2:8.3f}{2.4:8.3f}{1.0:6.2f}{20.0:6.2f}          {atom[0]:>2s}  ")
    pdb = root / "reference.pdb"
    pdb.write_text("\n".join(lines) + "\nTER\n", encoding="utf-8")
    manifests = root / "manifests"
    prepare_structure(index, batch.assay_id, pdb, output_dir=manifests)
    config = replace(small_config(), feature_selection={name: "synthetic/v1" for name in names},
                     score_sources=batch.score_names[:1], structure_manifest_dir=str(manifests))
    return features, config, batch, index
