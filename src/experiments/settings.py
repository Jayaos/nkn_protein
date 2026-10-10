"""The six mutation-domain settings from Kermut Section 4.2."""

from __future__ import annotations

import torch

from data_loading import FeatureBatch


SETTING_SCHEMES = {
    "singles_random": "random",
    "singles_modulo": "modulo",
    "singles_contiguous": "contiguous",
    "mixed_random_singles": "random_multiples",
    "mixed_random_doubles": "random_multiples",
    "single_to_double": None,
}
SETTINGS = tuple(SETTING_SCHEMES)
CV_SETTINGS = tuple(name for name, scheme in SETTING_SCHEMES.items() if scheme is not None)


def select_setting_batch(batch: FeatureBatch, setting: str) -> FeatureBatch:
    """Select a setting's training/evaluation population, preserving input order.

    Mixed CV retains both depths regardless of which test domain is requested.
    Full assay coverage must be checked before this explicit domain selection.
    """
    if setting not in SETTING_SCHEMES:
        raise ValueError(f"Unknown experiment setting: {setting}")
    counts = torch.bincount(batch.owners, minlength=batch.num_variants)
    singles = counts == 1
    if setting.startswith("singles_"):
        if not bool(singles.any()):
            raise ValueError(f"{setting} requires single mutants")
        mask = singles
    else:
        doubles = counts == 2
        if not bool(singles.any()) or not bool(doubles.any()):
            raise ValueError(f"{setting} requires both single and double mutants")
        mask = singles | doubles
    return batch.select(torch.nonzero(mask, as_tuple=False).flatten())
