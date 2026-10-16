"""Assays whose available structures cannot define full-reference scores."""

STRUCTURE_DISTANCE_DISABLED_ASSAYS = frozenset({"BRCA2_HUMAN_Erwood_2022_HEK293T"})
PROTEINMPNN_SCORE_SKIP_REASONS = {
    "BRCA2_HUMAN_Erwood_2022_HEK293T": "BRCA2 structures represent separate reference fragments",
    "A0A140D2T1_ZIKV_Sourisseau_2019": "structure covers reference positions 291-794 only",
    "POLG_HCVJF_Qi_2014": "structure covers reference positions 1982-2225 only",
}
