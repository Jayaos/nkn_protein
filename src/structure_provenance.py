"""Check cached feature identities against shared or independent structures."""

from collections.abc import Mapping


def validate_feature_structure(name, assay_id, geometry, cached_structure, recorded_provenance,
                               consistency="shared"):
    """Compare provenance after the caller validates structure metadata/mappings.

    Geometry is always validated separately against its selected manifest. Under
    per_cache, a feature's declared structure supplies its expected identity;
    checks of its tensor/config/sidecar provenance remain mandatory.
    """
    if consistency not in ("shared", "per_cache"):
        raise ValueError("structure_consistency must be 'shared' or 'per_cache'")
    expected = geometry
    if cached_structure is not None:
        if not isinstance(cached_structure, Mapping):
            raise ValueError(f"{name} structure provenance must be an object")
        if cached_structure.get("assay_id") != assay_id:
            raise ValueError(f"{name} structure provenance belongs to a different assay")
        if consistency == "shared" and cached_structure["input_signature"] != geometry["input_signature"]:
            raise ValueError(f"Selected structure differs from the {name} feature structure")
        if consistency == "per_cache":
            expected = cached_structure
    for recorded in recorded_provenance:
        if not isinstance(recorded, Mapping):
            raise ValueError(f"{name} structure provenance must be an object")
        for key, value in (("structure_sha256", expected["structure_sha256"]),
                           ("structure_input_signature", expected["input_signature"])):
            if key in recorded and recorded[key] != value:
                raise ValueError(f"{name} provenance disagrees with expected structure: {key}")
