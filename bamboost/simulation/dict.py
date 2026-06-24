from __future__ import annotations

from bamboost.exceptions import ForbiddenParameterKeyError

RESERVED_KEYS = {
    "id",
    "collection_uid",
    "name",
    "created_at",
    "modified_at",
    "description",
    "tags",
    "status",
    "submitted",
    "links",
}


def validate_parameter_key(key: str) -> None:
    """Validate a parameter key.

    Args:
        key: The key to validate.

    Raises:
        ValueError: If the key is reserved or contains a period.
    """
    # only top-level keys are checked for reservation and periods.
    # nested keys (e.g., 'params.nested_key') are already handled by the flattening logic
    # but we want to prevent users from manually passing a key with a period
    # which would interfere with our flattening/unflattening.
    if key in RESERVED_KEYS:
        raise ForbiddenParameterKeyError(f"Parameter key '{key}' is reserved.")
    if "." in key:
        raise ForbiddenParameterKeyError(
            f"Parameter key '{key}' cannot contain a period ('.')."
        )
