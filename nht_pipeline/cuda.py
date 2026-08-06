"""CUDA device selection that preserves an inherited visibility mask."""

from __future__ import annotations

from collections.abc import Mapping


def select_cuda_environment(
    base_environment: Mapping[str, str], logical_index: int
) -> tuple[dict[str, str], str]:
    """Return an environment exposing exactly one configured logical device.

    When a parent process has already set ``CUDA_VISIBLE_DEVICES``, indices are
    interpreted inside that inherited token list. Without a parent mask, the
    index is the bare-host CUDA ordinal.
    """

    if type(logical_index) is not int or logical_index < 0:
        raise ValueError("CUDA logical device index must be a non-negative integer")
    inherited_mask = base_environment.get("CUDA_VISIBLE_DEVICES")
    if inherited_mask is None:
        selected_token = str(logical_index)
    else:
        tokens = [token.strip() for token in inherited_mask.split(",")]
        if not tokens or any(not token for token in tokens):
            raise ValueError("CUDA_VISIBLE_DEVICES contains an empty device token")
        if logical_index >= len(tokens):
            raise ValueError(
                f"CUDA logical device {logical_index} is outside inherited mask "
                f"{inherited_mask!r}"
            )
        selected_token = tokens[logical_index]
    environment = dict(base_environment)
    environment["CUDA_VISIBLE_DEVICES"] = selected_token
    return environment, selected_token
