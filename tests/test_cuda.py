from __future__ import annotations

import pytest

from nht_pipeline.cuda import select_cuda_environment


@pytest.mark.parametrize(
    "mask,logical_index,selected_token",
    [
        (None, 1, "1"),
        ("3", 0, "3"),
        ("2,5", 1, "5"),
    ],
)
def test_cuda_selection_preserves_inherited_mask_tokens(
    mask, logical_index, selected_token
) -> None:
    base = {"UNCHANGED": "yes"}
    if mask is not None:
        base["CUDA_VISIBLE_DEVICES"] = mask

    environment, selected = select_cuda_environment(base, logical_index)

    assert selected == selected_token
    assert environment["CUDA_VISIBLE_DEVICES"] == selected_token
    assert environment["UNCHANGED"] == "yes"
    assert base.get("CUDA_VISIBLE_DEVICES") == mask


@pytest.mark.parametrize(
    "mask,index,match",
    [
        ("3", 1, "outside inherited mask"),
        ("2,,5", 0, "empty device token"),
    ],
)
def test_cuda_selection_rejects_invalid_logical_selection(mask, index, match) -> None:
    with pytest.raises(ValueError, match=match):
        select_cuda_environment({"CUDA_VISIBLE_DEVICES": mask}, index)
