from __future__ import annotations

import numpy as np
from PIL import Image

from nht_pipeline.frames import preprocess_frames


def test_preprocessing_rejects_only_extreme_blur(tmp_path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "images"
    source.mkdir()
    rng = np.random.default_rng(42)
    for index in range(20):
        if index == 19:
            pixels = np.full((64, 64, 3), 128, dtype=np.uint8)
        else:
            pixels = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(source / f"frame_{index:06}.jpg")

    result = preprocess_frames(source, output, frames_per_second=1.0)
    assert result["accepted_frame_count"] == 19
    assert result["rejected_frame_count"] == 1
    assert result["frames"][-1]["rejection_reasons"] == ["extreme_blur"]
    assert len(list(output.glob("*.jpg"))) == 19


def test_preprocessing_rejects_near_duplicate_frame(tmp_path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "images"
    source.mkdir()
    rng = np.random.default_rng(7)
    first = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    third = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    fourth = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(first).save(source / "frame_000000.png")
    Image.fromarray(first).save(source / "frame_000001.png")
    Image.fromarray(third).save(source / "frame_000002.png")
    Image.fromarray(fourth).save(source / "frame_000003.png")

    result = preprocess_frames(source, output, frames_per_second=1.0)

    assert result["accepted_frame_count"] == 3
    assert result["frames"][1]["rejection_reasons"] == ["near_duplicate"]
