"""Frame extraction, image-quality diagnostics, and deterministic preprocessing."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def extract_frames(
    video: Path,
    output_dir: Path,
    frames_per_second: float = 1.0,
    jpeg_quality: int = 2,
) -> dict[str, Any]:
    if frames_per_second <= 0:
        raise ValueError("frames_per_second must be positive")
    if not video.is_file():
        raise FileNotFoundError(video)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-vf",
        f"fps={frames_per_second:g}",
        "-q:v",
        str(jpeg_quality),
        "-start_number",
        "0",
        str(output_dir / "frame_%06d.jpg"),
    ]
    subprocess.run(command, check=True)
    images = list_images(output_dir)
    if not images:
        raise RuntimeError("ffmpeg produced no frames")
    return {
        "schema": "nht_frame_extraction_v1",
        "input_video": str(video.resolve()),
        "frames_per_second": frames_per_second,
        "frame_count": len(images),
        "image_directory": str(output_dir),
        "command": command,
    }


def image_quality(path: Path) -> dict[str, float]:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
    center = gray[1:-1, 1:-1]
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * center
    )
    clipped = np.count_nonzero((gray <= 2.0) | (gray >= 253.0))
    return {
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "sharpness_laplacian_variance": float(laplacian.var()),
        "clipped_fraction": float(clipped / gray.size),
    }


def _image_signature(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            image.convert("L").resize((32, 18), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )


def preprocess_frames(
    source_dir: Path,
    output_dir: Path,
    frames_per_second: float,
    absolute_minimum_sharpness: float = 10.0,
    p05_sharpness_fraction: float = 0.15,
    maximum_clipped_fraction: float = 0.60,
    minimum_temporal_difference: float = 1.0,
) -> dict[str, Any]:
    sources = list_images(source_dir)
    if not sources:
        raise RuntimeError(f"No images found in {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    quality = [image_quality(path) for path in sources]
    signatures = [_image_signature(path) for path in sources]
    sharpness = np.asarray(
        [record["sharpness_laplacian_variance"] for record in quality]
    )
    sharpness_threshold = max(
        absolute_minimum_sharpness,
        float(np.percentile(sharpness, 5)) * p05_sharpness_fraction,
    )
    records: list[dict[str, Any]] = []
    rejected = 0
    previous_accepted_signature: np.ndarray | None = None
    for index, (source, measures, signature) in enumerate(
        zip(sources, quality, signatures, strict=True)
    ):
        reasons: list[str] = []
        if measures["sharpness_laplacian_variance"] < sharpness_threshold:
            reasons.append("extreme_blur")
        if measures["clipped_fraction"] > maximum_clipped_fraction:
            reasons.append("extreme_exposure_clipping")
        temporal_difference = (
            float(np.abs(signature - previous_accepted_signature).mean())
            if previous_accepted_signature is not None
            else None
        )
        if (
            temporal_difference is not None
            and temporal_difference < minimum_temporal_difference
        ):
            reasons.append("near_duplicate")
        accepted = not reasons
        if accepted:
            destination = output_dir / source.name
            destination.hardlink_to(source)
            previous_accepted_signature = signature
        else:
            rejected += 1
        records.append(
            {
                "source_frame_index": index,
                "source_time_seconds": index / frames_per_second,
                "filename": source.name,
                "accepted": accepted,
                "rejection_reasons": reasons,
                "temporal_difference_from_previous_accepted": temporal_difference,
                **measures,
            }
        )
    if len(sources) - rejected < 3:
        raise RuntimeError("Preprocessing rejected too many frames for SfM")
    return {
        "schema": "nht_frames_v1",
        "input_frame_count": len(sources),
        "accepted_frame_count": len(sources) - rejected,
        "rejected_frame_count": rejected,
        "sharpness_threshold": sharpness_threshold,
        "p05_sharpness": float(np.percentile(sharpness, 5)),
        "median_sharpness": float(np.median(sharpness)),
        "minimum_temporal_difference": minimum_temporal_difference,
        "frames": records,
    }


def downsample_images(
    source_dir: Path, output_dir: Path, factor: int
) -> dict[str, Any]:
    if factor < 1:
        raise ValueError("factor must be positive")
    sources = list_images(source_dir)
    if not sources:
        raise RuntimeError(f"No images found in {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    target_size: tuple[int, int] | None = None
    for source in sources:
        with Image.open(source) as image:
            target_size = (
                round(image.width / factor),
                round(image.height / factor),
            )
            resized = image.convert("RGB").resize(target_size, Image.Resampling.BICUBIC)
            resized.save(output_dir / f"{source.stem}.png", format="PNG")
    return {
        "factor": factor,
        "source_count": len(sources),
        "output_count": len(list_images(output_dir)),
        "target_size": list(target_size) if target_size else None,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)
