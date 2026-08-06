"""Compare two SfM camera trajectories after robustly neutralizing similarity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def similarity_alignment(
    source: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scale, rotation, translation mapping source onto target."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Source and target must be Nx3 arrays with equal shape")
    if len(source) < 3:
        raise ValueError("At least three shared cameras are required")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    signs = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        signs[-1] = -1
    rotation = u @ np.diag(signs) @ vt
    source_variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if source_variance <= 0:
        raise ValueError("Source camera centers have zero variance")
    scale = float(np.dot(singular_values, signs) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _camera_centers(model_path: Path) -> dict[str, np.ndarray]:
    try:
        import pycolmap
    except ImportError as error:  # pragma: no cover - depends on runtime
        raise RuntimeError("pycolmap is required to compare SfM models") from error
    reconstruction = pycolmap.Reconstruction(model_path)
    return {
        image.name: np.asarray(image.projection_center(), dtype=np.float64)
        for image in reconstruction.images.values()
    }


def compare_reconstructions(reference: Path, candidate: Path) -> dict[str, Any]:
    reference_centers = _camera_centers(reference)
    candidate_centers = _camera_centers(candidate)
    shared_names = sorted(set(reference_centers) & set(candidate_centers))
    reference_array = np.asarray(
        [reference_centers[name] for name in shared_names], dtype=np.float64
    )
    candidate_array = np.asarray(
        [candidate_centers[name] for name in shared_names], dtype=np.float64
    )
    scale, rotation, translation = similarity_alignment(
        candidate_array, reference_array
    )
    aligned = scale * (candidate_array @ rotation.T) + translation
    errors = np.linalg.norm(aligned - reference_array, axis=1)
    reference_steps = np.linalg.norm(np.diff(reference_array, axis=0), axis=1)
    normalization = float(np.median(reference_steps))
    worst = np.argsort(errors)[-10:][::-1]
    return {
        "reference": str(reference.resolve()),
        "candidate": str(candidate.resolve()),
        "shared_cameras": len(shared_names),
        "reference_only_cameras": sorted(
            set(reference_centers) - set(candidate_centers)
        ),
        "candidate_only_cameras": sorted(
            set(candidate_centers) - set(reference_centers)
        ),
        "alignment": {
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        },
        "camera_center_error": {
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "median": float(np.median(errors)),
            "p95": float(np.percentile(errors, 95)),
            "maximum": float(errors.max()),
            "median_normalized_by_reference_step": float(
                np.median(errors) / normalization
            ),
            "p95_normalized_by_reference_step": float(
                np.percentile(errors, 95) / normalization
            ),
        },
        "worst_camera_center_errors": [
            {"name": shared_names[index], "error": float(errors[index])}
            for index in worst
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_reconstructions(args.reference, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
