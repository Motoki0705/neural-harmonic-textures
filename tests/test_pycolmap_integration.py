from __future__ import annotations

import numpy as np
from PIL import Image

from nht_pipeline.sfm.classic import run_sift_incremental
from nht_pipeline.sfm.metrics import evaluate_reconstruction


def _write_multiview_fixture(image_root) -> None:
    image_root.mkdir()
    random = np.random.default_rng(7)
    points = np.column_stack(
        [
            random.uniform(-2, 2, 350),
            random.uniform(-1.3, 1.3, 350),
            random.uniform(4, 9, 350),
        ]
    )
    patches = random.integers(0, 256, (350, 13, 13), dtype=np.uint8)
    for patch in patches:
        patch[::3, :] = 255 - patch[::3, :]
        patch[:, ::4] = 255 - patch[:, ::4]
    for frame, camera_x in enumerate(np.linspace(-0.7, 0.7, 14)):
        canvas = np.full((480, 640), 25, dtype=np.uint8)
        camera_points = points - np.asarray([camera_x, 0, 0])
        x = (520 * camera_points[:, 0] / camera_points[:, 2] + 320).round()
        y = (520 * camera_points[:, 1] / camera_points[:, 2] + 240).round()
        for index in np.argsort(camera_points[:, 2])[::-1]:
            column, row = int(x[index]), int(y[index])
            if 7 <= column < 633 and 7 <= row < 473:
                canvas[row - 6 : row + 7, column - 6 : column + 7] = patches[index]
        Image.fromarray(canvas).save(image_root / f"frame_{frame:06d}.png")


def test_real_pycolmap_reconstructs_small_multiview_fixture(tmp_path) -> None:
    image_root = tmp_path / "images"
    _write_multiview_fixture(image_root)
    candidate_root = tmp_path / "candidate"
    reconstruction, runtime = run_sift_incremental(
        image_root,
        candidate_root,
        {
            "camera_model": "PINHOLE",
            "camera_sharing": "single",
            "sequential_overlap": 6,
            "quadratic_overlap": True,
            "max_features": 4096,
            "max_image_size": 640,
            "num_threads": 1,
        },
        42,
    )

    assert reconstruction.num_reg_images() >= 10
    assert runtime["reconstructed_cameras"] == 1
    assert runtime["effective_seed"] == 42
    metrics = evaluate_reconstruction(
        candidate_root / "model",
        image_root,
        input_image_count=14,
        minimum_supported_points_per_image=1,
        thresholds={
            "minimum_registration_ratio": 0.5,
            "minimum_supported_registration_ratio": 0.5,
            "minimum_sparse_points": 10,
            "minimum_median_track_length": 2,
            "maximum_p95_reprojection_error_px": 5,
            "maximum_trajectory_step_ratio": 20,
            "maximum_trajectory_step_outliers": 5,
            "maximum_planarity_ratio": 1,
            "maximum_rotation_step_deg": 180,
            "maximum_mapping_components": 1,
            "maximum_near_duplicate_fraction": 1,
            "minimum_focal_to_width": 0.1,
            "maximum_focal_to_width": 5,
            "minimum_points_per_supported_camera": 1,
            "minimum_spatial_voxel_coverage_fraction": 0.001,
            "minimum_median_triangulation_angle_deg": 0.1,
            "maximum_focal_length_coefficient_of_variation": 0.2,
        },
    )
    assert metrics["mapping_components"] == 1
    assert metrics["triangulation"]["median_angle_deg"] > 0
    assert (
        metrics["intrinsics_stability"][
            "focal_length_coefficient_of_variation"
        ]
        < 1.0e-12
    )
