"""Classic SIFT plus COLMAP incremental reconstruction candidate."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def camera_mode_from_config(config: dict[str, Any], pycolmap: Any) -> Any:
    sharing = config.get("camera_sharing", "single")
    modes = {
        "single": pycolmap.CameraMode.SINGLE,
        "per_folder": pycolmap.CameraMode.PER_FOLDER,
        "per_image": pycolmap.CameraMode.PER_IMAGE,
        "segments": pycolmap.CameraMode.PER_IMAGE,
    }
    try:
        return modes[sharing]
    except KeyError as error:
        raise ValueError(f"Unsupported camera_sharing: {sharing}") from error


def _share_segment_cameras(database_path: Path, segment_size: int, pycolmap: Any) -> None:
    if segment_size < 2:
        raise ValueError("segment_size must be at least 2")
    database = pycolmap.Database(database_path)
    images = sorted(database.read_all_images(), key=lambda image: image.name)
    for start in range(0, len(images), segment_size):
        group = images[start : start + segment_size]
        shared_camera_id = group[0].camera_id
        for image in group[1:]:
            image.camera_id = shared_camera_id
            database.update_image(image)
    database.close()


def run_sift_incremental(
    image_dir: Path,
    candidate_dir: Path,
    config: dict[str, Any],
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    try:
        import pycolmap
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise RuntimeError("pycolmap is required for the SIFT candidate") from error

    started = time.monotonic()
    database_path = candidate_dir / "database.db"
    sparse_root = candidate_dir / "models"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    sparse_root.mkdir(parents=True, exist_ok=True)

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = config.get("camera_model", "OPENCV")
    reader_options.default_focal_length_factor = 1.0
    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.max_image_size = int(config.get("max_image_size", 1920))
    extraction_options.num_threads = int(config.get("num_threads", 4))
    extraction_options.use_gpu = False
    sift_extraction = pycolmap.SiftExtractionOptions()
    sift_extraction.max_num_features = int(config.get("max_features", 4096))
    sift_extraction.max_num_orientations = 1
    sift_extraction.peak_threshold = 0.006
    extraction_options.sift = sift_extraction
    pycolmap.extract_features(
        database_path,
        image_dir,
        camera_mode=camera_mode_from_config(config, pycolmap),
        reader_options=reader_options,
        extraction_options=extraction_options,
        device=pycolmap.Device.cpu,
    )
    if config.get("camera_sharing", "single") == "segments":
        _share_segment_cameras(
            database_path, int(config.get("camera_segment_size", 30)), pycolmap
        )

    pairing_options = pycolmap.SequentialPairingOptions()
    pairing_options.overlap = int(config.get("sequential_overlap", 10))
    pairing_options.quadratic_overlap = bool(config.get("quadratic_overlap", True))
    pairing_options.loop_detection = False
    pairing_options.num_threads = int(config.get("num_threads", 4))
    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.num_threads = int(config.get("num_threads", 4))
    matching_options.use_gpu = False
    matching_options.guided_matching = True
    matching_options.max_num_matches = 8192
    sift_matching = pycolmap.SiftMatchingOptions()
    sift_matching.max_ratio = 0.85
    sift_matching.cross_check = True
    matching_options.sift = sift_matching
    pycolmap.match_sequential(
        database_path,
        matching_options=matching_options,
        pairing_options=pairing_options,
        device=pycolmap.Device.cpu,
    )

    mapping_options = pycolmap.IncrementalPipelineOptions()
    mapping_options.multiple_models = False
    mapping_options.min_model_size = 10
    mapping_options.ba_refine_focal_length = True
    mapping_options.ba_refine_principal_point = False
    mapping_options.ba_refine_extra_params = True
    mapping_options.random_seed = seed
    mapping_options.num_threads = int(config.get("num_threads", 4))
    mapping_options.mapper.num_threads = int(config.get("num_threads", 4))
    reconstructions = pycolmap.incremental_mapping(
        database_path,
        image_dir,
        sparse_root,
        options=mapping_options,
    )
    if not reconstructions:
        raise RuntimeError("COLMAP produced no sparse reconstruction")
    reconstruction = max(
        reconstructions.values(),
        key=lambda item: (item.num_reg_images(), item.num_points3D()),
    )
    model_dir = candidate_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(model_dir)
    return reconstruction, {
        "backend": "pycolmap_sift_incremental",
        "elapsed_seconds": time.monotonic() - started,
        "models_produced": len(reconstructions),
        "selected_registered_images": reconstruction.num_reg_images(),
        "selected_sparse_points": reconstruction.num_points3D(),
        "camera_sharing": config.get("camera_sharing", "single"),
        "reconstructed_cameras": reconstruction.num_cameras(),
        "effective_seed": seed,
    }
