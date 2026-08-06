"""Classic SIFT plus COLMAP incremental reconstruction candidate."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any


def camera_mode_from_config(config: dict[str, Any], pycolmap: Any) -> Any:
    sharing = config.get("camera_sharing", "single")
    modes = {
        "single": pycolmap.CameraMode.SINGLE,
        "per_folder": pycolmap.CameraMode.PER_FOLDER,
        "per_image": pycolmap.CameraMode.PER_IMAGE,
        "segments": pycolmap.CameraMode.PER_FOLDER,
    }
    try:
        return modes[sharing]
    except KeyError as error:
        raise ValueError(f"Unsupported camera_sharing: {sharing}") from error


def _segment_image_root(
    image_dir: Path, candidate_dir: Path, segment_size: int
) -> Path:
    if segment_size < 2:
        raise ValueError("segment_size must be at least 2")
    images = sorted(path for path in image_dir.iterdir() if path.is_file())
    segmented = candidate_dir / "segmented-images"
    for index, image in enumerate(images):
        group = segmented / f"segment-{index // segment_size:04d}"
        group.mkdir(parents=True, exist_ok=True)
        destination = group / image.name
        try:
            destination.symlink_to(image.resolve())
        except OSError:
            try:
                destination.hardlink_to(image)
            except OSError:
                shutil.copy2(image, destination)
    return segmented


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
    extraction_image_dir = image_dir
    if config.get("camera_sharing", "single") == "segments":
        extraction_image_dir = _segment_image_root(
            image_dir,
            candidate_dir,
            int(config.get("camera_segment_size", 30)),
        )

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
        extraction_image_dir,
        camera_mode=camera_mode_from_config(config, pycolmap),
        reader_options=reader_options,
        extraction_options=extraction_options,
        device=pycolmap.Device.cpu,
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
        extraction_image_dir,
        sparse_root,
        options=mapping_options,
    )
    if not reconstructions:
        raise RuntimeError("COLMAP produced no sparse reconstruction")
    reconstruction = max(
        reconstructions.values(),
        key=lambda item: (item.num_reg_images(), item.num_points3D()),
    )
    if config.get("camera_sharing", "single") == "segments":
        for image in reconstruction.images.values():
            image.name = Path(image.name).name
    model_dir = candidate_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(model_dir)
    if extraction_image_dir != image_dir:
        shutil.rmtree(extraction_image_dir)
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
