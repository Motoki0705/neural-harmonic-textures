"""Opt-in real NHT adapter/export/render round-trip verification."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from nht_pipeline.export import create_scene_export


def _required_environment(name: str, *, preserve_symlink: bool = False) -> Path:
    value = os.environ.get(name)
    if value is None:
        pytest.skip(
            "real GPU smoke requires NHT_GPU_PYTHON, NHT_GPU_TRAINER, and "
            "NHT_GPU_SOURCE_WORKSPACE"
        )
    path = (
        Path(value).expanduser().absolute()
        if preserve_symlink
        else Path(value).resolve()
    )
    if not path.exists():
        pytest.fail(f"{name} does not exist: {path}")
    return path


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=source.is_dir())


@pytest.mark.gpu
def test_real_nht_export_render_round_trip(tmp_path: Path) -> None:
    nht_python = _required_environment("NHT_GPU_PYTHON", preserve_symlink=True)
    trainer = _required_environment("NHT_GPU_TRAINER")
    source = _required_environment("NHT_GPU_SOURCE_WORKSPACE")
    repository = Path(__file__).resolve().parents[1]
    adapter = repository / "nht_pipeline/nht_adapter.py"
    runtime_probe = subprocess.run(
        [str(nht_python), str(adapter), "probe", "--trainer", str(trainer)],
        cwd=trainer.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    runtime = json.loads(runtime_probe.stdout)

    dataset = tmp_path / "3dgs/dataset"
    _link(source / "sfm/model", dataset / "sparse/0")
    _link(source / "frames/images", dataset / "images")
    _link(source / "frames/training-images", dataset / "images_2")
    _link(source / "sfm/model", tmp_path / "sfm/model")
    _link(source / "sfm/reconstruction.json", tmp_path / "sfm/reconstruction.json")
    _link(source / "frames/frames.json", tmp_path / "frames/frames.json")

    model = tmp_path / "3dgs/model"
    command = [
        str(nht_python),
        str(adapter),
        "train",
        "--trainer",
        str(trainer),
        "--seed",
        "137",
        "--seed-output",
        str(tmp_path / "3dgs/effective-seed.json"),
        "--metadata-output",
        str(tmp_path / "3dgs/scene-metadata.json"),
        "--observed-image-root",
        str(tmp_path / "3dgs/observed-images"),
        "--runtime-output",
        str(model / "runtime-config.json"),
        "--",
        "default",
        "--data_dir",
        str(dataset),
        "--data_factor",
        "2",
        "--native_images_factor",
        "--result_dir",
        str(model),
        "--camera_model",
        "pinhole",
        "--near_plane",
        "0.125",
        "--far_plane",
        "456.0",
        "--test_every",
        "8",
        "--max_steps",
        "1",
        "--eval_steps",
        "1",
        "--save_steps",
        "1",
        "--disable_video",
        "--lpips_net",
        "alex",
        "--strategy.cap-max",
        "300000",
        "--num_workers",
        "0",
        "--disable_viewer",
    ]
    subprocess.run(command, cwd=trainer.parent, check=True)

    seed = json.loads((tmp_path / "3dgs/effective-seed.json").read_text())
    assert seed["requested_seed"] == seed["effective_seed"] == 137
    metadata = json.loads((tmp_path / "3dgs/scene-metadata.json").read_text())
    assert metadata["normalization"]["applied"] is True
    assert np.allclose(
        np.asarray(metadata["sfm_to_scene"]) @ np.asarray(metadata["scene_to_sfm"]),
        np.eye(4),
        atol=1e-6,
    )
    runtime_config = json.loads((model / "runtime-config.json").read_text())
    assert runtime_config["camera_model"] == "pinhole"
    assert runtime_config["pose_opt"] is False
    assert runtime_config["post_processing"] is None
    assert runtime_config["near_plane"] == 0.125
    assert runtime_config["far_plane"] == 456.0

    training = {
        "schema": "nht_training_v1",
        "status": "completed",
        "checkpoint": "model/ckpts/ckpt_0_rank0.pt",
        "scene_metadata": "scene-metadata.json",
        "observed_images": "observed-images",
        "runtime_config": "model/runtime-config.json",
        "seed": 137,
        "effective_seed": 137,
        "validation_metrics": [],
    }
    (tmp_path / "3dgs/training.json").write_text(json.dumps(training, indent=2) + "\n")
    scene = create_scene_export(tmp_path, "gpu-round-trip", "nht_standard_scene_v1")

    cameras = json.loads((tmp_path / "export/cameras.json").read_text())["cameras"]
    observed = next(camera for camera in cameras if camera["split"] == "validation")
    render_environment = os.environ.copy()
    render_environment["PYTHONPATH"] = str(repository)
    observed_output = tmp_path / "render-observed"
    subprocess.run(
        [
            str(nht_python),
            "-m",
            "nht_pipeline.render",
            "--scene",
            str(tmp_path / "export/scene.json"),
            "--camera-id",
            observed["camera_id"],
            "--output",
            str(observed_output),
        ],
        cwd=repository,
        env=render_environment,
        check=True,
    )

    rendered = np.load(
        observed_output / observed["camera_id"] / "rgb.npy", allow_pickle=False
    )
    alpha = np.load(
        observed_output / observed["camera_id"] / "alpha.npy", allow_pickle=False
    )
    depth = np.load(
        observed_output / observed["camera_id"] / "depth.npy", allow_pickle=False
    )
    trainer_canvas = (
        np.asarray(Image.open(model / "renders/val_step0_0000.png"), dtype=np.float32)
        / 255.0
    )
    trainer_render = trainer_canvas[:, rendered.shape[1] :]
    error = rendered - trainer_render
    mean_absolute_error = float(np.mean(np.abs(error)))
    psnr = float(-10.0 * np.log10(np.mean(error**2)))
    assert mean_absolute_error < 0.01
    assert psnr > 40.0
    assert rendered.shape == (observed["height"], observed["width"], 3)
    assert (
        alpha.shape
        == depth.shape
        == (
            observed["height"],
            observed["width"],
            1,
        )
    )
    assert np.isfinite(rendered).all()
    assert np.isfinite(alpha).all()
    assert np.isfinite(depth).all()
    assert (depth[alpha > 0.01] > 0).all()

    arbitrary = {
        "schema": "nht_render_request_v1",
        "cameras": [
            {
                "camera_id": "arbitrary-copy",
                "width": observed["width"],
                "height": observed["height"],
                "intrinsics": observed["intrinsics"],
                "camera_to_scene": observed["camera_to_scene"],
            }
        ],
    }
    request_path = tmp_path / "arbitrary-request.json"
    request_path.write_text(json.dumps(arbitrary, indent=2) + "\n")
    arbitrary_output = tmp_path / "render-arbitrary"
    subprocess.run(
        [
            str(nht_python),
            "-m",
            "nht_pipeline.render",
            "--scene",
            str(tmp_path / "export/scene.json"),
            "--cameras",
            str(request_path),
            "--output",
            str(arbitrary_output),
        ],
        cwd=repository,
        env=render_environment,
        check=True,
    )
    result = json.loads((arbitrary_output / "render.json").read_text())
    assert result["renders"][0]["request_source"] == "arbitrary"
    for output_name in ("rgb", "alpha", "depth"):
        arbitrary_array = np.load(
            arbitrary_output / "arbitrary-copy" / f"{output_name}.npy",
            allow_pickle=False,
        )
        assert np.isfinite(arbitrary_array).all()

    report = {
        "schema": "nht_gpu_round_trip_evidence_v1",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "source_workspace": str(source),
        "runtime": runtime,
        "effective_seed": seed["effective_seed"],
        "camera_id": observed["camera_id"],
        "camera_count": len(cameras),
        "point_count": scene["point_cloud"]["shape"][0],
        "trainer_render_mae": mean_absolute_error,
        "trainer_render_psnr_db": psnr,
        "acceptance_thresholds": {
            "maximum_trainer_render_mae": 0.01,
            "minimum_trainer_render_psnr_db": 40.0,
        },
        "observed_rgb_alpha_depth": True,
        "arbitrary_rgb_alpha_depth": True,
        "near_plane": runtime_config["near_plane"],
        "far_plane": runtime_config["far_plane"],
    }
    report_path = os.environ.get("NHT_GPU_REPORT")
    if report_path:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
