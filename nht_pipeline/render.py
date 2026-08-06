"""Render RGB, alpha, and depth using only a standard exported scene."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .export import validate_scene_export


def _safe_identifier(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Camera ID is not path-safe: {value!r}")
    return value


def _load_requests(
    export_root: Path,
    camera_ids: list[str] | None,
    request_path: Path | None,
) -> list[dict[str, Any]]:
    scene = json.loads((export_root / "scene.json").read_text())
    observed_payload = json.loads((export_root / scene["cameras"]).read_text())
    observed = {
        camera["camera_id"]: {**camera, "request_source": "observed"}
        for camera in observed_payload["cameras"]
    }
    requests: list[dict[str, Any]] = []
    if camera_ids:
        missing = [identifier for identifier in camera_ids if identifier not in observed]
        if missing:
            raise ValueError(f"Unknown observed camera IDs: {missing}")
        requests.extend(observed[identifier] for identifier in camera_ids)
    if request_path is not None:
        payload = json.loads(request_path.read_text())
        if payload.get("schema") != "nht_render_request_v1":
            raise ValueError("Unsupported arbitrary camera request schema")
        for camera in payload.get("cameras", []):
            requests.append({**camera, "request_source": "arbitrary"})
    if not requests:
        requests = list(observed.values())
    identifiers: set[str] = set()
    for request in requests:
        identifier = _safe_identifier(str(request["camera_id"]))
        if identifier in identifiers:
            raise ValueError(f"Duplicate render camera ID: {identifier}")
        identifiers.add(identifier)
        matrix = np.asarray(request["camera_to_scene"], dtype=np.float64)
        intrinsics = np.asarray(request["intrinsics"]["matrix"], dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid camera_to_scene for {request['camera_id']}")
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            raise ValueError(f"Invalid intrinsics for {request['camera_id']}")
        if request["intrinsics"].get("model") != "PINHOLE":
            raise ValueError(f"Unsupported projection model for {identifier}")
        if request["intrinsics"].get("distortion_model") != "NONE":
            raise ValueError(f"Arbitrary camera must be undistorted: {identifier}")
        if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
            raise ValueError(f"Non-positive focal length for {identifier}")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-5
        ):
            raise ValueError(f"camera_to_scene is not a proper rigid pose: {identifier}")
        if int(request["width"]) <= 0 or int(request["height"]) <= 0:
            raise ValueError(f"Invalid resolution for {request['camera_id']}")
    return requests


def _render_one(
    checkpoint: Path,
    runtime_config: Path,
    request: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import torch
        from gsplat.nht.deferred_shader import DeferredShaderModule
        from gsplat.rendering import rasterization
    except ImportError as error:  # pragma: no cover - GPU runtime dependency
        raise RuntimeError(
            "nht-render must run in an environment containing torch and the NHT "
            "gsplat runtime"
        ) from error

    if not torch.cuda.is_available():
        raise RuntimeError("nht-render requires a CUDA device")
    config = json.loads(runtime_config.read_text())
    if config.get("schema") != "nht_runtime_config_v1":
        raise ValueError("Unsupported NHT runtime configuration")
    device = torch.device("cuda:0")
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    splats = {name: value.to(device) for name, value in payload["splats"].items()}
    shader = DeferredShaderModule(
        feature_dim=int(config["deferred_opt_feature_dim"]),
        enable_view_encoding=bool(config["deferred_opt_enable_view_encoding"]),
        view_encoding_type=str(config["deferred_opt_view_encoding_type"]),
        mlp_hidden_dim=int(config["deferred_mlp_hidden_dim"]),
        mlp_num_layers=int(config["deferred_mlp_num_layers"]),
        sh_degree=int(config["deferred_opt_sh_degree"]),
        sh_scale=float(config["deferred_opt_sh_scale"]),
        fourier_num_freqs=int(config["deferred_opt_fourier_num_freqs"]),
        primitive_type=str(config["primitive_type"]),
        center_ray_encoding=bool(config["deferred_opt_center_ray_encoding"]),
        decode_activation=str(config["deferred_decode_activation"]),
    ).to(device)
    shader.load_state_dict(payload.get("deferred_ema", payload["deferred_module"]))
    shader.eval()
    camera_to_scene = torch.as_tensor(
        request["camera_to_scene"], dtype=torch.float32, device=device
    ).unsqueeze(0)
    intrinsics = torch.as_tensor(
        request["intrinsics"]["matrix"], dtype=torch.float32, device=device
    ).unsqueeze(0)
    width = int(request["width"])
    height = int(request["height"])
    with torch.inference_mode():
        rendered, alpha, _ = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=splats["features"],
            sh_degree=None,
            viewmats=torch.linalg.inv(camera_to_scene),
            Ks=intrinsics,
            width=width,
            height=height,
            tile_size=int(config["tile_size"]),
            packed=bool(config["packed"]),
            rasterize_mode=(
                "antialiased" if config["antialiased"] else "classic"
            ),
            render_mode="RGB+ED",
            distributed=False,
            camera_model=str(config["camera_model"]),
            with_ut=bool(config["with_ut"]),
            with_eval3d=bool(config["with_eval3d"]),
            nht=True,
            center_ray_mode=bool(config["deferred_opt_center_ray_encoding"]),
            ray_dir_scale=shader.ray_dir_scale,
        )
        rgb, extras = shader(rendered)
    if extras is None or extras.shape[-1] < 1:
        raise RuntimeError("NHT runtime did not return expected depth")
    return (
        rgb[0].clamp(0, 1).float().cpu().numpy(),
        alpha[0].clamp(0, 1).float().cpu().numpy(),
        extras[0, ..., :1].float().cpu().numpy(),
    )


def render_scene(
    scene_path: Path,
    output: Path,
    *,
    camera_ids: list[str] | None = None,
    request_path: Path | None = None,
) -> dict[str, Any]:
    scene_path = scene_path.resolve()
    export_root = scene_path.parent
    validation = validate_scene_export(export_root)
    scene = json.loads(scene_path.read_text())
    renderer = scene["renderer"]
    requests = _load_requests(export_root, camera_ids, request_path)
    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    try:
        for request in requests:
            identifier = _safe_identifier(str(request["camera_id"]))
            camera_root = staging / identifier
            camera_root.mkdir()
            rgb, alpha, depth = _render_one(
                export_root / renderer["checkpoint"],
                export_root / renderer["runtime_config"],
                request,
            )
            expected_shapes = (
                (int(request["height"]), int(request["width"]), 3),
                (int(request["height"]), int(request["width"]), 1),
                (int(request["height"]), int(request["width"]), 1),
            )
            for name, value, expected_shape in zip(
                ("rgb", "alpha", "depth"),
                (rgb, alpha, depth),
                expected_shapes,
                strict=True,
            ):
                if value.shape != expected_shape or not np.isfinite(value).all():
                    raise RuntimeError(
                        f"Renderer returned invalid {name} for {identifier}: "
                        f"shape={value.shape}"
                    )
            if rgb.min() < 0 or rgb.max() > 1 or alpha.min() < 0 or alpha.max() > 1:
                raise RuntimeError(f"Renderer returned RGB/alpha outside [0,1]: {identifier}")
            if depth.min() < 0:
                raise RuntimeError(f"Renderer returned negative depth: {identifier}")
            np.save(camera_root / "rgb.npy", rgb.astype(np.float32))
            np.save(camera_root / "alpha.npy", alpha.astype(np.float32))
            np.save(camera_root / "depth.npy", depth.astype(np.float32))
            Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8)).save(
                camera_root / "rgb.png"
            )
            Image.fromarray(
                (alpha[..., 0] * 255.0 + 0.5).astype(np.uint8), mode="L"
            ).save(camera_root / "alpha.png")
            records.append(
                {
                    "camera_id": identifier,
                    "request_source": request["request_source"],
                    "width": int(request["width"]),
                    "height": int(request["height"]),
                    "rgb": f"{identifier}/rgb.npy",
                    "rgb_preview": f"{identifier}/rgb.png",
                    "alpha": f"{identifier}/alpha.npy",
                    "alpha_preview": f"{identifier}/alpha.png",
                    "depth": f"{identifier}/depth.npy",
                }
            )
        manifest = {
            "schema": "nht_render_result_v1",
            "scene_schema": scene["schema"],
            "scene_id": scene["scene_id"],
            "coordinate_space": "canonical NHT scene space",
            "export_validation": validation,
            "renders": records,
        }
        (staging / "render.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--camera-id", action="append", dest="camera_ids")
    parser.add_argument("--cameras", type=Path, help="nht_render_request_v1 JSON")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    result = render_scene(
        args.scene,
        args.output,
        camera_ids=args.camera_ids,
        request_path=args.cameras,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
