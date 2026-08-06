"""Render RGB, alpha, and depth using only a standard exported scene."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .export import (
    ValidatedSceneExport,
    load_validated_scene_export,
    validate_pinhole_camera,
)
from .schema import validate_schema_payload


def _safe_identifier(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Camera ID is not path-safe: {value!r}")
    return value


def _load_requests(
    observed_cameras: list[dict[str, Any]],
    camera_ids: list[str] | None,
    request_path: Path | None,
) -> list[dict[str, Any]]:
    observed = {
        camera["camera_id"]: {**camera, "request_source": "observed"}
        for camera in observed_cameras
    }
    requests: list[dict[str, Any]] = []
    if camera_ids:
        missing = [
            identifier for identifier in camera_ids if identifier not in observed
        ]
        if missing:
            raise ValueError(f"Unknown observed camera IDs: {missing}")
        requests.extend(observed[identifier] for identifier in camera_ids)
    if request_path is not None:
        payload = json.loads(request_path.read_text())
        validate_schema_payload(
            "render-request", payload, context="Arbitrary camera request"
        )
        arbitrary = payload["cameras"]
        for camera in arbitrary:
            requests.append({**camera, "request_source": "arbitrary"})
    if not requests:
        requests = list(observed.values())
    identifiers: set[str] = set()
    for request in requests:
        identifier = _safe_identifier(str(request["camera_id"]))
        if identifier in identifiers:
            raise ValueError(f"Duplicate render camera ID: {identifier}")
        identifiers.add(identifier)
        validate_pinhole_camera(request, identifier)
    return requests


def _render_one(
    checkpoint: Path,
    config: dict[str, Any],
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
            rasterize_mode=("antialiased" if config["antialiased"] else "classic"),
            render_mode="RGB+ED",
            distributed=False,
            camera_model=str(config["camera_model"]),
            with_ut=bool(config["with_ut"]),
            with_eval3d=bool(config["with_eval3d"]),
            near_plane=float(config["near_plane"]),
            far_plane=float(config["far_plane"]),
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


def _validate_replaceable_output(output: Path, scene_id: str) -> None:
    if not output.exists():
        return
    if output.is_symlink() or not output.is_dir():
        raise ValueError("Render output changed to an unsafe target")
    if not any(output.iterdir()):
        return
    marker = output / "render.json"
    if marker.is_symlink() or not marker.is_file():
        raise ValueError(
            "Non-empty render output has no ordinary NHT render ownership marker"
        )
    try:
        ownership = json.loads(marker.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Render output ownership marker is invalid") from error
    validate_schema_payload(
        "render-result", ownership, context="Render output ownership marker"
    )
    if ownership.get("scene_id") != scene_id:
        raise ValueError("Render output ownership marker belongs to another scene")


def _safe_output_path(output: Path, validated: ValidatedSceneExport) -> Path:
    if output.is_symlink():
        raise ValueError("Render output cannot be a symbolic link")
    resolved = output.expanduser().resolve(strict=False)
    if resolved.parent == resolved:
        raise ValueError("Render output cannot be the filesystem root")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError("Render output must be a directory path, not an ordinary file")
    export_root = validated.export_root
    workspace = export_root.parent
    if workspace.is_relative_to(resolved):
        raise ValueError("Render output cannot be the scene workspace or its ancestor")
    if resolved.is_relative_to(export_root):
        raise ValueError("Render output cannot be the export root or its descendants")
    _validate_replaceable_output(resolved, str(validated.scene["scene_id"]))
    return resolved


def render_scene(
    scene_path: Path,
    output: Path,
    *,
    camera_ids: list[str] | None = None,
    request_path: Path | None = None,
) -> dict[str, Any]:
    validated = load_validated_scene_export(scene_path)
    scene = validated.scene
    requests = _load_requests(validated.cameras, camera_ids, request_path)
    output = _safe_output_path(output, validated)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".staging", dir=output.parent
        )
    )
    records: list[dict[str, Any]] = []
    try:
        for request in requests:
            identifier = _safe_identifier(str(request["camera_id"]))
            camera_root = staging / identifier
            camera_root.mkdir()
            rgb, alpha, depth = _render_one(
                validated.checkpoint_path,
                validated.runtime_config,
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
                raise RuntimeError(
                    f"Renderer returned RGB/alpha outside [0,1]: {identifier}"
                )
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
            "export_validation": validated.validation,
            "renders": records,
        }
        validate_schema_payload(
            "render-result", manifest, context="Generated render result"
        )
        (staging / "render.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output.exists():
            _validate_replaceable_output(output, str(scene["scene_id"]))
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
