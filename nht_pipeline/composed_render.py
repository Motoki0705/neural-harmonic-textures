"""Jointly rasterize a static NHT checkpoint and a dynamic Gaussian asset."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, Protocol, cast

import numpy as np
from PIL import Image

from .export import load_validated_scene_export
from .render import (
    _load_requests,
    _safe_identifier,
    _safe_output_path,
    _validate_replaceable_output,
)
from .schema import validate_schema_payload

_ASSET_ARRAY_KEYS = {
    "means_m",
    "quats_wxyz",
    "log_scales_m",
    "opacity_logits",
    "colors_linear_rgb",
}
_TIMELINE_ARRAY_KEYS = {
    "transforms_nht_from_asset",
    "present",
    "instance_ids",
}

# Default eval3d CUDA template instantiations from gsplat/cuda/csrc/Config.h.
# The vendored Python wrapper advertises additional NHT-only channel counts,
# so composed eval3d rendering must target this conservative compiled set
# before entering the wrapper.
_EVAL3D_CUDA_TOTAL_CHANNELS = (
    1,
    2,
    3,
    4,
    5,
    8,
    9,
    16,
    17,
    32,
    33,
    64,
    65,
    128,
    129,
    256,
    257,
    512,
    513,
)


@dataclass(frozen=True)
class _AssetArrays:
    means_m: np.ndarray
    quats_wxyz: np.ndarray
    log_scales_m: np.ndarray
    opacity_logits: np.ndarray
    colors_linear_rgb: np.ndarray

    @property
    def gaussian_count(self) -> int:
        return int(self.means_m.shape[0])


@dataclass(frozen=True)
class _TimelineArrays:
    transforms_nht_from_asset: np.ndarray
    present: np.ndarray
    instance_ids: np.ndarray


@dataclass(frozen=True)
class _CompositionRequest:
    payload: dict[str, Any]
    asset: _AssetArrays
    timeline: _TimelineArrays
    chunks: tuple[tuple[int, ...], ...]

    @property
    def frame_count(self) -> int:
        return int(self.timeline.present.shape[0])

    @property
    def object_count(self) -> int:
        return int(self.timeline.present.shape[1])


@dataclass(frozen=True)
class _JointEval3DChannelLayout:
    logical_input_channels: int
    physical_input_channels: int
    physical_total_channels: int


def _joint_eval3d_channel_layout(object_count: int) -> _JointEval3DChannelLayout:
    """Resolve RGB, semantic, and ED channels against compiled CUDA templates."""
    if isinstance(object_count, bool) or not isinstance(object_count, int):
        raise TypeError("Joint eval3d object_count must be an integer")
    if object_count <= 0:
        raise ValueError("Joint eval3d object_count must be positive")
    logical_input = 3 + object_count
    logical_total = logical_input + 1
    physical_total = next(
        (
            supported
            for supported in _EVAL3D_CUDA_TOTAL_CHANNELS
            if supported >= logical_total
        ),
        None,
    )
    if physical_total is None:
        maximum_objects = _EVAL3D_CUDA_TOTAL_CHANNELS[-1] - 4
        raise ValueError(
            "Joint eval3d composition exceeds the compiled CUDA channel limit "
            f"of {maximum_objects} objects"
        )
    return _JointEval3DChannelLayout(
        logical_input_channels=logical_input,
        physical_input_channels=physical_total - 1,
        physical_total_channels=physical_total,
    )


def _pad_joint_eval3d_channels(
    torch: Any,
    channels: Any,
    *,
    layout: _JointEval3DChannelLayout,
) -> Any:
    """Tail-pad logical RGB/semantic features before RGB+ED adds depth."""
    if channels.shape[-1] != layout.logical_input_channels:
        raise RuntimeError("Joint eval3d logical input channel count is inconsistent")
    padding = layout.physical_input_channels - layout.logical_input_channels
    if padding == 0:
        return channels
    zeros = torch.zeros(
        *channels.shape[:-1],
        padding,
        dtype=channels.dtype,
        device=channels.device,
    )
    return torch.cat((channels, zeros), dim=-1)


def _split_joint_eval3d_output(
    rendered: Any,
    *,
    object_count: int,
    layout: _JointEval3DChannelLayout,
) -> tuple[Any, Any, Any]:
    """Discard physical padding while preserving ED at the final channel."""
    if rendered.shape[-1] != layout.physical_total_channels:
        raise RuntimeError("Joint eval3d physical output channel count is inconsistent")
    semantic_stop = 3 + object_count
    return (
        rendered[..., :3],
        rendered[..., 3:semantic_stop],
        rendered[..., -1:],
    )


class _ComposedRenderer(Protocol):
    @property
    def cuda_peak_bytes(self) -> int: ...

    def render_background(
        self, request: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def render_frame(
        self, request: dict[str, Any], frame_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


class _TorchComposedRenderer:
    """One-process CUDA renderer that keeps the NHT scene resident."""

    def __init__(
        self,
        checkpoint: Path,
        config: dict[str, Any],
        composition: _CompositionRequest,
        _requests: list[dict[str, Any]],
    ) -> None:
        try:
            import torch
            from gsplat.nht.deferred_shader import DeferredShaderModule
            from gsplat.rendering import rasterization
        except ImportError as error:  # pragma: no cover - GPU runtime dependency
            raise RuntimeError(
                "Composed nht-render requires torch and the NHT gsplat runtime"
            ) from error
        if not torch.cuda.is_available():
            raise RuntimeError("Composed nht-render requires a CUDA device")
        if config["primitive_type"] != "3dgs":
            raise RuntimeError(
                "Joint direct-color Gaussian composition requires a 3DGS NHT scene"
            )
        if (
            bool(config["packed"])
            or not bool(config["with_ut"])
            or not bool(config["with_eval3d"])
            or str(config["camera_model"]) != "pinhole"
        ):
            raise RuntimeError(
                "Composed NHT rendering requires packed=False, with_ut=True, "
                "with_eval3d=True, and the pinhole camera model"
            )

        self.torch = torch
        self.rasterization = rasterization
        self.config = config
        self.composition = composition
        self.device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats(self.device)
        payload = torch.load(checkpoint, map_location=self.device, weights_only=True)
        required = {"means", "quats", "scales", "opacities", "features"}
        if not isinstance(payload.get("splats"), dict) or not required.issubset(
            payload["splats"]
        ):
            raise ValueError("NHT checkpoint is missing required Gaussian tensors")
        self.background = {
            name: payload["splats"][name].to(self.device)
            for name in sorted(required)
        }
        feature_dim = int(config["deferred_opt_feature_dim"])
        if self.background["features"].ndim != 2 or self.background[
            "features"
        ].shape[1] != feature_dim:
            raise ValueError("NHT checkpoint feature dimension disagrees with runtime config")
        self.shader = DeferredShaderModule(
            feature_dim=feature_dim,
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
        ).to(self.device)
        shader_state = payload.get("deferred_ema")
        if shader_state is None:
            shader_state = payload.get("deferred_module")
        if shader_state is None:
            raise ValueError("NHT checkpoint is missing the deferred shader state")
        self.shader.load_state_dict(shader_state)
        self.shader.eval()
        for parameter in self.shader.parameters():
            parameter.requires_grad_(False)

        asset = composition.asset
        self.asset = {
            "means": torch.as_tensor(
                asset.means_m, dtype=torch.float32, device=self.device
            ),
            "quats": torch.as_tensor(
                asset.quats_wxyz, dtype=torch.float32, device=self.device
            ),
            "scales": torch.as_tensor(
                asset.log_scales_m, dtype=torch.float32, device=self.device
            ),
            "opacities": torch.as_tensor(
                asset.opacity_logits, dtype=torch.float32, device=self.device
            ),
            "colors": torch.as_tensor(
                asset.colors_linear_rgb, dtype=torch.float32, device=self.device
            ),
        }
        self.transforms = torch.as_tensor(
            composition.timeline.transforms_nht_from_asset,
            dtype=torch.float32,
            device=self.device,
        )
        self.instance_ids = torch.as_tensor(
            composition.timeline.instance_ids,
            dtype=torch.int32,
            device=self.device,
        )
        self._background_cache: dict[str, tuple[Any, Any, Any]] = {}

    @property
    def cuda_peak_bytes(self) -> int:
        self.torch.cuda.synchronize(self.device)
        return int(self.torch.cuda.max_memory_allocated(self.device))

    def render_background(
        self, request: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb, alpha, depth = self._cached_background(request)
        return self._to_numpy(rgb, alpha, depth)

    def render_frame(
        self, request: dict[str, Any], frame_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        background_rgb, background_alpha, background_depth = (
            self._cached_background(request)
        )
        active = np.flatnonzero(
            self.composition.timeline.present[frame_index]
        ).tolist()
        if not active:
            labels = self.torch.zeros(
                background_rgb.shape[:-1],
                dtype=self.torch.int32,
                device=self.device,
            )
            rgb_value, alpha_value, depth_value = self._to_numpy(
                background_rgb,
                background_alpha,
                background_depth,
            )
            return rgb_value, alpha_value, depth_value, labels[0].cpu().numpy()

        splats, direct_colors, semantics = self._frame_splats(
            frame_index,
            active=active,
        )
        direct_rgb, semantic_weights, alpha, depth = self._render_joint_asset_pass(
            request,
            splats,
            direct_colors=direct_colors,
            semantics=semantics,
        )
        foreground_weight = semantic_weights.sum(dim=-1, keepdim=True).clamp(0, 1)
        rgb = (
            direct_rgb + background_rgb * (1.0 - foreground_weight)
        ).clamp(0, 1)
        confidence, object_indices = self.torch.max(semantic_weights, dim=-1)
        threshold = float(self.composition.payload["visibility_threshold"])
        labels = self.torch.where(
            confidence >= threshold,
            self.instance_ids[object_indices],
            0,
        )
        rgb_value, alpha_value, depth_value = self._to_numpy(rgb, alpha, depth)
        return (
            rgb_value,
            alpha_value,
            depth_value,
            labels[0].to(dtype=self.torch.int32).cpu().numpy(),
        )

    def _frame_splats(
        self,
        frame_index: int,
        *,
        active: list[int],
    ) -> tuple[dict[str, Any], Any, Any]:
        torch = self.torch
        geometry_keys = ("means", "quats", "scales", "opacities")
        parts: dict[str, list[Any]] = {
            name: [self.background[name]] for name in geometry_keys
        }
        object_count = self.composition.object_count
        background_colors = torch.zeros(
            (self.background["means"].shape[0], 3),
            dtype=torch.float32,
            device=self.device,
        )
        background_semantics = torch.zeros(
            (self.background["means"].shape[0], object_count),
            dtype=torch.float32,
            device=self.device,
        )
        color_parts = [background_colors]
        semantic_parts = [background_semantics]
        for object_index in active:
            transform = self.transforms[frame_index, object_index]
            linear = transform[:3, :3]
            scale = torch.linalg.det(linear).pow(1.0 / 3.0)
            rotation = linear / scale
            rotation_quaternion = _rotation_matrix_to_quaternion(torch, rotation)
            means = self.asset["means"] @ linear.T + transform[:3, 3]
            quats = torch.nn.functional.normalize(
                _quaternion_multiply(
                    torch,
                    rotation_quaternion.expand_as(self.asset["quats"]),
                    self.asset["quats"],
                ),
                dim=1,
            )
            parts["means"].append(means)
            parts["quats"].append(quats)
            parts["scales"].append(self.asset["scales"] + torch.log(scale))
            parts["opacities"].append(self.asset["opacities"])
            color_parts.append(self.asset["colors"])
            semantics = torch.zeros(
                (self.composition.asset.gaussian_count, object_count),
                dtype=torch.float32,
                device=self.device,
            )
            semantics[:, object_index] = 1.0
            semantic_parts.append(semantics)
        return (
            {name: torch.cat(values, dim=0) for name, values in parts.items()},
            torch.cat(color_parts, dim=0),
            torch.cat(semantic_parts, dim=0),
        )

    def _cached_background(self, request: dict[str, Any]) -> tuple[Any, Any, Any]:
        camera_id = str(request["camera_id"])
        cached = self._background_cache.get(camera_id)
        if cached is None:
            cached = self._render_appearance(request, self.background)
            self._background_cache[camera_id] = cached
        return cached

    def _render_appearance(
        self, request: dict[str, Any], splats: dict[str, Any]
    ) -> tuple[Any, Any, Any]:
        torch = self.torch
        camera_to_scene, intrinsics, width, height = self._camera_tensors(request)
        with torch.inference_mode():
            rendered, alpha, _ = self.rasterization(
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
                tile_size=int(self.config["tile_size"]),
                packed=bool(self.config["packed"]),
                rasterize_mode=(
                    "antialiased" if self.config["antialiased"] else "classic"
                ),
                render_mode="RGB+ED",
                distributed=False,
                camera_model=str(self.config["camera_model"]),
                with_ut=bool(self.config["with_ut"]),
                with_eval3d=bool(self.config["with_eval3d"]),
                near_plane=float(self.config["near_plane"]),
                far_plane=float(self.config["far_plane"]),
                nht=True,
                center_ray_mode=bool(
                    self.config["deferred_opt_center_ray_encoding"]
                ),
                ray_dir_scale=self.shader.ray_dir_scale,
            )
            rgb, extras = self.shader(rendered)
        if extras is None or extras.shape[-1] < 1:
            raise RuntimeError("NHT joint rasterizer did not return expected depth")
        return rgb.clamp(0, 1), alpha.clamp(0, 1), extras[..., :1]

    def _render_joint_asset_pass(
        self,
        request: dict[str, Any],
        splats: dict[str, Any],
        *,
        direct_colors: Any,
        semantics: Any,
    ) -> tuple[Any, Any, Any, Any]:
        torch = self.torch
        camera_to_scene, intrinsics, width, height = self._camera_tensors(request)
        layout = _joint_eval3d_channel_layout(self.composition.object_count)
        logical_channels = torch.cat((direct_colors, semantics), dim=1)
        channels = _pad_joint_eval3d_channels(
            torch,
            logical_channels,
            layout=layout,
        )
        with torch.inference_mode():
            rendered, alpha, _ = self.rasterization(
                means=splats["means"],
                quats=splats["quats"],
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=channels,
                sh_degree=None,
                viewmats=torch.linalg.inv(camera_to_scene),
                Ks=intrinsics,
                width=width,
                height=height,
                tile_size=int(self.config["tile_size"]),
                packed=bool(self.config["packed"]),
                rasterize_mode=(
                    "antialiased" if self.config["antialiased"] else "classic"
                ),
                render_mode="RGB+ED",
                distributed=False,
                camera_model=str(self.config["camera_model"]),
                with_ut=bool(self.config["with_ut"]),
                with_eval3d=bool(self.config["with_eval3d"]),
                near_plane=float(self.config["near_plane"]),
                far_plane=float(self.config["far_plane"]),
                nht=False,
            )
        direct_rgb, semantic_weights, depth = _split_joint_eval3d_output(
            rendered,
            object_count=self.composition.object_count,
            layout=layout,
        )
        return (
            direct_rgb,
            semantic_weights,
            alpha.clamp(0, 1),
            depth,
        )

    def _camera_tensors(self, request: dict[str, Any]) -> tuple[Any, Any, int, int]:
        torch = self.torch
        camera_to_scene = torch.as_tensor(
            request["camera_to_scene"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        intrinsics = torch.as_tensor(
            request["intrinsics"]["matrix"],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        return camera_to_scene, intrinsics, int(request["width"]), int(request["height"])

    @staticmethod
    def _to_numpy(rgb: Any, alpha: Any, depth: Any) -> tuple[np.ndarray, ...]:
        return (
            rgb[0].float().cpu().numpy(),
            alpha[0].float().cpu().numpy(),
            depth[0].float().cpu().numpy(),
        )


def render_composed_scene(
    scene_path: Path,
    output: Path,
    *,
    camera_ids: list[str] | None,
    request_path: Path | None,
    composition_path: Path,
) -> dict[str, Any]:
    """Render every requested frame with background and asset in the same scene."""
    validated = load_validated_scene_export(scene_path)
    requests = _load_requests(validated.cameras, camera_ids, request_path)
    composition = _load_composition_request(composition_path)
    output = _safe_output_path(output, validated)
    if composition_path.resolve(strict=True).is_relative_to(output):
        raise ValueError("Composition request cannot live inside the render output")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".staging", dir=output.parent
        )
    )
    try:
        renderer = _build_renderer(
            validated.checkpoint_path,
            validated.runtime_config,
            composition,
            requests,
        )
        background_records = _write_backgrounds(
            staging / "background",
            renderer=renderer,
            requests=requests,
            scene=validated.scene,
            export_validation=validated.validation,
        )
        chunk_records = _write_chunks(
            staging / "chunks",
            renderer=renderer,
            requests=requests,
            composition=composition,
        )
        manifest = {
            "schema": "nht_composed_render_result_v1",
            "scene_schema": validated.scene["schema"],
            "scene_id": validated.scene["scene_id"],
            "coordinate_space": "canonical NHT scene space",
            "background": "background/render.json",
            "composition": {
                "request_schema": composition.payload["schema"],
                "frame_count": composition.frame_count,
                "object_count": composition.object_count,
                "asset_gaussian_count": composition.asset.gaussian_count,
                "appearance_model": composition.payload["asset"][
                    "appearance_model"
                ],
                "rasterization": "joint_3dgs_eval3d_transmittance_v1",
                "visibility_threshold": composition.payload["visibility_threshold"],
            },
            "chunks": chunk_records,
            "cuda_peak_bytes": renderer.cuda_peak_bytes,
        }
        if not background_records:
            raise RuntimeError("Composed renderer did not publish any backgrounds")
        validate_schema_payload(
            "composed-render-result",
            manifest,
            context="Generated composed render result",
        )
        (staging / "render.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output.exists():
            _validate_replaceable_output(output, str(validated.scene["scene_id"]))
            shutil.rmtree(output)
        staging.replace(output)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _build_renderer(
    checkpoint: Path,
    config: dict[str, Any],
    composition: _CompositionRequest,
    requests: list[dict[str, Any]],
) -> _ComposedRenderer:
    return _TorchComposedRenderer(checkpoint, config, composition, requests)


def _write_backgrounds(
    root: Path,
    *,
    renderer: _ComposedRenderer,
    requests: list[dict[str, Any]],
    scene: dict[str, Any],
    export_validation: dict[str, Any],
) -> list[dict[str, Any]]:
    root.mkdir()
    records: list[dict[str, Any]] = []
    for request in requests:
        identifier = _safe_identifier(str(request["camera_id"]))
        camera_root = root / identifier
        camera_root.mkdir()
        rgb, alpha, depth = renderer.render_background(request)
        _validate_render_arrays(request, rgb, alpha, depth)
        _write_dense_render(camera_root, rgb=rgb, alpha=alpha, depth=depth)
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
        "export_validation": export_validation,
        "renders": records,
    }
    validate_schema_payload(
        "render-result", manifest, context="Generated composed background result"
    )
    (root / "render.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return records


def _write_chunks(
    root: Path,
    *,
    renderer: _ComposedRenderer,
    requests: list[dict[str, Any]],
    composition: _CompositionRequest,
) -> list[dict[str, Any]]:
    root.mkdir()
    camera_ids = tuple(str(request["camera_id"]) for request in requests)
    records: list[dict[str, Any]] = []
    for chunk_index, frame_indices in enumerate(composition.chunks):
        chunk_id = f"chunk-{chunk_index:06d}"
        directory = root / chunk_id
        directory.mkdir()
        frame_values: list[int] = []
        camera_values: list[int] = []
        pixels: list[np.ndarray] = []
        rgbs: list[np.ndarray] = []
        alphas: list[np.ndarray] = []
        depths: list[np.ndarray] = []
        instance_ids: list[np.ndarray] = []
        offsets = [0]
        for frame_index in frame_indices:
            for camera_index, request in enumerate(requests):
                rgb, alpha, depth, labels = renderer.render_frame(request, frame_index)
                _validate_render_arrays(request, rgb, alpha, depth)
                expected_shape = (int(request["height"]), int(request["width"]))
                if labels.dtype != np.int32 or labels.shape != expected_shape:
                    raise RuntimeError("Joint instance rasterization returned an invalid mask")
                active_ids = {
                    int(value)
                    for value in composition.timeline.instance_ids[
                        composition.timeline.present[frame_index]
                    ]
                }
                rendered_ids = {int(value) for value in np.unique(labels)} - {0}
                if not rendered_ids.issubset(active_ids):
                    raise RuntimeError(
                        "Joint instance rasterization returned an absent object identity"
                    )
                visible = labels.reshape(-1) > 0
                indices = np.flatnonzero(visible).astype(np.int32, copy=False)
                selected_depth = depth.reshape(-1)[visible]
                if np.any(selected_depth <= 0.0):
                    raise RuntimeError("Visible joint Gaussian pixels require positive depth")
                frame_values.append(frame_index)
                camera_values.append(camera_index)
                pixels.append(indices)
                rgbs.append(rgb.reshape(-1, 3)[visible])
                alphas.append(alpha.reshape(-1)[visible])
                depths.append(selected_depth)
                instance_ids.append(labels.reshape(-1)[visible])
                offsets.append(offsets[-1] + len(indices))
        arrays_path = directory / "composed.npz"
        _save_composed_arrays(
            arrays_path,
            frame_indices=np.asarray(frame_values, dtype=np.int64),
            camera_indices=np.asarray(camera_values, dtype=np.int32),
            offsets=np.asarray(offsets, dtype=np.int64),
            pixel_indices=_concatenate(pixels, dtype=np.int32),
            rgb=_concatenate(rgbs, dtype=np.float32, trailing=(3,)),
            alpha=_concatenate(alphas, dtype=np.float32),
            depth=_concatenate(depths, dtype=np.float32),
            instance_ids=_concatenate(instance_ids, dtype=np.int32),
        )
        records.append(
            {
                "chunk_id": chunk_id,
                "frame_indices": list(frame_indices),
                "camera_ids": list(camera_ids),
                "sample_count": len(frame_values),
                "pixel_count": offsets[-1],
                "arrays": f"chunks/{chunk_id}/composed.npz",
            }
        )
    return records


def _load_composition_request(path: Path) -> _CompositionRequest:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("Composition request must be an ordinary JSON file")
    payload_raw = _load_json(path)
    validate_schema_payload(
        "composed-render-request",
        payload_raw,
        context="Gaussian composition request",
    )
    payload = cast(dict[str, Any], payload_raw)
    asset_raw = payload["asset"]
    timeline_raw = payload["timeline"]
    asset_path = _ordinary_sibling(path, asset_raw["tensors"])
    timeline_path = _ordinary_sibling(path, timeline_raw["tensors"])
    with np.load(asset_path, allow_pickle=False) as archive:
        if set(archive.files) != _ASSET_ARRAY_KEYS:
            raise ValueError("Gaussian asset NPZ has unknown or missing arrays")
        asset = _AssetArrays(
            means_m=np.array(archive["means_m"], copy=True),
            quats_wxyz=np.array(archive["quats_wxyz"], copy=True),
            log_scales_m=np.array(archive["log_scales_m"], copy=True),
            opacity_logits=np.array(archive["opacity_logits"], copy=True),
            colors_linear_rgb=np.array(archive["colors_linear_rgb"], copy=True),
        )
    _validate_asset(asset, expected_count=int(asset_raw["gaussian_count"]))
    with np.load(timeline_path, allow_pickle=False) as archive:
        if set(archive.files) != _TIMELINE_ARRAY_KEYS:
            raise ValueError("Gaussian timeline NPZ has unknown or missing arrays")
        timeline = _TimelineArrays(
            transforms_nht_from_asset=np.array(
                archive["transforms_nht_from_asset"], copy=True
            ),
            present=np.array(archive["present"], copy=True),
            instance_ids=np.array(archive["instance_ids"], copy=True),
        )
    chunks = tuple(
        tuple(int(value) for value in chunk["frame_indices"])
        for chunk in timeline_raw["chunks"]
    )
    _validate_timeline(timeline, timeline_raw=timeline_raw, chunks=chunks)
    return _CompositionRequest(
        payload=payload,
        asset=asset,
        timeline=timeline,
        chunks=chunks,
    )


def _validate_asset(asset: _AssetArrays, *, expected_count: int) -> None:
    count = expected_count
    expected_shapes = {
        "means_m": (count, 3),
        "quats_wxyz": (count, 4),
        "log_scales_m": (count, 3),
        "opacity_logits": (count,),
        "colors_linear_rgb": (count, 3),
    }
    for name, expected_shape in expected_shapes.items():
        value = getattr(asset, name)
        if value.dtype != np.float32 or value.shape != expected_shape:
            raise ValueError(f"Gaussian asset {name} must be float32 {expected_shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"Gaussian asset {name} contains a non-finite value")
    norms = np.linalg.norm(asset.quats_wxyz, axis=1)
    if not np.allclose(norms, 1.0, atol=1.0e-5, rtol=0.0):
        raise ValueError("Gaussian asset quaternions must be normalized wxyz values")
    scales = np.exp(asset.log_scales_m)
    if not np.isfinite(scales).all() or not np.all(scales > 0.0):
        raise ValueError("Gaussian asset scales must be positive")
    if np.any(asset.colors_linear_rgb < 0.0) or np.any(asset.colors_linear_rgb > 1.0):
        raise ValueError("Gaussian asset RGB must lie in [0,1]")


def _validate_timeline(
    timeline: _TimelineArrays,
    *,
    timeline_raw: dict[str, Any],
    chunks: tuple[tuple[int, ...], ...],
) -> None:
    frame_count = int(timeline_raw["frame_count"])
    object_count = int(timeline_raw["object_count"])
    if len(timeline_raw["object_ids"]) != object_count:
        raise ValueError("Gaussian timeline object_ids count is inconsistent")
    if timeline.transforms_nht_from_asset.dtype != np.float64 or timeline.transforms_nht_from_asset.shape != (
        frame_count,
        object_count,
        4,
        4,
    ):
        raise ValueError("Gaussian timeline transforms have the wrong dtype or shape")
    if timeline.present.dtype != np.bool_ or timeline.present.shape != (
        frame_count,
        object_count,
    ):
        raise ValueError("Gaussian timeline presence has the wrong dtype or shape")
    if timeline.instance_ids.dtype != np.int32 or timeline.instance_ids.shape != (
        object_count,
    ):
        raise ValueError("Gaussian timeline instance_ids have the wrong dtype or shape")
    expected_ids = np.arange(1, object_count + 1, dtype=np.int32)
    if not np.array_equal(timeline.instance_ids, expected_ids) or timeline_raw[
        "instance_ids"
    ] != expected_ids.tolist():
        raise ValueError("Gaussian timeline instance_ids must equal 1..object_count")
    if not np.isfinite(timeline.transforms_nht_from_asset).all():
        raise ValueError("Gaussian timeline transforms contain a non-finite value")
    for frame_index, object_index in np.argwhere(timeline.present):
        _validate_similarity(timeline.transforms_nht_from_asset[frame_index, object_index])
    flattened = tuple(value for chunk in chunks for value in chunk)
    if flattened != tuple(range(frame_count)):
        raise ValueError("Gaussian chunks must cover every frame exactly once in order")
    if tuple(chunk["chunk_index"] for chunk in timeline_raw["chunks"]) != tuple(
        range(len(chunks))
    ):
        raise ValueError("Gaussian chunk indices must equal 0..chunk_count-1")


def _validate_similarity(matrix: np.ndarray) -> None:
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-8, rtol=0.0):
        raise ValueError("Active Gaussian transform is not homogeneous")
    linear = matrix[:3, :3]
    scale = float(np.cbrt(np.linalg.det(linear)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Active Gaussian transform must have positive scale")
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5, rtol=0.0):
        raise ValueError("Active Gaussian transform must have uniform scale")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-5, rtol=0.0):
        raise ValueError("Active Gaussian transform rotation must be proper")


def _ordinary_sibling(owner: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "/" in relative or "\\" in relative:
        raise ValueError("Composition tensor reference must be one sibling filename")
    owner = owner.resolve(strict=True)
    path = owner.parent / relative
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Composition tensor file is unavailable: {path}")
    resolved = path.resolve(strict=True)
    if resolved.parent != owner.parent:
        raise ValueError("Composition tensor file must remain beside its request")
    return resolved


def _load_json(path: Path) -> object:
    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"Non-finite JSON number {value!r} is forbidden")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} is forbidden")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON file: {path}") from error


def _validate_render_arrays(
    request: dict[str, Any], rgb: np.ndarray, alpha: np.ndarray, depth: np.ndarray
) -> None:
    height = int(request["height"])
    width = int(request["width"])
    expected = ((height, width, 3), (height, width, 1), (height, width, 1))
    for name, value, shape in zip(("rgb", "alpha", "depth"), (rgb, alpha, depth), expected):
        if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
            raise RuntimeError(f"Joint renderer returned invalid {name}")
    if rgb.min() < 0.0 or rgb.max() > 1.0 or alpha.min() < 0.0 or alpha.max() > 1.0:
        raise RuntimeError("Joint renderer returned RGB/alpha outside [0,1]")
    if depth.min() < 0.0:
        raise RuntimeError("Joint renderer returned negative depth")


def _write_dense_render(
    root: Path, *, rgb: np.ndarray, alpha: np.ndarray, depth: np.ndarray
) -> None:
    np.save(root / "rgb.npy", rgb.astype(np.float32, copy=False))
    np.save(root / "alpha.npy", alpha.astype(np.float32, copy=False))
    np.save(root / "depth.npy", depth.astype(np.float32, copy=False))
    Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8)).save(root / "rgb.png")
    Image.fromarray((alpha[..., 0] * 255.0 + 0.5).astype(np.uint8), mode="L").save(
        root / "alpha.png"
    )


def _save_composed_arrays(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".npz.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _concatenate(
    values: list[np.ndarray], *, dtype: type[np.generic], trailing: tuple[int, ...] = ()
) -> np.ndarray:
    if not values or not any(value.size for value in values):
        return np.empty((0, *trailing), dtype=dtype)
    return np.concatenate(values, axis=0).astype(dtype, copy=False)


def _rotation_matrix_to_quaternion(torch: Any, rotation: Any) -> Any:
    values = torch.stack(
        (
            rotation.trace(),
            rotation[0, 0] - rotation[1, 1] - rotation[2, 2],
            rotation[1, 1] - rotation[0, 0] - rotation[2, 2],
            rotation[2, 2] - rotation[0, 0] - rotation[1, 1],
        )
    )
    largest_index = int(torch.argmax(values))
    largest = torch.sqrt(values[largest_index] + 1.0) * 0.5
    multiplier = 0.25 / largest
    if largest_index == 0:
        quaternion = torch.stack(
            (
                largest,
                (rotation[2, 1] - rotation[1, 2]) * multiplier,
                (rotation[0, 2] - rotation[2, 0]) * multiplier,
                (rotation[1, 0] - rotation[0, 1]) * multiplier,
            )
        )
    elif largest_index == 1:
        quaternion = torch.stack(
            (
                (rotation[2, 1] - rotation[1, 2]) * multiplier,
                largest,
                (rotation[1, 0] + rotation[0, 1]) * multiplier,
                (rotation[0, 2] + rotation[2, 0]) * multiplier,
            )
        )
    elif largest_index == 2:
        quaternion = torch.stack(
            (
                (rotation[0, 2] - rotation[2, 0]) * multiplier,
                (rotation[1, 0] + rotation[0, 1]) * multiplier,
                largest,
                (rotation[2, 1] + rotation[1, 2]) * multiplier,
            )
        )
    else:
        quaternion = torch.stack(
            (
                (rotation[1, 0] - rotation[0, 1]) * multiplier,
                (rotation[0, 2] + rotation[2, 0]) * multiplier,
                (rotation[2, 1] + rotation[1, 2]) * multiplier,
                largest,
            )
        )
    quaternion = torch.nn.functional.normalize(quaternion, dim=0)
    return torch.where(quaternion[0] < 0, -quaternion, quaternion)


def _quaternion_multiply(torch: Any, left: Any, right: Any) -> Any:
    left_w, left_x, left_y, left_z = left.unbind(dim=-1)
    right_w, right_x, right_y, right_z = right.unbind(dim=-1)
    return torch.stack(
        (
            left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
            left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
            left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
            left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
        ),
        dim=-1,
    )


__all__ = ["render_composed_scene"]
