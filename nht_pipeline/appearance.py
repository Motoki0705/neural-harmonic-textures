"""Progressive appearance controls for staged ``nht-render`` requests.

A request may ask the renderer to show the learned scene in an intermediate
state between a sparse point cloud and the final NHT render. Every control is a
scalar in ``[0, 1]`` and ``1`` everywhere reproduces the learned scene exactly:

* ``splat_progress``: Gaussians start as opaque isotropic spheres of
  ``policy.point_radius`` and each grows to its learned scale and learned
  opacity. Growth is staggered by a fixed size order (``largest_first``), so
  coarse structure appears before fine detail. Opacity follows each Gaussian's
  own growth because large Gaussians held opaque would hide the whole scene.
* ``view_dependence``: the deferred shader's ray-direction input moves from the
  camera optical axis (one direction for every pixel) to the true per-pixel ray.

The blend functions use only arithmetic and ``.clip``, so the same code runs on
NumPy arrays (tests) and Torch tensors (the CUDA renderer).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

SPLAT_ORDER_LARGEST_FIRST = "largest_first"
SPLAT_ORDERS = (SPLAT_ORDER_LARGEST_FIRST,)

def _unit_interval(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class AppearancePolicy:
    """Request-wide parameters shared by every staged camera."""

    point_radius: float
    splat_order: str
    splat_stagger: float

    def __post_init__(self) -> None:
        radius = float(self.point_radius)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("point_radius must be positive and finite")
        if self.splat_order not in SPLAT_ORDERS:
            raise ValueError(f"Unsupported splat_order: {self.splat_order!r}")
        stagger = float(self.splat_stagger)
        if not math.isfinite(stagger) or not 0.0 < stagger <= 1.0:
            raise ValueError("splat_stagger must lie in (0, 1]")

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> AppearancePolicy:
        return cls(
            point_radius=float(payload["point_radius"]),
            splat_order=str(payload["splat_order"]),
            splat_stagger=float(payload["splat_stagger"]),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "point_radius": self.point_radius,
            "splat_order": self.splat_order,
            "splat_stagger": self.splat_stagger,
        }


@dataclass(frozen=True, slots=True)
class AppearanceControls:
    """Per-camera progress values; all ones reproduce the learned scene."""

    splat_progress: float
    view_dependence: float

    def __post_init__(self) -> None:
        for name in ("splat_progress", "view_dependence"):
            object.__setattr__(self, name, _unit_interval(getattr(self, name), name))

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> AppearanceControls:
        return cls(
            splat_progress=payload["splat_progress"],
            view_dependence=payload["view_dependence"],
        )

    def to_payload(self) -> dict[str, float]:
        return {
            "splat_progress": self.splat_progress,
            "view_dependence": self.view_dependence,
        }


def largest_first_ranks(log_scales: np.ndarray) -> np.ndarray:
    """Return each Gaussian's normalized rank by its largest learned axis.

    Rank ``0`` is the largest Gaussian and ``1`` the smallest. Ties keep the
    checkpoint order, so the ranks are a deterministic function of the model.
    """
    values = np.asarray(log_scales)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise ValueError("log_scales must have shape (N, 3) with N > 0")
    if not np.isfinite(values).all():
        raise ValueError("log_scales must be finite")
    count = values.shape[0]
    order = np.argsort(-values.max(axis=1), kind="stable")
    ranks = np.empty(count, dtype=np.float32)
    ranks[order] = np.arange(count, dtype=np.float32) / max(count - 1, 1)
    return ranks


def gaussian_splat_progress(ranks: Any, splat_progress: float, stagger: float) -> Any:
    """Map the global progress to each Gaussian's own growth in ``[0, 1]``.

    A Gaussian with rank ``r`` starts growing at global progress
    ``r / (1 + stagger)`` and finishes ``stagger / (1 + stagger)`` later, so
    progress ``0`` keeps every Gaussian a point and ``1`` completes all of them.
    """
    return ((splat_progress * (1.0 + stagger) - ranks) / stagger).clip(0.0, 1.0)


def blended_log_scales(log_scales: Any, progress: Any, point_radius: float) -> Any:
    """Interpolate log-scales from the point sphere to the learned ellipsoid."""
    weight = progress[..., None]
    return log_scales * weight + (1.0 - weight) * math.log(point_radius)


def blended_opacities(learned: Any, progress: Any) -> Any:
    """Interpolate activated opacity from fully opaque to the learned value."""
    return learned * progress + (1.0 - progress)


def optical_axis_ray_channels(
    camera_to_scene: np.ndarray, ray_dir_scale: float
) -> np.ndarray:
    """Encode the camera +Z axis exactly as the rasterizer encodes pixel rays."""
    transform = np.asarray(camera_to_scene, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("camera_to_scene must be a finite 4x4 matrix")
    axis = transform[:3, 2]
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise ValueError("camera_to_scene has a degenerate optical axis")
    return ((axis / norm) * ray_dir_scale + 1.0) * 0.5


def blended_ray_channels(actual: Any, fixed: Any, view_dependence: float) -> Any:
    """Interpolate the shader ray input from the optical axis to the true ray."""
    return actual * view_dependence + fixed * (1.0 - view_dependence)
