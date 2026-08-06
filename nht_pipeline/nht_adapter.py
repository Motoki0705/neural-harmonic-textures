"""Adapter executed inside the NHT environment for controlled training metadata."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _load_trainer(path: Path) -> ModuleType:
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("nht_pipeline_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load NHT trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    # ``tyro`` inspects the dataclass source when it builds the CLI. Dynamic
    # modules must be registered or inspect treats their classes as built-ins.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _run_probe(args: argparse.Namespace) -> None:
    import torch

    trainer = _load_trainer(args.trainer.resolve())
    required = ("Config", "Parser", "MCMCStrategy", "cli", "main", "set_random_seed")
    missing = [name for name in required if not hasattr(trainer, name)]
    if missing:
        raise RuntimeError(f"NHT trainer is missing required symbols: {missing}")
    print(
        json.dumps(
            {
                "schema": "nht_runtime_probe_v1",
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_devices": torch.cuda.device_count(),
                "cuda_names": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
                "trainer": str(args.trainer.resolve()),
                "trainer_importable": True,
            }
        )
    )


def _instrument_parser(
    trainer: Any,
    metadata_output: Path,
    observed_image_root: Path,
) -> None:
    import cv2
    import imageio.v2 as imageio
    import numpy as np
    from datasets import colmap as colmap_dataset

    components: dict[str, Any] = {}
    original_similarity = colmap_dataset.similarity_from_cameras
    original_alignment = colmap_dataset.align_principal_axes

    def capture_similarity(cameras: Any) -> Any:
        transform = original_similarity(cameras)
        components["camera_similarity"] = np.asarray(transform).tolist()
        return transform

    def capture_alignment(points: Any) -> Any:
        transform = original_alignment(points)
        components["principal_axis_alignment"] = np.asarray(transform).tolist()
        return transform

    colmap_dataset.similarity_from_cameras = capture_similarity
    colmap_dataset.align_principal_axes = capture_alignment
    original_parser = trainer.Parser

    class InstrumentedParser(original_parser):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            total = np.asarray(self.transform, dtype=np.float64)
            similarity = np.asarray(
                components.get("camera_similarity", np.eye(4)), dtype=np.float64
            )
            alignment = np.asarray(
                components.get("principal_axis_alignment", np.eye(4)),
                dtype=np.float64,
            )
            before_flip = alignment @ similarity
            upside_down = total @ np.linalg.inv(before_flip)
            observed_image_root.mkdir(parents=True, exist_ok=True)
            cameras: list[dict[str, Any]] = []
            for index, image_name in enumerate(self.image_names):
                image = imageio.imread(self.image_paths[index])[..., :3]
                camera_id = self.camera_ids[index]
                params = self.params_dict[camera_id]
                source_height, source_width = image.shape[:2]
                if len(params) > 0:
                    image = cv2.remap(
                        image,
                        self.mapx_dict[camera_id],
                        self.mapy_dict[camera_id],
                        cv2.INTER_LINEAR,
                    )
                    x, y, width, height = self.roi_undist_dict[camera_id]
                    image = image[y : y + height, x : x + width]
                    crop = [int(x), int(y), int(width), int(height)]
                else:
                    crop = [0, 0, int(source_width), int(source_height)]
                destination_name = f"{Path(image_name).stem}.png"
                imageio.imwrite(observed_image_root / destination_name, image)
                height, width = image.shape[:2]
                cameras.append(
                    {
                        "image_name": image_name,
                        "observed_image": destination_name,
                        "camera_id": int(camera_id),
                        "camera_index": int(self.camera_indices[index]),
                        "split": (
                            "validation" if index % self.test_every == 0 else "train"
                        ),
                        "width": int(width),
                        "height": int(height),
                        "source_resolution": [
                            int(source_width),
                            int(source_height),
                        ],
                        "crop_xywh": crop,
                        "projection_model": "PINHOLE",
                        "distortion_model": "NONE",
                        "intrinsics": np.asarray(
                            self.Ks_dict[camera_id], dtype=np.float64
                        ).tolist(),
                        "camera_to_scene": np.asarray(
                            self.camtoworlds[index], dtype=np.float64
                        ).tolist(),
                    }
                )
            _atomic_json(
                metadata_output,
                {
                    "schema": "nht_training_scene_v1",
                    "canonical_scene_space": (
                        "NHT parser normalized world space; model means, cameras, "
                        "and exported sparse points use this right-handed space"
                    ),
                    "sfm_to_scene": total.tolist(),
                    "scene_to_sfm": np.linalg.inv(total).tolist(),
                    "normalization": {
                        "applied": bool(self.normalize),
                        "camera_similarity": similarity.tolist(),
                        "principal_axis_alignment": alignment.tolist(),
                        "upside_down_correction": upside_down.tolist(),
                    },
                    "data_factor": int(self.factor),
                    "scene_scale": float(self.scene_scale),
                    "test_every": int(self.test_every),
                    "camera_count": len(cameras),
                    "cameras": cameras,
                },
            )

    trainer.Parser = InstrumentedParser


def _run_training(args: argparse.Namespace) -> None:
    trainer: Any = _load_trainer(args.trainer.resolve())
    effective_seeds: list[int] = []
    original_seed_function = trainer.set_random_seed

    def controlled_seed(trainer_seed: int) -> None:
        rank_offset = int(trainer_seed) - 42
        effective = int(args.seed) + rank_offset
        original_seed_function(effective)
        effective_seeds.append(effective)
        _atomic_json(
            args.seed_output,
            {
                "schema": "nht_effective_seed_v1",
                "requested_seed": int(args.seed),
                "effective_seed": effective,
                "rank_offset": rank_offset,
            },
        )

    trainer.set_random_seed = controlled_seed
    _instrument_parser(trainer, args.metadata_output, args.observed_image_root)
    trainer_arguments = list(args.trainer_arguments)
    if trainer_arguments and trainer_arguments[0] == "--":
        trainer_arguments.pop(0)
    sys.argv = [str(args.trainer), *trainer_arguments]
    configs = {
        "default": (
            "NHT training with MCMC densification (default).",
            trainer.Config(strategy=trainer.MCMCStrategy(verbose=True)),
        ),
    }
    cfg = trainer.tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)
    if cfg.camera_model != "pinhole":
        raise ValueError("The standard renderer requires camera_model=pinhole")
    if bool(cfg.pose_opt):
        raise ValueError("The standard renderer requires pose_opt=false")
    if cfg.post_processing is not None:
        raise ValueError(
            "The standard renderer currently requires NHT post_processing=null"
        )
    _atomic_json(
        args.runtime_output,
        {
            "schema": "nht_runtime_config_v1",
            "camera_model": cfg.camera_model,
            "pose_opt": False,
            "primitive_type": cfg.primitive_type,
            "antialiased": bool(cfg.antialiased),
            "packed": bool(cfg.packed),
            "tile_size": int(cfg.tile_size),
            "with_ut": bool(cfg.with_ut),
            "with_eval3d": bool(cfg.with_eval3d),
            "near_plane": float(cfg.near_plane),
            "far_plane": float(cfg.far_plane),
            "deferred_opt_feature_dim": int(cfg.deferred_opt_feature_dim),
            "deferred_opt_enable_view_encoding": bool(
                cfg.deferred_opt_enable_view_encoding
            ),
            "deferred_opt_view_encoding_type": cfg.deferred_opt_view_encoding_type,
            "deferred_mlp_hidden_dim": int(cfg.deferred_mlp_hidden_dim),
            "deferred_mlp_num_layers": int(cfg.deferred_mlp_num_layers),
            "deferred_opt_sh_degree": int(cfg.deferred_opt_sh_degree),
            "deferred_opt_sh_scale": float(cfg.deferred_opt_sh_scale),
            "deferred_opt_fourier_num_freqs": int(cfg.deferred_opt_fourier_num_freqs),
            "deferred_opt_center_ray_encoding": bool(
                cfg.deferred_opt_center_ray_encoding
            ),
            "deferred_decode_activation": cfg.deferred_decode_activation,
            "post_processing": None,
        },
    )
    if cfg.compression == "png":
        import plas  # noqa: F401
        import torchpq  # noqa: F401
    if cfg.with_ut and not cfg.with_eval3d:
        raise ValueError("Training with UT requires with_eval3d")
    trainer.cli(trainer.main, cfg, verbose=True)
    if not effective_seeds:
        raise RuntimeError("Trainer did not invoke the controlled seed adapter")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--trainer", type=Path, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--seed-output", type=Path, required=True)
    train.add_argument("--metadata-output", type=Path, required=True)
    train.add_argument("--observed-image-root", type=Path, required=True)
    train.add_argument("--runtime-output", type=Path, required=True)
    train.add_argument("trainer_arguments", nargs=argparse.REMAINDER)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--trainer", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.command == "train":
        _run_training(args)
    elif args.command == "probe":
        _run_probe(args)


if __name__ == "__main__":
    main()
