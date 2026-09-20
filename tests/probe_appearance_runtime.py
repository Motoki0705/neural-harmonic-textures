"""Queued GPU-runtime check: parse CLI and selected metadata, without training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from nht_pipeline.cuda import select_cuda_environment
from nht_pipeline.nht_adapter import _instrument_parser, _load_trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--source-workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()
    environment, _ = select_cuda_environment(os.environ, args.cuda_device)
    os.environ["CUDA_VISIBLE_DEVICES"] = environment["CUDA_VISIBLE_DEVICES"]
    trainer = _load_trainer(args.trainer)
    names = [f"frame_{round(i * 248 / 49):06d}.jpg" for i in range(50)]
    sys.argv = [
        str(args.trainer),
        "default",
        "--image_names",
        *names,
        "--max_steps",
        "7000",
        "--disable_viewer",
    ]
    cfg = trainer.tyro.extras.overridable_config_cli(
        {
            "default": (
                "Probe",
                trainer.Config(strategy=trainer.MCMCStrategy(verbose=False)),
            )
        }
    )
    assert cfg.image_names == names and cfg.max_steps == 7000
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = args.output / "scene-metadata.json"
    _instrument_parser(trainer, metadata, args.output / "observed-original-images")
    selected = trainer.Parser(
        str(args.source_workspace / "3dgs/dataset"),
        factor=2,
        normalize=True,
        test_every=8,
        native_images_factor=True,
        image_names=names,
    )
    payload = json.loads(metadata.read_text())
    assert payload["camera_count"] == 50 and payload["full_image_count"] == 491
    assert sum(camera["split"] == "validation" for camera in payload["cameras"]) == 8
    assert len(trainer.Dataset(selected, split="train")) == 42
    result = {
        "purpose": "original-image runtime contract check; no generated images or training",
        "cuda_available": trainer.torch.cuda.is_available(),
        "model_arguments_parsed": True,
        "selected": 50,
        "train": 42,
        "validation": 8,
        "scene_scale": payload["scene_scale"],
    }
    assert result["cuda_available"]
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
