"""GPU acceptance for persistent, bounded public render batches (no training)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from nht_pipeline.export import load_validated_scene_export
from nht_pipeline.render import ResidentSceneRenderer, render_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"gpu": torch.cuda.get_device_name(), "scenes": {}}
    for scene_id in ("B00", "B01", "B02", "B03"):
        root = args.scenes / scene_id
        scene_path = root / "reconstruction/export/scene.json"
        # Observed cameras use the exact public scene coordinate contract.
        validated = load_validated_scene_export(scene_path)
        requests = [
            dict(c)
            for c in validated.cameras[:: max(1, len(validated.cameras) // 8)][:8]
        ]
        keys = {"camera_id", "width", "height", "intrinsics", "camera_to_scene"}
        requests = [{k: v for k, v in c.items() if k in keys} for c in requests]
        request_path = args.output / f"{scene_id}-cameras.json"
        request_path.write_text(
            json.dumps({"schema": "nht_render_request_v1", "cameras": requests})
        )
        started = time.perf_counter()
        renderer = ResidentSceneRenderer(
            validated.checkpoint_path, validated.runtime_config
        )
        torch.cuda.synchronize()
        item = {
            "load_seconds": time.perf_counter() - started,
            "cameras": len(requests),
            "batches": {},
        }
        references = [renderer.render(request) for request in requests]
        for size in (1, 4, 8):
            timings = []
            torch.cuda.reset_peak_memory_stats()
            max_errors = [0.0, 0.0, 0.0]
            for repeat in range(4):
                started = time.perf_counter()
                outputs = []
                for offset in range(0, len(requests), size):
                    outputs.extend(
                        renderer.render_batch(requests[offset : offset + size])
                    )
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - started)
                for expected, actual in zip(references, outputs, strict=True):
                    for channel, (a, b) in enumerate(
                        zip(expected, actual, strict=True)
                    ):
                        assert np.isfinite(b).all()
                        max_errors[channel] = max(
                            max_errors[channel], float(np.max(np.abs(a - b)))
                        )
                        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
            item["batches"][str(size)] = {
                "seconds": timings,
                "warm_median_seconds": statistics.median(timings[1:]),
                "max_abs_error_rgb_alpha_depth": max_errors,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        del renderer
        torch.cuda.empty_cache()
        # Exercise the actual transactional public entry, count checkpoint loads.
        original_load = torch.load
        started = time.perf_counter()
        with patch("torch.load", wraps=original_load) as loads:
            manifest = render_scene(
                scene_path, args.output / scene_id, request_path=request_path
            )
        item["public_seconds"] = time.perf_counter() - started
        item["checkpoint_load_count"] = loads.call_count
        assert loads.call_count == 1
        for expected, record in zip(references, manifest["renders"], strict=True):
            for value, field in zip(expected, ("rgb", "alpha", "depth"), strict=True):
                np.testing.assert_allclose(
                    value,
                    np.load(args.output / scene_id / record[field]),
                    rtol=1e-5,
                    atol=1e-5,
                )
        report["scenes"][scene_id] = item
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({scene_id: item}), flush=True)


if __name__ == "__main__":
    main()
