"""Research JSONL worker; files remain the public process boundary."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from nht_pipeline.export import load_validated_scene_export
from nht_pipeline.render import ResidentSceneRenderer, _load_requests

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=Path, required=True)
parser.add_argument("--cameras", type=Path, required=True)
parser.add_argument("--buffer", type=Path, required=True)
parser.add_argument("--auxiliary", action="store_true")
args = parser.parse_args()
start = time.perf_counter()
validated = load_validated_scene_export(args.scene)
requests = _load_requests(validated.cameras, None, args.cameras)
renderer = ResidentSceneRenderer(validated.checkpoint_path, validated.runtime_config)
torch.cuda.synchronize()
print(
    json.dumps(
        {
            "ready_seconds": time.perf_counter() - start,
            "allocated_bytes": torch.cuda.memory_allocated(),
        }
    ),
    flush=True,
)
for line in sys.stdin:
    message = json.loads(line)
    if message["op"] == "stop":
        break
    if (
        message["op"] != "render"
        or not message["indices"]
        or any(
            type(i) is not int or not 0 <= i < len(requests) for i in message["indices"]
        )
    ):
        raise ValueError("Invalid render operation or camera indices")
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    output = []
    alphas, depths = [], []
    for index in message["indices"]:
        rgb, alpha, depth = renderer.render(requests[index])
        output.append(rgb)
        if args.auxiliary:
            alphas.append(alpha)
            depths.append(depth)
    torch.cuda.synchronize()
    render_seconds = time.perf_counter() - start
    np.save(args.buffer, np.stack(output), allow_pickle=False)
    if args.auxiliary:
        np.save(
            args.buffer.with_suffix(".alpha.npy"), np.stack(alphas), allow_pickle=False
        )
        np.save(
            args.buffer.with_suffix(".depth.npy"), np.stack(depths), allow_pickle=False
        )
    print(
        json.dumps(
            {
                "render_seconds": render_seconds,
                "total_seconds": time.perf_counter() - start,
                "peak_bytes": torch.cuda.max_memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
            }
        ),
        flush=True,
    )
