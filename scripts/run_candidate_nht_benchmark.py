#!/usr/bin/env python3
"""Run the same NHT recipe for one already reconstructed SfM candidate."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from nht_pipeline.config import load_config, write_resolved_config
from nht_pipeline.pipeline import PipelineContext, run_pipeline
from nht_pipeline.run_state import RunState
from nht_pipeline.workspace import link_or_copy


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workspace", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _load_candidate(source: Path, identifier: str) -> dict[str, Any]:
    manifest_path = source / "sfm/candidates/candidates.json"
    manifest = json.loads(manifest_path.read_text())
    matches = [
        record for record in manifest["candidates"] if record["id"] == identifier
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one SfM candidate {identifier!r}")
    candidate = matches[0]
    if candidate.get("status") != "completed":
        raise ValueError(f"SfM candidate {identifier!r} did not complete")
    if not candidate.get("metrics", {}).get("accepted"):
        raise ValueError(f"SfM candidate {identifier!r} failed semantic gates")
    return candidate


def _prepare_workspace(
    source: Path,
    destination: Path,
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> RunState:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Benchmark workspace is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    frames_method = link_or_copy(source / "frames", destination / "frames", directory=True)
    candidate_root = destination / "sfm/candidates" / candidate["id"]
    candidate_root.mkdir(parents=True)
    shutil.copytree(
        source / "sfm/candidates" / candidate["id"] / "model",
        candidate_root / "model",
    )
    selected = {**candidate, "model": f"{candidate['id']}/model"}
    (candidate_root / "candidate.json").write_text(
        json.dumps(selected, indent=2) + "\n"
    )
    candidate_manifest = {
        "schema": "nht_sfm_candidates_v1",
        "input_images": candidate["metrics"]["input_images"],
        "candidates": [selected],
        "completed_candidates": 1,
        "effective_seeds": {
            "sfm": int(config["seed"]),
            "pair_generation": int(config["seed"]),
        },
    }
    (destination / "sfm/candidates/candidates.json").write_text(
        json.dumps(candidate_manifest, indent=2) + "\n"
    )

    shutil.copytree(candidate_root / "model", destination / "sfm/model")
    reconstruction = {
        "schema": "nht_sfm_selection_v1",
        "selected_candidate": candidate["id"],
        "selected_backend": candidate["backend"],
        "metrics": candidate["metrics"],
        "model": "model",
    }
    (destination / "sfm/reconstruction.json").write_text(
        json.dumps(reconstruction, indent=2) + "\n"
    )
    diagnostics = destination / "sfm/diagnostics"
    diagnostics.mkdir()
    (diagnostics / "selection.json").write_text(
        json.dumps(reconstruction, indent=2) + "\n"
    )

    source_state = RunState.load(source)
    state = RunState.create_or_load(
        destination,
        f"{source_state.payload['scene_id']}--{candidate['id']}",
        Path(source_state.payload["input_video"]),
    )
    state.mark_completed("frames", {"imported_from": str(source), "method": frames_method})
    state.mark_completed("preprocess", {"imported_from": str(source)})
    state.mark_completed("sfm", candidate_manifest)
    state.mark_completed(
        "sfm_selection",
        {
            "selected_candidate": candidate["id"],
            "selected_backend": candidate["backend"],
            "registered_images": candidate["metrics"]["registered_images"],
            "supported_registered_images": candidate["metrics"][
                "supported_registered_images"
            ],
            "sparse_points": candidate["metrics"]["sparse_points"],
        },
    )
    provenance = {
        "schema": "nht_candidate_downstream_benchmark_v1",
        "source_workspace": str(source),
        "candidate_id": candidate["id"],
        "candidate_backend": candidate["backend"],
        "sfm_metrics": candidate["metrics"],
        "nht_recipe": config["nht_training"],
        "effective_seed": int(config["seed"]),
    }
    (destination / "candidate-benchmark.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    write_resolved_config(destination / "resolved-config.yaml", config)
    return state


def main() -> None:
    args = _arguments()
    repository_root = Path(__file__).resolve().parents[1]
    source = args.source_workspace.resolve()
    destination = args.workspace.resolve()
    config = load_config(args.config)
    candidate = _load_candidate(source, args.candidate_id)
    state = _prepare_workspace(source, destination, candidate, config)
    state.request("nht_training")
    context = PipelineContext(
        workspace=destination,
        state=state,
        config=config,
        repository_root=repository_root,
    )
    run_pipeline(context, "nht_training")


if __name__ == "__main__":
    main()
