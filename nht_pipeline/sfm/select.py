"""Semantic gating and deterministic SfM candidate selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class NoValidCandidateError(RuntimeError):
    pass


def candidate_rank(metrics: dict[str, Any]) -> tuple[float, ...]:
    supported = int(metrics.get("supported_registered_images", 0))
    point_density = float(metrics.get("sparse_points", 0)) / max(supported, 1)
    reprojection = metrics.get("p95_reprojection_error_px")
    track = metrics.get("median_track_length")
    trajectory = metrics.get("trajectory", {})
    return (
        float(metrics.get("supported_registration_ratio", 0.0)),
        point_density,
        -float(reprojection) if reprojection is not None else float("-inf"),
        float(track) if track is not None else float("-inf"),
        -float(trajectory.get("maximum_step_to_median", float("inf"))),
    )


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise NoValidCandidateError("No SfM candidates were produced")
    eligible = [
        candidate for candidate in candidates if candidate["metrics"]["accepted"]
    ]
    rejected = [
        candidate for candidate in candidates if not candidate["metrics"]["accepted"]
    ]
    if not eligible:
        failures = {
            candidate["id"]: [
                name
                for name, gate in candidate["metrics"].get("gates", {}).items()
                if not gate.get("passed", False)
            ]
            for candidate in rejected
        }
        raise NoValidCandidateError(
            f"All SfM candidates failed semantic gates: {failures}"
        )
    ranked = sorted(
        eligible,
        key=lambda candidate: (candidate_rank(candidate["metrics"]), candidate["id"]),
        reverse=True,
    )
    selected = ranked[0]
    return {
        "schema": "nht_sfm_selection_v1",
        "selected_candidate": selected["id"],
        "selected_backend": selected["backend"],
        "selection_policy": [
            "all semantic gates must pass",
            "maximum supported registration ratio",
            "maximum sparse point density per supported camera",
            "minimum p95 reprojection error",
            "maximum median track length",
            "minimum trajectory step ratio",
        ],
        "ranked_eligible_candidates": [
            {"id": candidate["id"], "rank": list(candidate_rank(candidate["metrics"]))}
            for candidate in ranked
        ],
        "rejected_candidates": [candidate["id"] for candidate in rejected],
        "metrics": selected["metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = [json.loads(path.read_text()) for path in args.candidates]
    result = select_candidate(candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
