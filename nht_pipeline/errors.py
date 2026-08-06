"""Stable operational failure categories for run manifests and diagnostics."""

from __future__ import annotations


def classify_error(error: BaseException | str) -> str:
    name = type(error).__name__.lower() if isinstance(error, BaseException) else ""
    message = str(error).lower()
    if "out of memory" in message or "cuda oom" in message:
        return "oom"
    if (
        "novalidcandidateerror" in name
        or "no sfm candidates" in message
        or "all sfm candidates" in message
        or "every configured sfm candidate" in message
        or "candidate" in message
        and "rejected" in message
    ):
        return "all_candidates_rejected"
    if "checkpoint" in message and any(
        marker in message for marker in ("missing", "no checkpoint", "without")
    ):
        return "checkpoint_missing"
    if "non-finite" in message or "nonfinite" in message:
        return "non_finite_metric"
    if "camera" in message and ("unsupported" in message or "invalid" in message):
        return "invalid_camera_model"
    if (
        "modulenotfound" in name
        or "importerror" in name
        or "dependency" in message
        or "not found" in message
    ):
        return "missing_dependency"
    if "signal" in message or "sigterm" in message or "sigint" in message:
        return "process_signal"
    if "calledprocesserror" in name or "exited with code" in message:
        return "process_nonzero_exit"
    if "interrupted" in name or "interrupted" in message:
        return "process_interrupted"
    return "unclassified"
