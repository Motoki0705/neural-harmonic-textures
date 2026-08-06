"""Exclusive workspace access and transactional stage publication."""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self, TextIO

from .stages import STAGE_BY_NAME, descendants


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_path(workspace: Path, relative: Path) -> Path:
    workspace = workspace.resolve()
    path = workspace / relative
    if (
        path == workspace
        or relative.name in {"", ".", ".."}
        or not path.parent.resolve().is_relative_to(workspace)
    ):
        raise RuntimeError(f"Unsafe workspace-owned path: {path}")
    return path


def remove_path(path: Path) -> None:
    """Remove a file, link, or directory without following its final symlink."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def invalidate_published_outputs(workspace: Path, from_stage: str) -> None:
    """Physically unpublish a stage and every descendant before execution."""
    for name in descendants(from_stage, include_self=True):
        for relative in STAGE_BY_NAME[name].owned_paths:
            remove_path(_safe_path(workspace, relative))


def stage_staging_root(workspace: Path, stage_name: str) -> Path:
    root = _safe_path(workspace, Path(".staging") / stage_name)
    remove_path(root)
    root.mkdir(parents=True)
    return root


def cleanup_staging(workspace: Path) -> None:
    remove_path(_safe_path(workspace, Path(".staging")))


def publish_stage(workspace: Path, staging_root: Path, stage_name: str) -> None:
    """Atomically rename validated, stage-owned outputs onto canonical paths."""
    for relative in STAGE_BY_NAME[stage_name].owned_paths:
        source = staging_root / relative
        if not source.exists() and not source.is_symlink():
            raise RuntimeError(f"Stage did not stage its owned output: {relative}")
        destination = _safe_path(workspace, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        remove_path(destination)
        source.replace(destination)
    remove_path(staging_root)
    staging_parent = workspace / ".staging"
    if staging_parent.is_dir() and not any(staging_parent.iterdir()):
        staging_parent.rmdir()


def link_or_copy(source: Path, destination: Path, *, directory: bool = False) -> str:
    """Prefer symlink/hardlink and provide an explicit portable copy fallback."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(source, target_is_directory=directory)
        return "symlink"
    except OSError:
        if directory:
            shutil.copytree(source, destination)
            return "copy"
        try:
            destination.hardlink_to(source)
            return "hardlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy"


@contextmanager
def capture_stage_log(path: Path) -> Iterator[TextIO]:
    """Capture Python and native-library process stdout/stderr to one stage log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    with path.open("w") as stream:
        try:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            yield stream
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


class WorkspaceLock(AbstractContextManager["WorkspaceLock"]):
    """Refuse concurrent mutation while recovering locks left by dead processes."""

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.path = self.workspace / ".pipeline.lock"
        self.acquired = False
        self.recovered_stale_lock = False

    @staticmethod
    def _process_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def __enter__(self) -> Self:
        self.workspace.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "nht_workspace_lock_v1",
            "pid": os.getpid(),
            "started_at_utc": _now(),
            "workspace": str(self.workspace),
        }
        for _ in range(2):
            try:
                descriptor = os.open(
                    self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
            except FileExistsError:
                try:
                    existing: dict[str, Any] = json.loads(self.path.read_text())
                    pid = int(existing["pid"])
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    pid = -1
                if pid > 0 and self._process_alive(pid):
                    raise RuntimeError(
                        f"Workspace is already locked by live process {pid}: "
                        f"{self.workspace}"
                    )
                self.path.unlink(missing_ok=True)
                self.recovered_stale_lock = True
                continue
            with os.fdopen(descriptor, "w") as stream:
                json.dump(payload, stream)
                stream.write("\n")
            self.acquired = True
            return self
        raise RuntimeError(f"Could not acquire workspace lock: {self.workspace}")

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False
