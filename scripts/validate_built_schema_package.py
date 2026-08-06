"""Verify canonical schemas and their runtime validator ship in built artifacts."""

from __future__ import annotations

import tarfile
from pathlib import Path
from zipfile import ZipFile

SCHEMA_PATHS = {
    "nht_pipeline/schemas/scene.schema.json",
    "nht_pipeline/schemas/cameras.schema.json",
    "nht_pipeline/schemas/render-request.schema.json",
    "nht_pipeline/schemas/render-result.schema.json",
    "nht_pipeline/schemas/run.schema.json",
}


def _single_artifact(pattern: str) -> Path:
    artifacts = sorted(Path("dist").glob(pattern))
    if len(artifacts) != 1:
        raise RuntimeError(f"Expected one {pattern} artifact, found {artifacts}")
    return artifacts[0]


def main() -> None:
    wheel = _single_artifact("*.whl")
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
        missing = SCHEMA_PATHS - names
        if missing:
            raise RuntimeError(f"Wheel is missing canonical schemas: {sorted(missing)}")
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = archive.read(metadata_name).decode()
        if "Requires-Dist: jsonschema>=4.25" not in metadata:
            raise RuntimeError("Wheel does not declare jsonschema as a runtime dependency")

    source_distribution = _single_artifact("*.tar.gz")
    with tarfile.open(source_distribution, "r:gz") as archive:
        names = {name.split("/", 1)[-1] for name in archive.getnames() if "/" in name}
        missing = SCHEMA_PATHS - names
        if missing:
            raise RuntimeError(f"Sdist is missing canonical schemas: {sorted(missing)}")

    print("wheel/sdist contain all canonical schemas and runtime jsonschema metadata")


if __name__ == "__main__":
    main()
