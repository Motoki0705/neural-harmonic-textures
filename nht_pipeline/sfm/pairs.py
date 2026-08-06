"""Build a deterministic video pair graph for SfM feature matching."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def image_names(image_dir: Path) -> list[str]:
    """Return image paths in deterministic temporal (filename) order."""
    names = sorted(
        path.relative_to(image_dir).as_posix()
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix in IMAGE_SUFFIXES
    )
    if not names:
        raise ValueError(f"No images found under {image_dir}")
    if len(names) != len(set(names)):
        raise ValueError("Image names are not unique")
    return names


def canonical_pair(name0: str, name1: str) -> tuple[str, str]:
    if name0 == name1:
        raise ValueError(f"Self-pair is not allowed: {name0}")
    return (name0, name1) if name0 < name1 else (name1, name0)


def sequential_pairs(names: list[str], overlap: int) -> set[tuple[str, str]]:
    if overlap < 1:
        raise ValueError("Sequential overlap must be at least 1")
    return {
        canonical_pair(names[index], names[other])
        for index in range(len(names))
        for other in range(index + 1, min(index + overlap + 1, len(names)))
    }


def read_pairs(path: Path) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected two image names")
        pairs.add(canonical_pair(*fields))
    return pairs


def validate_names(pairs: Iterable[tuple[str, str]], known_names: set[str]) -> None:
    unknown = sorted(
        {name for pair in pairs for name in pair if name not in known_names}
    )
    if unknown:
        raise ValueError(f"Pair graph refers to unknown images: {unknown[:10]}")


def write_pair_graph(
    image_dir: Path,
    output: Path,
    sequential_overlap: int,
    retrieval_pairs: Path | None = None,
    summary_output: Path | None = None,
) -> dict[str, object]:
    names = image_names(image_dir)
    sequential = sequential_pairs(names, sequential_overlap)
    retrieval = read_pairs(retrieval_pairs) if retrieval_pairs else set()
    validate_names(retrieval, set(names))
    combined = sequential | retrieval

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{a} {b}\n" for a, b in sorted(combined)))

    temporal_index = {name: index for index, name in enumerate(names)}
    temporal_distances = Counter(
        abs(temporal_index[a] - temporal_index[b]) for a, b in combined
    )
    summary: dict[str, object] = {
        "image_count": len(names),
        "sequential_overlap": sequential_overlap,
        "sequential_pair_count": len(sequential),
        "retrieval_pair_count": len(retrieval),
        "retrieval_pairs_added": len(combined - sequential),
        "combined_pair_count": len(combined),
        "long_range_pair_count": sum(
            count
            for distance, count in temporal_distances.items()
            if distance > sequential_overlap
        ),
        "maximum_temporal_distance": max(temporal_distances, default=0),
        "output": str(output),
    }
    if summary_output:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequential-overlap", type=int, default=10)
    parser.add_argument("--retrieval-pairs", type=Path)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    summary = write_pair_graph(
        args.image_dir,
        args.output,
        args.sequential_overlap,
        args.retrieval_pairs,
        args.summary_output,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
