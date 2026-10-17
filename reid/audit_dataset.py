"""Audit ``data_reid`` and build a reproducible image manifest.

The folder name is used only as ground-truth identity metadata.  This script
does not feed it into any matching algorithm.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


SPLIT_PATTERN = re.compile(r"^(training|validation|testing)[_-]", re.IGNORECASE)


def read_image(path: Path) -> np.ndarray | None:
    """Decode an image without assuming that OpenCV supports the path text."""
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def source_split(filename: str) -> str:
    match = SPLIT_PATTERN.match(filename)
    return match.group(1).lower() if match else "unknown"


def rotation_hash(image: np.ndarray) -> str:
    """Exact decoded-pixel hash, canonicalized over quarter turns."""
    variants = []
    for turns in range(4):
        rotated = np.ascontiguousarray(np.rot90(image, turns))
        payload = f"{rotated.shape}:{rotated.dtype}".encode() + rotated.tobytes()
        variants.append(hashlib.sha256(payload).hexdigest())
    return min(variants)


def perceptual_rotation_hashes(image: np.ndarray) -> tuple[str, ...]:
    """Return 63-bit DCT perceptual hashes for all quarter turns."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants = []
    for turns in range(4):
        rotated = np.ascontiguousarray(np.rot90(gray, turns))
        low_frequency = cv2.dct(
            cv2.resize(rotated, (32, 32)).astype(np.float32)
        )[:8, :8].ravel()[1:]
        bits = low_frequency > np.median(low_frequency)
        variants.append("".join("1" if value else "0" for value in bits))
    return tuple(variants)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def root(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, first: int, second: int) -> None:
        first_root = self.root(first)
        second_root = self.root(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def relative(path: Path, repository_root: Path) -> str:
    return path.relative_to(repository_root).as_posix()


def audit(data_root: Path, repository_root: Path) -> tuple[list[dict], dict]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_root}")

    files = sorted(path for path in data_root.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"No files found below {data_root}")

    records = []
    exact_groups: dict[str, list[int]] = defaultdict(list)
    dimensions = Counter()
    extensions = Counter()

    for path in files:
        extensions[path.suffix.lower() or "<none>"] += 1
        image = read_image(path)
        record = {
            "path": path,
            "image": image,
            "exact_hash": None,
            "perceptual_hashes": None,
        }
        index = len(records)
        records.append(record)
        if image is None:
            continue
        height, width = image.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]
        dimensions[(width, height, channels, str(image.dtype))] += 1
        record["exact_hash"] = rotation_hash(image)
        record["perceptual_hashes"] = perceptual_rotation_hashes(image)
        exact_groups[record["exact_hash"]].append(index)

    duplicate_of: dict[int, tuple[str, int]] = {}
    exact_sets = []
    for indices in exact_groups.values():
        if len(indices) < 2:
            continue
        canonical = indices[0]
        exact_sets.append([relative(records[i]["path"], repository_root) for i in indices])
        for index in indices[1:]:
            duplicate_of[index] = ("exact_duplicate", canonical)

    # Connect images from one identity when any quarter-turn pHash is exactly
    # equal.  Comparing only one canonical orientation can miss a shared hash.
    union_find = UnionFind(len(records))
    perceptual_buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record["perceptual_hashes"] is None:
            continue
        identity = record["path"].parent.name
        for image_hash in record["perceptual_hashes"]:
            perceptual_buckets[(identity, image_hash)].append(index)
    for indices in perceptual_buckets.values():
        for index in indices[1:]:
            union_find.union(indices[0], index)
    perceptual_groups: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record["perceptual_hashes"] is not None:
            perceptual_groups[union_find.root(index)].append(index)

    near_sets = []
    for indices in perceptual_groups.values():
        remaining = [index for index in sorted(indices) if index not in duplicate_of]
        if len(remaining) < 2:
            continue
        # A zero-distance perceptual hash is only a candidate.  Identity was
        # already used to restrict union operations above.
        canonical = remaining[0]
        near_sets.append([relative(records[i]["path"], repository_root) for i in remaining])
        for index in remaining[1:]:
            duplicate_of[index] = ("near_duplicate", canonical)

    manifest = []
    for index, record in enumerate(records):
        path = record["path"]
        if record["image"] is None:
            status = "unreadable"
        elif index in duplicate_of:
            kind, canonical = duplicate_of[index]
            canonical_path = relative(records[canonical]["path"], repository_root)
            status = f"{kind}_of:{canonical_path}"
        else:
            status = "ok"
        manifest.append(
            {
                "image_path": relative(path, repository_root),
                "pothole_id": path.parent.name,
                "source_split": source_split(path.name),
                "source_filename": path.name,
                # Numeric frame names and adjacency do not prove a capture session.
                "capture_id": "unknown",
                "quality_status": status,
            }
        )

    quality_counts = Counter(row["quality_status"].split("_of:", 1)[0] for row in manifest)
    report = {
        "data_root": relative(data_root, repository_root),
        "total_files": len(files),
        "identity_counts": dict(sorted(Counter(row["pothole_id"] for row in manifest).items())),
        "extensions": dict(sorted(extensions.items())),
        "decoded_image_layouts": [
            {
                "width": width,
                "height": height,
                "channels": channels,
                "dtype": dtype,
                "count": count,
            }
            for (width, height, channels, dtype), count in sorted(dimensions.items())
        ],
        "unreadable_files": [
            row["image_path"] for row in manifest if row["quality_status"] == "unreadable"
        ],
        "quality_status_counts": dict(sorted(quality_counts.items())),
        "exact_duplicate_sets": exact_sets,
        "near_duplicate_candidate_sets": near_sets,
        "metadata_notes": [
            "Folder names are ground-truth identities and must not be used by a predictor.",
            "Current filenames contain no training/validation/testing prefix, so source_split is unknown.",
            "Numeric filenames and adjacency do not establish an independent capture; capture_id is unknown.",
            "Near-duplicate candidates have identical rotation-invariant perceptual hashes within one identity.",
            "The dataset has no masks, so the number of visible potholes cannot be certified automatically.",
        ],
    }
    return manifest, report


def write_outputs(manifest: list[dict], report: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    columns = (
        "image_path",
        "pothole_id",
        "source_split",
        "source_filename",
        "capture_id",
        "quality_status",
    )
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(manifest)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit data_reid and write manifest.csv")
    parser.add_argument("--data-root", type=Path, default=Path("data_reid"))
    parser.add_argument("--output", type=Path, default=Path("results/reid_audit"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = Path.cwd().resolve()
    data_root = args.data_root.resolve()
    try:
        data_root.relative_to(repository_root)
    except ValueError as error:
        raise SystemExit("--data-root must be inside the repository") from error
    manifest, report = audit(data_root, repository_root)
    write_outputs(manifest, report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Written: {args.output.resolve()}")


if __name__ == "__main__":
    main()
