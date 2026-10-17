"""Render representative inlier matches from pair cross-validation results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from reid.matching import METHODS, create_detector, load_baseline_configs


def read_color(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image


def render(first, second, detector, norm, config, title: str) -> np.ndarray:
    gray_a = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
    keypoints_a, descriptors_a = detector.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = detector.detectAndCompute(gray_b, None)
    good = []
    if descriptors_a is not None and descriptors_b is not None and len(descriptors_b) >= 2:
        rows = cv2.BFMatcher(norm).knnMatch(descriptors_a, descriptors_b, k=2)
        good = [
            nearest
            for row in rows
            if len(row) == 2
            for nearest, second_nearest in [row]
            if nearest.distance < float(config["ratio"]) * second_nearest.distance
        ]
    verified = good
    if len(good) >= 4:
        source = np.float32(
            [keypoints_a[match.queryIdx].pt for match in good]
        ).reshape(-1, 1, 2)
        destination = np.float32(
            [keypoints_b[match.trainIdx].pt for match in good]
        ).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(
            source,
            destination,
            cv2.RANSAC,
            float(config["ransac_threshold"]),
        )
        if mask is not None:
            verified = [match for match, flag in zip(good, mask.ravel()) if flag]
    visualization = cv2.drawMatches(
        first,
        keypoints_a,
        second,
        keypoints_b,
        verified[:80],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.rectangle(visualization, (0, 0), (visualization.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        visualization,
        title[:120],
        (6, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return visualization


def save_jpeg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise OSError(f"Cannot encode {path}")
    encoded.tofile(str(path))


def select_examples(rows: list[dict]) -> dict[str, dict]:
    groups = {
        "true_positive": [
            row for row in rows if int(row["label"]) == 1 and row["predicted"] == "True"
        ],
        "false_negative": [
            row for row in rows if int(row["label"]) == 1 and row["predicted"] == "False"
        ],
        "true_negative_hard": [
            row for row in rows if int(row["label"]) == 0 and row["predicted"] == "False"
        ],
        "false_positive": [
            row for row in rows if int(row["label"]) == 0 and row["predicted"] == "True"
        ],
    }
    selected = {}
    for category, candidates in groups.items():
        if not candidates:
            continue
        if category == "true_positive":
            selected[category] = max(candidates, key=lambda row: float(row["score"]))
        else:
            # Closest difficult example: highest score among errors/negatives.
            selected[category] = max(candidates, key=lambda row: float(row["score"]))
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render pair matching examples")
    parser.add_argument(
        "--pairs", type=Path, default=Path("results/reid_pair_cv/test_pairs.csv")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("legacy/results/step05/validation/best_configs.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/reid_pair_cv/examples")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.pairs.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    configs = load_baseline_configs(args.config)
    root = Path.cwd()
    written = []
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        detector, norm = create_detector(method, configs[method])
        for category, row in select_examples(method_rows).items():
            first = read_color(root / row["image_a"])
            second = read_color(root / row["image_b"])
            title = (
                f"{method} {category} score={float(row['score']):.3f} "
                f"threshold={float(row['threshold']):.3f} inliers={row['inliers']}"
            )
            destination = args.output / f"{method.lower()}_{category}.jpg"
            save_jpeg(
                destination,
                render(first, second, detector, norm, configs[method], title),
            )
            written.append(destination.as_posix())
    print("\n".join(written))


if __name__ == "__main__":
    main()
