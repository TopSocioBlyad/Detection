"""Feature extraction and pair verification for real-image pothole re-ID."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


METHODS = ("SIFT", "ORB", "AKAZE", "BRISK")


@dataclass(frozen=True)
class ImageRecord:
    image_path: str
    pothole_id: str
    source_split: str
    source_filename: str
    capture_id: str
    quality_status: str


@dataclass
class Features:
    keypoints: tuple
    descriptors: np.ndarray | None
    image_shape: tuple[int, int]
    extraction_ms: float


def load_manifest(path: Path, *, include_flagged: bool = False) -> list[ImageRecord]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "image_path",
        "pothole_id",
        "source_split",
        "source_filename",
        "capture_id",
        "quality_status",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest is empty or misses columns: {sorted(required)}")
    records = [ImageRecord(**{key: row[key] for key in required}) for row in rows]
    if not include_flagged:
        records = [record for record in records if record.quality_status == "ok"]
    if not records:
        raise ValueError("Manifest has no usable images")
    return sorted(records, key=lambda record: record.image_path)


def load_baseline_configs(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    configs = data.get("best_configs")
    if not isinstance(configs, dict) or set(configs) != set(METHODS):
        raise ValueError(f"Expected baseline configurations for exactly {METHODS}")
    return configs


def create_detector(method: str, config: dict):
    nfeatures = int(config["nfeatures"])
    if method == "SIFT":
        return cv2.SIFT_create(nfeatures=nfeatures), cv2.NORM_L2
    if method == "ORB":
        return cv2.ORB_create(nfeatures=nfeatures), cv2.NORM_HAMMING
    if method == "AKAZE":
        threshold = config.get("akaze_threshold")
        if threshold is None:
            raise ValueError("AKAZE baseline requires akaze_threshold")
        return cv2.AKAZE_create(threshold=float(threshold)), cv2.NORM_HAMMING
    if method == "BRISK":
        return cv2.BRISK_create(), cv2.NORM_HAMMING
    raise ValueError(f"Unsupported method: {method}")


def read_gray(repository_root: Path, record: ImageRecord) -> np.ndarray:
    path = repository_root / record.image_path
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode image: {record.image_path}")
    return image


def extract_features(detector, gray: np.ndarray) -> Features:
    started = time.perf_counter()
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return Features(tuple(keypoints), descriptors, gray.shape[:2], elapsed_ms)


def _empty_result(reason: str, first: Features, second: Features) -> dict:
    return {
        "score": 0.0,
        "baseline_found": False,
        "reason": reason,
        "kp_a": len(first.keypoints),
        "kp_b": len(second.keypoints),
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "median_reprojection_px": None,
        "projected_area_ratio": None,
        "matching_ms": None,
    }


def verify_pair(first: Features, second: Features, norm: int, config: dict) -> dict:
    """Compare two feature sets without identity labels or ground-truth masks."""
    started = time.perf_counter()
    result = _empty_result("no_descriptors", first, second)
    descriptors_a = first.descriptors
    descriptors_b = second.descriptors
    if (
        descriptors_a is None
        or descriptors_b is None
        or len(descriptors_a) < 2
        or len(descriptors_b) < 2
    ):
        result["matching_ms"] = (time.perf_counter() - started) * 1000
        return result

    ratio_threshold = float(config["ratio"])
    rows = cv2.BFMatcher(norm, crossCheck=False).knnMatch(
        descriptors_a, descriptors_b, k=2
    )
    good = [
        nearest
        for row in rows
        if len(row) == 2
        for nearest, second_nearest in [row]
        if nearest.distance < ratio_threshold * second_nearest.distance
    ]
    result["good_matches"] = len(good)
    if len(good) < 4:
        result["reason"] = "few_matches"
        result["matching_ms"] = (time.perf_counter() - started) * 1000
        return result

    source = np.float32(
        [first.keypoints[match.queryIdx].pt for match in good]
    ).reshape(-1, 1, 2)
    destination = np.float32(
        [second.keypoints[match.trainIdx].pt for match in good]
    ).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(
        source,
        destination,
        cv2.RANSAC,
        float(config["ransac_threshold"]),
    )
    if homography is None or inlier_mask is None or not np.isfinite(homography).all():
        result["reason"] = "no_homography"
        result["matching_ms"] = (time.perf_counter() - started) * 1000
        return result

    inlier_flags = inlier_mask.ravel().astype(bool)
    inliers = int(inlier_flags.sum())
    inlier_ratio = inliers / len(good)
    result["inliers"] = inliers
    result["inlier_ratio"] = inlier_ratio
    projected_points = cv2.perspectiveTransform(source, homography)
    errors = np.linalg.norm(
        (projected_points - destination).reshape(-1, 2), axis=1
    )
    if inliers:
        result["median_reprojection_px"] = float(np.median(errors[inlier_flags]))

    height, width = first.image_shape
    corners = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    ).reshape(-1, 1, 2)
    projected_corners = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    area = abs(float(cv2.contourArea(projected_corners.astype(np.float32))))
    source_area = float(width * height)
    area_ratio = area / source_area if source_area else math.inf
    result["projected_area_ratio"] = area_ratio
    geometry_valid = (
        np.isfinite(projected_corners).all()
        and 0.05 <= area_ratio <= 20.0
        and cv2.isContourConvex(projected_corners.astype(np.float32))
    )
    if not geometry_valid:
        result["reason"] = "invalid_geometry"
        result["matching_ms"] = (time.perf_counter() - started) * 1000
        return result

    result["reason"] = "ok"
    result["score"] = float(inliers * inlier_ratio)
    result["baseline_found"] = bool(
        inliers >= int(config["min_inliers"])
        and inlier_ratio >= float(config["min_inlier_ratio"])
    )
    result["matching_ms"] = (time.perf_counter() - started) * 1000
    return result
