"""Binary verification metrics without external machine-learning packages."""

from __future__ import annotations

import math

import numpy as np


def confusion(labels, scores, threshold: float) -> dict[str, int]:
    truth = np.asarray(labels, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    predicted = values >= threshold
    positive = truth == 1
    return {
        "tp": int(np.count_nonzero(predicted & positive)),
        "fp": int(np.count_nonzero(predicted & ~positive)),
        "fn": int(np.count_nonzero(~predicted & positive)),
        "tn": int(np.count_nonzero(~predicted & ~positive)),
    }


def binary_metrics(labels, scores, threshold: float) -> dict:
    counts = confusion(labels, scores, threshold)
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "far": fpr,
        "specificity": specificity,
        "threshold": float(threshold),
        "samples": len(labels),
        "positives": int(sum(int(value) for value in labels)),
        "negatives": int(len(labels) - sum(int(value) for value in labels)),
    }


def select_f1_threshold(labels, scores) -> tuple[float, dict]:
    values = np.asarray(scores, dtype=np.float64)
    if not len(values):
        raise ValueError("Cannot select a threshold from no scores")
    candidates = sorted(set(float(value) for value in values))
    candidates.append(float(np.nextafter(values.max(), math.inf)))
    best_threshold = candidates[0]
    best_metrics = binary_metrics(labels, scores, best_threshold)
    for threshold in candidates[1:]:
        metrics = binary_metrics(labels, scores, threshold)
        candidate_key = (metrics["f1"], -metrics["fpr"], threshold)
        best_key = (best_metrics["f1"], -best_metrics["fpr"], best_threshold)
        if candidate_key > best_key:
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics


def roc_auc(labels, scores) -> float | None:
    truth = np.asarray(labels, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    positives = values[truth == 1]
    negatives = values[truth == 0]
    if not len(positives) or not len(negatives):
        return None
    comparisons = positives[:, None] - negatives[None, :]
    wins = np.count_nonzero(comparisons > 0)
    ties = np.count_nonzero(comparisons == 0)
    return float((wins + 0.5 * ties) / comparisons.size)


def curve_points(labels, scores) -> tuple[list[dict], list[dict]]:
    truth = np.asarray(labels, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    thresholds = [math.inf] + sorted(
        set(float(value) for value in values), reverse=True
    )
    roc = []
    precision_recall = []
    for threshold in thresholds:
        metrics = binary_metrics(truth, values, threshold)
        roc.append(
            {
                "threshold": threshold,
                "fpr": metrics["fpr"],
                "tpr": metrics["recall"],
            }
        )
        precision_recall.append(
            {
                "threshold": threshold,
                "recall": metrics["recall"],
                "precision": metrics["precision"],
            }
        )
    return roc, precision_recall


def average_precision(labels, scores) -> float | None:
    truth = np.asarray(labels, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    positive_count = int(np.count_nonzero(truth == 1))
    if not positive_count:
        return None
    order = np.argsort(-values, kind="stable")
    sorted_truth = truth[order]
    sorted_scores = values[order]
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    area = 0.0
    for index, label in enumerate(sorted_truth):
        if label:
            true_positives += 1
        else:
            false_positives += 1
        is_last_at_score = (
            index == len(sorted_truth) - 1
            or sorted_scores[index + 1] != sorted_scores[index]
        )
        if not is_last_at_score:
            continue
        recall = true_positives / positive_count
        precision = true_positives / (true_positives + false_positives)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return float(area)
