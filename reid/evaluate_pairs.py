"""Grouped cross-validation for real-image pair verification."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from reid.folds import build_identity_folds, partition_records
from reid.matching import (
    METHODS,
    create_detector,
    extract_features,
    load_baseline_configs,
    load_manifest,
    read_gray,
    verify_pair,
)
from reid.metrics import (
    average_precision,
    binary_metrics,
    curve_points,
    roc_auc,
    select_f1_threshold,
)
from reid.pairs import Pair, build_pairs


def evaluate_pair_list(pairs, features, norm, config, cache):
    rows = []
    for pair in pairs:
        key = (pair.image_a, pair.image_b)
        if key not in cache:
            cache[key] = verify_pair(
                features[pair.image_a], features[pair.image_b], norm, config
            )
        result = cache[key]
        rows.append(
            {
                "image_a": pair.image_a,
                "image_b": pair.image_b,
                "identity_a": pair.identity_a,
                "identity_b": pair.identity_b,
                "pair_type": pair.pair_type,
                "label": pair.label,
                "capture_relation": pair.capture_relation,
                **result,
                "feature_a_ms": features[pair.image_a].extraction_ms,
                "feature_b_ms": features[pair.image_b].extraction_ms,
                "cold_pair_ms": (
                    features[pair.image_a].extraction_ms
                    + features[pair.image_b].extraction_ms
                    + float(result["matching_ms"])
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_for(rows: list[dict]) -> dict:
    labels = [row["label"] for row in rows]
    scores = [row["score"] for row in rows]
    # Each row carries the threshold selected without its test identities.
    predictions = [bool(row["predicted"]) for row in rows]
    tp = sum(prediction and label for prediction, label in zip(predictions, labels))
    fp = sum(prediction and not label for prediction, label in zip(predictions, labels))
    fn = sum(not prediction and label for prediction, label in zip(predictions, labels))
    tn = sum(not prediction and not label for prediction, label in zip(predictions, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    baseline = binary_metrics(
        labels, [int(bool(row["baseline_found"])) for row in rows], 0.5
    )
    return {
        "samples": len(rows),
        "positives": int(sum(labels)),
        "negatives": len(rows) - int(sum(labels)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "far": fpr,
        "roc_auc": roc_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "mean_matching_ms": float(np.mean([row["matching_ms"] for row in rows])),
        "median_matching_ms": float(np.median([row["matching_ms"] for row in rows])),
        "mean_cold_pair_ms": float(np.mean([row["cold_pair_ms"] for row in rows])),
        "baseline_fixed_rule": baseline,
        "capture_relations": dict(sorted(Counter(row["capture_relation"] for row in rows).items())),
    }


def draw_plots(test_rows: list[dict], summaries: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method in METHODS:
        rows = [row for row in test_rows if row["method"] == method]
        labels = [row["label"] for row in rows]
        scores = [row["score"] for row in rows]
        roc, pr = curve_points(labels, scores)
        axes[0].plot(
            [point["fpr"] for point in roc],
            [point["tpr"] for point in roc],
            label=f"{method} ({summaries[method]['roc_auc']:.3f})",
        )
        axes[1].plot(
            [point["recall"] for point in pr],
            [point["precision"] for point in pr],
            label=f"{method} ({summaries[method]['average_precision']:.3f})",
        )
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–Recall")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "roc_pr_curves.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(METHODS, [summaries[method]["f1"] for method in METHODS])
    axes[0].set(ylabel="Out-of-fold F1", ylim=(0, 1.02))
    axes[1].bar(
        METHODS, [summaries[method]["mean_matching_ms"] for method in METHODS]
    )
    axes[1].set(ylabel="Mean matching time, ms")
    fig.tight_layout()
    fig.savefig(output / "f1_runtime.png", dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-image grouped pair verification")
    parser.add_argument("--manifest", type=Path, default=Path("results/reid_audit/manifest.csv"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("legacy/results/step05/validation/best_configs.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/reid_pair_cv"))
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--max-positive-per-identity", type=int, default=30)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    core_outputs = ("summary.json", "test_pairs.csv", "training_pairs.csv")
    if not args.overwrite and any((args.output / name).exists() for name in core_outputs):
        raise FileExistsError(
            f"Results already exist in {args.output}; use --overwrite only intentionally"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    repository_root = Path.cwd().resolve()
    records = load_manifest(args.manifest)
    configs = load_baseline_configs(args.config)
    folds = build_identity_folds(records, n_splits=args.folds, seed=args.seed)
    fold_description = []
    for fold_index, test_identities in enumerate(folds):
        training, testing = partition_records(records, test_identities)
        fold_description.append(
            {
                "fold": fold_index,
                "training_identities": sorted({record.pothole_id for record in training}),
                "test_identities": list(test_identities),
                "training_images": len(training),
                "test_images": len(testing),
            }
        )

    all_training_rows = []
    all_test_rows = []
    threshold_reports = []
    feature_reports = {}
    cv2.setRNGSeed(args.seed)
    for method in METHODS:
        print(f"{method}: extracting features for {len(records)} images", flush=True)
        config = configs[method]
        detector, norm = create_detector(method, config)
        features = {}
        for index, record in enumerate(records, 1):
            features[record.image_path] = extract_features(
                detector, read_gray(repository_root, record)
            )
            if index % 50 == 0 or index == len(records):
                print(f"  features {index}/{len(records)}", flush=True)
        feature_reports[method] = {
            "mean_ms": float(np.mean([item.extraction_ms for item in features.values()])),
            "median_ms": float(np.median([item.extraction_ms for item in features.values()])),
            "mean_keypoints": float(np.mean([len(item.keypoints) for item in features.values()])),
        }
        result_cache = {}
        for fold_index, test_identities in enumerate(folds):
            training_records, test_records = partition_records(records, test_identities)
            training_pairs = build_pairs(
                training_records,
                max_positive_per_identity=args.max_positive_per_identity,
                negative_ratio=args.negative_ratio,
                seed=args.seed + fold_index * 100 + 1,
            )
            test_pairs = build_pairs(
                test_records,
                max_positive_per_identity=args.max_positive_per_identity,
                negative_ratio=args.negative_ratio,
                seed=args.seed + fold_index * 100 + 2,
            )
            training_rows = evaluate_pair_list(
                training_pairs, features, norm, config, result_cache
            )
            threshold, tuning_metrics = select_f1_threshold(
                [row["label"] for row in training_rows],
                [row["score"] for row in training_rows],
            )
            test_rows = evaluate_pair_list(test_pairs, features, norm, config, result_cache)
            for partition, rows in (("training", training_rows), ("test", test_rows)):
                for row in rows:
                    row.update(
                        {
                            "method": method,
                            "fold": fold_index,
                            "partition": partition,
                            "threshold": threshold,
                            "predicted": bool(row["score"] >= threshold),
                        }
                    )
            all_training_rows.extend(training_rows)
            all_test_rows.extend(test_rows)
            test_metrics = binary_metrics(
                [row["label"] for row in test_rows],
                [row["score"] for row in test_rows],
                threshold,
            )
            threshold_reports.append(
                {
                    "method": method,
                    "fold": fold_index,
                    "threshold": threshold,
                    "training_metrics": tuning_metrics,
                    "test_metrics": test_metrics,
                }
            )
            print(
                f"  fold {fold_index}: threshold={threshold:.4f} "
                f"train_F1={tuning_metrics['f1']:.3f} test_F1={test_metrics['f1']:.3f}",
                flush=True,
            )

    write_csv(args.output / "training_pairs.csv", all_training_rows)
    write_csv(args.output / "test_pairs.csv", all_test_rows)
    summaries = {
        method: summary_for([row for row in all_test_rows if row["method"] == method])
        for method in METHODS
    }
    summary = {
        "protocol": "identity_disjoint_grouped_cross_validation",
        "images_in_manifest": len(records),
        "identity_count": len({record.pothole_id for record in records}),
        "folds": fold_description,
        "pair_settings": {
            "max_positive_per_identity": args.max_positive_per_identity,
            "negative_ratio": args.negative_ratio,
            "seed": args.seed,
            "duplicate_policy": "only quality_status=ok",
        },
        "frozen_baseline_configs": configs,
        "feature_extraction": feature_reports,
        "threshold_selection": threshold_reports,
        "results": summaries,
        "notes": [
            "Positive pairs are different real files of the same manually grouped physical pothole.",
            "All capture_id values are unknown; pairs may be correlated frames from one recording.",
            "Exact and perceptual duplicate copies flagged by the manifest are excluded.",
            "Identity labels construct folds and score ground truth but are never passed to verify_pair().",
            "Each threshold is selected on training identities only and then applied to held-out identities.",
            "IoU is omitted because data_reid has no corresponding ground-truth masks.",
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    draw_plots(all_test_rows, summaries, args.output)
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)
    print(f"Written: {args.output.resolve()}")


if __name__ == "__main__":
    main()
