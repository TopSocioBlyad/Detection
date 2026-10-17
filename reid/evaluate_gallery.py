"""Open-set gallery/query evaluation on held-out pothole identities."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from reid.matching import (
    METHODS,
    create_detector,
    extract_features,
    load_baseline_configs,
    load_manifest,
    read_gray,
    verify_pair,
)


def select_gallery(records, count: int):
    ordered = sorted(records, key=lambda record: record.image_path)
    if len(ordered) <= count:
        raise ValueError("An identity needs more images than gallery_per_identity")
    indices = np.linspace(0, len(ordered) - 1, num=count, dtype=int)
    selected = [ordered[int(index)] for index in indices]
    selected_paths = {record.image_path for record in selected}
    queries = [record for record in ordered if record.image_path not in selected_paths]
    return selected, queries


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    known = [row for row in rows if row["target_known"]]
    unknown = [row for row in rows if not row["target_known"]]
    statuses = Counter(row["status"] for row in rows)
    known_correct = statuses["correct_identity"]
    false_rejects = statuses["false_reject"]
    misidentifications = statuses["misidentification"]
    correct_rejects = statuses["correct_reject"]
    false_accepts = statuses["false_accept"]
    return {
        "samples": len(rows),
        "known_queries": len(known),
        "unknown_queries": len(unknown),
        "rank1_identification_rate": known_correct / len(known) if known else None,
        "false_reject_rate": false_rejects / len(known) if known else None,
        "misidentification_rate": misidentifications / len(known) if known else None,
        "correct_rejection_rate": correct_rejects / len(unknown) if unknown else None,
        "false_accept_rate": false_accepts / len(unknown) if unknown else None,
        "overall_accuracy": (known_correct + correct_rejects) / len(rows) if rows else None,
        "status_counts": dict(sorted(statuses.items())),
        "mean_query_matching_ms": float(np.mean([row["query_matching_ms"] for row in rows])),
        "mean_query_total_ms": float(np.mean([row["query_total_ms"] for row in rows])),
    }


def draw_plot(summaries: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(METHODS))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(
        x - width / 2,
        [summaries[method]["rank1_identification_rate"] for method in METHODS],
        width,
        label="Known: correct identity",
    )
    axis.bar(
        x + width / 2,
        [summaries[method]["correct_rejection_rate"] for method in METHODS],
        width,
        label="Unknown: correct rejection",
    )
    axis.set_xticks(x, METHODS)
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Rate")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "gallery_query_rates.png", dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open-set gallery/query evaluation")
    parser.add_argument("--manifest", type=Path, default=Path("results/reid_audit/manifest.csv"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("legacy/results/step05/validation/best_configs.json"),
    )
    parser.add_argument(
        "--pair-summary", type=Path, default=Path("results/reid_pair_cv/summary.json")
    )
    parser.add_argument("--output", type=Path, default=Path("results/reid_gallery"))
    parser.add_argument("--gallery-per-identity", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gallery_per_identity < 1:
        raise ValueError("gallery-per-identity must be positive")
    if not args.overwrite and (args.output / "summary.json").exists():
        raise FileExistsError(f"Results already exist in {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    repository_root = Path.cwd().resolve()
    records = load_manifest(args.manifest)
    configs = load_baseline_configs(args.config)
    pair_summary = json.loads(args.pair_summary.read_text(encoding="utf-8"))
    folds = {
        int(item["fold"]): tuple(item["test_identities"])
        for item in pair_summary["folds"]
    }
    thresholds = {
        (item["method"], int(item["fold"])): float(item["threshold"])
        for item in pair_summary["threshold_selection"]
    }
    records_by_identity = defaultdict(list)
    for record in records:
        records_by_identity[record.pothole_id].append(record)

    predictions = []
    cv2.setRNGSeed(args.seed)
    for method in METHODS:
        print(f"{method}: gallery/query feature extraction", flush=True)
        config = configs[method]
        detector, norm = create_detector(method, config)
        features = {
            record.image_path: extract_features(
                detector, read_gray(repository_root, record)
            )
            for record in records
        }
        match_cache = {}
        for fold, fold_identities in folds.items():
            threshold = thresholds[(method, fold)]
            for episode, unknown_identity in enumerate(fold_identities):
                known_identities = [
                    identity for identity in fold_identities if identity != unknown_identity
                ]
                gallery = {}
                queries = []
                for identity in known_identities:
                    selected, known_queries = select_gallery(
                        records_by_identity[identity], args.gallery_per_identity
                    )
                    gallery[identity] = selected
                    queries.extend((query, True) for query in known_queries)
                queries.extend(
                    (query, False) for query in records_by_identity[unknown_identity]
                )

                for query, target_known in queries:
                    identity_scores = {}
                    matching_time = 0.0
                    comparison_count = 0
                    for identity in known_identities:
                        scores = []
                        for reference in gallery[identity]:
                            key = (query.image_path, reference.image_path)
                            if key not in match_cache:
                                match_cache[key] = verify_pair(
                                    features[query.image_path],
                                    features[reference.image_path],
                                    norm,
                                    config,
                                )
                            comparison = match_cache[key]
                            scores.append(float(comparison["score"]))
                            matching_time += float(comparison["matching_ms"])
                            comparison_count += 1
                        identity_scores[identity] = max(scores)
                    best_identity, best_score = max(
                        identity_scores.items(), key=lambda item: (item[1], item[0])
                    )
                    predicted_identity = (
                        best_identity if best_score >= threshold else "no_match"
                    )
                    if target_known:
                        if predicted_identity == query.pothole_id:
                            status = "correct_identity"
                        elif predicted_identity == "no_match":
                            status = "false_reject"
                        else:
                            status = "misidentification"
                    else:
                        status = (
                            "correct_reject"
                            if predicted_identity == "no_match"
                            else "false_accept"
                        )
                    predictions.append(
                        {
                            "method": method,
                            "fold": fold,
                            "episode": episode,
                            "unknown_identity": unknown_identity,
                            "gallery_identities": ";".join(known_identities),
                            "gallery_paths": json.dumps(
                                {
                                    identity: [record.image_path for record in gallery[identity]]
                                    for identity in known_identities
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            "query_path": query.image_path,
                            "true_identity": query.pothole_id,
                            "target_known": target_known,
                            "predicted_identity": predicted_identity,
                            "best_score": best_score,
                            "threshold": threshold,
                            "status": status,
                            "identity_scores": json.dumps(
                                identity_scores, ensure_ascii=False, sort_keys=True
                            ),
                            "comparisons": comparison_count,
                            "query_feature_ms": features[query.image_path].extraction_ms,
                            "query_matching_ms": matching_time,
                            "query_total_ms": (
                                features[query.image_path].extraction_ms + matching_time
                            ),
                            "capture_id": query.capture_id,
                        }
                    )
            print(f"  fold {fold} complete", flush=True)

    write_csv(args.output / "predictions.csv", predictions)
    summaries = {
        method: summarize([row for row in predictions if row["method"] == method])
        for method in METHODS
    }
    report = {
        "protocol": "held_out_identity_open_set_gallery_query",
        "gallery_per_identity": args.gallery_per_identity,
        "episode_design": (
            "Each three-identity test fold is evaluated in three episodes. "
            "One identity is absent from the gallery and the other two are known."
        ),
        "threshold_source": "training identities from results/reid_pair_cv/summary.json",
        "results": summaries,
        "notes": [
            "Gallery references and queries are different files, with manifest duplicates excluded.",
            "Capture IDs are unknown, so gallery/query files may come from one recording.",
            "Episodes reuse queries under different known/unknown gallery compositions and are correlated.",
            "The predictor sees image features and calibrated scores, never pothole_id.",
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    draw_plot(summaries, args.output)
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)
    print(f"Written: {args.output.resolve()}")


if __name__ == "__main__":
    main()
