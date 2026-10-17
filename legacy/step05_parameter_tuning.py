"""Step 05: reproducible parameter tuning on Pothole-600 validation only.

Requirements: step03_batch.py and step04_compare.py in the same directory.
Do NOT use testing for selection. Detection decisions never use true masks.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import step03_batch as base
import step04_compare as comp

METHODS = ("SIFT", "ORB", "AKAZE", "BRISK")
SCENARIOS = ("identity", "rotate_60", "scale_0.75", "perspective",
             "darker", "blur_sigma_0.8", "blur_sigma_1.5", "blur_sigma_2.5")
AKAZE_THRESHOLDS = (0.001, 0.0005, 0.0001, 0.00005)
RATIOS = (0.65, 0.70, 0.75, 0.80, 0.85)
MIN_INLIERS = (6, 10, 15, 20)


def create_detector(method, nfeatures, threshold=None):
    if method == "AKAZE":
        return cv2.AKAZE_create(threshold=threshold if threshold is not None else .001), cv2.NORM_HAMMING
    return comp.make_method(method, nfeatures)


def build_cases(pairs, selected, count, seed, scenarios):
    """Same deterministic positive/negative pairs for all configurations."""
    rng = np.random.default_rng(seed)
    cases = []
    skipped = []
    for stem, rgb, label in selected:
        try:
            image, mask = base.read_data(rgb, label)
        except (OSError, ValueError, cv2.error) as err:
            skipped.append({"image": stem, "reason": str(err)})
            continue
        try:
            template, tm, _ = base.make_template(image, mask, 45)
        except (ValueError, cv2.error) as err:
            skipped.append({"image": stem, "reason": str(err)})
            continue
        targets = []
        for scenario in scenarios:
            scene, true_mask = comp.make_scene(image, mask, scenario)
            if true_mask.any():
                targets.append(("positive", scenario, scene, true_mask))
            else:
                skipped.append({"image": stem, "scenario": scenario, "reason": "out_of_frame"})
        other = [p for p in pairs if p[0] != stem]
        if other and count:
            choices = rng.choice(len(other), size=min(count, len(other)), replace=False)
            for i in choices:
                neg_id, neg_rgb, _ = other[int(i)]
                scene = cv2.imread(str(neg_rgb), cv2.IMREAD_COLOR)
                if scene is not None:
                    targets.append(("negative", f"other_{neg_id}", scene, None))
        cases.append((stem, template, tm, targets))
    return cases, skipped


def calculate_summary(rows, iou_threshold):
    pos = [r for r in rows if r["pair_type"] == "positive"]
    neg = [r for r in rows if r["pair_type"] == "negative"]
    tp = sum(bool(r["found"]) and r["iou"] >= iou_threshold for r in pos)
    fn = len(pos) - tp
    fp_wrong = sum(bool(r["found"]) and r["iou"] < iou_threshold for r in pos)
    fp_neg = sum(bool(r["found"]) for r in neg)
    fp = fp_wrong + fp_neg
    tn = len(neg) - fp_neg
    p = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * p * recall / (p + recall) if p + recall else 0.
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, fp_wrong=fp_wrong, fp_negative=fp_neg,
                precision=round(p, 6), recall=round(recall, 6), f1=round(f1, 6),
                mean_iou=round(float(np.mean([r["iou"] for r in pos])), 6) if pos else 0.,
                mean_keypoints=round(float(np.mean([r["kp_template"] for r in rows])), 2) if rows else 0.,
                mean_matches=round(float(np.mean([r["good_matches"] for r in rows])), 2) if rows else 0.,
                mean_inliers=round(float(np.mean([r["inliers"] for r in rows])), 2) if rows else 0.,
                mean_runtime_ms=round(float(np.mean([r["runtime_ms"] for r in rows])), 2) if rows else 0.,
                positives=len(pos), negatives=len(neg))


def evaluate(cases, method, *, threshold=None, ratio=.75, min_inliers=10,
             ransac_threshold=3., min_inlier_ratio=.35, nfeatures=2000, iou_threshold=.5):
    detector, norm = create_detector(method, nfeatures, threshold)
    rows = []
    for stem, template, tm, targets in cases:
        features = comp.compute_features(detector, template)
        for typ, name, scene, ground_truth in targets:
            metrics, predicted = comp.detect(features, tm, scene, detector, norm,
                ratio=ratio, ransac_threshold=ransac_threshold,
                min_inliers=min_inliers, min_inlier_ratio=min_inlier_ratio)
            iou = float(base.iou_score(predicted, ground_truth)) if typ == "positive" and metrics["found"] else 0.
            rows.append(dict(image=stem, pair_type=typ, scenario=name, iou=iou, **metrics))
    return calculate_summary(rows, iou_threshold), rows


def csv_write(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def charts(akaze, matching, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if akaze:
        fig, ax = plt.subplots(figsize=(8, 5))
        xx = [str(r["threshold"]) for r in akaze]
        ax.plot(xx, [r["f1"] for r in akaze], "o-", label="F1")
        ax.plot(xx, [r["recall"] for r in akaze], "s--", label="Recall")
        ax.set(xlabel="AKAZE threshold", ylabel="Score", ylim=(0, 1.05))
        ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
        fig.savefig(out / "akaze_threshold_quality.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xx, [r["mean_keypoints"] for r in akaze], "o-")
        ax.set(xlabel="AKAZE threshold", ylabel="Mean keypoints in template")
        ax.grid(alpha=.3); fig.tight_layout()
        fig.savefig(out / "akaze_threshold_keypoints.png", dpi=160); plt.close(fig)
    if matching:
        for method in METHODS:
            ms = [r for r in matching if r["method"] == method]
            if not ms: continue
            ratios = sorted({r["ratio"] for r in ms})
            minimums = sorted({r["min_inliers"] for r in ms})
            heat = np.full((len(minimums), len(ratios)), np.nan)
            for r in ms:
                heat[minimums.index(r["min_inliers"]), ratios.index(r["ratio"])] = r["f1"]
            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(heat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(ratios)), ratios)
            ax.set_yticks(range(len(minimums)), minimums)
            ax.set(xlabel="Lowe ratio", ylabel="Minimum RANSAC inliers", title=method)
            fig.colorbar(im, ax=ax, label="F1")
            fig.tight_layout(); fig.savefig(out / f"heatmap_{method.lower()}.png", dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Step 05 — tune keypoint methods on validation only")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--limit", type=int, default=10, help="Number of validation images; 0 = all")
    parser.add_argument("--phase", choices=("akaze", "matching", "all"), default="all")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--negatives", type=int, default=1)
    parser.add_argument("--nfeatures", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/step05/validation")
    args = parser.parse_args()
    if args.limit < 0 or args.negatives < 0: parser.error("limit and negatives must be >= 0")
    pairs = base.load_pairs(Path(args.data_root), "validation")
    selected = pairs[:args.limit] if args.limit else pairs
    cases, skipped = build_cases(pairs, selected, args.negatives, args.seed, args.scenarios)
    if not cases: raise RuntimeError("Нет пригодных эталонов. Проверь data/validation/rgb и label")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    akaze_results = []; matching_results = []
    selected_threshold = .001
    if args.phase in ("akaze", "all"):
        print("Phase A: AKAZE threshold sweep", flush=True)
        for threshold in AKAZE_THRESHOLDS:
            stats, _ = evaluate(cases, "AKAZE", threshold=threshold, nfeatures=args.nfeatures)
            record = dict(method="AKAZE", threshold=threshold, **stats)
            akaze_results.append(record)
            print(f"  threshold={threshold:.5g} F1={stats['f1']:.4f} KP={stats['mean_keypoints']}", flush=True)
        selected_threshold = sorted(akaze_results, key=lambda r:(-r["f1"], r["mean_runtime_ms"], -r["threshold"]))[0]["threshold"]
        csv_write(out / "akaze_thresholds.csv", akaze_results)
    if args.phase in ("matching", "all"):
        print("Phase B: ratio/min-inliers sweep", flush=True)
        if args.phase == "matching":
            # Standalone phase: use a fixed conservative AKAZE threshold; in 'all', use phase-A selected threshold.
            selected_threshold = .0001
        for method in METHODS:
            print(" ", method, flush=True)
            for ratio in RATIOS:
                for min_inliers in MIN_INLIERS:
                    stats, _ = evaluate(cases, method,
                        threshold=selected_threshold if method == "AKAZE" else None,
                        ratio=ratio, min_inliers=min_inliers, nfeatures=args.nfeatures)
                    row = dict(method=method, akaze_threshold=selected_threshold if method == "AKAZE" else "",
                               ratio=ratio, min_inliers=min_inliers, **stats)
                    matching_results.append(row)
            best = max((r for r in matching_results if r["method"] == method), key=lambda r:(r["f1"], -r["mean_runtime_ms"]))
            print(f"    best F1={best['f1']:.4f} ratio={best['ratio']} inliers={best['min_inliers']}", flush=True)
        csv_write(out / "matching_grid.csv", matching_results)
    best_configs = {}
    for method in METHODS:
        valid = [r for r in matching_results if r["method"] == method]
        if valid:
            best = max(valid, key=lambda r:(r["f1"], -r["mean_runtime_ms"]))
            best_configs[method] = {"ratio": best["ratio"], "min_inliers": best["min_inliers"],
                                    "ransac_threshold": 3., "min_inlier_ratio": .35,
                                    "nfeatures": args.nfeatures,
                                    "akaze_threshold": selected_threshold if method == "AKAZE" else None,
                                    "validation_f1": best["f1"]}
    report = {"split": "validation", "records": len(pairs), "templates_evaluated": len(cases),
              "phase": args.phase, "scenarios": args.scenarios, "negatives_per_template": args.negatives,
              "seed": args.seed, "selected_akaze_threshold": selected_threshold,
              "best_configs": best_configs, "skipped": skipped,
              "notes": ["Synthetic positives are correlated; do not treat as independent photographs.",
                        "Different file IDs may show the same real pothole; manually inspect negative false positives.",
                        "No ground truth mask is used when the algorithm decides 'found'.",
                        "Validation F1 is selection score, NOT an unbiased final test result.",
                        "SIFT/ORB/AKAZE/BRISK share numerical ratio/inlier grids; other settings differ by detector."]}
    (out / "best_configs.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    charts(akaze_results, matching_results, out)
    print("Done:", out.resolve(), flush=True)


if __name__ == "__main__":
    main()
