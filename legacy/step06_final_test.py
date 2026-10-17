"""Step 06. Final, frozen-parameter evaluation on Pothole-600 TESTING.

Uses unmodified logic from step03_batch.py and step04_compare.py, located
beside this script. Hyperparameters are read from Step 05 validation JSON.

Examples (PowerShell, from the project root):
  python step06_final_test.py --check-only
  python step06_final_test.py --data-root data --limit 0

Do not tune on testing; do not interpret synthetic views as independent photos.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import step03_batch as base
import step04_compare as comp

METHODS = ("SIFT", "ORB", "AKAZE", "BRISK")
# Intentionally the same synthetic scenarios as Step 05, not Step 04.
EXPECTED_SCENARIOS = (
    "identity", "rotate_60", "scale_0.75", "perspective", "darker",
    "blur_sigma_0.8", "blur_sigma_1.5", "blur_sigma_2.5",
)
FIELDS = (
    "method", "image_id", "scene_id", "pair_type", "scenario", "found",
    "success", "reason", "iou", "kp_template", "kp_scene", "good_matches",
    "inliers", "inlier_ratio", "reprojection_px", "template_ms",
    "runtime_ms", "total_ms",
)


def load_frozen_config(path: Path):
    """Validate metadata AND each actual detector configuration; do not modify it."""
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    if data.get("split") != "validation" or data.get("phase") != "matching":
        raise ValueError("Нужен JSON этапа 5: split=validation, phase=matching")
    if tuple(data.get("scenarios", ())) != EXPECTED_SCENARIOS:
        raise ValueError("Сценарии в JSON отличаются от протокола этапа 5")
    if data.get("negatives_per_template") != 1 or data.get("seed") != 42:
        raise ValueError("Ожидались negatives_per_template=1 и seed=42")
    if data.get("templates_evaluated", 0) <= 0:
        raise ValueError("Отсутствует число проверенных эталонов validation")
    configs = data.get("best_configs", {})
    if set(configs) != set(METHODS):
        raise ValueError(f"best_configs должен содержать ровно {METHODS}")
    for method in METHODS:
        cfg = configs[method]
        for key in ("ratio", "min_inliers", "ransac_threshold", "min_inlier_ratio", "nfeatures"):
            if key not in cfg:
                raise ValueError(f"{method}: нет обязательного поля {key}")
        if not 0 < float(cfg["ratio"]) < 1:
            raise ValueError(f"{method}: некорректный Lowe ratio")
        if int(cfg["min_inliers"]) < 4 or float(cfg["ransac_threshold"]) <= 0:
            raise ValueError(f"{method}: некорректные параметры RANSAC")
        if not 0 <= float(cfg["min_inlier_ratio"]) <= 1 or int(cfg["nfeatures"]) < 1:
            raise ValueError(f"{method}: некорректные пороги/число точек")
        if method == "AKAZE" and not (cfg.get("akaze_threshold") is not None and
                                      0 < float(cfg["akaze_threshold"]) < 1):
            raise ValueError("AKAZE: не задан корректный threshold")
    return data, hashlib.sha256(raw).hexdigest()


def detector_for(method: str, cfg: dict):
    if method == "AKAZE":
        return cv2.AKAZE_create(threshold=float(cfg["akaze_threshold"])), cv2.NORM_HAMMING
    return comp.make_method(method, int(cfg["nfeatures"]))


def select_negatives(pairs, selected, seed: int, count: int):
    """One common, reproducible set of negatives for all four methods."""
    rng = np.random.default_rng(seed)
    choices = {}
    for stem, _, _ in selected:
        other = [p for p in pairs if p[0] != stem]
        if not other or count == 0:
            choices[stem] = []
        else:
            indices = rng.choice(len(other), size=min(count, len(other)), replace=False)
            choices[stem] = [other[int(i)] for i in indices]
    return choices


def numeric_stats(values):
    v = np.asarray([float(x) for x in values if x is not None], dtype=np.float64)
    if not len(v):
        return {"mean": None, "median": None, "p95": None}
    return {"mean": round(float(v.mean()), 3),
            "median": round(float(np.median(v)), 3),
            "p95": round(float(np.percentile(v, 95)), 3)}


def score(rows, threshold):
    """Same TP/FP/FN/TN semantics as Step 05; wrong localization = FP + FN."""
    positives = [r for r in rows if r["pair_type"] == "positive"]
    negatives = [r for r in rows if r["pair_type"] == "negative"]
    tp = sum(bool(r["found"]) and float(r["iou"]) >= threshold for r in positives)
    fn = len(positives) - tp
    fp_wrong = sum(bool(r["found"]) and float(r["iou"]) < threshold for r in positives)
    fp_negative = sum(bool(r["found"]) for r in negatives)
    fp = fp_wrong + fp_negative
    tn = len(negatives) - fp_negative
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    results = {
        "samples": len(rows), "positives": len(positives), "negatives": len(negatives),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "fp_wrong_localization": fp_wrong, "fp_negative_pairs": fp_negative,
        "precision": round(precision, 6), "recall": round(recall, 6),
        "f1": round(f1, 6),
        "localization_success_rate": round(tp / len(positives), 6) if positives else None,
        "mean_iou_including_failures": round(
            float(np.mean([r["iou"] for r in positives])), 6) if positives else None,
        "mean_iou_successful": round(float(np.mean([
            r["iou"] for r in positives if r["success"]])), 6) if tp else None,
        "mean_good_matches": numeric_stats(r["good_matches"] for r in rows)["mean"],
        "mean_inliers": numeric_stats(r["inliers"] for r in rows)["mean"],
        "mean_inlier_ratio": numeric_stats(r["inlier_ratio"] for r in rows)["mean"],
        "mean_kp_template": numeric_stats(r["kp_template"] for r in rows)["mean"],
        "mean_kp_scene": numeric_stats(r["kp_scene"] for r in rows)["mean"],
        "runtime_ms": numeric_stats(r["runtime_ms"] for r in rows),
        "template_ms": numeric_stats(r["template_ms"] for r in rows),
        "total_ms": numeric_stats(r["total_ms"] for r in rows),
        "reasons": dict(sorted(Counter(str(r["reason"]) for r in rows).items())),
    }
    return results


def save_csv(path: Path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def draw_plots(rows, summaries, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib отсутствует; PNG-графики пропущены", flush=True)
        return
    method_names = list(METHODS)
    for field, name, ylabel in (
        ("f1", "f1_comparison.png", "F1 (test)"),
        ("total_ms", "runtime_comparison.png", "Mean total time per pair, ms"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ys = [summaries[m][field]["mean"] if field == "total_ms" else summaries[m][field]
              for m in method_names]
        ax.bar(method_names, [v if v is not None else 0 for v in ys])
        ax.set_ylabel(ylabel)
        if field == "f1":
            ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(out / name, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(EXPECTED_SCENARIOS))
    width = 0.8 / len(METHODS)
    for j, m in enumerate(METHODS):
        values = []
        for scenario in EXPECTED_SCENARIOS:
            subset = [r for r in rows if r["method"] == m and
                      r["pair_type"] == "positive" and r["scenario"] == scenario]
            values.append(float(np.mean([r["success"] for r in subset])) if subset else 0)
        ax.bar(x - 0.4 + width * (j + 0.5), values, width, label=m)
    ax.set_xticks(x, EXPECTED_SCENARIOS, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Successful localization rate (IoU >= 0.5)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "success_by_scenario.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Step 06: TESTING with frozen Step 05 parameters")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path,
                        default=Path("results/step05/validation/best_configs.json"))
    parser.add_argument("--out", type=Path, default=Path("results/step06/testing"))
    parser.add_argument("--limit", type=int, default=0, help="0 = весь testing (по умолчанию)")
    parser.add_argument("--check-only", action="store_true",
                        help="Проверить конфигурацию, версии и наличие данных, не оценивая testing")
    parser.add_argument("--overwrite", action="store_true",
                        help="Явно разрешить заменить предыдущие результаты testing")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit не может быть отрицательным")

    config, sha256 = load_frozen_config(args.config)
    for method in METHODS:
        try:
            detector_for(method, config["best_configs"][method])
        except (AttributeError, cv2.error) as exc:
            raise RuntimeError(f"Не удалось создать {method}; проверь opencv-contrib-python 4.11: {exc}") from exc
    pairs = base.load_pairs(args.data_root, "testing")  # Deliberately never validation/training.
    selected = pairs[:args.limit] if args.limit else pairs
    if not selected:
        raise RuntimeError("В testing нет пригодных rgb/label пар")
    print(f"OpenCV {cv2.__version__}; testing пар: {len(pairs)}; будет обработано: {len(selected)}")
    print(f"Frozen config: {args.config.resolve()} | sha256={sha256}")
    print("Параметры: валидированы на validation; на testing подбор ЗАПРЕЩЁН.")
    for method in METHODS:
        cfg = config["best_configs"][method]
        print(f"  {method}: ratio={cfg['ratio']}, min_inliers={cfg['min_inliers']}, "
              f"AKAZE_threshold={cfg.get('akaze_threshold')}")
    if args.check_only:
        print("Проверка завершена. Изображения testing не анализировались.")
        return

    out = args.out
    expected_files = ("comparison.csv", "summary.json", "config_snapshot.json")
    if not args.overwrite and any((out / n).exists() for n in expected_files):
        raise FileExistsError(f"В {out} уже есть результаты. Для намеренного перезапуска "
                              "передай --overwrite (НЕ подбирай параметры на testing).")
    out.mkdir(parents=True, exist_ok=True)
    # Copy byte-for-byte to show exactly which frozen config was used.
    (out / "config_snapshot.json").write_bytes(args.config.read_bytes())

    negative_choices = select_negatives(pairs, selected, seed=42, count=1)
    # Detector for each method is constructed once; template extraction once per
    # method/template; same scene/negative pair is evaluated for every method.
    detectors = {m: detector_for(m, config["best_configs"][m]) for m in METHODS}
    rows = []
    skipped = []
    for index, (stem, rgb_path, mask_path) in enumerate(selected, 1):
        try:
            image, mask = base.read_data(rgb_path, mask_path)
            template, tm, _ = base.make_template(image, mask, 45)
            scenes = []
            for scenario in EXPECTED_SCENARIOS:
                scene, truth = comp.make_scene(image, mask, scenario)
                if truth.any():
                    scenes.append(("positive", scenario, stem, scene, truth))
                else:
                    skipped.append({"image_id": stem, "scenario": scenario,
                                    "reason": "object_out_of_frame"})
            for neg_stem, neg_path, _ in negative_choices[stem]:
                scene = cv2.imread(str(neg_path), cv2.IMREAD_COLOR)
                if scene is None:
                    skipped.append({"image_id": stem, "scene_id": neg_stem,
                                    "reason": "unreadable_negative"})
                    continue
                scenes.append(("negative", "other_pothole", neg_stem, scene, None))
        except (OSError, ValueError, cv2.error) as exc:
            skipped.append({"image_id": stem, "reason": str(exc)})
            continue

        for method in METHODS:
            cfg = config["best_configs"][method]
            detector, norm = detectors[method]
            try:
                t0 = time.perf_counter()
                template_features = comp.compute_features(detector, template)
                template_ms = (time.perf_counter() - t0) * 1000
                for pair_type, scenario, scene_stem, scene, truth in scenes:
                    # Crucially, truth is NOT passed to detect().
                    metrics, pred = comp.detect(
                        template_features, tm, scene, detector, norm,
                        ratio=float(cfg["ratio"]),
                        min_inliers=int(cfg["min_inliers"]),
                        ransac_threshold=float(cfg["ransac_threshold"]),
                        min_inlier_ratio=float(cfg["min_inlier_ratio"]),
                    )
                    overlap = (float(base.iou_score(pred, truth))
                               if pair_type == "positive" and metrics["found"] else 0.0)
                    success = (bool(metrics["found"] and overlap >= 0.5)
                               if pair_type == "positive" else not bool(metrics["found"]))
                    rows.append({
                        "method": method, "image_id": stem, "scene_id": scene_stem,
                        "pair_type": pair_type, "scenario": scenario,
                        "found": bool(metrics["found"]), "success": success,
                        "reason": metrics["reason"],
                        "iou": round(overlap, 6) if pair_type == "positive" else None,
                        "kp_template": int(metrics["kp_template"]),
                        "kp_scene": int(metrics["kp_scene"]),
                        "good_matches": int(metrics["good_matches"]),
                        "inliers": int(metrics["inliers"]),
                        "inlier_ratio": round(float(metrics["inlier_ratio"]), 6),
                        "reprojection_px": metrics["reprojection_px"],
                        "template_ms": round(template_ms, 3),
                        "runtime_ms": metrics["runtime_ms"],
                        "total_ms": round(template_ms + float(metrics["runtime_ms"]), 3),
                    })
            except (cv2.error, ValueError, OSError) as exc:
                # Fail loudly; silently dropping method-specific cases biases comparison.
                raise RuntimeError(f"Ошибка метода {method} на testing/{stem}: {exc}") from exc
        if index % 10 == 0 or index == len(selected):
            print(f"Эталоны: {index}/{len(selected)}; обработано сравнений: {len(rows)}", flush=True)

    if not rows:
        raise RuntimeError("Нет результатов; проверь наличие изображений и масок")

    # Assert comparable numbers of sample pairs across detectors.
    pair_keys = {
        m: {(r["image_id"], r["scene_id"], r["scenario"], r["pair_type"])
            for r in rows if r["method"] == m} for m in METHODS
    }
    if any(pair_keys[m] != pair_keys[METHODS[0]] for m in METHODS[1:]):
        raise RuntimeError("Разные наборы тестовых пар у методов — результат некорректен")

    save_csv(out / "comparison.csv", rows, FIELDS)
    summaries = {m: score([r for r in rows if r["method"] == m], 0.5) for m in METHODS}
    per_scenario_rows = []
    for method in METHODS:
        for scenario in EXPECTED_SCENARIOS:
            subset = [r for r in rows if r["method"] == method and r["scenario"] == scenario]
            s = score(subset, 0.5)
            per_scenario_rows.append({
                "method": method, "scenario": scenario, "positives": s["positives"],
                "tp": s["tp"], "fn": s["fn"], "fp_wrong_localization": s["fp_wrong_localization"],
                "success_rate": s["localization_success_rate"],
                "mean_iou_including_failures": s["mean_iou_including_failures"],
                "mean_inliers": s["mean_inliers"],
                "mean_runtime_ms": s["runtime_ms"]["mean"],
                "mean_total_ms": s["total_ms"]["mean"],
            })
    save_csv(out / "by_scenario.csv", per_scenario_rows, list(per_scenario_rows[0]))
    (out / "skipped.json").write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "split": "testing", "protocol": "step05_matching_fixed", "full_testing": args.limit == 0,
        "data_root": str(args.data_root), "dataset_records": len(pairs),
        "templates_requested": len(selected),
        "templates_evaluated": len(set(r["image_id"] for r in rows)),
        "scenarios": list(EXPECTED_SCENARIOS), "negatives_per_template": 1,
        "seed": 42, "padding": 45, "iou_success_threshold": 0.5,
        "frozen_config_file": str(args.config), "frozen_config_sha256": sha256,
        "opencv_version": cv2.__version__, "numpy_version": np.__version__,
        "best_configs_used": config["best_configs"], "results": summaries,
        "skipped_count": len(skipped),
        "notes": [
            "Положительные пары: искусственные преобразования ОДНОГО RGB кадра, не независимые съёмки.",
            "Отрицательные пары: другие файлы testing; идентичность физических ям неизвестна.",
            "FP за неверную локализацию на положительной паре одновременно учитывается как FN.",
            "IoU вычисляется после детекции; истинная маска не передаётся алгоритму принятия решения.",
            "runtime_ms не включает построение шаблона и извлечение его признаков; "
            "total_ms = runtime_ms + template_ms (стоимость извлечения признаков шаблона "
            "отнесена к каждой паре, без подготовки ROI и генерации сцены).",
            "При тестировании параметры не изменялись. Не выбирать новые параметры по этому split.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_plots(rows, summaries, out)
    print("\nИТОГ TESTING (не validation):")
    for method in METHODS:
        s = summaries[method]
        print(f"{method:6}  P={s['precision']:.4f} R={s['recall']:.4f} F1={s['f1']:.4f} "
              f"IoU={s['mean_iou_including_failures']!s:>7} "
              f"total={s['total_ms']['mean']} ms")
    print(f"\nРезультаты сохранены: {out.resolve()}")


if __name__ == "__main__":
    main()
