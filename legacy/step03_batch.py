"""Пакетные эксперименты: поиск ТОЙ ЖЕ ямы Pothole-600 через SIFT+BF+RANSAC.

Примеры:
  python step03_batch.py --split validation --limit 10
  python step03_batch.py --split testing --limit 0
  python step03_batch.py --split training --limit 0 --negatives 1

Для положительных тестов применяются ИСКУССТВЕННЫЕ геометрические преобразования.
Отрицательные тесты используют другие изображения той же выборки.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}


def images_by_stem(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f'Папка не найдена: {directory}')
    return {p.stem: p for p in sorted(directory.iterdir()) if p.suffix.lower() in EXTS}


def load_pairs(root: Path, split: str) -> list[tuple[str, Path, Path]]:
    rgb = images_by_stem(root / split / 'rgb')
    masks = images_by_stem(root / split / 'label')
    pairs = [(name, path, masks[name]) for name, path in sorted(rgb.items()) if name in masks]
    if not pairs:
        raise RuntimeError(f'Нет совпадающих по имени rgb и label в {root / split}')
    return pairs


def read_data(rgb: Path, mask: Path) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(rgb), cv2.IMREAD_COLOR)
    gray_mask = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    if image is None or gray_mask is None or gray_mask.shape != image.shape[:2]:
        raise ValueError(f'Ошибка чтения изображения/маски: {rgb}, {mask}')
    return image, (gray_mask > 127).astype(np.uint8) * 255


def make_template(image: np.ndarray, mask: np.ndarray, padding: int):
    ys, xs = np.where(mask != 0)
    if len(xs) == 0:
        raise ValueError('Маска не содержит яму')
    h, w = mask.shape
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(w, int(xs.max()) + 1 + padding)
    y2 = min(h, int(ys.max()) + 1 + padding)
    return image[y1:y2, x1:x2].copy(), mask[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def synthetic_scene(image: np.ndarray, mask: np.ndarray, scenario: str):
    """Возвращает тестовый кадр и известную истинную маску в нём."""
    h, w = image.shape[:2]
    H = np.eye(3, dtype=np.float64)
    if scenario == 'identity':
        pass
    elif scenario.startswith('rotate_'):
        degrees = float(scenario.split('_')[1])
        H = np.vstack([cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), degrees, 1.0), [0, 0, 1]])
    elif scenario.startswith('scale_'):
        factor = float(scenario.split('_')[1])
        H = np.vstack([cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), 0, factor), [0, 0, 1]])
    elif scenario == 'perspective':
        src = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        dst = np.float32([[0.07*w, 0.03*h], [0.91*w, 0.08*h],
                          [0.96*w, 0.93*h], [0.03*w, 0.96*h]])
        H = cv2.getPerspectiveTransform(src, dst)
    elif scenario == 'darker':
        return cv2.convertScaleAbs(image, alpha=0.7, beta=0), mask.copy()
    elif scenario == 'blur':
        return cv2.GaussianBlur(image, (7, 7), 1.5), mask.copy()
    else:
        raise ValueError(f'Неизвестный сценарий: {scenario}')
    scene = cv2.warpPerspective(image, H, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(110, 110, 110))
    truth = cv2.warpPerspective(mask, H, (w, h), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return scene, truth


def run_detector(template, template_mask, scene, *, ratio, reproj_threshold,
                 min_inliers, min_inlier_ratio, sift):
    start = time.perf_counter()
    result = {'found': False, 'reason': '', 'kp_template': 0, 'kp_scene': 0,
              'good_matches': 0, 'inliers': 0, 'inlier_ratio': 0.0,
              'reprojection_px': None, 'runtime_ms': None}
    pred = np.zeros(scene.shape[:2], dtype=np.uint8)
    g1 = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)
    result['kp_template'], result['kp_scene'] = len(kp1), len(kp2)
    if des1 is None or des2 is None or len(des2) < 2:
        result['reason'] = 'no_descriptors'
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        pairs = matcher.knnMatch(des1, des2, k=2)
        good = [m for pair in pairs if len(pair) == 2
                for m, n in [pair] if m.distance < ratio * n.distance]
        result['good_matches'] = len(good)
        if len(good) < 4:
            result['reason'] = 'few_matches'
        else:
            src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, flags = cv2.findHomography(src, dst, cv2.RANSAC, reproj_threshold)
            n = int(flags.sum()) if flags is not None else 0
            result['inliers'] = n
            result['inlier_ratio'] = n / len(good)
            if H is None or not np.isfinite(H).all():
                result['reason'] = 'no_homography'
            elif n < min_inliers or result['inlier_ratio'] < min_inlier_ratio:
                result['reason'] = 'few_inliers'
            else:
                inlier = flags.ravel().astype(bool)
                projected = cv2.perspectiveTransform(src, H).reshape(-1, 2)
                errors = np.linalg.norm(projected - dst.reshape(-1, 2), axis=1)
                result['reprojection_px'] = float(errors[inlier].mean())
                th, tw = template.shape[:2]
                corners = np.float32([[0, 0], [tw-1, 0], [tw-1, th-1], [0, th-1]])
                projected_corners = cv2.perspectiveTransform(corners.reshape(1,-1,2), H)[0]
                area = float(abs(cv2.contourArea(projected_corners)))
                h, w = scene.shape[:2]
                if not np.isfinite(projected_corners).all() or not (20 <= area <= 4*w*h):
                    result['reason'] = 'invalid_geometry'
                else:
                    pred = cv2.warpPerspective(template_mask, H, (w, h),
                                               flags=cv2.INTER_NEAREST,
                                               borderMode=cv2.BORDER_CONSTANT,
                                               borderValue=0)
                    if np.count_nonzero(pred) == 0:
                        result['reason'] = 'empty_prediction'
                    else:
                        result['found'] = True
                        result['reason'] = 'ok'
    result['runtime_ms'] = round(1000*(time.perf_counter()-start), 2)
    return result, pred


def iou_score(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.count_nonzero((a > 0) & (b > 0))
    union = np.count_nonzero((a > 0) | (b > 0))
    return intersection / union if union else 0.0


def summarize(rows: list[dict], threshold: float):
    tp = fp = fn = tn = 0
    for row in rows:
        positive = row['pair_type'] == 'positive'
        valid_detection = row['found'] and (row['iou'] is not None and row['iou'] >= threshold)
        if positive:
            if valid_detection:
                tp += 1
            else:
                fn += 1
                if row['found']:
                    fp += 1  # ложная локализация также является FP при объектном сопоставлении
        else:
            if row['found']:
                fp += 1
            else:
                tn += 1
    precision = tp/(tp+fp) if (tp+fp) else 0.0
    recall = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    positive_rows = [r for r in rows if r['pair_type']=='positive']
    return {'samples':len(rows),'positives':len(positive_rows),'negatives':len(rows)-len(positive_rows),
            'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':round(precision,4),
            'recall':round(recall,4),'f1':round(f1,4),
            'positive_success_rate':round(tp/len(positive_rows),4) if positive_rows else None,
            'mean_positive_iou_including_failures':round(float(np.mean([r['iou'] or 0.0 for r in positive_rows])),4) if positive_rows else None,
            'mean_runtime_ms':round(float(np.mean([r['runtime_ms'] for r in rows])),2) if rows else None,
            'note':'FP на положительном примере с неверной локализацией и FN учитываются одновременно; синтетические виды не независимы.'}


def save_plots(rows, out, threshold):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib не установлен: графики пропущены')
        return
    groups = defaultdict(list)
    for row in rows:
        if row['pair_type'] == 'positive':
            groups[row['scenario']].append(row)
    if not groups:
        return
    scenarios = list(groups)
    mean_iou = [np.mean([r['iou'] or 0.0 for r in groups[s]]) for s in scenarios]
    success = [np.mean([bool(r['found']) and (r['iou'] or 0) >= threshold for r in groups[s]]) for s in scenarios]
    for values, filename, ylabel in ((mean_iou,'mean_iou_by_scenario.png','Mean IoU (failures = 0)'),
                                    (success,'success_rate_by_scenario.png',f'Detection success rate (IoU >= {threshold})')):
        fig, ax = plt.subplots(figsize=(max(8, len(scenarios)*1.1), 4.6))
        ax.bar(scenarios, values)
        ax.set_ylim(0,1.05)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=35)
        fig.tight_layout()
        fig.savefig(out/filename, dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Пакетная оценка SIFT + BF + RANSAC на Pothole-600')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--split', choices=['training','validation','testing'], default='validation')
    parser.add_argument('--limit', type=int, default=10, help='0 = все пары RGB/label')
    parser.add_argument('--negatives', type=int, default=1, help='Число других изображений для каждого эталона')
    parser.add_argument('--scenarios', nargs='+', default=['identity','rotate_20','rotate_60','scale_0.75','scale_1.25','perspective','darker','blur'])
    parser.add_argument('--padding', type=int, default=45)
    parser.add_argument('--ratio', type=float, default=0.75)
    parser.add_argument('--ransac-threshold', type=float, default=3.0)
    parser.add_argument('--min-inliers', type=int, default=10)
    parser.add_argument('--min-inlier-ratio', type=float, default=0.35)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    if not (0 < args.ratio < 1 and 0 < args.iou_threshold <= 1 and 0 <= args.min_inlier_ratio <= 1):
        parser.error('Проверьте значения ratio, iou-threshold и min-inlier-ratio')
    if args.limit < 0 or args.negatives < 0 or args.padding < 0 or args.min_inliers < 4 or args.ransac_threshold <= 0:
        parser.error('Неверные параметры limit/negatives/padding/RANSAC')
    pairs = load_pairs(Path(args.data_root), args.split)
    rng = np.random.default_rng(args.seed)
    selected = pairs[:args.limit] if args.limit else pairs
    out = Path(args.out) if args.out else Path('results')/'step03'/args.split
    out.mkdir(parents=True, exist_ok=True)
    sift = cv2.SIFT_create(nfeatures=2000)
    rows = []
    failures = []
    print(f'Раздел: {args.split}; эталонов: {len(selected)}; сценариев: {len(args.scenarios)}; отрицательных на эталон: {args.negatives}', flush=True)
    for counter, (stem, rgb_path, mask_path) in enumerate(selected, 1):
        try:
            image, mask = read_data(rgb_path, mask_path)
            template, template_mask, bbox = make_template(image, mask, args.padding)
            for scenario in args.scenarios:
                scene, truth = synthetic_scene(image, mask, scenario)
                result, pred = run_detector(template, template_mask, scene, ratio=args.ratio,
                                            reproj_threshold=args.ransac_threshold,
                                            min_inliers=args.min_inliers, min_inlier_ratio=args.min_inlier_ratio, sift=sift)
                overlap = iou_score(pred, truth) if result['found'] and truth.any() else 0.0
                # При исчезновении ямы с кадра случай помечается отдельно, но не учитывается как положительный.
                if not truth.any():
                    failures.append({'image':stem,'reason':f'{scenario}: объект вне кадра'})
                    continue
                rows.append({'image_id':stem,'scene_id':stem,'pair_type':'positive',
                             'scenario':scenario,'iou':round(overlap,5),**result})
            if args.negatives:
                other_indices = [j for j in range(len(pairs)) if pairs[j][0] != stem]
                if not other_indices:
                    failures.append({'image':stem,'reason':'Нет другого изображения для отрицательной пары'})
                else:
                    for j in rng.choice(other_indices, size=min(args.negatives,len(other_indices)), replace=False):
                        other_stem, other_rgb, _ = pairs[int(j)]
                        scene = cv2.imread(str(other_rgb), cv2.IMREAD_COLOR)
                        if scene is None:
                            failures.append({'image':other_stem,'reason':'Нельзя прочитать отрицательную сцену'})
                            continue
                        result, _ = run_detector(template, template_mask, scene,ratio=args.ratio,
                                                 reproj_threshold=args.ransac_threshold,
                                                 min_inliers=args.min_inliers,
                                                 min_inlier_ratio=args.min_inlier_ratio,sift=sift)
                        rows.append({'image_id':stem,'scene_id':other_stem,'pair_type':'negative',
                                     'scenario':'other_pothole','iou':None,**result})
        except (ValueError, cv2.error, OSError) as exc:
            failures.append({'image':stem,'reason':str(exc)})
        if counter % 10 == 0 or counter == len(selected):
            print(f'Обработано эталонов: {counter}/{len(selected)}; тестов: {len(rows)}', flush=True)
    if not rows:
        raise RuntimeError('Нет обработанных случаев; проверьте маски и имена файлов')
    columns = ['image_id','scene_id','pair_type','scenario','found','reason','iou',
               'kp_template','kp_scene','good_matches','inliers','inlier_ratio',
               'reprojection_px','runtime_ms']
    with (out/'results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with (out/'skipped.json').open('w',encoding='utf-8') as f:
        json.dump(failures,f,ensure_ascii=False,indent=2)
    summary = summarize(rows,args.iou_threshold)
    summary.update({'split':args.split,'dataset_records':len(pairs),'templates_processed':len(selected),
                    'settings':vars(args),'skipped':len(failures),
                    'caution':'Тесты на поворот и перспективу синтетические. Отрицательные пары могут содержать повторные снимки одной ямы, если источник не размечает идентичность физических дефектов.'})
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    save_plots(rows,out,args.iou_threshold)
    print(json.dumps({k:v for k,v in summary.items() if k!='settings'},ensure_ascii=False,indent=2))
    print(f'Результаты: {out.resolve()}')


if __name__ == '__main__':
    main()
