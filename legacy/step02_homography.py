"""Шаг 2: Pothole-600, SIFT + Lowe + RANSAC, перенос маски и IoU.

Запуск из каталога проекта:
python step02_homography.py --split testing --index 0050
"""
from pathlib import Path
import argparse
import json
import time

import cv2
import numpy as np

EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')


def find_pair(root: Path, split: str, index: str):
    rgb_dir, label_dir = root / split / 'rgb', root / split / 'label'
    for directory in (rgb_dir, label_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f'Не найдена папка: {directory}')

    def find(stem, folder):
        return next((folder / (stem + ext) for ext in EXTENSIONS if (folder / (stem + ext)).is_file()), None)

    rgb, mask = find(index, rgb_dir), find(index, label_dir)
    if rgb is None or mask is None:
        raise FileNotFoundError(f'Не найдена пара rgb/label для {index} в {root / split}')
    return rgb, mask


def project_polygon(H, points):
    transformed = cv2.perspectiveTransform(np.float32(points).reshape(1, -1, 2), H)[0]
    return transformed


def main():
    parser = argparse.ArgumentParser(description='SIFT + RANSAC: поиск той же ямы и IoU')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--split', default='testing', choices=['training', 'validation', 'testing'])
    parser.add_argument('--index', default='0050')
    parser.add_argument('--padding', type=int, default=45)
    parser.add_argument('--ratio', type=float, default=0.75)
    parser.add_argument('--ransac-threshold', type=float, default=3.0)
    parser.add_argument('--min-inliers', type=int, default=10)
    parser.add_argument('--min-inlier-ratio', type=float, default=0.35)
    parser.add_argument('--out', default=None, help='Папка результатов; по умолчанию results/step02/<split>_<index>')
    args = parser.parse_args()

    if not (0 < args.ratio < 1):
        parser.error('--ratio должен быть между 0 и 1')
    if args.padding < 0 or args.ransac_threshold <= 0 or args.min_inliers < 4:
        parser.error('Неверные параметры padding / ransac-threshold / min-inliers')
    if not (0 <= args.min_inlier_ratio <= 1):
        parser.error('--min-inlier-ratio должен быть в диапазоне [0, 1]')

    rgb_file, label_file = find_pair(Path(args.data_root), args.split, args.index)
    image = cv2.imread(str(rgb_file), cv2.IMREAD_COLOR)
    label = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
    if image is None or label is None:
        raise ValueError('Не удалось прочитать исходные изображения')
    h, w = image.shape[:2]
    if label.shape != (h, w):
        raise ValueError('Маска и изображение имеют разные размеры')
    full_mask = np.uint8(label > 127) * 255
    ys, xs = np.where(full_mask > 0)
    if len(xs) == 0:
        raise ValueError('Маска ямы пуста')
    x1, y1 = max(0, int(xs.min()) - args.padding), max(0, int(ys.min()) - args.padding)
    x2, y2 = min(w, int(xs.max()) + args.padding + 1), min(h, int(ys.max()) + args.padding + 1)
    template = image[y1:y2, x1:x2].copy()
    template_mask = full_mask[y1:y2, x1:x2].copy()
    th, tw = template.shape[:2]

    # Тот же контролируемый синтетический перспективный сценарий, что и в шаге 1.
    source_corners = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    destination_corners = np.float32([[18, 10], [w-24, 21], [w-13, h-18], [13, h-8]])
    known_H = cv2.getPerspectiveTransform(source_corners, destination_corners)
    scene = cv2.warpPerspective(image, known_H, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(110, 110, 110))
    # Маску преобразуем методом ближайшего соседа, чтобы сохранить бинарные значения.
    truth_mask = cv2.warpPerspective(full_mask, known_H, (w, h), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    translation_to_full = np.array([[1, 0, x1], [0, 1, y1], [0, 0, 1]], dtype=np.float64)
    true_local_H = known_H @ translation_to_full

    start = time.perf_counter()
    sift = cv2.SIFT_create(nfeatures=2000)
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY), None)
    good = []
    if des1 is not None and des2 is not None and len(des2) >= 2:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        pairs = matcher.knnMatch(des1, des2, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < args.ratio * pair[1].distance]

    H = None
    inlier_flags = np.zeros(len(good), dtype=bool)
    reason = 'Недостаточно соответствий для расчёта гомографии'
    mean_reprojection_error = None
    if len(good) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, args.ransac_threshold)
        if inliers is not None:
            inlier_flags = inliers.ravel().astype(bool)
        if H is None or not np.all(np.isfinite(H)):
            H = None
            reason = 'Гомография не найдена'
        elif int(inlier_flags.sum()) < args.min_inliers:
            reason = 'Недостаточно инлайеров'
        elif float(inlier_flags.mean()) < args.min_inlier_ratio:
            reason = 'Низкая доля инлайеров'
        else:
            projected = cv2.perspectiveTransform(src, H).reshape(-1, 2)
            measured = dst.reshape(-1, 2)
            mean_reprojection_error = float(np.linalg.norm(projected[inlier_flags] - measured[inlier_flags], axis=1).mean())
            reason = 'ok'

    n_inliers = int(inlier_flags.sum())
    fraction = float(inlier_flags.mean()) if len(good) else 0.0
    predicted_mask = np.zeros((h, w), np.uint8)
    found = False
    iou = None
    projected_corners = None
    template_corners = np.float32([[0, 0], [tw-1, 0], [tw-1, th-1], [0, th-1]])
    true_corners = project_polygon(true_local_H, template_corners)
    if reason == 'ok':
        projected_corners = project_polygon(H, template_corners)
        if not np.all(np.isfinite(projected_corners)) or abs(cv2.contourArea(projected_corners)) < 20:
            reason = 'Вырожденная или слишком малая проекция эталона'
        else:
            found = True
            predicted_mask = cv2.warpPerspective(template_mask, H, (w, h), flags=cv2.INTER_NEAREST,
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            intersection = np.count_nonzero((truth_mask > 0) & (predicted_mask > 0))
            union = np.count_nonzero((truth_mask > 0) | (predicted_mask > 0))
            iou = float(intersection / union) if union else None
    elapsed_ms = (time.perf_counter() - start) * 1000

    out = Path(args.out) if args.out else Path('results') / 'step02' / f'{args.split}_{args.index}'
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / '01_template.png'), template)
    cv2.imwrite(str(out / '02_scene.png'), scene)
    cv2.imwrite(str(out / '03_true_mask.png'), truth_mask)
    cv2.imwrite(str(out / '04_predicted_mask.png'), predicted_mask)
    raw_lines = cv2.drawMatches(template, kp1, scene, kp2, good, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(str(out / '05_matches_before_ransac.png'), raw_lines)
    verified = [m for m, flag in zip(good, inlier_flags) if flag]
    inlier_lines = cv2.drawMatches(template, kp1, scene, kp2, verified, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(str(out / '06_inlier_matches.png'), inlier_lines)

    overlay = scene.copy()
    cv2.polylines(overlay, [np.rint(true_corners).astype(np.int32)], True, (0, 255, 0), 2)
    if found:
        cv2.polylines(overlay, [np.rint(projected_corners).astype(np.int32)], True, (255, 0, 0), 2)
    cv2.putText(overlay, 'GREEN: truth, BLUE: prediction', (6, max(17, h-12)),
                cv2.FONT_HERSHEY_SIMPLEX, .44, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, 'GREEN: truth, BLUE: prediction', (6, max(17, h-12)),
                cv2.FONT_HERSHEY_SIMPLEX, .44, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / '07_contours.png'), overlay)
    colored = scene.copy()
    colored[truth_mask > 0] = np.uint8(0.55 * colored[truth_mask > 0] + 0.45 * np.array([0,255,0]))
    colored[predicted_mask > 0] = np.uint8(0.55 * colored[predicted_mask > 0] + 0.45 * np.array([255,0,0]))
    cv2.imwrite(str(out / '08_mask_overlay.png'), colored)

    metrics = {
        'rgb_path': str(rgb_file), 'label_path': str(label_file),
        'template_bbox_xyxy': [x1,y1,x2,y2],
        'template_keypoints': len(kp1), 'scene_keypoints': len(kp2),
        'matches_after_ratio': len(good), 'ransac_inliers': n_inliers,
        'inlier_ratio': round(fraction, 4),
        'mean_inlier_reprojection_error_px': round(mean_reprojection_error, 4) if mean_reprojection_error is not None else None,
        'found': found, 'reason': reason, 'iou': round(iou, 4) if iou is not None else None,
        'runtime_ms': round(elapsed_ms, 2),
        'ratio_threshold': args.ratio, 'ransac_threshold_px': args.ransac_threshold,
        'estimated_homography': H.tolist() if H is not None else None,
        'note': 'IoU вычисляется по синтетически преобразованной маске. Это не тест реального изменения ракурса.'
    }
    (out / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in metrics.items() if k != 'estimated_homography'}, ensure_ascii=False, indent=2))
    print(f'\nИзображения и метрики сохранены в {out.resolve()}')


if __name__ == '__main__':
    main()
