"""Шаг 1: выделение эталона по маске, синтетическая смена ракурса, SIFT-сопоставление.
Запуск: python step01_pothole600.py --split testing --index 0000
"""
from pathlib import Path
import argparse
import json
import time

import cv2
import numpy as np


def locate(root: Path, split: str, index: str):
    rgb_dir, label_dir = root / split / 'rgb', root / split / 'label'
    if not rgb_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f'Не найдены каталоги {rgb_dir} и {label_dir}. Укажите --data-root.')
    rgb = next((p for ext in ('.png', '.jpg', '.jpeg', '.bmp') if (p := rgb_dir / (index + ext)).exists()), None)
    label = next((p for ext in ('.png', '.jpg', '.jpeg', '.bmp') if (p := label_dir / (index + ext)).exists()), None)
    if rgb is None or label is None:
        raise FileNotFoundError(f'Не найдена пара {index} в {rgb_dir} и {label_dir}.')
    return rgb, label


def main():
    parser = argparse.ArgumentParser(description='Pothole-600: маска → эталон → SIFT → BF KNN')
    parser.add_argument('--data-root', default='data', help='Корневая папка с testing/training/validation')
    parser.add_argument('--split', default='testing', choices=['training', 'testing', 'validation'])
    parser.add_argument('--index', default='0000', help='Имя файла без расширения')
    parser.add_argument('--padding', type=int, default=45, help='Запас асфальта вокруг маски, пиксели')
    parser.add_argument('--ratio', type=float, default=0.75, help='Порог Lowe ratio test')
    parser.add_argument('--out', default='results/step01_p600')
    args = parser.parse_args()
    rgb_file, label_file = locate(Path(args.data_root), args.split, args.index)
    img = cv2.imread(str(rgb_file), cv2.IMREAD_COLOR)
    label = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
    if img is None or label is None:
        raise RuntimeError('Не удалось открыть изображение или маску.')
    h, w = img.shape[:2]
    if label.shape != (h, w):
        raise ValueError(f'Размеры изображения {w}x{h} и маски {label.shape[::-1]} не совпадают.')
    mask = np.uint8(label > 127)
    ys, xs = np.where(mask != 0)
    if len(xs) == 0:
        raise ValueError('Маска пуста: белые пиксели не найдены.')
    x1, y1 = max(0, int(xs.min()) - args.padding), max(0, int(ys.min()) - args.padding)
    x2, y2 = min(w, int(xs.max()) + args.padding + 1), min(h, int(ys.max()) + args.padding + 1)
    template = img[y1:y2, x1:x2].copy()

    # Контролируемое перспективное преобразование: точки углов известны заранее.
    corners_src = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    corners_dst = np.float32([[18, 10], [w-24, 21], [w-13, h-18], [13, h-8]])
    known_H = cv2.getPerspectiveTransform(corners_src, corners_dst)
    scene = cv2.warpPerspective(img, known_H, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(110, 110, 110))

    start = time.perf_counter()
    sift = cv2.SIFT_create(nfeatures=2000)
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY), None)
    knn = []
    good = []
    if des1 is not None and des2 is not None and len(des2) >= 2:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn = matcher.knnMatch(des1, des2, k=2)
        good = [pair[0] for pair in knn if len(pair) == 2 and pair[0].distance < args.ratio * pair[1].distance]
    elapsed_ms = (time.perf_counter() - start) * 1000

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / '01_template.png'), template)
    cv2.imwrite(str(out / '02_scene_perspective.png'), scene)
    cv2.imwrite(str(out / '03_template_keypoints.png'), cv2.drawKeypoints(template, kp1, None,
                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    cv2.imwrite(str(out / '04_scene_keypoints.png'), cv2.drawKeypoints(scene, kp2, None,
                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    matches_image = cv2.drawMatches(template, kp1, scene, kp2, good, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(str(out / '05_matches.png'), matches_image)

    # Справочный истинный контур эталона (не вычисленный по сопоставлениям).
    template_corners = np.float32([[x1, y1], [x2-1, y1], [x2-1, y2-1], [x1, y2-1]])
    true_corners = cv2.perspectiveTransform(template_corners[None, :, :], known_H)[0]
    overlay = scene.copy()
    cv2.polylines(overlay, [np.round(true_corners).astype(np.int32)], True, (0,255,0), 2)
    cv2.imwrite(str(out / '06_true_location_reference.png'), overlay)

    metrics = {
        'rgb_path': str(rgb_file), 'label_path': str(label_file),
        'template_bbox_xyxy': [x1,y1,x2,y2], 'image_size': [w,h],
        'template_keypoints': len(kp1), 'scene_keypoints': len(kp2),
        'knn_pairs': len(knn), 'matches_after_ratio': len(good),
        'ratio_threshold': args.ratio, 'runtime_ms': round(elapsed_ms,2),
        'note': 'Гомография пока не оценивается алгоритмом. Зелёный контур на 06 — известное истинное положение.'
    }
    (out / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f'\nРезультаты: {out.resolve()}')

if __name__ == '__main__':
    main()
