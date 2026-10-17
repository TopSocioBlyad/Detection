"""Шаг 1 курсовой: выделение SIFT-признаков и сопоставление двух изображений.

Пример:
    python step01_sift.py --select-roi
    python step01_sift.py --roi 120 100 240 190
    python step01_sift.py --template data/template.png --scene data/scene.png

Это исследовательская визуализация. Количество хороших совпадений само
по себе НЕ является подтверждением обнаружения той же самой ямы.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_image(path: Path) -> np.ndarray:
    """Чтение RGB/BGR-изображения, включая пути с кириллицей на Windows."""
    if not path.is_file():
        raise FileNotFoundError(f"Файл изображения не найден: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")
    return image


def save_png(path: Path, image: np.ndarray) -> None:
    """Сохранение PNG, включая пути с кириллицей на Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"Не удалось закодировать изображение: {path}")
    encoded.tofile(str(path))


def get_template_roi(
    image: np.ndarray,
    roi: tuple[int, int, int, int] | None,
    select_roi: bool,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Вырезать область (x, y, w, h) или интерактивно выбрать её мышью."""
    height, width = image.shape[:2]
    if roi is not None and select_roi:
        raise ValueError("Используйте либо --roi, либо --select-roi, не оба одновременно.")

    if select_roi:
        window_name = "Select pothole AND surrounding asphalt; ENTER to confirm"
        print("Выделите мышью яму вместе с окружающим асфальтом; нажмите Enter.")
        try:
            rectangle = cv2.selectROI(
                window_name, image, showCrosshair=True, fromCenter=False
            )
            cv2.destroyWindow(window_name)
        except cv2.error as exc:
            raise RuntimeError(
                "Не удалось открыть окно выбора. Запустите проект на компьютере с "
                "графическим интерфейсом или задайте координаты через --roi x y w h."
            ) from exc
        roi = tuple(int(value) for value in rectangle)

    if roi is None:
        roi = (0, 0, width, height)
        print("Область не задана: в качестве эталона используется весь первый кадр.")

    x, y, w, h = roi
    if w <= 0 or h <= 0:
        raise ValueError("Размер эталонной области должен быть положительным.")
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError(
            f"Область {roi} выходит за пределы изображения {width}x{height}."
        )
    return image[y : y + h, x : x + w].copy(), roi


def run(args: argparse.Namespace) -> None:
    template_full = load_image(Path(args.template))
    scene = load_image(Path(args.scene))
    template, roi = get_template_roi(template_full, args.roi, args.select_roi)

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=args.nfeatures)
    keypoints_template, descriptors_template = sift.detectAndCompute(
        template_gray, None
    )
    keypoints_scene, descriptors_scene = sift.detectAndCompute(scene_gray, None)

    if descriptors_template is None or descriptors_scene is None:
        raise RuntimeError(
            "SIFT не нашёл дескрипторы хотя бы в одном изображении. "
            "Выберите более текстурированную область, включающую асфальт вокруг ямы."
        )
    if len(descriptors_scene) < 2:
        raise RuntimeError("На втором изображении недостаточно признаков для KNN (k=2).")

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    candidate_pairs = matcher.knnMatch(
        descriptors_template, descriptors_scene, k=2
    )

    # Lowe's ratio test: ближайший сосед должен быть заметно ближе второго.
    good_matches = []
    valid_pairs = 0
    for pair in candidate_pairs:
        if len(pair) < 2:
            continue
        valid_pairs += 1
        nearest, second_nearest = pair
        if nearest.distance < args.ratio * second_nearest.distance:
            good_matches.append(nearest)

    good_matches.sort(key=lambda match: match.distance)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_png(output / "01_template_crop.png", template)
    save_png(
        output / "02_template_keypoints.png",
        cv2.drawKeypoints(
            template, keypoints_template, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        ),
    )
    save_png(
        output / "03_scene_keypoints.png",
        cv2.drawKeypoints(
            scene, keypoints_scene, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        ),
    )
    visualization = cv2.drawMatches(
        template, keypoints_template,
        scene, keypoints_scene,
        good_matches[: args.max_draw], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    save_png(output / "04_good_matches.png", visualization)

    summary = {
        "template_image": str(Path(args.template)),
        "scene_image": str(Path(args.scene)),
        "template_roi_xywh": list(roi),
        "detector": "SIFT",
        "matcher": "BFMatcher(NORM_L2), knnMatch(k=2)",
        "lowe_ratio": args.ratio,
        "n_keypoints_template": len(keypoints_template),
        "n_keypoints_scene": len(keypoints_scene),
        "n_knn_pairs": valid_pairs,
        "n_good_matches": len(good_matches),
        "n_drawn_matches": min(args.max_draw, len(good_matches)),
        "warning": "Совпадения еще не проверены геометрически. Детекция не подтверждена.",
    }
    (output / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== РЕЗУЛЬТАТ ПЕРВОГО ШАГА ===")
    print(f"Эталонная область (x, y, w, h): {roi}")
    print(f"Ключевых точек на эталоне: {len(keypoints_template)}")
    print(f"Ключевых точек на сцене:   {len(keypoints_scene)}")
    print(f"KNN-пар для проверки:     {valid_pairs}")
    print(f"После Lowe's ratio test:  {len(good_matches)}")
    print(f"Изображения и метрики:    {output.resolve()}")
    print("Важно: обнаружение конкретной ямы на этом шаге не подтверждается.")
    if not good_matches:
        print("Нет хороших совпадений. Попробуйте выбрать более широкий/текстурный эталон.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SIFT: сопоставление эталонного фрагмента дороги со вторым кадром."
    )
    parser.add_argument("--template", default="data/template.png", help="Первое изображение")
    parser.add_argument("--scene", default="data/scene.png", help="Второе изображение")
    parser.add_argument("--roi", nargs=4, metavar=("X", "Y", "W", "H"),
                        type=int, help="Область ямы на первом изображении")
    parser.add_argument("--select-roi", action="store_true",
                        help="Выбрать область ямы мышью (нужен графический интерфейс)")
    parser.add_argument("--ratio", type=float, default=0.75,
                        help="Порог Lowe's ratio test (по умолчанию 0.75)")
    parser.add_argument("--nfeatures", type=int, default=2000,
                        help="Целевое максимальное число SIFT-признаков")
    parser.add_argument("--max-draw", type=int, default=60,
                        help="Максимальное число рисуемых соответствий")
    parser.add_argument("--output", default="results/step01", help="Папка результатов")
    args = parser.parse_args()
    if not 0 < args.ratio < 1:
        parser.error("--ratio должен быть между 0 и 1")
    if args.nfeatures < 1 or args.max_draw < 1:
        parser.error("--nfeatures и --max-draw должны быть положительными")
    if args.roi is not None:
        args.roi = tuple(args.roi)
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"Ошибка: {error}") from error
