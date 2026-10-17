"""Извлечь два кадра одного видеоролика для шага 1.

Пример: python extract_frames.py --video data/pothole.mp4 --first 0 --second 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def save_frame(video: cv2.VideoCapture, index: int, destination: Path) -> None:
    video.set(cv2.CAP_PROP_POS_FRAMES, index)
    success, image = video.read()
    if not success or image is None:
        raise RuntimeError(f"Не удалось прочитать кадр {index}. Выберите другой номер.")
    success, data = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"Не удалось закодировать кадр {index}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.tofile(str(destination))
    print(f"Кадр {index} сохранён: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Извлечение двух кадров из видео")
    parser.add_argument("--video", required=True, help="Путь к mp4/avi/mov")
    parser.add_argument("--first", type=int, default=0, help="Первый кадр (с нуля)")
    parser.add_argument("--second", type=int, default=12, help="Второй кадр (с нуля)")
    parser.add_argument("--output", default="data", help="Папка для пары кадров")
    args = parser.parse_args()

    if args.first < 0 or args.second < 0 or args.first == args.second:
        parser.error("Номера кадров должны быть неотрицательными и различаться")
    if not Path(args.video).is_file():
        parser.error(f"Файл видео не найден: {args.video}")

    video = cv2.VideoCapture(args.video)
    if not video.isOpened():
        parser.error(f"Не удалось открыть видео: {args.video}")
    try:
        count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if count > 0 and max(args.first, args.second) >= count:
            parser.error(f"В видео {count} кадров; выберите кадр от 0 до {count-1}")
        destination = Path(args.output)
        save_frame(video, args.first, destination / "template.png")
        save_frame(video, args.second, destination / "scene.png")
    finally:
        video.release()


if __name__ == "__main__":
    main()
