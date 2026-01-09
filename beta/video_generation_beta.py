"""Прототип text-to-video пайплайна для беты.

- Принимает текстовый промт из CLI и генерирует набор кадров-плейсхолдеров.
- При наличии ffmpeg собирает их в mp4.
- Используется как испытательный стенд перед подключением реальной модели (например, Wan 2.2).
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LOG_FORMAT = "[%(levelname)s] %(asctime)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("beta.video_generation")


def ensure_output_dir(base_output: Path) -> Path:
    base_output.mkdir(parents=True, exist_ok=True)
    run_dir = base_output / f"run_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Создана папка для вывода: %s", run_dir)
    return run_dir


def _color_cycle(idx: int, total: int) -> tuple[int, int, int]:
    angle = 2 * math.pi * (idx / max(total, 1))
    r = int((math.sin(angle) * 0.5 + 0.5) * 255)
    g = int((math.sin(angle + 2) * 0.5 + 0.5) * 255)
    b = int((math.sin(angle + 4) * 0.5 + 0.5) * 255)
    return r, g, b


def generate_placeholder_frames(prompt: str, run_dir: Path, num_frames: int, seed: int, size=(512, 512)) -> List[Path]:
    rng = random.Random(seed)
    font = ImageFont.load_default()
    frames: List[Path] = []
    logger.info("Генерирую %s кадров-плейсхолдеров (%sx%s)...", num_frames, size[0], size[1])
    start = time.time()
    for i in range(num_frames):
        bg_color = _color_cycle(i, num_frames)
        noise = np.uint8(rng.random() * 30)
        base = np.full((size[1], size[0], 3), noise, dtype=np.uint8)
        img = Image.fromarray(base)
        overlay = Image.new("RGBA", size, bg_color + (90,))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

        draw = ImageDraw.Draw(img)
        text = f"{prompt[:60]}" if prompt else "(пустой промт)"
        draw.text((16, 16), f"Frame {i+1}/{num_frames}", fill=(255, 255, 255), font=font)
        draw.text((16, 40), text, fill=(255, 255, 255), font=font)

        frame_path = run_dir / f"frame_{i:03d}.png"
        img.save(frame_path, format="PNG", optimize=True)
        frames.append(frame_path)
    logger.info("Кадры готовы за %.2f c", time.time() - start)
    return frames


def build_video_ffmpeg(frames: List[Path], fps: int, output_mp4: Path) -> bool:
    if not frames:
        logger.error("Нет кадров для сборки видео")
        return False
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg не найден в PATH. Сохранены только кадры PNG: %s", frames[0].parent)
        return False

    tmp_pattern = frames[0].parent / "frame_%03d.png"
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(tmp_pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_mp4),
    ]
    logger.info("Собираю видео через ffmpeg: %s", output_mp4)
    start = time.time()
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Видео собрано за %.2f c", time.time() - start)
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("ffmpeg завершился с ошибкой: %s", exc.stderr.decode(errors="ignore") if exc.stderr else exc)
        return False


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Beta: text-to-video прототип")
    parser.add_argument("--prompt", required=True, help="Текстовый промт")
    parser.add_argument("--frames", type=int, default=24, help="Количество кадров (по умолчанию 24)")
    parser.add_argument("--fps", type=int, default=8, help="Кадров в секунду (по умолчанию 8)")
    parser.add_argument("--output", default="beta/output", help="Папка для вывода (по умолчанию beta/output)")
    parser.add_argument("--seed", type=int, default=None, help="Сид для детерминированности")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    seed = args.seed if args.seed is not None else random.randint(0, 2_000_000_000)
    logger.info("Промт: %s", args.prompt)
    logger.info("Параметры: frames=%s, fps=%s, seed=%s", args.frames, args.fps, seed)

    base_output = Path(args.output)
    run_dir = ensure_output_dir(base_output)

    frames = generate_placeholder_frames(args.prompt, run_dir, args.frames, seed)
    output_mp4 = run_dir / "video.mp4"
    built = build_video_ffmpeg(frames, args.fps, output_mp4)

    if built:
        logger.info("Готово. Видео: %s", output_mp4)
    else:
        logger.info("Готово. Кадры лежат в: %s", run_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
