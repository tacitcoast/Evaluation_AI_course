# homework_multimodal_images.py
"""Практика 2: подготовка третьей группы примеров — контролируемые искажения.

Группы 1 и 2 собраны из готовых снимков лекции. Группа 3 строится из
original.png известными преобразованиями нарастающей силы: только так видно,
монотонна ли метрика по силе искажения и на что она вообще реагирует.

Запуск:
    .venv/bin/python homework_multimodal_images.py
"""

import sys

from PIL import Image, ImageFilter, ImageOps
import numpy as np

import homework_config as cfg

DERIVED_DIR = cfg.IMAGES_DIR / "derived"

# Преобразования: имя -> (описание, функция)
TRANSFORMS = {
    "blur_light": ("слабое размытие, радиус 2",
                   lambda im: im.filter(ImageFilter.GaussianBlur(2))),
    "blur_heavy": ("сильное размытие, радиус 8",
                   lambda im: im.filter(ImageFilter.GaussianBlur(8))),
    "noise": ("гауссов шум, сигма 25", None),
    "grayscale": ("оттенки серого",
                  lambda im: ImageOps.grayscale(im).convert("RGB")),
    "mirror": ("отражение по горизонтали", ImageOps.mirror),
    "jpeg_low": ("повторное сжатие JPEG, качество 10", None),
}


def add_noise(image: Image.Image, sigma: float = 25.0) -> Image.Image:
    rng = np.random.default_rng(20261127)
    array = np.asarray(image).astype(np.float32)
    noisy = array + rng.normal(0.0, sigma, array.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))


def jpeg_roundtrip(image: Image.Image, quality: int = 10) -> Image.Image:
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def main() -> int:
    source = cfg.IMAGES_DIR / "original.png"
    if not source.exists():
        raise SystemExit(f"Нет исходного изображения: {source}")

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    base = Image.open(source).convert("RGB")
    print(f"Исходное: {source.name}, {base.size[0]}×{base.size[1]}")

    for name, (description, func) in TRANSFORMS.items():
        if name == "noise":
            result = add_noise(base)
        elif name == "jpeg_low":
            result = jpeg_roundtrip(base)
        else:
            result = func(base)
        out = DERIVED_DIR / f"{name}.png"
        result.save(out)
        print(f"  {name:<12} {description:<38} -> {out.name}")

    print(f"\nГотово: {len(TRANSFORMS)} изображений в {DERIVED_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
