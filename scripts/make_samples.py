"""Generate synthetic real-world-style test images into samples/.

Usage: python scripts/make_samples.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def base(text: str, width: int = 1000, height: int = 300) -> np.ndarray:
    img = np.full((height, width, 3), 245, np.uint8)
    cv2.putText(img, text, (40, height // 2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 4)
    return img


def add_noise(img: np.ndarray, sigma: float = 22) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_shadow(img: np.ndarray, strength: int = 150) -> np.ndarray:
    gradient = np.tile(np.linspace(0, strength, img.shape[1]), (img.shape[0], 1))
    return np.clip(img.astype(np.float32) - gradient[:, :, None], 0, 255).astype(np.uint8)


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderValue=(245, 245, 245))


def main() -> None:
    SAMPLES.mkdir(exist_ok=True)
    variants = {
        "clean.png": base("INVOICE TOTAL 1250"),
        "noisy.png": add_noise(base("RECEIPT NO 4471")),
        "shadow.png": add_shadow(base("EXIT GATE 2")),
        "blurred.png": cv2.GaussianBlur(base("PLATFORM NO 5"), (7, 7), 0),
        "rotated.png": rotate(base("BUS STOP AIROLI"), 6),
        "hard.png": add_noise(add_shadow(rotate(base("SHOP NO 14 PUNE"), -4), 110), 15),
    }
    for name, image in variants.items():
        cv2.imwrite(str(SAMPLES / name), image)
        print("wrote", SAMPLES / name)


if __name__ == "__main__":
    main()
