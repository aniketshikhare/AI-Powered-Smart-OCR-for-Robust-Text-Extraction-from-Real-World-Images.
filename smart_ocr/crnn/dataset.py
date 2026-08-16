"""Synthetic text-line dataset.

Real-world degradations (blur, noise, shadow, rotation, textured background,
low contrast) are applied on the fly so the recogniser sees the same conditions
the preprocessing module is designed to survive.
"""
from __future__ import annotations

import random
import string
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

from .charset import CHARSET
from .model import IMAGE_HEIGHT, IMAGE_WIDTH

FONT_DIRS = [
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/truetype/liberation"),
    Path("/usr/share/fonts/truetype/freefont"),
]

WORDS = [
    "INVOICE", "TOTAL", "AMOUNT", "RECEIPT", "DATE", "GST", "BILL", "CASH",
    "SHOP", "STORE", "MARKET", "ROAD", "STREET", "PUNE", "MUMBAI", "AIROLI",
    "EXIT", "ENTRY", "GATE", "PLATFORM", "STATION", "BUS", "STOP", "PARKING",
    "OPEN", "CLOSED", "NOTICE", "WARNING", "DANGER", "OFFICE", "COLLEGE",
    "LIBRARY", "HOSPITAL", "PHARMACY", "HOTEL", "CAFE", "MENU", "PRICE",
    "QTY", "ITEM", "SUBTOTAL", "DISCOUNT", "THANK", "YOU", "VISIT", "AGAIN",
]


def available_fonts() -> list[Path]:
    fonts = [f for d in FONT_DIRS if d.exists() for f in sorted(d.glob("*.ttf"))]
    if not fonts:
        raise RuntimeError("No TrueType fonts found; install fonts-dejavu.")
    return fonts


def random_text(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.35:
        return rng.choice(WORDS)
    if kind < 0.55:
        return f"{rng.choice(WORDS)} {rng.randint(1, 9999)}"
    if kind < 0.70:
        return f"{rng.randint(1, 99999)}"
    if kind < 0.85:
        return " ".join(rng.choice(WORDS) for _ in range(rng.randint(2, 3)))
    length = rng.randint(3, 10)
    alphabet = string.ascii_uppercase + string.digits + ".-/"
    return "".join(rng.choice(alphabet) for _ in range(length))


def render_text(text: str, font_path: Path, rng: random.Random) -> np.ndarray:
    font_size = rng.randint(28, 46)
    font = ImageFont.truetype(str(font_path), font_size)
    padding = rng.randint(6, 18)
    dummy = Image.new("L", (10, 10))
    box = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
    width = box[2] - box[0] + 2 * padding
    height = box[3] - box[1] + 2 * padding

    background = rng.randint(150, 255)
    foreground = rng.randint(0, max(0, background - 70))
    image = Image.new("L", (max(width, 8), max(height, 8)), background)
    ImageDraw.Draw(image).text((padding - box[0], padding - box[1]), text, font=font, fill=foreground)
    return np.array(image)


def degrade(image: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = image.shape
    noise_rng = np.random.default_rng(rng.randrange(2**32))
    if rng.random() < 0.5:  # textured background
        texture = noise_rng.normal(0, rng.uniform(3, 12), (h, w))
        image = np.clip(image.astype(np.float32) + texture, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:  # shadow / light gradient
        direction = rng.choice([0, 1])
        gradient = np.linspace(0, rng.uniform(20, 90), w if direction else h)
        gradient = np.tile(gradient, (h, 1)) if direction else np.tile(gradient[:, None], (1, w))
        image = np.clip(image.astype(np.float32) - gradient, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:  # motion / defocus blur
        k = rng.choice([3, 5])
        image = cv2.GaussianBlur(image, (k, k), 0)
    if rng.random() < 0.35:  # rotation
        angle = rng.uniform(-4, 4)
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.3:  # resolution loss
        scale = rng.uniform(0.4, 0.8)
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        image = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return image


def fit_to_input(image: np.ndarray, height: int = IMAGE_HEIGHT, width: int = IMAGE_WIDTH) -> np.ndarray:
    """Scale to the model height, then pad or crop to the fixed width."""
    h, w = image.shape[:2]
    scale = height / max(h, 1)
    new_w = max(1, min(width, int(w * scale)))
    resized = cv2.resize(image, (new_w, height), interpolation=cv2.INTER_AREA)
    if new_w == width:
        return resized
    pad_value = int(np.median(resized[:, -1]))
    canvas = np.full((height, width), pad_value, np.uint8)
    canvas[:, :new_w] = resized
    return canvas


def to_tensor(image: np.ndarray) -> torch.Tensor:
    normalised = image.astype(np.float32) / 255.0
    return torch.from_numpy((normalised - 0.5) / 0.5).unsqueeze(0)


class SyntheticTextDataset(Dataset):
    """Deterministic per-index generation so workers stay reproducible."""

    def __init__(self, size: int = 40000, seed: int = 0, degrade_prob: float = 0.9) -> None:
        self.size = size
        self.seed = seed
        self.degrade_prob = degrade_prob
        self.fonts = available_fonts()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        rng = random.Random(self.seed * 1_000_003 + index)
        text = random_text(rng)
        image = render_text(text, rng.choice(self.fonts), rng)
        if rng.random() < self.degrade_prob:
            image = degrade(image, rng)
        return to_tensor(fit_to_input(image)), text


def collate(batch):
    images = torch.stack([b[0] for b in batch])
    texts = [b[1] for b in batch]
    return images, texts


def unsupported_characters(text: str) -> set[str]:
    return {c for c in text if c not in CHARSET}
