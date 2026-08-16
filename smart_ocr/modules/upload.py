"""Module 1 - Image Upload.

Validates an incoming image, stores it safely on disk and returns its metadata.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class UploadError(Exception):
    """Raised when an uploaded file cannot be accepted."""


@dataclass
class UploadedImage:
    original_name: str
    stored_name: str
    path: Path
    width: int
    height: int
    size_bytes: int


def _extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def is_allowed(filename: str, allowed: set[str]) -> bool:
    return _extension(filename) in allowed


def save_upload(
    data: bytes,
    filename: str,
    upload_dir: Path,
    allowed: set[str],
    max_bytes: int,
) -> UploadedImage:
    """Validate raw bytes and persist them as an image file."""
    if not filename:
        raise UploadError("No file selected.")
    if not is_allowed(filename, allowed):
        raise UploadError(
            f"Unsupported file type '.{_extension(filename)}'. "
            f"Allowed: {', '.join(sorted(allowed))}."
        )
    if not data:
        raise UploadError("Uploaded file is empty.")
    if len(data) > max_bytes:
        raise UploadError(f"File too large ({len(data) / 1e6:.1f} MB). Limit is {max_bytes / 1e6:.0f} MB.")

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise UploadError("File is not a readable image.")

    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}.{_extension(filename)}"
    path = upload_dir / stored_name
    path.write_bytes(data)

    height, width = image.shape[:2]
    return UploadedImage(
        original_name=Path(filename).name,
        stored_name=stored_name,
        path=path,
        width=width,
        height=height,
        size_bytes=len(data),
    )


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise UploadError(f"Could not read image at {path}")
    return image
