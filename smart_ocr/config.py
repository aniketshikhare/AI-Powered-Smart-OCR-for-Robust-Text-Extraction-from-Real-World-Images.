"""Application configuration for the AI-Powered Smart OCR system."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    SECRET_KEY = os.environ.get("SMART_OCR_SECRET_KEY", "smart-ocr-dev-key")

    UPLOAD_DIR = Path(os.environ.get("SMART_OCR_UPLOAD_DIR", BASE_DIR / "uploads"))
    DEBUG_DIR = UPLOAD_DIR / "debug"
    DATABASE_PATH = Path(os.environ.get("SMART_OCR_DB", BASE_DIR / "smart_ocr.db"))

    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

    # Preprocessing defaults
    TARGET_MIN_WIDTH = 1000
    TARGET_MAX_WIDTH = 2400

    # Recognition defaults
    OCR_ENGINE = os.environ.get("SMART_OCR_ENGINE", "tesseract")
    OCR_LANGUAGES = os.environ.get("SMART_OCR_LANGS", "eng")
    MIN_CONFIDENCE = float(os.environ.get("SMART_OCR_MIN_CONFIDENCE", "40"))

    @classmethod
    def ensure_dirs(cls) -> None:
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        cls.DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
