import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smart_ocr import create_app
from smart_ocr.config import Config


def make_text_image(text="HELLO WORLD 2026", width=900, height=220, noise=False, angle=0.0):
    img = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(img, text, (30, height // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4)
    if angle:
        m = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        img = cv2.warpAffine(img, m, (width, height), borderValue=(255, 255, 255))
    if noise:
        gauss = np.random.normal(0, 18, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + gauss, 0, 255).astype(np.uint8)
    return img


@pytest.fixture
def text_image():
    return make_text_image()


@pytest.fixture
def app(tmp_path):
    class TestConfig(Config):
        TESTING = True
        UPLOAD_DIR = tmp_path / "uploads"
        DEBUG_DIR = tmp_path / "uploads" / "debug"
        DATABASE_PATH = tmp_path / "test.db"

    return create_app(TestConfig)


@pytest.fixture
def client(app):
    return app.test_client()
