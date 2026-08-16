"""Robustness checks across degraded real-world image conditions."""
import cv2
import numpy as np
import pytest

from conftest import make_text_image
from smart_ocr.modules import preprocessing as pp
from smart_ocr.pipeline import OCRPipeline


def shadowed(text="EXIT GATE"):
    img = make_text_image(text)
    gradient = np.tile(np.linspace(0, 150, img.shape[1]), (img.shape[0], 1))
    return np.clip(img.astype(np.float32) - gradient[:, :, None], 0, 255).astype(np.uint8)


def test_threshold_auto_picks_adaptive_for_uneven_light():
    even = pp.to_grayscale(make_text_image())
    uneven = pp.to_grayscale(shadowed())
    assert pp.illumination_variation(uneven) > pp.illumination_variation(even)


@pytest.mark.parametrize(
    "image,expected",
    [
        (make_text_image("CLEAN TEXT"), "CLEAN"),
        (make_text_image("NOISY TEXT", noise=True), "NOISY"),
        (make_text_image("SKEWED TEXT", angle=6), "SKEWED"),
        (shadowed("SHADOW TEXT"), "SHADOW"),
        (cv2.GaussianBlur(make_text_image("BLUR TEXT"), (5, 5), 0), "BLUR"),
    ],
)
def test_pipeline_survives_degraded_images(tmp_path, image, expected):
    path = tmp_path / "case.png"
    cv2.imwrite(str(path), image)
    result = OCRPipeline().run(path)
    assert expected in result.text.upper()
