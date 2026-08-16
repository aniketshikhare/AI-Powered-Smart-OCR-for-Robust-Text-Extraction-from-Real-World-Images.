import numpy as np

from conftest import make_text_image
from smart_ocr.modules import preprocessing as pp


def test_resize_scales_into_band():
    small = np.zeros((100, 400, 3), np.uint8)
    assert pp.resize_for_ocr(small, 1000, 2400).shape[1] == 1000
    big = np.zeros((100, 4000, 3), np.uint8)
    assert pp.resize_for_ocr(big, 1000, 2400).shape[1] == 2400
    ok = np.zeros((100, 1500, 3), np.uint8)
    assert pp.resize_for_ocr(ok, 1000, 2400).shape[1] == 1500


def test_grayscale_and_binarize(text_image):
    gray = pp.to_grayscale(text_image)
    assert gray.ndim == 2
    binary = pp.binarize(gray)
    assert set(np.unique(binary)).issubset({0, 255})


def test_denoise_reduces_variance():
    noisy = pp.to_grayscale(make_text_image(noise=True))
    clean = pp.remove_noise(noisy)
    assert clean.std() <= noisy.std()


def test_contrast_enhancement_widens_range():
    flat = np.full((200, 200), 120, np.uint8)
    flat[50:150, 50:150] = 135
    assert np.ptp(pp.enhance_contrast(flat)) >= np.ptp(flat)


def test_deskew_reduces_angle():
    skewed = pp.to_grayscale(make_text_image(angle=7))
    assert abs(pp.estimate_skew(skewed)) > 1.0
    corrected, angle = pp.deskew(skewed)
    assert angle != 0.0
    assert abs(pp.estimate_skew(corrected)) < abs(pp.estimate_skew(skewed))


def test_preprocess_records_steps(text_image):
    result = pp.preprocess(text_image)
    assert result.binary is not None
    assert {"grayscale", "denoise", "threshold"}.issubset(set(result.steps))
    assert result.image.ndim == 2


def test_preprocess_can_disable_steps(text_image):
    result = pp.preprocess(text_image, pp.PreprocessOptions(threshold=False, denoise=False))
    assert result.binary is None
    assert "denoise" not in result.steps
