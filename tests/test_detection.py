from conftest import make_text_image
from smart_ocr.modules import detection
from smart_ocr.modules.detection import TextRegion


def test_merge_overlapping_combines_boxes():
    boxes = [TextRegion(0, 0, 100, 20), TextRegion(50, 0, 100, 20), TextRegion(400, 200, 60, 20)]
    merged = detection.merge_overlapping(boxes)
    assert len(merged) == 2
    assert merged[0].w == 150


def test_padded_stays_inside_image():
    region = TextRegion(2, 2, 10, 10).padded(10, (30, 30))
    assert region.x == 0 and region.y == 0
    assert region.x + region.w <= 30


def test_morphology_detects_text_line(text_image):
    regions = detection.detect_text_regions(text_image, method="morphology")
    assert regions
    assert max(r.area for r in regions) > 1000


def test_mser_detects_text():
    regions = detection.detect_text_regions(make_text_image("SIGNBOARD"), method="mser")
    assert regions


def test_blank_image_yields_no_regions():
    import numpy as np

    blank = np.full((300, 300, 3), 255, np.uint8)
    assert detection.detect_text_regions(blank) == []
