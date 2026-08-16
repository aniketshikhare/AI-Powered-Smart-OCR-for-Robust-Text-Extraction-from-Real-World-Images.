"""Module 3 - Text Detection.

Locates candidate text regions before recognition. Two detectors are provided:

* ``morphology``  - classic CV: gradient + morphological closing (default, no model files)
* ``mser``        - MSER character blobs grouped into lines

Both return axis-aligned boxes merged into text lines.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TextRegion:
    x: int
    y: int
    w: int
    h: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    def padded(self, pad: int, shape: tuple[int, int]) -> "TextRegion":
        max_h, max_w = shape[:2]
        x = max(0, self.x - pad)
        y = max(0, self.y - pad)
        w = min(max_w - x, self.w + 2 * pad)
        h = min(max_h - y, self.h + 2 * pad)
        return TextRegion(x, y, w, h)

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y : self.y + self.h, self.x : self.x + self.w]


def _to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _filter_boxes(boxes: list[TextRegion], shape: tuple[int, int]) -> list[TextRegion]:
    h, w = shape[:2]
    kept = []
    for b in boxes:
        if b.w < 8 or b.h < 8:
            continue
        if b.h > 0.9 * h and b.w > 0.9 * w:
            continue
        aspect = b.w / max(b.h, 1)
        if aspect < 0.1 or aspect > 60:
            continue
        kept.append(b)
    return kept


def merge_overlapping(boxes: list[TextRegion], iou_threshold: float = 0.1) -> list[TextRegion]:
    """Greedily merge boxes that overlap, so a line is recognised as one region."""
    remaining = sorted(boxes, key=lambda b: b.area, reverse=True)
    merged: list[TextRegion] = []
    while remaining:
        current = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            for other in list(remaining):
                if _overlap_ratio(current, other) > iou_threshold:
                    current = _union(current, other)
                    remaining.remove(other)
                    changed = True
        merged.append(current)
    return sorted(merged, key=lambda b: (b.y, b.x))


def _overlap_ratio(a: TextRegion, b: TextRegion) -> float:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    return inter / min(a.area, b.area)


def _union(a: TextRegion, b: TextRegion) -> TextRegion:
    x1, y1 = min(a.x, b.x), min(a.y, b.y)
    x2 = max(a.x + a.w, b.x + b.w)
    y2 = max(a.y + a.h, b.y + b.h)
    return TextRegion(x1, y1, x2 - x1, y2 - y1)


def detect_morphology(image: np.ndarray) -> list[TextRegion]:
    """Text has high local gradient energy; close it into blobs and take contours."""
    gray = _to_gray(image)
    grad = cv2.morphologyEx(
        gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel_w = max(9, image.shape[1] // 60)
    closed = cv2.morphologyEx(
        bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [TextRegion(*cv2.boundingRect(c)) for c in contours]
    return merge_overlapping(_filter_boxes(boxes, gray.shape))


def detect_mser(image: np.ndarray) -> list[TextRegion]:
    """MSER finds stable character blobs; group them into line-level boxes."""
    gray = _to_gray(image)
    mser = cv2.MSER_create()
    mser.setMinArea(30)
    mser.setMaxArea(int(0.2 * gray.shape[0] * gray.shape[1]))
    regions, _ = mser.detectRegions(gray)
    boxes = [TextRegion(*cv2.boundingRect(r.reshape(-1, 1, 2))) for r in regions]
    boxes = _filter_boxes(boxes, gray.shape)
    if not boxes:
        return []
    # Dilate horizontally so characters of one line overlap, then merge.
    grown = [
        TextRegion(max(0, b.x - b.h // 2), b.y, b.w + b.h, b.h) for b in boxes
    ]
    return merge_overlapping(grown, iou_threshold=0.05)


DETECTORS = {"morphology": detect_morphology, "mser": detect_mser}


def detect_text_regions(
    image: np.ndarray, method: str = "morphology", pad: int = 4
) -> list[TextRegion]:
    detector = DETECTORS.get(method, detect_morphology)
    regions = detector(image)
    return [r.padded(pad, image.shape) for r in regions]


def draw_regions(image: np.ndarray, regions: list[TextRegion]) -> np.ndarray:
    canvas = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    canvas = canvas.copy()
    for r in regions:
        cv2.rectangle(canvas, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 180, 255), 2)
    return canvas
