"""Module 2 - Image Preprocessing.

Each step is a small pure function so it can be tested, tuned or disabled
independently. `preprocess` chains them into the pipeline described in the
synopsis: resize -> grayscale -> denoise -> contrast -> sharpen -> deskew ->
threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class PreprocessOptions:
    resize: bool = True
    grayscale: bool = True
    denoise: bool = True
    enhance_contrast: bool = True
    sharpen: bool = True
    deskew: bool = True
    threshold: bool = True
    perspective_correct: bool = False
    min_width: int = 1000
    max_width: int = 2400
    threshold_method: str = "auto"  # auto | otsu | adaptive


@dataclass
class PreprocessResult:
    image: np.ndarray            # image handed to detection/recognition
    gray: np.ndarray             # cleaned grayscale image
    binary: np.ndarray | None    # thresholded image (None if disabled)
    skew_angle: float = 0.0
    steps: list[str] = field(default_factory=list)


def resize_for_ocr(image: np.ndarray, min_width: int = 1000, max_width: int = 2400) -> np.ndarray:
    """Scale the image into a width band where OCR engines perform best."""
    h, w = image.shape[:2]
    if w == 0:
        return image
    if w < min_width:
        scale = min_width / w
    elif w > max_width:
        scale = max_width / w
    else:
        return image
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=interp)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(gray: np.ndarray) -> np.ndarray:
    """Edge preserving denoise: bilateral filter + light median blur."""
    filtered = cv2.bilateralFilter(gray, d=7, sigmaColor=55, sigmaSpace=55)
    return cv2.medianBlur(filtered, 3)


def enhance_contrast(gray: np.ndarray, clip_limit: float = 2.5, tile: int = 8) -> np.ndarray:
    """CLAHE handles uneven / poor lighting better than global equalisation."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def sharpen(gray: np.ndarray, amount: float = 0.8) -> np.ndarray:
    """Unsharp masking to recover strokes lost to blur.

    A light median blur follows because sharpening also amplifies sensor noise,
    which otherwise survives thresholding as speckle.
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(gray, 1 + amount, blurred, -amount, 0)
    return cv2.medianBlur(sharpened, 3)


def estimate_skew(gray: np.ndarray) -> float:
    """Estimate page skew in degrees using the minimum-area box of dark pixels."""
    inverted = cv2.bitwise_not(gray)
    _, mask = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coords = cv2.findNonZero(mask)
    if coords is None or len(coords) < 50:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    return float(angle)


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.1:
        return image
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    border = cv2.BORDER_REPLICATE
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=border)


def deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    angle = estimate_skew(gray)
    if abs(angle) > 15:  # implausible for text lines; ignore
        return gray, 0.0
    return rotate(gray, angle), angle


def illumination_variation(gray: np.ndarray, grid: int = 8) -> float:
    """Spread of block-wise median brightness: high means uneven lighting."""
    h, w = gray.shape[:2]
    bh, bw = max(1, h // grid), max(1, w // grid)
    medians = [
        float(np.median(gray[y : y + bh, x : x + bw]))
        for y in range(0, h - bh + 1, bh)
        for x in range(0, w - bw + 1, bw)
    ]
    return float(np.std(medians)) if medians else 0.0


def binarize(
    gray: np.ndarray, method: str = "auto", block_size: int = 51, c: int = 15
) -> np.ndarray:
    """Binarise with the threshold that suits the image.

    Otsu keeps thin strokes intact on evenly lit images, while adaptive
    thresholding is the only one that survives shadows and light gradients, so
    ``auto`` picks between them from the illumination spread.
    """
    if method == "auto":
        method = "adaptive" if illumination_variation(gray) > 12 else "otsu"
    if method == "otsu":
        _, binary = cv2.threshold(
            cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
    else:
        block_size = max(3, block_size | 1)
        binary = cv2.adaptiveThreshold(
            cv2.GaussianBlur(gray, (5, 5), 0),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c,
        )
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def order_corners(points: np.ndarray) -> np.ndarray:
    pts = points.reshape(4, 2).astype("float32")
    ordered = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    ordered[0] = pts[np.argmin(s)]       # top-left
    ordered[2] = pts[np.argmax(s)]       # bottom-right
    ordered[1] = pts[np.argmin(diff)]    # top-right
    ordered[3] = pts[np.argmax(diff)]    # bottom-left
    return ordered


def correct_perspective(image: np.ndarray) -> np.ndarray:
    """Flatten a document photographed at an angle, if a quad is detectable."""
    gray = to_grayscale(image)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    image_area = image.shape[0] * image.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(contour) < 0.25 * image_area:
            break
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) != 4:
            continue
        src = order_corners(approx)
        (tl, tr, br, bl) = src
        width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        if width < 50 or height < 50:
            continue
        dst = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, matrix, (width, height))
    return image


def preprocess(image: np.ndarray, options: PreprocessOptions | None = None) -> PreprocessResult:
    opts = options or PreprocessOptions()
    steps: list[str] = []
    work = image

    if opts.perspective_correct:
        work = correct_perspective(work)
        steps.append("perspective_correct")
    if opts.resize:
        work = resize_for_ocr(work, opts.min_width, opts.max_width)
        steps.append("resize")

    gray = to_grayscale(work)
    steps.append("grayscale")

    if opts.denoise:
        gray = remove_noise(gray)
        steps.append("denoise")
    if opts.enhance_contrast:
        gray = enhance_contrast(gray)
        steps.append("enhance_contrast")
    if opts.sharpen:
        gray = sharpen(gray)
        steps.append("sharpen")

    angle = 0.0
    if opts.deskew:
        gray, angle = deskew(gray)
        steps.append("deskew")

    binary = None
    if opts.threshold:
        binary = binarize(gray, opts.threshold_method)
        steps.append("threshold")

    return PreprocessResult(
        image=binary if binary is not None else gray,
        gray=gray,
        binary=binary,
        skew_angle=angle,
        steps=steps,
    )
