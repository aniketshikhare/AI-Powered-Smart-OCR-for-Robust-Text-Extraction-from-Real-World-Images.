"""Functional modules of the Smart OCR system (one file per synopsis module)."""

from . import detection, postprocessing, preprocessing, recognition, result, upload

__all__ = ["upload", "preprocessing", "detection", "recognition", "postprocessing", "result"]
