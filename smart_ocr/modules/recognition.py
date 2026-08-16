"""Module 4 - Text Recognition.

An engine abstraction so the OCR backend can be swapped without touching the
rest of the pipeline. Tesseract (pytesseract) is the default; EasyOCR is used
when installed and selected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class RecognisedWord:
    text: str
    confidence: float
    box: tuple[int, int, int, int]
    line_id: int = 0


@dataclass
class RecognitionResult:
    words: list[RecognisedWord] = field(default_factory=list)
    engine: str = ""

    @property
    def mean_confidence(self) -> float:
        scored = [w.confidence for w in self.words if w.text.strip()]
        return round(sum(scored) / len(scored), 2) if scored else 0.0

    @property
    def raw_text(self) -> str:
        lines: dict[int, list[str]] = {}
        for w in self.words:
            if w.text.strip():
                lines.setdefault(w.line_id, []).append(w.text)
        return "\n".join(" ".join(lines[k]) for k in sorted(lines))


class OCREngine:
    name = "base"

    def recognise(self, image: np.ndarray, languages: str = "eng") -> RecognitionResult:
        raise NotImplementedError


class TesseractEngine(OCREngine):
    name = "tesseract"

    def __init__(self, psm: int = 6, oem: int = 3) -> None:
        import pytesseract  # imported here so the dependency is optional per engine

        self._pytesseract = pytesseract
        self.config = f"--oem {oem} --psm {psm}"

    def recognise(self, image: np.ndarray, languages: str = "eng") -> RecognitionResult:
        data = self._pytesseract.image_to_data(
            image,
            lang=languages,
            config=self.config,
            output_type=self._pytesseract.Output.DICT,
        )
        words: list[RecognisedWord] = []
        line_counter: dict[tuple[int, int, int], int] = {}
        for i, text in enumerate(data["text"]):
            if not text.strip():
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            line_id = line_counter.setdefault(key, len(line_counter))
            conf = float(data["conf"][i])
            words.append(
                RecognisedWord(
                    text=text,
                    confidence=max(conf, 0.0),
                    box=(data["left"][i], data["top"][i], data["width"][i], data["height"][i]),
                    line_id=line_id,
                )
            )
        return RecognitionResult(words=words, engine=self.name)


class EasyOCREngine(OCREngine):
    name = "easyocr"

    def __init__(self, languages: str = "en", gpu: bool = False) -> None:
        import easyocr

        self._reader = easyocr.Reader(languages.split("+"), gpu=gpu)

    def recognise(self, image: np.ndarray, languages: str = "en") -> RecognitionResult:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        words: list[RecognisedWord] = []
        for line_id, (bbox, text, conf) in enumerate(self._reader.readtext(image)):
            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            words.append(
                RecognisedWord(
                    text=text,
                    confidence=float(conf) * 100.0,
                    box=(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)),
                    line_id=line_id,
                )
            )
        return RecognitionResult(words=words, engine=self.name)


_ENGINE_CACHE: dict[str, OCREngine] = {}


def get_engine(name: str = "tesseract", **kwargs) -> OCREngine:
    """Engines are cached because EasyOCR model loading is expensive."""
    if name not in _ENGINE_CACHE:
        if name == "easyocr":
            _ENGINE_CACHE[name] = EasyOCREngine(**kwargs)
        else:
            _ENGINE_CACHE[name] = TesseractEngine(**kwargs)
    return _ENGINE_CACHE[name]


def recognise_image(
    image: np.ndarray, engine: str = "tesseract", languages: str = "eng"
) -> RecognitionResult:
    return get_engine(engine).recognise(image, languages)


def recognise_regions(
    image: np.ndarray, regions, engine: str = "tesseract", languages: str = "eng"
) -> RecognitionResult:
    """Recognise each detected region separately and stitch results together.

    Cropping keeps a noisy background out of the recogniser's receptive field,
    which is where full-page OCR usually fails on scene images.
    """
    ocr = get_engine(engine)
    words: list[RecognisedWord] = []
    for line_id, region in enumerate(regions):
        crop = region.crop(image)
        if crop.size == 0:
            continue
        crop = cv2.copyMakeBorder(crop, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        result = ocr.recognise(crop, languages)
        for word in result.words:
            wx, wy, ww, wh = word.box
            words.append(
                RecognisedWord(
                    text=word.text,
                    confidence=word.confidence,
                    box=(region.x + wx - 10, region.y + wy - 10, ww, wh),
                    line_id=line_id,
                )
            )
    return RecognitionResult(words=words, engine=ocr.name)
