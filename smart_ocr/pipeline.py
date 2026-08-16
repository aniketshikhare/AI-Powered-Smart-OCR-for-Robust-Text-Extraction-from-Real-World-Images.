"""Orchestrates modules 2-6 into the workflow described in the synopsis."""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import cv2

from .modules import detection, postprocessing, preprocessing, recognition, upload
from .modules.result import OCRResult


class OCRPipeline:
    def __init__(
        self,
        engine: str = "tesseract",
        languages: str = "eng",
        min_confidence: float = 40.0,
        debug_dir: Path | None = None,
    ) -> None:
        self.engine = engine
        self.languages = languages
        self.min_confidence = min_confidence
        self.debug_dir = debug_dir
        self._dictionary = postprocessing.load_dictionary()

    def _read(self, image, options, detector, spell_correct, use_detection, engine):
        pre = preprocessing.preprocess(image, options)

        regions: list[detection.TextRegion] = []
        if use_detection:
            regions = detection.detect_text_regions(pre.image, method=detector)

        if regions:
            rec = recognition.recognise_regions(pre.image, regions, engine, self.languages)
        else:
            rec = recognition.recognise_image(pre.image, engine, self.languages)

        # Region-based OCR can miss text when detection fragments a noisy image.
        if not rec.words and regions:
            rec = recognition.recognise_image(pre.image, engine, self.languages)

        post = postprocessing.postprocess(
            rec,
            min_confidence=self.min_confidence,
            spell_correct=spell_correct,
            dictionary=self._dictionary if spell_correct else set(),
        )
        return pre, regions, rec, post

    @staticmethod
    def _score(post) -> float:
        """Rank attempts by confidence weighted by how much text survived."""
        return post.confidence * min(post.word_count, 40)

    def run(
        self,
        image_path: str | Path,
        image_name: str | None = None,
        options: preprocessing.PreprocessOptions | None = None,
        detector: str = "morphology",
        spell_correct: bool = True,
        use_detection: bool = True,
        engine: str | None = None,
    ) -> OCRResult:
        started = time.perf_counter()
        engine = engine or self.engine
        image_path = Path(image_path)
        image = upload.load_image(image_path)
        options = options or preprocessing.PreprocessOptions()

        pre, regions, rec, post = self._read(
            image, options, detector, spell_correct, use_detection, engine
        )

        # Binarisation is the step most likely to destroy a hard image; when the
        # result looks weak, retry on the grayscale image and keep the better read.
        if options.threshold and post.confidence < 70:
            alt_options = replace(options, threshold=False)
            alt = self._read(
                image, alt_options, detector, spell_correct, use_detection, engine
            )
            if self._score(alt[3]) > self._score(post):
                pre, regions, rec, post = alt

        preview_name = None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            preview_name = f"preview_{image_path.stem}.png"
            cv2.imwrite(str(self.debug_dir / preview_name), detection.draw_regions(pre.image, regions))

        return OCRResult(
            image_name=image_name or image_path.name,
            stored_name=image_path.name,
            text=post.text,
            confidence=post.confidence,
            word_count=post.word_count,
            region_count=len(regions),
            engine=rec.engine,
            skew_angle=round(pre.skew_angle, 2),
            preprocess_steps=pre.steps,
            postprocess_steps=post.steps,
            corrections=post.corrections,
            removed_tokens=post.removed_tokens,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            preview_url=preview_name,
        )
