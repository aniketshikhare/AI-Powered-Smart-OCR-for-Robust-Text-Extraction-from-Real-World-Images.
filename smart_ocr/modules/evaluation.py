"""OCR evaluation utilities for comparing extracted text with ground truth.

This module is intentionally independent from the OCR pipeline.  The OCR system
can produce a result, while this module answers a different question: how close
is that result to a known correct transcription?
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    """Accuracy metrics for one OCR prediction against ground truth."""

    expected_text: str
    predicted_text: str
    character_errors: int
    word_errors: int
    character_count: int
    word_count: int
    cer: float
    wer: float
    character_accuracy: float
    word_accuracy: float
    exact_match: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "expected_text": self.expected_text,
            "predicted_text": self.predicted_text,
            "character_errors": self.character_errors,
            "word_errors": self.word_errors,
            "character_count": self.character_count,
            "word_count": self.word_count,
            "cer": self.cer,
            "wer": self.wer,
            "character_accuracy": self.character_accuracy,
            "word_accuracy": self.word_accuracy,
            "exact_match": self.exact_match,
        }


def normalize_text(text: str) -> str:
    """Normalize whitespace/case so formatting does not affect evaluation."""
    return re.sub(r"\s+", " ", text.strip()).casefold()


def _edit_distance(left: list[str], right: list[str]) -> int:
    """Levenshtein distance using O(min(n, m)) memory."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (left_item != right_item)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def evaluate(expected_text: str, predicted_text: str) -> EvaluationResult:
    """Calculate CER/WER and exact/accuracy scores.

    CER/WER are error rates, so 0% is perfect. Character/word accuracy are the
    corresponding intuitive scores, so 100% is perfect. Values are clamped to
    0..100 for presentation.
    """
    expected = normalize_text(expected_text)
    predicted = normalize_text(predicted_text)
    expected_chars = list(expected.replace(" ", ""))
    predicted_chars = list(predicted.replace(" ", ""))
    expected_words = expected.split() if expected else []
    predicted_words = predicted.split() if predicted else []

    char_errors = _edit_distance(expected_chars, predicted_chars)
    word_errors = _edit_distance(expected_words, predicted_words)
    char_count = len(expected_chars)
    word_count = len(expected_words)

    cer = (char_errors / char_count * 100.0) if char_count else (0.0 if not predicted_chars else 100.0)
    wer = (word_errors / word_count * 100.0) if word_count else (0.0 if not predicted_words else 100.0)

    return EvaluationResult(
        expected_text=expected_text,
        predicted_text=predicted_text,
        character_errors=char_errors,
        word_errors=word_errors,
        character_count=char_count,
        word_count=word_count,
        cer=round(min(cer, 100.0), 2),
        wer=round(min(wer, 100.0), 2),
        character_accuracy=round(max(0.0, 100.0 - min(cer, 100.0)), 2),
        word_accuracy=round(max(0.0, 100.0 - min(wer, 100.0)), 2),
        exact_match=expected == predicted,
    )
