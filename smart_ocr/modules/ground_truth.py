"""Helpers for attaching and evaluating ground-truth text.

Ground truth is optional: it is useful for project demonstrations and benchmark
experiments where the correct transcription is known in advance.
"""
from __future__ import annotations

from .evaluation import EvaluationResult, evaluate


def evaluate_against_ground_truth(predicted_text: str, ground_truth: str) -> EvaluationResult:
    """Compare an OCR prediction with a supplied reference transcription."""
    return evaluate(ground_truth, predicted_text)
