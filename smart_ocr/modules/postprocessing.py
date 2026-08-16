"""Module 5 - Post-Processing.

Cleans the recogniser output: drops low-confidence noise, repairs common OCR
confusions, normalises whitespace/punctuation and optionally corrects words
against a dictionary using edit distance.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DICTIONARY_PATH = Path("/usr/share/dict/words")

# Confusions are applied only inside tokens whose character class is unambiguous.
DIGIT_TO_ALPHA = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"}
ALPHA_TO_DIGIT = {v: k for k, v in DIGIT_TO_ALPHA.items()}

NOISE_TOKEN = re.compile(r"^[^\w]{1,2}$")


@dataclass
class PostProcessResult:
    text: str
    confidence: float
    removed_tokens: int = 0
    corrections: list[tuple[str, str]] = field(default_factory=list)
    word_count: int = 0
    steps: list[str] = field(default_factory=list)


def load_dictionary(path: Path = DEFAULT_DICTIONARY_PATH, min_len: int = 3) -> set[str]:
    if not path.exists():
        return set()
    words = set()
    for line in path.read_text(errors="ignore").splitlines():
        w = line.strip().lower()
        if len(w) >= min_len and w.isalpha():
            words.add(w)
    return words


def filter_by_confidence(words, min_confidence: float):
    """Drop words the recogniser itself is unsure about."""
    kept = [w for w in words if w.confidence >= min_confidence]
    return kept, len(words) - len(kept)


def drop_noise_tokens(words):
    kept = [w for w in words if not NOISE_TOKEN.match(w.text.strip())]
    return kept, len(words) - len(kept)


def fix_character_confusions(token: str) -> str:
    """`H0USE` -> `HOUSE`, `1O0` -> `100`: push a token to a single char class."""
    letters = sum(c.isalpha() for c in token)
    digits = sum(c.isdigit() for c in token)
    if letters >= 2 and digits and digits < letters:
        return "".join(DIGIT_TO_ALPHA.get(c, c) for c in token)
    if digits >= 2 and letters and letters < digits:
        return "".join(ALPHA_TO_DIGIT.get(c.upper(), c) for c in token)
    return token


def normalise_whitespace(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    text = "\n".join(lines)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])(?=[^\s\d])", r"\1 ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _edits1(word: str) -> set[str]:
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits if b for c in letters]
    inserts = [a + c + b for a, b in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def correct_word(word: str, dictionary: set[str]) -> str:
    """Single edit-distance correction; unknown or short words are left alone."""
    if not dictionary or len(word) < 4 or not word.isalpha():
        return word
    lower = word.lower()
    if lower in dictionary:
        return word
    candidates = _edits1(lower) & dictionary
    if not candidates:
        return word
    best = min(candidates, key=lambda c: (abs(len(c) - len(lower)), c))
    if word.isupper():
        return best.upper()
    if word[0].isupper():
        return best.capitalize()
    return best


def postprocess(
    recognition,
    min_confidence: float = 40.0,
    spell_correct: bool = True,
    dictionary: set[str] | None = None,
) -> PostProcessResult:
    steps: list[str] = []
    words = list(recognition.words)

    words, removed_conf = filter_by_confidence(words, min_confidence)
    steps.append("confidence_filter")
    words, removed_noise = drop_noise_tokens(words)
    steps.append("noise_filter")

    corrections: list[tuple[str, str]] = []
    vocab = dictionary if dictionary is not None else (load_dictionary() if spell_correct else set())

    lines: dict[int, list[str]] = {}
    for w in words:
        token = fix_character_confusions(w.text.strip())
        if spell_correct and vocab:
            fixed = correct_word(token, vocab)
            if fixed != token:
                corrections.append((token, fixed))
            token = fixed
        if token:
            lines.setdefault(w.line_id, []).append(token)
    if spell_correct and vocab:
        steps.append("spell_correction")
    steps.append("character_confusion_fix")

    text = normalise_whitespace("\n".join(" ".join(lines[k]) for k in sorted(lines)))
    steps.append("whitespace_normalisation")

    confidences = [w.confidence for w in words]
    confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

    return PostProcessResult(
        text=text,
        confidence=confidence,
        removed_tokens=removed_conf + removed_noise,
        corrections=corrections,
        word_count=len(text.split()),
        steps=steps,
    )
