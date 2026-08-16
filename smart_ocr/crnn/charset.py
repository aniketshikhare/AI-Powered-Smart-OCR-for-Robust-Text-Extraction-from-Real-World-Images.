"""Character set and CTC encode/decode helpers.

Index 0 is reserved for the CTC blank, so a class index maps to
``CHARSET[index - 1]``.
"""
from __future__ import annotations

import torch

CHARSET = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    " .,:;!?-/()&#%@'\""
)
BLANK = 0
NUM_CLASSES = len(CHARSET) + 1

_CHAR_TO_INDEX = {c: i + 1 for i, c in enumerate(CHARSET)}


def encode(text: str) -> list[int]:
    """Map a string to CTC target indices, dropping unsupported characters."""
    return [_CHAR_TO_INDEX[c] for c in text if c in _CHAR_TO_INDEX]


def encode_batch(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [encode(t) for t in texts]
    lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    flat = torch.tensor([i for e in encoded for i in e], dtype=torch.long)
    return flat, lengths


def decode_greedy(logits: torch.Tensor) -> list[str]:
    """Best-path CTC decode.

    ``logits`` is (T, N, C). Repeats are collapsed first, then blanks removed —
    the order matters, otherwise doubled letters like "LL" are lost.
    """
    best = logits.argmax(dim=2).permute(1, 0)  # (N, T)
    texts = []
    for sequence in best.tolist():
        chars = []
        previous = None
        for index in sequence:
            if index != previous and index != BLANK:
                chars.append(CHARSET[index - 1])
            previous = index
        texts.append("".join(chars))
    return texts


def sequence_confidence(logits: torch.Tensor) -> list[float]:
    """Mean best-path probability per sample, used as the OCR confidence."""
    probs = logits.softmax(dim=2).max(dim=2).values  # (T, N)
    return [float(v) * 100.0 for v in probs.mean(dim=0)]
