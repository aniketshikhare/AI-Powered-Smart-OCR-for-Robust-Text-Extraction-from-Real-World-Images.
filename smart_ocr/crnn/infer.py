"""Inference helper around a trained CRNN checkpoint."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .charset import decode_greedy, sequence_confidence
from .dataset import fit_to_input, to_tensor
from .model import CRNN
from .train import DEFAULT_MODEL_PATH


class CRNNRecogniser:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No CRNN checkpoint at {self.model_path}. Train one with "
                "`python -m smart_ocr.crnn.train`."
            )
        checkpoint = torch.load(self.model_path, map_location="cpu")
        self.model = CRNN()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.cer = float(checkpoint.get("cer", float("nan")))

    def read_batch(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        if not crops:
            return []
        gray = [c if c.ndim == 2 else c[..., 0] for c in crops]
        batch = torch.stack([to_tensor(fit_to_input(c)) for c in gray])
        with torch.no_grad():
            logits = self.model(batch)
        return list(zip(decode_greedy(logits), sequence_confidence(logits)))

    def read(self, crop: np.ndarray) -> tuple[str, float]:
        return self.read_batch([crop])[0]
