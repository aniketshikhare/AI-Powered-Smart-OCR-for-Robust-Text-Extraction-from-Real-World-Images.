"""Tests for the from-scratch CRNN recogniser."""
import numpy as np
import pytest
import torch

from smart_ocr.crnn import charset
from smart_ocr.crnn.dataset import SyntheticTextDataset, degrade, fit_to_input, render_text
from smart_ocr.crnn.model import IMAGE_HEIGHT, IMAGE_WIDTH, CRNN
from smart_ocr.crnn.train import DEFAULT_MODEL_PATH, char_error_rate, levenshtein


def test_encode_skips_unsupported_characters():
    assert charset.encode("AB") == [charset.CHARSET.index("A") + 1, charset.CHARSET.index("B") + 1]
    assert charset.encode("A\u00e9B") == charset.encode("AB")


def test_greedy_decode_collapses_repeats_before_blanks():
    def index(ch):
        return charset.CHARSET.index(ch) + 1

    # H H blank E blank L L blank L O  ->  "HELLO"
    sequence = [index("H"), index("H"), 0, index("E"), 0, index("L"), index("L"), 0, index("L"), index("O")]
    logits = torch.zeros(len(sequence), 1, charset.NUM_CLASSES)
    for t, i in enumerate(sequence):
        logits[t, 0, i] = 10.0
    assert charset.decode_greedy(logits) == ["HELLO"]


def test_sequence_confidence_is_percentage():
    logits = torch.zeros(5, 2, charset.NUM_CLASSES)
    logits[:, :, 1] = 20.0
    confidences = charset.sequence_confidence(logits)
    assert len(confidences) == 2
    assert all(90 <= c <= 100 for c in confidences)


def test_model_output_shape_matches_ctc_expectations():
    model = CRNN()
    logits = model(torch.zeros(3, 1, IMAGE_HEIGHT, IMAGE_WIDTH))
    assert logits.shape == (IMAGE_WIDTH // 4, 3, charset.NUM_CLASSES)
    # CTC needs at least as many time steps as target characters.
    assert logits.shape[0] >= 25


def test_ctc_loss_decreases_when_overfitting_one_batch():
    torch.manual_seed(0)
    dataset = SyntheticTextDataset(size=4, seed=3, degrade_prob=0.0)
    images = torch.stack([dataset[i][0] for i in range(4)])
    texts = [dataset[i][1] for i in range(4)]
    model = CRNN()
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3)
    targets, lengths = charset.encode_batch(texts)

    losses = []
    for _ in range(12):
        logits = model(images)
        loss = criterion(logits, targets, torch.full((4,), logits.size(0)), lengths)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_dataset_is_deterministic_and_shaped():
    a, text_a = SyntheticTextDataset(size=10, seed=5)[3]
    b, text_b = SyntheticTextDataset(size=10, seed=5)[3]
    assert text_a == text_b
    assert torch.equal(a, b)
    assert a.shape == (1, IMAGE_HEIGHT, IMAGE_WIDTH)
    assert -1.01 <= float(a.min()) and float(a.max()) <= 1.01


def test_fit_to_input_pads_and_crops():
    wide = np.full((40, 2000), 255, np.uint8)
    narrow = np.full((40, 30), 255, np.uint8)
    assert fit_to_input(wide).shape == (IMAGE_HEIGHT, IMAGE_WIDTH)
    assert fit_to_input(narrow).shape == (IMAGE_HEIGHT, IMAGE_WIDTH)


def test_degradations_change_the_image():
    import random

    from smart_ocr.crnn.dataset import available_fonts

    clean = render_text("SAMPLE TEXT", available_fonts()[0], random.Random(1))
    noisy = degrade(clean.copy(), random.Random(2))
    assert noisy.shape == clean.shape
    assert not np.array_equal(noisy, clean)


def test_error_metrics():
    assert levenshtein("kitten", "sitting") == 3
    assert char_error_rate(["ABC"], ["ABC"]) == 0.0
    assert char_error_rate(["ABD"], ["ABC"]) == pytest.approx(1 / 3)


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="CRNN checkpoint not trained yet")
def test_trained_engine_reads_a_rendered_line():
    import cv2

    from smart_ocr.modules.recognition import get_engine

    image = np.full((80, 520, 3), 245, np.uint8)
    cv2.putText(image, "INVOICE 2026", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (15, 15, 15), 3)
    result = get_engine("crnn").recognise(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    assert "INVOICE" in result.raw_text.upper()
