# AI-Powered Smart OCR for Robust Text Extraction from Real-World Images

Final-year project implementation of the synopsis. Every module listed in the
synopsis is implemented from scratch in its own file, so each one can be
explained, demonstrated and tested independently.

## Module map

| Synopsis module | File | What it does |
|---|---|---|
| 1. Image Upload | `smart_ocr/modules/upload.py` | Type/size/decodability validation, safe storage with a UUID name |
| 2. Image Preprocessing | `smart_ocr/modules/preprocessing.py` | Resize, grayscale, denoise, CLAHE contrast, unsharp mask, deskew, perspective correction, auto Otsu/adaptive threshold |
| 3. Text Detection | `smart_ocr/modules/detection.py` | Morphological-gradient and MSER detectors, box filtering and line merging |
| 4. Text Recognition | `smart_ocr/modules/recognition.py` | Engine abstraction (Tesseract default, EasyOCR optional, own CRNN+CTC model), per-word text + confidence |
| 4b. Own recogniser | `smart_ocr/crnn/` | CRNN (CNN + BiLSTM + CTC) written and trained from scratch in PyTorch, with its own synthetic data generator |
| 5. Post-Processing | `smart_ocr/modules/postprocessing.py` | Confidence filtering, noise-token removal, O/0 I/1 confusion fixes, whitespace normalisation, dictionary spell correction |
| 6. Result | `smart_ocr/modules/result.py` | Result object, quality label, TXT/JSON export |
| Database Collection | `smart_ocr/database.py` | SQLite table: OCR ID, User ID, Image Name, Extracted Text, Confidence Score, Date and Time |
| Workflow | `smart_ocr/pipeline.py` | Chains modules 2-6 with fallbacks |
| Frontend | `templates/`, `static/` | Upload UI, live options, result panel with copy/download, history page |

## Setup

```bash
sudo apt-get install -y tesseract-ocr          # OCR engine
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py                                  # http://localhost:5000
```

Optional: `SMART_OCR_ENGINE=easyocr` (after `pip install easyocr`) switches the
recognition backend without changing any other module.

Generate synthetic noisy/blurred/skewed/shadowed test images:

```bash
python scripts/make_samples.py                 # writes to samples/
```

## Tests

```bash
python -m pytest -q                            # 48 tests
```

`tests/test_robustness.py` runs the whole pipeline over clean, noisy, skewed,
shadowed and blurred images — this is the evidence for the "robust" claim in the
synopsis.

## How the accuracy is actually gained

Three decisions do most of the work, and are worth calling out in the report:

1. **Threshold selection is automatic.** Otsu keeps thin strokes intact on
   evenly lit images but collapses under a light gradient; adaptive thresholding
   survives shadows but adds speckle. `binarize(method="auto")` measures
   block-wise brightness spread (`illumination_variation`) and picks one.
2. **Detection before recognition.** Each detected line is cropped, padded and
   recognised on its own, keeping complex background out of the recogniser's
   view — the main failure mode of plain OCR on scene images.
3. **A second read when the first looks weak.** If mean confidence is under 70,
   the pipeline re-runs on the grayscale (non-binarised) image and keeps the
   better result, scored by confidence weighted by surviving word count.

## Own recogniser: CRNN + CTC (trained from scratch)

Besides the off-the-shelf engines, `smart_ocr/crnn/` contains a text-line
recogniser built from nothing but PyTorch primitives:

| File | Role |
|---|---|
| `charset.py` | Character vocabulary, CTC blank at index 0, encoder and greedy decoder with per-sequence confidence |
| `model.py` | CRNN: 5 conv blocks (32x160 crop → 1x40 feature strip) → 2-layer BiLSTM(192) → linear head over the vocabulary |
| `dataset.py` | Synthetic text-line generator: random words/numbers/labels rendered with system fonts, then blurred, noised, shadowed, rotated and contrast-shifted |
| `train.py` | CTC training loop (AdamW + OneCycle), CER/exact-match evaluation, checkpointing to `models/crnn.pt` |
| `infer.py` | Checkpoint loading and batched line recognition |

Train it (CPU is enough):

```bash
pip install -r requirements-crnn.txt
python -m smart_ocr.crnn.train --steps 9000 --batch-size 64
```

Use it: pick "CRNN" in the web UI, send `-F engine=crnn` to `/api/ocr`, or set
`SMART_OCR_ENGINE=crnn`. The detection module feeds it one cropped line at a
time, which is exactly the input distribution it was trained on.

## API

| Method | Route | Purpose |
|---|---|---|
| POST | `/api/ocr` | multipart `image` + option flags (`engine=tesseract|crnn`, `detector`, preprocessing toggles) → JSON result |
| GET | `/api/records` | Stored OCR records |
| DELETE | `/api/records/<id>` | Delete a record |
| GET | `/download/<id>.txt` / `.json` | Export |
| GET | `/health` | Health check |

Example:

```bash
curl -F image=@samples/noisy.png -F detector=morphology http://localhost:5000/api/ocr
```

## Future scope hooks

Multilingual OCR (`SMART_OCR_LANGS=eng+hin`), handwriting and camera input plug
in at the recognition and upload modules respectively; no other module changes.
