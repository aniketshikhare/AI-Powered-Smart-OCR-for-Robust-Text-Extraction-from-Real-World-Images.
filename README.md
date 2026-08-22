# AI-Powered Smart OCR for Robust Text Extraction from Real-World Images

Final-year project implementation of the synopsis. The system is a complete web-based OCR pipeline for difficult real-world images, with upload validation, preprocessing, text detection, multiple recognition engines, post-processing, SQLite history, exports, and measurable accuracy evaluation.

## Module map

| Synopsis module | File | What it does |
|---|---|---|
| 1. Image Upload | `smart_ocr/modules/upload.py` | Type/size/decodability validation, safe storage with a UUID name |
| 2. Image Preprocessing | `smart_ocr/modules/preprocessing.py` | Resize, grayscale, denoise, CLAHE contrast, unsharp mask, deskew, perspective correction, auto Otsu/adaptive threshold |
| 3. Text Detection | `smart_ocr/modules/detection.py` | Morphological-gradient and MSER detectors, box filtering and line merging |
| 4. Text Recognition | `smart_ocr/modules/recognition.py` | Tesseract default, EasyOCR optional, and own CRNN+CTC model; per-word text + confidence |
| 4b. Own recogniser | `smart_ocr/crnn/` | CRNN (CNN + BiLSTM + CTC) written and trained from scratch in PyTorch |
| 5. Post-Processing | `smart_ocr/modules/postprocessing.py` | Confidence filtering, noise-token removal, O/0 and I/1 context fixes, whitespace normalisation, dictionary spell correction |
| 6. Result | `smart_ocr/modules/result.py` | Result object, quality label, TXT/JSON export |
| Accuracy Evaluation | `smart_ocr/modules/evaluation.py` | Ground-truth comparison using CER, WER, character accuracy, word accuracy and exact match |
| Database Collection | `smart_ocr/database.py` | SQLite table for OCR ID, User ID, Image Name, Extracted Text, Confidence Score, Date and Time |
| Workflow | `smart_ocr/pipeline.py` | Chains modules with a low-confidence retry on non-binarised input |
| Frontend | `templates/`, `static/` | Upload UI, preprocessing controls, result metrics, accuracy metrics, copy/download and history |

## Setup on Windows/Linux

**Recommended Python version: 3.11 (64-bit).** Python 3.14 is not recommended for this project because the pinned NumPy version and the optional CRNN/PyTorch stack may not provide compatible Windows wheels.

Install Tesseract OCR and make sure the `tesseract` command is available in PATH.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
pip install -r requirements.txt
python run.py
```

The `python --version` command should report Python 3.11.x before installing dependencies.

### Linux/macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
pip install -r requirements.txt
python run.py
```

Open `http://localhost:5000`.

If Tesseract is missing, the web API now returns a clear setup error instead of an unhandled server error.

Optional CRNN engine:

```bash
pip install -r requirements-crnn.txt --index-url https://download.pytorch.org/whl/cpu
```

Then choose **CRNN** in the web UI or set `SMART_OCR_ENGINE=crnn`.

## Accuracy verification

OCR confidence is not the same as correctness. To demonstrate whether an extraction is actually correct, paste the known correct transcription into **Ground-truth text** before extraction. The response then reports:

- Character Accuracy — higher is better; 100% means no character edits are needed.
- Word Accuracy — higher is better; 100% means no word edits are needed.
- CER (Character Error Rate) — lower is better; 0% is perfect.
- WER (Word Error Rate) — lower is better; 0% is perfect.
- Exact Match — whether normalized predicted and expected text are identical.

The same calculation is available through `POST /api/evaluate` with `ground_truth` and `predicted_text` fields.

## API

| Method | Route | Purpose |
|---|---|---|
| POST | `/api/ocr` | multipart `image` + options → OCR JSON result; optional `ground_truth` adds evaluation metrics |
| POST | `/api/evaluate` | Compare supplied OCR text with ground truth |
| GET | `/api/records` | Stored OCR records |
| DELETE | `/api/records/<id>` | Delete a record |
| GET | `/download/<id>.txt` / `.json` | Export |
| GET | `/health` | Health check |

Example:

```bash
curl -F image=@samples/noisy.png -F detector=morphology http://localhost:5000/api/ocr
```

## Robustness design

Three decisions do most of the work:

1. **Automatic threshold selection.** The preprocessing module selects between Otsu and adaptive thresholding based on illumination variation.
2. **Detection before recognition.** Detected text lines are cropped before recognition, keeping complex backgrounds out of the recogniser.
3. **Low-confidence retry.** If thresholded OCR is weak, the pipeline retries the grayscale image and keeps the better result using confidence weighted by surviving word count.

## Own recogniser: CRNN + CTC

`smart_ocr/crnn/` contains a line recogniser built from PyTorch primitives: CNN feature extraction, a 2-layer BiLSTM sequence model and a CTC output head. Synthetic training data includes blur, noise, shadows, rotation, contrast changes, varied tracking and digits.

The repository includes a trained checkpoint at `models/crnn.pt`.

## Tests

```bash
python -m pytest -q
```

The test suite covers preprocessing, detection, recognition, CRNN, upload validation, post-processing, pipeline behaviour, robustness samples and the ground-truth accuracy evaluator.

## Docker deployment

A production-oriented Dockerfile is included. It installs Tesseract, Python dependencies and Gunicorn:

```bash
docker build -t smart-ocr .
docker run -p 5000:5000 smart-ocr
```

Then open `http://localhost:5000`.

## Samples and benchmark

Generate synthetic difficult samples:

```bash
python scripts/make_samples.py
```

Run the benchmark:

```bash
python scripts/benchmark.py
```

The benchmark compares the recognition engines while keeping the same preprocessing, detection and post-processing pipeline.
