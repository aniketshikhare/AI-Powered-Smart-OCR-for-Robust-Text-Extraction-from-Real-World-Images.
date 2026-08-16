"""Compare recognition engines over samples/ and print a markdown table.

Usage: python scripts/benchmark.py [--engines tesseract,crnn]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from smart_ocr.pipeline import OCRPipeline  # noqa: E402

GROUND_TRUTH = {
    "clean.png": "INVOICE TOTAL 1250",
    "noisy.png": "RECEIPT NO 4471",
    "shadow.png": "EXIT GATE 2",
    "blurred.png": "PLATFORM NO 5",
    "rotated.png": "BUS STOP AIROLI",
    "hard.png": "SHOP NO 14 PUNE",
}


def levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def normalise(text: str) -> str:
    return " ".join(text.upper().split())


def run_engine(engine: str, samples: Path) -> list[dict]:
    pipeline = OCRPipeline()
    rows = []
    for name, truth in GROUND_TRUTH.items():
        path = samples / name
        if not path.exists():
            continue
        started = time.perf_counter()
        result = pipeline.run(str(path), image_name=name, engine=engine)
        elapsed = time.perf_counter() - started
        predicted = normalise(result.text)
        distance = levenshtein(predicted, truth)
        rows.append(
            {
                "sample": name.removesuffix(".png"),
                "truth": truth,
                "predicted": predicted,
                "cer": distance / max(len(truth), 1),
                "exact": predicted == truth,
                "confidence": result.confidence,
                "seconds": elapsed,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", default="tesseract,crnn")
    parser.add_argument("--samples", default=str(ROOT / "samples"))
    args = parser.parse_args()

    samples = Path(args.samples)
    for engine in args.engines.split(","):
        engine = engine.strip()
        rows = run_engine(engine, samples)
        mean_cer = sum(r["cer"] for r in rows) / max(len(rows), 1)
        exact = sum(r["exact"] for r in rows)
        mean_time = sum(r["seconds"] for r in rows) / max(len(rows), 1)

        print(f"\n### {engine}\n")
        print("| Sample | Ground truth | Output | CER | Exact | Confidence |")
        print("|---|---|---|---|---|---|")
        for r in rows:
            mark = "yes" if r["exact"] else "no"
            print(
                f"| {r['sample']} | {r['truth']} | {r['predicted']} | "
                f"{r['cer']:.3f} | {mark} | {r['confidence']:.1f} |"
            )
        print(
            f"\nmean CER {mean_cer:.3f} | exact {exact}/{len(rows)} | "
            f"mean {mean_time:.2f}s per image"
        )


if __name__ == "__main__":
    main()
