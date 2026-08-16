"""Train the CRNN recogniser on synthetic degraded text lines.

    python -m smart_ocr.crnn.train --steps 6000 --batch-size 64
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .charset import decode_greedy, encode_batch
from .dataset import SyntheticTextDataset, collate
from .model import CRNN

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "crnn.pt"


def char_error_rate(predictions: list[str], targets: list[str]) -> float:
    total_distance = total_length = 0
    for pred, target in zip(predictions, targets):
        total_distance += levenshtein(pred, target)
        total_length += len(target)
    return total_distance / max(total_length, 1)


def levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


@torch.no_grad()
def evaluate(model: CRNN, loader: DataLoader, max_batches: int = 8) -> tuple[float, float]:
    model.eval()
    predictions: list[str] = []
    targets: list[str] = []
    for i, (images, texts) in enumerate(loader):
        if i >= max_batches:
            break
        predictions.extend(decode_greedy(model(images)))
        targets.extend(texts)
    exact = sum(p == t for p, t in zip(predictions, targets)) / max(len(targets), 1)
    return char_error_rate(predictions, targets), exact


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    train_loader = DataLoader(
        SyntheticTextDataset(size=args.steps * args.batch_size, seed=args.seed),
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        SyntheticTextDataset(size=args.batch_size * 8, seed=9999),
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=collate,
    )

    model = CRNN()
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=args.steps, pct_start=0.1
    )

    model_path = Path(args.output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    best_cer = float("inf")
    started = time.time()
    running = 0.0

    for step, (images, texts) in enumerate(train_loader, start=1):
        model.train()
        logits = model(images)  # (T, N, C)
        targets, target_lengths = encode_batch(texts)
        input_lengths = torch.full((images.size(0),), logits.size(0), dtype=torch.long)
        loss = criterion(logits, targets, input_lengths, target_lengths)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        scheduler.step()
        running += loss.item()

        if step % args.log_every == 0:
            print(
                f"step {step}/{args.steps} loss {running / args.log_every:.4f} "
                f"elapsed {time.time() - started:.0f}s",
                flush=True,
            )
            running = 0.0

        if step % args.eval_every == 0 or step == args.steps:
            cer, exact = evaluate(model, val_loader)
            print(f"  eval step {step}: CER {cer:.4f} exact {exact:.3f}", flush=True)
            if cer < best_cer:
                best_cer = cer
                torch.save({"state_dict": model.state_dict(), "cer": cer, "step": step}, model_path)
                print(f"  saved {model_path} (CER {cer:.4f})", flush=True)

        if step >= args.steps:
            break

    print(f"done. best CER {best_cer:.4f} -> {model_path}")
    return model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--output", default=str(DEFAULT_MODEL_PATH))
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
