#!/usr/bin/env python3
"""Plot training loss curves from logs/train.log -> docs/training_loss.png.

Training is step-based (batches sampled with replacement), so the x-axis is
optimizer steps rather than epochs.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRAIN_RE = re.compile(r"\[TRAIN\] step (\d+)/\d+ loss=([\d.]+)")
VAL_RE = re.compile(r"\[VAL\] step (\d+) loss=([\d.]+)")


def main() -> None:
    train: list[tuple[int, float]] = []
    val: list[tuple[int, float]] = []
    for line in (ROOT / "logs" / "train.log").read_text().splitlines():
        m = TRAIN_RE.search(line)
        if m:
            train.append((int(m[1]), float(m[2])))
        m = VAL_RE.search(line)
        if m:
            val.append((int(m[1]), float(m[2])))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs_train, ys_train = zip(*train)
    ax.plot(xs_train, ys_train, label="train", color="#c94c4c", lw=1.2, alpha=0.75)
    xs_val, ys_val = zip(*val)
    ax.plot(xs_val, ys_val, label="val", color="#222222", lw=1.8, marker="o", ms=2.5)
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss (log)")
    ax.set_title("neural-doom training loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = ROOT / "docs" / "training_loss.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
