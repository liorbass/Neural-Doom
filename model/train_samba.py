#!/usr/bin/env python3
"""Trainer for Samba Basic Block Predictor on DOOM traces."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from block_schema import BlockTransition
from samba import DoomSambaPredictor

ROOT = Path(__file__).resolve().parent.parent


class BlockBatch(TypedDict):
    start_pcs: torch.Tensor       # (B, L)
    regs_in: torch.Tensor         # (B, L, 32)
    next_pc_offsets: torch.Tensor # (B, L, 1)
    branch_taken: torch.Tensor    # (B, L, 1)
    reg_masks: torch.Tensor       # (B, L, 32)
    reg_vals: torch.Tensor        # (B, L, 32)
    mem_has_write: torch.Tensor   # (B, L, 2)
    mem_addrs: torch.Tensor       # (B, L, 2)
    mem_vals: torch.Tensor        # (B, L, 2)
    mem_sizes: torch.Tensor       # (B, L, 2)


def load_dataset(jsonl_path: Path) -> list[BlockTransition]:
    print(f"Loading transitions from {jsonl_path}...")
    blocks: list[BlockTransition] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                blocks.append(BlockTransition.model_validate_json(line))
    print(f"Loaded {len(blocks)} basic block transitions.")
    return blocks


def create_batches(
    blocks: list[BlockTransition],
    seq_len: int = 16,
    batch_size: int = 16,
    device: torch.device = torch.device("cpu"),
) -> list[BlockBatch]:
    """Slice sequential execution stream into Batches of sequence length L."""
    N = len(blocks)
    samples: list[dict[str, list[Any]]] = []

    for i in range(0, N - seq_len, seq_len):
        chunk = blocks[i : i + seq_len]
        sample_pcs: list[int] = []
        sample_regs: list[list[int]] = []
        sample_pc_offsets: list[float] = []
        sample_branch_taken: list[float] = []
        sample_reg_masks: list[list[float]] = []
        sample_reg_vals: list[list[float]] = []
        sample_mem_flags: list[list[float]] = []
        sample_mem_addrs: list[list[float]] = []
        sample_mem_vals: list[list[float]] = []
        sample_mem_sizes: list[list[int]] = []

        for b in chunk:
            sample_pcs.append(b.start_pc)
            sample_regs.append(b.regs_in)

            # PC offset: (next_pc - start_pc) / 4
            offset = (b.next_pc - b.start_pc) // 4
            sample_pc_offsets.append(float(offset))

            # Branch taken: 1.0 if branch taken (next_pc != end_pc + 4), 0.0 if fallthrough
            taken = 1.0 if b.next_pc != (b.end_pc + 4) else 0.0
            sample_branch_taken.append(taken)

            # Reg mask & vals
            rmask = [0.0] * 32
            rvals = [0.0] * 32
            for rd, val in b.reg_writes.items():
                if 0 <= rd < 32:
                    rmask[rd] = 1.0
                    rvals[rd] = (float(val) - 2147483648.0) / 2147483648.0
            sample_reg_masks.append(rmask)
            sample_reg_vals.append(rvals)

            # Mem writes (up to 2 slots)
            mflags = [0.0, 0.0]
            maddrs = [0.0, 0.0]
            mvals = [0.0, 0.0]
            msizes = [0, 0]
            for slot, mw in enumerate(b.mem_writes[:2]):
                mflags[slot] = 1.0
                maddrs[slot] = (float(mw[0]) - 2147483648.0) / 2147483648.0
                mvals[slot] = (float(mw[1]) - 2147483648.0) / 2147483648.0
                msizes[slot] = {1: 0, 2: 1, 4: 2}.get(mw[2], 2)
            sample_mem_flags.append(mflags)
            sample_mem_addrs.append(maddrs)
            sample_mem_vals.append(mvals)
            sample_mem_sizes.append(msizes)

        samples.append(
            {
                "pcs": sample_pcs,
                "regs": sample_regs,
                "pc_offsets": sample_pc_offsets,
                "branch_taken": sample_branch_taken,
                "reg_masks": sample_reg_masks,
                "reg_vals": sample_reg_vals,
                "mem_flags": sample_mem_flags,
                "mem_addrs": sample_mem_addrs,
                "mem_vals": sample_mem_vals,
                "mem_sizes": sample_mem_sizes,
            }
        )

    # Collate into batches
    batches: list[BlockBatch] = []
    for i in range(0, len(samples) - batch_size + 1, batch_size):
        b_samples = samples[i : i + batch_size]
        b_pcs = torch.tensor([s["pcs"] for s in b_samples], device=device, dtype=torch.int64)
        b_regs = torch.tensor([s["regs"] for s in b_samples], device=device, dtype=torch.int64)
        b_pc_offsets = torch.tensor(
            [[[off] for off in s["pc_offsets"]] for s in b_samples],
            device=device,
            dtype=torch.float32,
        )
        b_branch_taken = torch.tensor(
            [[[t] for t in s["branch_taken"]] for s in b_samples],
            device=device,
            dtype=torch.float32,
        )
        b_reg_masks = torch.tensor([s["reg_masks"] for s in b_samples], device=device, dtype=torch.float32)
        b_reg_vals = torch.tensor([s["reg_vals"] for s in b_samples], device=device, dtype=torch.float32)
        b_mem_flags = torch.tensor([s["mem_flags"] for s in b_samples], device=device, dtype=torch.float32)
        b_mem_addrs = torch.tensor([s["mem_addrs"] for s in b_samples], device=device, dtype=torch.float32)
        b_mem_vals = torch.tensor([s["mem_vals"] for s in b_samples], device=device, dtype=torch.float32)
        b_mem_sizes = torch.tensor([s["mem_sizes"] for s in b_samples], device=device, dtype=torch.int64)

        batches.append(
            {
                "start_pcs": b_pcs,
                "regs_in": b_regs,
                "next_pc_offsets": b_pc_offsets,
                "branch_taken": b_branch_taken,
                "reg_masks": b_reg_masks,
                "reg_vals": b_reg_vals,
                "mem_has_write": b_mem_flags,
                "mem_addrs": b_mem_addrs,
                "mem_vals": b_mem_vals,
                "mem_sizes": b_mem_sizes,
            }
        )
    return batches


def train(
    data_path: Path,
    save_path: Path,
    epochs: int = 15,
    d_model: int = 128,
    num_layers: int = 4,
    num_heads: int = 4,
    lr: float = 1e-3,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    blocks = load_dataset(data_path)
    batches = create_batches(blocks, seq_len=16, batch_size=16, device=device)
    print(f"Created {len(batches)} batches (seq_len=16, batch_size=16).")

    if not batches:
        print("Not enough data to form batches!")
        return

    model = DoomSambaPredictor(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_state=16,
        window_size=32,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,}")

    print("\n--- Starting Samba Training ---")
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        branch_loss_sum = 0.0
        reg_loss_sum = 0.0

        for b in batches:
            optimizer.zero_grad()
            pred, _, _ = model(b["start_pcs"], b["regs_in"])

            # 1. Branch classification loss (BCE)
            l_branch = F.binary_cross_entropy_with_logits(
                pred.branch_taken_logits, b["branch_taken"]
            )

            # 2. PC offset loss (Smooth L1, scaled)
            l_pc = F.smooth_l1_loss(pred.next_pc_offset, b["next_pc_offsets"])

            # 3. Reg mask loss (BCE)
            l_reg_mask = F.binary_cross_entropy_with_logits(
                pred.reg_mask_logits, b["reg_masks"]
            )

            # 4. Reg vals loss (masked by ground-truth writes)
            l_reg_vals = (
                F.smooth_l1_loss(pred.reg_vals, b["reg_vals"], reduction="none")
                * b["reg_masks"]
            ).sum() / (b["reg_masks"].sum() + 1e-6)

            # 5. Mem flags loss
            l_mem_flags = F.binary_cross_entropy_with_logits(
                pred.mem_has_write, b["mem_has_write"]
            )

            loss = 5.0 * l_branch + 0.01 * l_pc + 2.0 * l_reg_mask + l_reg_vals + l_mem_flags
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            branch_loss_sum += l_branch.item()
            reg_loss_sum += l_reg_mask.item()

        avg_loss = total_loss / len(batches)
        avg_branch = branch_loss_sum / len(batches)
        avg_reg = reg_loss_sum / len(batches)
        if epoch % 3 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} "
                f"(Branch: {avg_branch:.4f}, RegMask: {avg_reg:.4f})"
            )

    elapsed = time.perf_counter() - t0
    print(f"\nTraining finished in {elapsed:.2f}s ({elapsed/epochs:.2f}s/epoch)")

    # Save model checkpoint
    checkpoint = {
        "model": model.state_dict(),
        "config": {
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
        },
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Samba Block Predictor")
    parser.add_argument("--data", type=Path, default=ROOT / "model" / "doom_blocks.jsonl")
    parser.add_argument("--save", type=Path, default=ROOT / "model" / "samba_blocks.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(
        args.data,
        args.save,
        epochs=args.epochs,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
