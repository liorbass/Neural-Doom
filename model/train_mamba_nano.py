#!/usr/bin/env python3
"""Train and export Mamba-Nano: Ultra-lightweight 2-layer Selective SSM for DOOM."""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from samba import MambaSSM
from block_schema import BlockTransition

ROOT = Path(__file__).resolve().parent.parent


class MambaNanoPredictor(nn.Module):
    """Ultra-lightweight 2-layer Selective SSM for fast branch & block prediction."""

    def __init__(self, d_model: int = 64, d_state: int = 16, num_layers: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_layers = num_layers

        self.pc_proj = nn.Linear(32, d_model // 4)
        self.regs_proj = nn.Linear(32, d_model - d_model // 4)
        self.drop = nn.Dropout(p=0.15)

        self.layers = nn.ModuleList([
            MambaSSM(d_model=d_model, d_state=d_state, dt_rank=8)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        self.head_branch = nn.Linear(d_model, 1)
        self.head_pc = nn.Linear(d_model, 1)

    def forward(
        self,
        pc_bits: torch.Tensor,       # (B, L, 32)
        regs_norm: torch.Tensor,     # (B, L, 32)
        states: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        h = torch.cat([self.pc_proj(pc_bits), self.regs_proj(self.drop(regs_norm))], dim=-1)

        new_states = []
        for i, layer in enumerate(self.layers):
            s_in = states[i] if states is not None and i < len(states) else None
            h_out, s_out = layer(h, prev_state=s_in)
            h = h + h_out
            new_states.append(s_out)

        h = self.ln_f(h)
        branch_logits = self.head_branch(h)
        next_pc_offset = self.head_pc(h)

        return branch_logits, next_pc_offset, new_states


class MambaNanoONNXWrapper(nn.Module):
    def __init__(self, model: MambaNanoPredictor) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        pc_bits: torch.Tensor,   # (1, 1, 32)
        regs_norm: torch.Tensor, # (1, 1, 32)
        s0: torch.Tensor,        # (1, 64, 16)
        s1: torch.Tensor,        # (1, 64, 16)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        branch_logits, pc_offset, new_states = self.model(
            pc_bits, regs_norm, [s0, s1]
        )
        return branch_logits, pc_offset, new_states[0], new_states[1]


def train_and_export():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    jsonl_path = ROOT / "model" / "doom_blocks.jsonl"
    print(f"Loading transitions from {jsonl_path}...")
    blocks: list[BlockTransition] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                blocks.append(BlockTransition.model_validate_json(line.strip()))

    print(f"Loaded {len(blocks)} transitions.")

    # Create model
    model = MambaNanoPredictor(d_model=64, d_state=16, num_layers=2).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Mamba-Nano Parameter Count: {param_count:,} (~{param_count * 4 / 1024:.1f} KB)")

    # Prepare training sequences (seq_len=16)
    seq_len = 16
    batch_size = 32
    samples_pc = []
    samples_regs = []
    samples_taken = []
    samples_offset = []

    for i in range(0, len(blocks) - seq_len, seq_len):
        chunk = blocks[i : i + seq_len]
        pcs = []
        regs = []
        taken = []
        offsets = []
        for b in chunk:
            # 32 bit PC features from end_pc (exact location of branch / terminator)
            bits = [(b.end_pc >> bit) & 1 for bit in range(32)]
            pcs.append(bits)
            # Register values at the branch (apply register writes that occurred inside the block)
            regs_at_branch = list(b.regs_in)
            for reg_idx, val in b.reg_writes.items():
                regs_at_branch[int(reg_idx)] = val
            # Non-linear log1p normalized registers (preserves 0, 1, 2, ..., small loops in float32)
            r_norm = [float(np.log1p(float(v)) / 22.2) for v in regs_at_branch]
            regs.append(r_norm)
            # taken
            is_taken = 1.0 if b.next_pc != (b.end_pc + 4) else 0.0
            taken.append([is_taken])
            offsets.append([(b.next_pc - b.end_pc) / 4.0])

        samples_pc.append(pcs)
        samples_regs.append(regs)
        samples_taken.append(taken)
        samples_offset.append(offsets)

    t_pcs = torch.tensor(samples_pc, dtype=torch.float32, device=device)
    t_regs = torch.tensor(samples_regs, dtype=torch.float32, device=device)
    t_taken = torch.tensor(samples_taken, dtype=torch.float32, device=device)
    t_offsets = torch.tensor(samples_offset, dtype=torch.float32, device=device)
    # Train / Val Split (90% / 10%)
    num_total = len(samples_pc)
    num_val = max(1, int(num_total * 0.1))
    num_train = num_total - num_val

    # Random split but fixed seed for reproducibility
    torch.manual_seed(42)
    indices = torch.randperm(num_total)
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]

    train_pcs = t_pcs[train_idx]
    train_regs = t_regs[train_idx]
    train_taken = t_taken[train_idx]
    train_offsets = t_offsets[train_idx]

    val_pcs = t_pcs[val_idx]
    val_regs = t_regs[val_idx]
    val_taken = t_taken[val_idx]
    val_offsets = t_offsets[val_idx]

    print(f"Dataset split: {num_train} train sequences, {num_val} validation sequences ({num_val * seq_len} transitions)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    epochs = 40
    batch_size = 64
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"Training Mamba-Nano for {epochs} epochs (batch_size={batch_size}) with log1p features and CosineAnnealing...")
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(num_train, device=device)
        total_loss = 0.0
        train_correct = 0
        train_total = 0
        batches = 0

        for b_idx in range(0, num_train, batch_size):
            idx = perm[b_idx : b_idx + batch_size]
            b_pcs = train_pcs[idx]
            b_regs = train_regs[idx]
            b_taken = train_taken[idx]
            b_offsets = train_offsets[idx]

            optimizer.zero_grad()
            branch_logits, pc_offsets, _ = model(b_pcs, b_regs)

            loss_branch = F.binary_cross_entropy_with_logits(branch_logits, b_taken)
            loss_pc = F.smooth_l1_loss(pc_offsets, b_offsets)
            loss = 5.0 * loss_branch + 0.01 * loss_pc

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

            preds = (branch_logits > 0.0).float()
            train_correct += (preds == b_taken).sum().item()
            train_total += b_taken.numel()

        scheduler.step()
        train_acc = (train_correct / train_total) * 100.0

        # Evaluate validation set
        model.eval()
        with torch.no_grad():
            val_branch_logits, val_pc_offsets, _ = model(val_pcs, val_regs)
            val_preds = (val_branch_logits > 0.0).float()
            val_acc = ((val_preds == val_taken).sum().item() / val_taken.numel()) * 100.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"  Epoch {epoch:2d}/{epochs} - Loss: {total_loss / batches:.4f} "
                f"| Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% (Best: {best_val_acc:.2f}%) "
                f"(LR: {scheduler.get_last_lr()[0]:.6f})"
            )

    print(f"\nFinal Best Validation Accuracy: {best_val_acc:.2f}%")

    # Export ONNX
    print("Exporting Mamba-Nano to ONNX...")
    model.eval().cpu()
    wrapper = MambaNanoONNXWrapper(model)

    dummy_pc = torch.zeros(1, 1, 32, dtype=torch.float32)
    dummy_regs = torch.zeros(1, 1, 32, dtype=torch.float32)
    dummy_s0 = torch.zeros(1, 64, 16, dtype=torch.float32)
    dummy_s1 = torch.zeros(1, 64, 16, dtype=torch.float32)

    onnx_path = ROOT / "web" / "mamba_nano.onnx"
    try:
        torch.onnx.export(
            wrapper,
            (dummy_pc, dummy_regs, dummy_s0, dummy_s1),
            str(onnx_path),
            input_names=["pc_bits", "regs_norm", "s0", "s1"],
            output_names=["branch_taken_logits", "next_pc_offset", "s0_out", "s1_out"],
            dynamic_axes=None,
            opset_version=18,
            dynamo=False,
            do_constant_folding=True,
        )
    except TypeError:
        torch.onnx.export(
            wrapper,
            (dummy_pc, dummy_regs, dummy_s0, dummy_s1),
            str(onnx_path),
            input_names=["pc_bits", "regs_norm", "s0", "s1"],
            output_names=["branch_taken_logits", "next_pc_offset", "s0_out", "s1_out"],
            dynamic_axes=None,
            opset_version=18,
            do_constant_folding=True,
        )

    print(f"Saved: {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    train_and_export()
