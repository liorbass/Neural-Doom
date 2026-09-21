#!/usr/bin/env python3
"""Closed-loop evaluation for Samba Basic Block Predictor on DOOM."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from libemulator import NativeCPU
from samba import DoomSambaPredictor

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIN = ROOT / "doomgeneric" / "doomgeneric" / "doom_rv32i.bin"
DEFAULT_CHECKPOINT = ROOT / "data" / "checkpoint.bin"
DEFAULT_MODEL = ROOT / "model" / "samba_blocks.pt"


def find_block_end(ram: bytes, start_pc: int, max_inst: int = 32) -> tuple[int, int]:
    curr_pc = start_pc
    for _ in range(max_inst):
        offset = curr_pc - 0x80000000
        if offset < 0 or offset + 4 > len(ram):
            break
        inst = int.from_bytes(ram[offset : offset + 4], "little")
        if inst == 0:
            break
        op = inst & 0x7F
        if op in (0x63, 0x6F, 0x67, 0x73):
            return curr_pc, inst
        curr_pc += 4
    return curr_pc - 4, 0


def evaluate_closed_loop(
    bin_path: Path,
    checkpoint_path: Path | None,
    model_path: Path,
    max_blocks: int = 500,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Samba model from {model_path} onto {device}...")

    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = DoomSambaPredictor(
        d_model=cfg.get("d_model", 128),
        num_layers=cfg.get("num_layers", 4),
        num_heads=cfg.get("num_heads", 4),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("Initializing Ground-Truth Oracle CPU and Neural Evaluator CPU...")
    oracle_cpu = NativeCPU(bin_path)
    neural_cpu = NativeCPU(bin_path)

    if checkpoint_path and checkpoint_path.exists():
        print(f"Restoring checkpoint from {checkpoint_path}...")
        oracle_cpu.load_checkpoint(checkpoint_path)
        neural_cpu.load_checkpoint(checkpoint_path)

    print(f"\n--- Running Closed-Loop Basic Block Evaluation (Max {max_blocks} blocks) ---")
    states: list[torch.Tensor] = [
        torch.zeros(1, model.d_model, 16, device=device)
        for _ in range(model.num_layers)
    ]
    kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None

    steps_matched = 0
    total_insts_matched = 0
    diverged = False
    divergence_reason = ""

    # Cache ram reference for fast instruction lookups
    ram = neural_cpu.get_ram()

    # Warmup GPU
    for _ in range(10):
        model.step_block(0x80000000, [0] * 32, states)

    t_start = time.perf_counter()
    latencies: list[float] = []

    for step in range(1, max_blocks + 1):
        curr_pc = neural_cpu.pc
        curr_regs = neural_cpu.regs

        # Find block end instruction from memory
        end_pc, end_inst = find_block_end(ram, curr_pc)

        # 1. Oracle ground truth execution
        oracle_block = oracle_cpu.step_block()
        if oracle_block is None:
            print(f"Oracle CPU halted at step {step}.")
            break

        # 2. Neural model prediction
        t0 = time.perf_counter()
        pred_pc, reg_writes, mem_writes, states, kv_caches = model.step_block(
            curr_pc,
            curr_regs,
            states,
            kv_caches,
            end_pc=end_pc,
            end_inst=end_inst,
        )
        lat = (time.perf_counter() - t0) * 1000.0  # ms
        latencies.append(lat)

        # 3. Check divergence
        # Check PC
        if pred_pc != oracle_block.next_pc:
            diverged = True
            divergence_reason = (
                f"PC mismatch: predicted 0x{pred_pc:08x} != ground truth 0x{oracle_block.next_pc:08x}"
            )
            break

        # Check register modifications
        reg_diffs: list[str] = []
        for rd, expected_val in oracle_block.reg_writes.items():
            got_val = reg_writes.get(rd)
            if got_val is None or got_val != expected_val:
                reg_diffs.append(f"x{rd}: pred={got_val} exp={expected_val}")

        if reg_diffs:
            diverged = True
            divergence_reason = f"Register mismatch: {', '.join(reg_diffs[:3])}"
            break

        # 4. Apply prediction to Neural CPU
        neural_cpu.apply_block_prediction(pred_pc, reg_writes, mem_writes)
        steps_matched += 1
        total_insts_matched += oracle_block.instruction_count

        if step % 50 == 0:
            avg_lat = sum(latencies[-50:]) / len(latencies[-50:])
            print(
                f"  Block {step:3d} matched! ({total_insts_matched} instructions, "
                f"{avg_lat:.2f} ms/block, {avg_lat / oracle_block.instruction_count:.3f} ms/inst)"
            )

    t_total = time.perf_counter() - t_start
    avg_block_lat = sum(latencies) / max(1, len(latencies))
    blocks_per_sec = 1000.0 / avg_block_lat if avg_block_lat > 0 else 0

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Target:                      Basic Block Execution (Samba)")
    print(f"Consecutive Blocks Matched:  {steps_matched} blocks")
    print(f"Equivalent Instructions:     {total_insts_matched} instructions")
    if diverged:
        print(f"Divergence At:               Block {steps_matched + 1}")
        print(f"Divergence Reason:           {divergence_reason}")
    else:
        print(f"Status:                      100% matched for all {max_blocks} blocks!")

    print("\n--- Latency & Throughput Benchmark ---")
    print(f"Average Block Latency (GPU): {avg_block_lat:.2f} ms per block")
    print(f"Throughput:                  {blocks_per_sec:.1f} blocks / sec")
    print(f"Effective Instruction Rate:  {blocks_per_sec * 9.86:.1f} instructions / sec")
    print(f"Speedup vs. Single Inst:     ~{48.0 / (avg_block_lat / 9.86):.1f}x faster per instruction")
    print("====================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Samba Basic Block Predictor")
    parser.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-blocks", type=int, default=500)
    args = parser.parse_args()

    evaluate_closed_loop(args.bin, args.checkpoint, args.model, max_blocks=args.max_blocks)


if __name__ == "__main__":
    main()
