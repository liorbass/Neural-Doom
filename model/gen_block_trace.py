#!/usr/bin/env python3
"""Generate Basic Block execution traces from DOOM binary using NativeCPU."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from block_schema import BlockTransition
from libemulator import NativeCPU

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIN = ROOT / "doomgeneric" / "doomgeneric" / "doom_rv32i.bin"


def generate_traces(
    bin_path: Path,
    out_path: Path,
    checkpoint_path: Path | None = None,
    max_blocks: int = 10000,
    max_inst_per_block: int = 32,
    skip_steps: int = 0,
    append: bool = False,
) -> None:
    print(f"Loading binary: {bin_path}")
    cpu = NativeCPU(bin_path)

    if checkpoint_path and str(checkpoint_path).lower() != "none" and checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        cpu.load_checkpoint(checkpoint_path)

    if skip_steps > 0:
        print(f"Fast-forwarding {skip_steps:,} steps before tracing...")
        cpu.run_steps(skip_steps)

    print(f"Tracing up to {max_blocks} basic blocks...")
    t0 = time.perf_counter()

    total_instructions = 0
    blocks_collected = 0
    unique_pcs: set[int] = set()

    mode = "a" if append else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        while blocks_collected < max_blocks and not cpu.halted:
            block = cpu.step_block(max_instructions=max_inst_per_block)
            if block is None:
                break

            total_instructions += block.instruction_count
            unique_pcs.add(block.start_pc)
            blocks_collected += 1

            # Serialize transition to JSONL
            line = block.model_dump_json()
            f.write(line + "\n")

            if blocks_collected % 2000 == 0:
                elapsed = time.perf_counter() - t0
                mips = (total_instructions / elapsed) / 1_000_000
                print(
                    f"  Collected {blocks_collected}/{max_blocks} blocks "
                    f"({total_instructions} insts, {mips:.2f} MIPS, "
                    f"avg {total_instructions / blocks_collected:.1f} inst/block)"
                )

    elapsed = time.perf_counter() - t0
    print("\n--- Trace Generation Complete ---")
    print(f"Total Basic Blocks:       {blocks_collected}")
    print(f"Total CPU Instructions:   {total_instructions}")
    print(f"Unique Basic Block PCs:   {len(unique_pcs)}")
    print(f"Average Block Length:     {total_instructions / max(1, blocks_collected):.2f} instructions")
    print(f"Wall Time:                {elapsed:.2f} seconds ({blocks_collected / elapsed:.1f} blocks/sec)")
    print(f"Output saved to:          {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DOOM Basic Block Traces")
    parser.add_argument("--bin", type=Path, default=DEFAULT_BIN, help="Path to doom_rv32i.bin")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data" / "checkpoint.bin", help="Path to checkpoint.bin")
    parser.add_argument("--out", type=Path, default=ROOT / "model" / "doom_blocks.jsonl", help="Output JSONL path")
    parser.add_argument("--blocks", type=int, default=10000, help="Number of basic blocks to trace")
    parser.add_argument("--max-inst", type=int, default=32, help="Max instructions per block")
    parser.add_argument("--skip-steps", type=int, default=0, help="Number of CPU steps to fast-forward before tracing")
    parser.add_argument("--append", action="store_true", help="Append to out file instead of overwriting")
    args = parser.parse_args()

    generate_traces(
        args.bin,
        args.out,
        checkpoint_path=args.checkpoint,
        max_blocks=args.blocks,
        max_inst_per_block=args.max_inst,
        skip_steps=args.skip_steps,
        append=args.append,
    )


if __name__ == "__main__":
    main()
