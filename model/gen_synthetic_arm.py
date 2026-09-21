#!/usr/bin/env python3
"""
Diverse synthetic ARM32 (ARMv4T) transition dataset.

Generates random valid ARM32 instructions, sets up random initial register and
CPSR flag states, executes them through NativeARMCPU, and records prompt-target pairs.
"""

from __future__ import annotations

import argparse
import json
import random
import struct
import sys
from pathlib import Path

from libemulator_arm import NativeARMCPU
from schema import Pair

MASK32 = 0xFFFFFFFF


def rand_val(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.5:
        return rng.getrandbits(32)
    if r < 0.75:
        return rng.randrange(0, 4096)
    if r < 0.90:
        return rng.choice([0, 1, 2, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xFFFF0000, 0x10000])
    return rng.randrange(0x1000, 0x10000)


def rand_cpsr(rng: random.Random) -> int:
    # Set random combinations of N, Z, C, V flags
    n = rng.choice([0, 1]) << 31
    z = rng.choice([0, 1]) << 30
    c = rng.choice([0, 1]) << 29
    v = rng.choice([0, 1]) << 28
    mode = 0x10  # User mode
    return n | z | c | v | mode


def gen_arm32_inst(rng: random.Random) -> int:
    # Pick a condition code (favor AL, but include all conditions)
    if rng.random() < 0.35:
        cond = 14  # AL
    else:
        cond = rng.randrange(15)  # 0..14

    category = rng.random()

    # Category 1: ALU Data Processing (65%)
    if category < 0.65:
        opcode = rng.randrange(16)
        s_bit = rng.choice([0, 1])
        rn = rng.randrange(14)
        rd = rng.randrange(14)

        # Immediate vs Register shift
        if rng.random() < 0.5:
            # Immediate operand2
            imm8 = rng.randrange(256)
            rot = rng.randrange(16)
            inst = (cond << 28) | (1 << 25) | (opcode << 21) | (s_bit << 20) | (rn << 16) | (rd << 12) | (rot << 8) | imm8
        else:
            # Register shifted operand2
            rm = rng.randrange(14)
            shift_type = rng.randrange(4)  # 0=LSL, 1=LSR, 2=ASR, 3=ROR
            shift_imm = rng.randrange(32)
            inst = (cond << 28) | (0 << 25) | (opcode << 21) | (s_bit << 20) | (rn << 16) | (rd << 12) | (shift_imm << 7) | (shift_type << 5) | (0 << 4) | rm
        return inst & MASK32

    # Category 2: Multiply / MLA (15%)
    elif category < 0.80:
        a_bit = rng.choice([0, 1])
        s_bit = rng.choice([0, 1])
        rd = rng.randrange(14)
        rn = rng.randrange(14)
        rs = rng.randrange(14)
        rm = rng.randrange(14)
        inst = (cond << 28) | (a_bit << 21) | (s_bit << 20) | (rd << 16) | (rn << 12) | (rs << 8) | (9 << 4) | rm
        return inst & MASK32

    # Category 3: Single Data Transfer LDR / STR (20%)
    else:
        l_bit = rng.choice([0, 1])
        b_bit = rng.choice([0, 1])
        rn = rng.randrange(14)
        rd = rng.randrange(14)
        offset12 = rng.randrange(0, 128) << 2
        inst = (cond << 28) | (1 << 26) | (1 << 24) | (1 << 23) | (b_bit << 22) | (l_bit << 20) | (rn << 16) | (rd << 12) | offset12
        return inst & MASK32


def generate_pair(cpu: NativeARMCPU, rng: random.Random) -> Pair | None:
    # Setup random CPU state
    pc = (rng.randrange(0x1000, 0x100000)) & ~3
    cpu.pc = pc
    cpu.cpsr = rand_cpsr(rng)

    for i in range(14):
        cpu.set_reg(i, rand_val(rng))
    cpu.set_reg(13, 0x01000000)  # SP
    cpu.set_reg(14, 0x00000000)  # LR

    inst = gen_arm32_inst(rng)
    code_bytes = struct.pack("<I", inst)

    # Write instruction at PC
    cpu.mem_write(pc, 4, inst)

    # In case instruction is a load, ensure target memory address is initialized
    rn = (inst >> 16) & 0xF
    base_addr = cpu.get_reg(rn)
    if base_addr < 0x02000000 - 0x1000:
        cpu.mem_write(base_addr, 4, rand_val(rng))

    trace = cpu.step_trace()
    if not trace:
        return None

    prompt, target = trace
    return Pair(state=prompt, next=target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ARM32 instruction dataset")
    parser.add_argument("--count", type=int, default=1000, help="Number of pairs to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="", help="Output JSONL path (or stdout if empty)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cpu = NativeARMCPU()

    out_file = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout

    generated = 0
    while generated < args.count:
        pair = generate_pair(cpu, rng)
        if pair:
            out_file.write(pair.model_dump_json() + "\n")
            generated += 1

    if args.output:
        out_file.close()
        print(f"Generated {generated} ARM32 synthetic pairs to {args.output}")


if __name__ == "__main__":
    main()
