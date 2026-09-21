#!/usr/bin/env python3
"""
Diverse synthetic RV32I transition dataset (same text format as emulator.c).

Random instructions, PCs, and operand values so the model must learn the
actual ISA semantics instead of memorizing one DOOM trace. Loads include
the memory value ([MEM addr] raw) so the transition is a pure function of
the input.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Callable

from schema import Pair

MASK = 0xFFFFFFFF
R: list[str] = [f"x{i}" for i in range(32)]

GenFunc = Callable[[random.Random, int], tuple[str, str]]


def sign_extend(v: int, bits: int) -> int:
    if v & (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def r32(rng: random.Random, kind: str = "any") -> int:
    r = rng.random()
    if r < 0.55 or kind == "full":
        return rng.getrandbits(32)
    if r < 0.75:
        return rng.randrange(0, 4096)
    if r < 0.88:  # sign boundaries
        return rng.choice([0x7FFFFFFF, 0x80000000, 0xFFFFFFFF,
                           0xFFFFFFF0 + rng.randrange(16),
                           rng.randrange(16), 0x10000, 0xFFFF0000])
    return 0x80000000 + rng.randrange(0x400000)  # DOOM-ish pointer


def reg(rng: random.Random) -> int:
    return rng.randrange(32)


def rval(rng: random.Random, r: int) -> int:
    """Register source value: x0 is hardwired to 0 (matches tracer)."""
    return 0 if r == 0 else r32(rng)


def imm12(rng: random.Random) -> int:
    return rng.randrange(-256, 256) if rng.random() < 0.7 \
        else rng.randrange(-2048, 2048)


def pc_gen(rng: random.Random) -> int:
    return 0x80000000 + (rng.randrange(0x100000) << 2)


def gen_r(rng: random.Random, pc: int) -> tuple[str, str]:
    m = rng.choice(["add", "sub", "sll", "slt", "sltu", "xor",
                    "srl", "sra", "or", "and"])
    rd, rs1, rs2 = reg(rng), reg(rng), reg(rng)
    a, b = rval(rng, rs1), rval(rng, rs2)
    sa, sb = sign_extend(a, 32), sign_extend(b, 32)
    if m == "add": res = a + b
    elif m == "sub": res = a - b
    elif m == "sll": res = a << (b & 0x1F)
    elif m == "slt": res = 1 if sa < sb else 0
    elif m == "sltu": res = 1 if a < b else 0
    elif m == "xor": res = a ^ b
    elif m == "srl": res = a >> (b & 0x1F)
    elif m == "sra": res = sa >> (b & 0x1F)
    elif m == "or": res = a | b
    else: res = a & b
    res &= MASK
    state = (f"[CMD] STEP\n[PC] {pc:08x}\n[INST] {m} {R[rd]}, {R[rs1]}, "
             f"{R[rs2]}\n[REG {R[rs1]}] {a:08x}\n[REG {R[rs2]}] {b:08x}")
    return state, f"[NPC] {pc+4:08x}\n[W_REG {R[rd]}] {res:08x}"


def gen_i(rng: random.Random, pc: int) -> tuple[str, str]:
    m = rng.choice(["addi", "slti", "sltiu", "xori", "ori", "andi",
                    "slli", "srli", "srai"])
    rd, rs1 = reg(rng), reg(rng)
    a = rval(rng, rs1)
    sa = sign_extend(a, 32)
    if m in ("slli", "srli", "srai"):
        sh = rng.randrange(32)
        imm_str = str(sh)
        if m == "slli": res = a << sh
        elif m == "srli": res = a >> sh
        else: res = sa >> sh
    else:
        imm = imm12(rng)
        imm_str = str(imm)
        if m == "addi": res = a + imm
        elif m == "slti": res = 1 if sa < imm else 0
        elif m == "sltiu": res = 1 if (a & MASK) < (imm & MASK) else 0
        elif m == "xori": res = a ^ (imm & MASK)
        elif m == "ori": res = a | (imm & MASK)
        else: res = a & (imm & MASK)
    res &= MASK
    state = (f"[CMD] STEP\n[PC] {pc:08x}\n[INST] {m} {R[rd]}, {R[rs1]}, "
             f"{imm_str}\n[REG {R[rs1]}] {a:08x}")
    return state, f"[NPC] {pc+4:08x}\n[W_REG {R[rd]}] {res:08x}"


def gen_load(rng: random.Random, pc: int) -> tuple[str, str]:
    m = rng.choice(["lb", "lh", "lw", "lbu", "lhu"])
    rd, rs1 = reg(rng), reg(rng)
    base = rval(rng, rs1)
    imm = imm12(rng)
    addr = (base + imm) & MASK
    if m in ("lb", "lbu"):
        raw = rng.getrandbits(8)
        val = sign_extend(raw, 8) & MASK if m == "lb" else raw
    elif m in ("lh", "lhu"):
        raw = rng.getrandbits(16)
        val = sign_extend(raw, 16) & MASK if m == "lh" else raw
    else:
        raw = rng.getrandbits(32)
        val = raw
    state = (f"[CMD] STEP\n[PC] {pc:08x}\n[INST] {m} {R[rd]}, {imm}({R[rs1]})"
             f"\n[REG {R[rs1]}] {base:08x}\n[MEM {addr:08x}] {raw:08x}")
    return state, f"[NPC] {pc+4:08x}\n[W_REG {R[rd]}] {val:08x}"


def gen_store(rng: random.Random, pc: int) -> tuple[str, str]:
    m = rng.choice(["sb", "sh", "sw"])
    rs1, rs2 = reg(rng), reg(rng)
    base, val = rval(rng, rs1), rval(rng, rs2)
    imm = imm12(rng)
    addr = (base + imm) & MASK
    state = (f"[CMD] STEP\n[PC] {pc:08x}\n[INST] {m} {R[rs2]}, {imm}({R[rs1]})"
             f"\n[REG {R[rs1]}] {base:08x}\n[REG {R[rs2]}] {val:08x}")
    return state, f"[NPC] {pc+4:08x}\n[W_MEM {addr:08x}] {val:08x}"


def gen_branch(rng: random.Random, pc: int) -> tuple[str, str]:
    m = rng.choice(["beq", "bne", "blt", "bge", "bltu", "bgeu"])
    rs1, rs2 = reg(rng), reg(rng)
    a, b = rval(rng, rs1), rval(rng, rs2)
    if rng.random() < 0.5:  # bias toward taken on purpose
        b = a
    sa, sb = sign_extend(a, 32), sign_extend(b, 32)
    imm = (rng.randrange(-256, 256) if rng.random() < 0.7
           else rng.randrange(-2048, 2048)) & ~1
    if m == "beq": take = a == b
    elif m == "bne": take = a != b
    elif m == "blt": take = sa < sb
    elif m == "bge": take = sa >= sb
    elif m == "bltu": take = a < b
    else: take = a >= b
    npc = ((pc + imm) if take else pc + 4) & MASK
    state = (f"[CMD] STEP\n[PC] {pc:08x}\n[INST] {m} {R[rs1]}, {R[rs2]}, "
             f"{imm}\n[REG {R[rs1]}] {a:08x}\n[REG {R[rs2]}] {b:08x}")
    return state, f"[NPC] {npc:08x}"


def gen_jal(rng: random.Random, pc: int) -> tuple[str, str]:
    rd = reg(rng)
    imm = (rng.randrange(-4096, 4096) if rng.random() < 0.7
           else rng.randrange(-524288, 524288)) & ~1
    npc = (pc + imm) & MASK
    state = f"[CMD] STEP\n[PC] {pc:08x}\n[INST] jal {R[rd]}, {imm}"
    return state, f"[NPC] {npc:08x}\n[W_REG {R[rd]}] {(pc+4) & MASK:08x}"


def gen_jalr(rng: random.Random, pc: int) -> tuple[str, str]:
    rd, rs1 = reg(rng), reg(rng)
    base = rval(rng, rs1)
    imm = imm12(rng)
    npc = ((base + imm) & ~1) & MASK
    state = (f"[CMD] STEP\n[PC] {pc:08x}\n[INST] jalr {R[rd]}, {R[rs1]}, "
             f"{imm}\n[REG {R[rs1]}] {base:08x}")
    return state, f"[NPC] {npc:08x}\n[W_REG {R[rd]}] {(pc+4) & MASK:08x}"


def gen_lui(rng: random.Random, pc: int) -> tuple[str, str]:
    rd = reg(rng)
    imm = rng.getrandbits(20) << 12
    state = f"[CMD] STEP\n[PC] {pc:08x}\n[INST] lui {R[rd]}, {imm:08x}"
    return state, f"[NPC] {pc+4:08x}\n[W_REG {R[rd]}] {imm:08x}"


def gen_auipc(rng: random.Random, pc: int) -> tuple[str, str]:
    rd = reg(rng)
    imm = rng.getrandbits(20) << 12
    res = (pc + imm) & MASK
    state = f"[CMD] STEP\n[PC] {pc:08x}\n[INST] auipc {R[rd]}, {imm:08x}"
    return state, f"[NPC] {pc+4:08x}\n[W_REG {R[rd]}] {res:08x}"


def main() -> None:
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 450000
    out = sys.argv[2] if len(sys.argv) > 2 else str(root / "data" / "dataset_synthetic.jsonl")
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    rng = random.Random(seed)
    gens: list[tuple[GenFunc, float]] = [
        (gen_r, 0.26), (gen_i, 0.20), (gen_load, 0.20),
        (gen_store, 0.10), (gen_branch, 0.14), (gen_jal, 0.03),
        (gen_jalr, 0.03), (gen_lui, 0.02), (gen_auipc, 0.02),
    ]
    funcs: list[GenFunc] = [f for f, _ in gens]
    weights: list[float] = [w for _, w in gens]
    max_len = 0
    with open(out, "w") as fh:
        for _ in range(n):
            fn = rng.choices(funcs, weights)[0]
            state, nxt = fn(rng, pc_gen(rng))
            text = state + "\n=>\n" + nxt
            max_len = max(max_len, len(text))
            fh.write(Pair(state=state, next=nxt).model_dump_json() + "\n")
    print(f"wrote {n} pairs to {out}, max text len {max_len}")


if __name__ == "__main__":
    main()
