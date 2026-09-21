"""Shared data schemas and types for neural-doom."""

from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel, Field, field_validator

MASK32 = 0xFFFFFFFF

REG_NAMES: list[str] = [f"x{i}" for i in range(32)]
REG_IDX: dict[str, int] = {n: i for i, n in enumerate(REG_NAMES)}
REG_IDX.update({"zero": 0, "ra": 1, "sp": 2, "s0": 8})

ARM32_REG_NAMES: list[str] = [f"r{i}" for i in range(16)]
REG_IDX.update({n: i for i, n in enumerate(ARM32_REG_NAMES)})
REG_IDX.update({"lr": 14, "pc": 15})

_HEX_WORD = re.compile(r"^[0-9a-fA-F]{1,8}$")


def sign_extend(value: int, bits: int) -> int:
    """Sign-extend `value` from `bits` width to Python int."""
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)


def hex_word(v: int | str) -> int:
    """Accept an int, or a hex word validated against the trace format."""
    if isinstance(v, str):
        if not _HEX_WORD.fullmatch(v):
            raise ValueError(f"not a hex word (1-8 digits): {v!r}")
        return int(v, 16)
    return v


class Pair(BaseModel):
    """One training record: pre-execution state -> post-execution state."""
    state: str
    next: str


class RegWrite(BaseModel):
    """A parsed `[W_REG <name>] <hex>` line."""
    reg: int = Field(ge=-1, le=31)  # -1 = unrecognized reg name
    value: int = Field(ge=0, le=MASK32)

    @field_validator("value", mode="before")
    @classmethod
    def _value_from_hex(cls, v: int | str) -> int:
        return hex_word(v)


class MemWrite(BaseModel):
    """A parsed `[W_MEM <hex>] <hex>` line."""
    addr: int = Field(ge=0, le=MASK32)
    value: int = Field(ge=0, le=MASK32)

    @field_validator("addr", "value", mode="before")
    @classmethod
    def _from_hex(cls, v: int | str) -> int:
        return hex_word(v)


class Transition(BaseModel):
    """Parsed transition block: [NPC] + optional [W_REG] + optional [W_MEM] + optional [W_CPSR]."""
    npc: int | None = Field(default=None, ge=0, le=MASK32)
    w_reg: RegWrite | None = None
    w_mem: MemWrite | None = None
    w_cpsr: int | None = Field(default=None, ge=0, le=MASK32)

    @field_validator("npc", "w_cpsr", mode="before")
    @classmethod
    def _hex_or_none(cls, v: int | str | None) -> int | None:
        return None if v is None else hex_word(v)

    @classmethod
    def parse(cls, text: str) -> Transition:
        npc: str | None = None
        w_reg: RegWrite | None = None
        w_mem: MemWrite | None = None
        w_cpsr: str | None = None
        for line in text.splitlines():
            m = re.match(r"\[NPC\] ([0-9a-fA-F]{1,8})", line)
            if m:
                npc = m.group(1)
                continue
            m = re.match(r"\[W_REG (\w+)\] ([0-9a-fA-F]{1,8})", line)
            if m:
                w_reg = RegWrite(reg=REG_IDX.get(m.group(1), -1),
                                 value=hex_word(m.group(2)))
                continue
            m = re.match(r"\[W_MEM ([0-9a-fA-F]{1,8})\] ([0-9a-fA-F]{1,8})", line)
            if m:
                w_mem = MemWrite(addr=hex_word(m.group(1)),
                                 value=hex_word(m.group(2)))
                continue
            m = re.match(r"\[W_CPSR\] ([0-9a-fA-F]{1,8})", line)
            if m:
                w_cpsr = m.group(1)
        return cls(npc=None if npc is None else hex_word(npc),
                   w_reg=w_reg, w_mem=w_mem,
                   w_cpsr=None if w_cpsr is None else hex_word(w_cpsr))
