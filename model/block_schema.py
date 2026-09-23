"""Data schemas and types for Basic Block prediction in DOOM."""

from __future__ import annotations

import re
from typing import NamedTuple
from pydantic import BaseModel, Field, field_validator

from schema import MASK32, hex_word, MemWrite


class BasicBlockInfo(BaseModel):
    """Static metadata for a basic block in the binary."""
    bb_id: int = Field(ge=0, le=MASK32)
    start_pc: int = Field(ge=0, le=MASK32)
    end_pc: int = Field(ge=0, le=MASK32)
    instruction_count: int = Field(ge=1)
    opcodes: list[int] = Field(default_factory=list)


class BlockRegDelta(NamedTuple):
    """Register write occurring within a basic block."""
    reg: int
    val: int


class BlockMemWrite(NamedTuple):
    """Memory write occurring within a basic block."""
    addr: int
    val: int
    size: int


class BlockTransition(BaseModel):
    """Dynamic execution trace of a single basic block step."""
    start_pc: int = Field(ge=0, le=MASK32)
    end_pc: int = Field(ge=0, le=MASK32)
    next_pc: int = Field(ge=0, le=MASK32)
    instruction_count: int = Field(ge=1)
    # Registers before execution (16 on ARM32 traces, 32 on RV32I; the
    # trainer zero-pads to the 32-wide feature vector)
    regs_in: list[int] = Field(min_length=16, max_length=32)
    # Register modifications during block: {reg_idx: new_val}
    reg_writes: dict[int, int] = Field(default_factory=dict)
    # Memory writes during block: list of (addr, val, size)
    mem_writes: list[list[int]] = Field(default_factory=list)
    halted: bool = False

    @field_validator("start_pc", "end_pc", "next_pc", mode="before")
    @classmethod
    def _validate_hex_pc(cls, v: int | str) -> int:
        return hex_word(v)
