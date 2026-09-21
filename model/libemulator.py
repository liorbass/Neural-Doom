"""Python ctypes wrapper for libemulator.so (native RV32I C emulator)."""

from __future__ import annotations

import ctypes
import os
import pickle
from pathlib import Path
from typing import Any

from schema import Transition
from block_schema import BlockTransition

ROOT = Path(__file__).resolve().parent.parent
LIB_PATH = ROOT / "emulator" / "libemulator.so"


class _LibEmulator:
    _instance: _LibEmulator | None = None

    def __init__(self, lib_path: Path) -> None:
        if not lib_path.exists():
            raise FileNotFoundError(
                f"libemulator.so not found at {lib_path}. Run 'make -C emulator libemulator.so'."
            )
        self.cdll = ctypes.CDLL(str(lib_path))

        # Define function signatures
        self.cdll.rv32i_create.restype = ctypes.c_void_p
        self.cdll.rv32i_create.argtypes = []

        self.cdll.rv32i_free.restype = None
        self.cdll.rv32i_free.argtypes = [ctypes.c_void_p]

        self.cdll.rv32i_init.restype = None
        self.cdll.rv32i_init.argtypes = [ctypes.c_void_p]

        self.cdll.rv32i_copy.restype = None
        self.cdll.rv32i_copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        self.cdll.rv32i_load_binary.restype = ctypes.c_int
        self.cdll.rv32i_load_binary.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        self.cdll.rv32i_load_symbols.restype = ctypes.c_int
        self.cdll.rv32i_load_symbols.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        self.cdll.rv32i_step.restype = ctypes.c_int
        self.cdll.rv32i_step.argtypes = [ctypes.c_void_p]

        self.cdll.rv32i_run_steps.restype = ctypes.c_uint32
        self.cdll.rv32i_run_steps.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

        self.cdll.rv32i_step_trace.restype = ctypes.c_int
        self.cdll.rv32i_step_trace.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]

        self.cdll.rv32i_decode_prompt.restype = ctypes.c_int
        self.cdll.rv32i_decode_prompt.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]

        self.cdll.rv32i_apply_prediction.restype = None
        self.cdll.rv32i_apply_prediction.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_int,
        ]

        self.cdll.rv32i_save_checkpoint.restype = ctypes.c_int
        self.cdll.rv32i_save_checkpoint.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        self.cdll.rv32i_load_checkpoint.restype = ctypes.c_int
        self.cdll.rv32i_load_checkpoint.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        self.cdll.rv32i_dump_frame.restype = None
        self.cdll.rv32i_dump_frame.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        self.cdll.rv32i_get_pc.restype = ctypes.c_uint32
        self.cdll.rv32i_get_pc.argtypes = [ctypes.c_void_p]

        self.cdll.rv32i_set_pc.restype = None
        self.cdll.rv32i_set_pc.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

        self.cdll.rv32i_get_steps.restype = ctypes.c_uint64
        self.cdll.rv32i_get_steps.argtypes = [ctypes.c_void_p]

        self.cdll.rv32i_set_steps.restype = None
        self.cdll.rv32i_set_steps.argtypes = [ctypes.c_void_p, ctypes.c_uint64]

        self.cdll.rv32i_get_reg.restype = ctypes.c_uint32
        self.cdll.rv32i_get_reg.argtypes = [ctypes.c_void_p, ctypes.c_int]

        self.cdll.rv32i_set_reg.restype = None
        self.cdll.rv32i_set_reg.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]

        self.cdll.rv32i_is_halted.restype = ctypes.c_int
        self.cdll.rv32i_is_halted.argtypes = [ctypes.c_void_p]

        self.cdll.rv32i_set_halted.restype = None
        self.cdll.rv32i_set_halted.argtypes = [ctypes.c_void_p, ctypes.c_int]

        self.cdll.rv32i_get_timer_ms.restype = ctypes.c_uint32
        self.cdll.rv32i_get_timer_ms.argtypes = [ctypes.c_void_p]

        self.cdll.rv32i_set_timer_ms.restype = None
        self.cdll.rv32i_set_timer_ms.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

        self.cdll.rv32i_get_keyboard.restype = ctypes.c_uint32
        self.cdll.rv32i_get_keyboard.argtypes = [ctypes.c_void_p]

        self.cdll.rv32i_set_keyboard.restype = None
        self.cdll.rv32i_set_keyboard.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

        self.cdll.rv32i_get_ram_ptr.restype = ctypes.c_void_p
        self.cdll.rv32i_get_ram_ptr.argtypes = [ctypes.c_void_p]

        self.cdll.rv32i_mem_read.restype = ctypes.c_uint32
        self.cdll.rv32i_mem_read.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int]

        self.cdll.rv32i_mem_write.restype = None
        self.cdll.rv32i_mem_write.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]

        self.cdll.rv32i_step_block.restype = ctypes.c_int
        self.cdll.rv32i_step_block.argtypes = [ctypes.c_void_p, ctypes.c_int]

        self.cdll.rv32i_get_write_reg.restype = ctypes.c_int
        self.cdll.rv32i_get_write_reg.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint32),
        ]

        self.cdll.rv32i_get_write_mem.restype = ctypes.c_int
        self.cdll.rv32i_get_write_mem.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_int),
        ]

    @classmethod
    def get(cls) -> _LibEmulator:
        if cls._instance is None:
            cls._instance = cls(LIB_PATH)
        return cls._instance


class NativeCPU:
    """High-performance RV32I CPU instance backed by libemulator.so."""

    def __init__(self, bin_path: str | Path | None = None, symbols_path: str | Path | None = None) -> None:
        self._lib = _LibEmulator.get()
        self._ptr = self._lib.cdll.rv32i_create()
        if not self._ptr:
            raise MemoryError("Failed to allocate native RV32I CPU")

        if symbols_path:
            self.load_symbols(symbols_path)
        elif bin_path:
            sym = Path(bin_path).with_name("symbols.txt")
            if sym.exists():
                self.load_symbols(sym)

        if bin_path:
            self.load_binary(bin_path)

    def __del__(self) -> None:
        if hasattr(self, "_ptr") and self._ptr:
            self._lib.cdll.rv32i_free(self._ptr)
            self._ptr = None

    def clone(self) -> NativeCPU:
        """Create a deep copy of this CPU state."""
        other = NativeCPU()
        self._lib.cdll.rv32i_copy(other._ptr, self._ptr)
        return other

    def load_binary(self, path: str | Path) -> int:
        n = self._lib.cdll.rv32i_load_binary(self._ptr, str(path).encode("utf-8"))
        if n < 0:
            raise IOError(f"Failed to load binary from {path}")
        return int(n)

    def load_symbols(self, path: str | Path) -> bool:
        ret = self._lib.cdll.rv32i_load_symbols(self._ptr, str(path).encode("utf-8"))
        return bool(ret != 0)

    @property
    def pc(self) -> int:
        return int(self._lib.cdll.rv32i_get_pc(self._ptr))

    @pc.setter
    def pc(self, val: int) -> None:
        self._lib.cdll.rv32i_set_pc(self._ptr, ctypes.c_uint32(val))

    @property
    def steps(self) -> int:
        return int(self._lib.cdll.rv32i_get_steps(self._ptr))

    @steps.setter
    def steps(self, val: int) -> None:
        self._lib.cdll.rv32i_set_steps(self._ptr, ctypes.c_uint64(val))

    def run_steps(self, count: int) -> int:
        """Run up to count instructions in native C loop at maximum speed."""
        return int(self._lib.cdll.rv32i_run_steps(self._ptr, ctypes.c_uint32(count)))

    @property
    def timer_ms(self) -> int:
        return int(self._lib.cdll.rv32i_get_timer_ms(self._ptr))

    @timer_ms.setter
    def timer_ms(self, val: int) -> None:
        self._lib.cdll.rv32i_set_timer_ms(self._ptr, ctypes.c_uint32(val))

    @property
    def keyboard(self) -> int:
        return int(self._lib.cdll.rv32i_get_keyboard(self._ptr))

    @keyboard.setter
    def keyboard(self, val: int) -> None:
        self._lib.cdll.rv32i_set_keyboard(self._ptr, ctypes.c_uint32(val))

    @property
    def halted(self) -> bool:
        return bool(self._lib.cdll.rv32i_is_halted(self._ptr) != 0)

    @halted.setter
    def halted(self, val: bool) -> None:
        self._lib.cdll.rv32i_set_halted(self._ptr, 1 if val else 0)

    @property
    def regs(self) -> list[int]:
        return [self.get_reg(i) for i in range(32)]

    @regs.setter
    def regs(self, values: list[int]) -> None:
        for i, val in enumerate(values[:32]):
            self.set_reg(i, val)

    def get_reg(self, idx: int) -> int:
        return int(self._lib.cdll.rv32i_get_reg(self._ptr, idx))

    def set_reg(self, idx: int, val: int) -> None:
        self._lib.cdll.rv32i_set_reg(self._ptr, idx, ctypes.c_uint32(val))

    def get_ram(self) -> bytes:
        ram_ptr = self._lib.cdll.rv32i_get_ram_ptr(self._ptr)
        ram_size = self._lib.cdll.rv32i_get_ram_size()
        return ctypes.string_at(ram_ptr, ram_size)

    def set_ram(self, data: bytes | bytearray) -> None:
        ram_ptr = self._lib.cdll.rv32i_get_ram_ptr(self._ptr)
        ram_size = self._lib.cdll.rv32i_get_ram_size()
        n = min(len(data), ram_size)
        ctypes.memmove(ram_ptr, bytes(data[:n]), n)

    def mem_read(self, addr: int, size: int) -> int:
        return int(self._lib.cdll.rv32i_mem_read(self._ptr, ctypes.c_uint32(addr), ctypes.c_int(size)))

    def mem_write(self, addr: int, size: int, val: int) -> None:
        self._lib.cdll.rv32i_mem_write(self._ptr, ctypes.c_uint32(addr), ctypes.c_int(size), ctypes.c_uint32(val))

    def step_block(self, max_instructions: int = 32) -> BlockTransition | None:
        """Step instructions until a branch/jump/ecall terminates the basic block.
        Returns BlockTransition or None if halted."""
        if self.halted:
            return None
        start_pc = self.pc
        regs_in = self.regs
        mem_writes: list[list[int]] = []
        inst_count = 0
        curr_pc = start_pc

        m_addr = ctypes.c_uint32()
        m_val = ctypes.c_uint32()
        m_sz = ctypes.c_int()

        ram_ptr = self._lib.cdll.rv32i_get_ram_ptr(self._ptr)

        while not self.halted and inst_count < max_instructions:
            curr_pc = self.pc
            if curr_pc < 0x80000000 or curr_pc + 4 > 0x80000000 + (17 * 1024 * 1024):
                self.halted = True
                break

            offset = curr_pc - 0x80000000
            inst = ctypes.c_uint32.from_address(ram_ptr + offset).value
            if inst == 0:
                self.halted = True
                break

            ok = self._lib.cdll.rv32i_step(self._ptr)
            if not ok:
                break
            inst_count += 1

            if self._lib.cdll.rv32i_get_write_mem(
                self._ptr, ctypes.byref(m_addr), ctypes.byref(m_val), ctypes.byref(m_sz)
            ):
                mem_writes.append([m_addr.value, m_val.value, m_sz.value])

            op = inst & 0x7F
            if op in (0x6F, 0x67, 0x63, 0x73):
                break

        if inst_count == 0:
            return None

        regs_out = self.regs
        reg_writes: dict[int, int] = {}
        for i in range(32):
            if regs_out[i] != regs_in[i]:
                reg_writes[i] = regs_out[i]

        return BlockTransition(
            start_pc=start_pc,
            end_pc=curr_pc,
            next_pc=self.pc,
            instruction_count=inst_count,
            regs_in=regs_in,
            reg_writes=reg_writes,
            mem_writes=mem_writes,
            halted=self.halted,
        )

    def apply_block_prediction(
        self, next_pc: int, reg_writes: dict[int, int], mem_writes: list[list[int]]
    ) -> None:
        """Apply model-predicted basic block next state (next_pc, reg_writes, mem_writes)."""
        self.pc = next_pc
        for rd, val in reg_writes.items():
            if 0 < rd < 32:
                self.set_reg(rd, val)
        for mw in mem_writes:
            if len(mw) >= 3:
                self.mem_write(mw[0], mw[2], mw[1])


    def step(self) -> str | None:
        """Step one instruction and return prompt + => + target, or None if halted."""
        prompt_buf = ctypes.create_string_buffer(1024)
        target_buf = ctypes.create_string_buffer(256)
        ok = self._lib.cdll.rv32i_step_trace(
            self._ptr, prompt_buf, len(prompt_buf), target_buf, len(target_buf)
        )
        if not ok:
            return None
        return (
            prompt_buf.value.decode("utf-8")
            + "\n=>\n"
            + target_buf.value.decode("utf-8")
            + "\n"
        )

    def decode_prompt(self) -> tuple[str, str] | None:
        """Decode prompt without executing. Returns (prompt, mnemonic) or None if halted."""
        prompt_buf = ctypes.create_string_buffer(1024)
        mnem_buf = ctypes.create_string_buffer(32)
        ok = self._lib.cdll.rv32i_decode_prompt(
            self._ptr, prompt_buf, len(prompt_buf), mnem_buf, len(mnem_buf)
        )
        if not ok:
            return None
        return prompt_buf.value.decode("utf-8"), mnem_buf.value.decode("utf-8")

    def apply_prediction(self, pred: Transition, mnemonic: str) -> None:
        """Apply model-predicted next state (NPC, W_REG, W_MEM) to CPU."""
        npc = pred.npc if pred.npc is not None else (self.pc + 4)
        w_reg = pred.w_reg
        rd = w_reg.reg if w_reg is not None else 0
        reg_val = w_reg.value if w_reg is not None else 0

        w_mem = pred.w_mem
        has_mem = 1 if w_mem is not None else 0
        mem_addr = w_mem.addr if w_mem is not None else 0
        mem_val = w_mem.value if w_mem is not None else 0
        mem_size = {"sb": 1, "sh": 2}.get(mnemonic, 4)

        self._lib.cdll.rv32i_apply_prediction(
            self._ptr,
            ctypes.c_uint32(npc),
            ctypes.c_int(rd),
            ctypes.c_uint32(reg_val),
            ctypes.c_int(has_mem),
            ctypes.c_uint32(mem_addr),
            ctypes.c_uint32(mem_val),
            ctypes.c_int(mem_size),
        )

    def save_checkpoint(self, path: str | Path) -> None:
        ret = self._lib.cdll.rv32i_save_checkpoint(self._ptr, str(path).encode("utf-8"))
        if ret != 0:
            raise IOError(f"Failed to save native checkpoint to {path} (error code {ret})")

    def load_checkpoint(self, path: str | Path) -> None:
        p = Path(path)
        if p.suffix == ".pkl":
            # Backward compatibility with legacy pickle checkpoints
            with open(p, "rb") as f:
                ckpt = pickle.load(f)
            self.pc = ckpt["pc"]
            self.steps = ckpt["steps"]
            self.timer_ms = ckpt["timer_ms"]
            self.keyboard = ckpt.get("keyboard", 0)
            self.halted = ckpt.get("halted", False)
            self.regs = ckpt["regs"]
            self.set_ram(ckpt["ram"])
        else:
            ret = self._lib.cdll.rv32i_load_checkpoint(self._ptr, str(p).encode("utf-8"))
            if ret != 0:
                raise IOError(f"Failed to load native checkpoint from {path} (error code {ret})")

    def dump_frame(self, path: str | Path) -> None:
        self._lib.cdll.rv32i_dump_frame(self._ptr, str(path).encode("utf-8"))
