"""Python ctypes wrapper for libemulator_arm32.so."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

from schema import Transition


class _LibEmulatorARM:
    _instance: _LibEmulatorARM | None = None

    def __init__(self, lib_path: Path) -> None:
        self.cdll = ctypes.CDLL(str(lib_path))

        self.cdll.arm32_create.restype = ctypes.c_void_p
        self.cdll.arm32_create.argtypes = []

        self.cdll.arm32_free.restype = None
        self.cdll.arm32_free.argtypes = [ctypes.c_void_p]

        self.cdll.arm32_init.restype = None
        self.cdll.arm32_init.argtypes = [ctypes.c_void_p]

        self.cdll.arm32_step.restype = ctypes.c_int
        self.cdll.arm32_step.argtypes = [ctypes.c_void_p]

        self.cdll.arm32_step_trace.restype = ctypes.c_int
        self.cdll.arm32_step_trace.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]

        self.cdll.arm32_decode_prompt.restype = ctypes.c_int
        self.cdll.arm32_decode_prompt.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]

        self.cdll.arm32_load_binary_data.restype = ctypes.c_int
        self.cdll.arm32_load_binary_data.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]

        self.cdll.arm32_get_pc.restype = ctypes.c_uint32
        self.cdll.arm32_get_pc.argtypes = [ctypes.c_void_p]

        self.cdll.arm32_set_pc.restype = None
        self.cdll.arm32_set_pc.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

        self.cdll.arm32_get_cpsr.restype = ctypes.c_uint32
        self.cdll.arm32_get_cpsr.argtypes = [ctypes.c_void_p]

        self.cdll.arm32_set_cpsr.restype = None
        self.cdll.arm32_set_cpsr.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

        self.cdll.arm32_get_reg.restype = ctypes.c_uint32
        self.cdll.arm32_get_reg.argtypes = [ctypes.c_void_p, ctypes.c_int]

        self.cdll.arm32_set_reg.restype = None
        self.cdll.arm32_set_reg.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]

        self.cdll.arm32_push_key.restype = None
        self.cdll.arm32_push_key.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int]

        self.cdll.arm32_mem_read.restype = ctypes.c_uint32
        self.cdll.arm32_mem_read.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int]

        self.cdll.arm32_mem_write.restype = None
        self.cdll.arm32_mem_write.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]

    @classmethod
    def get(cls) -> _LibEmulatorARM:
        if cls._instance is None:
            here = Path(__file__).resolve().parent
            candidates = [
                here / "../emulator/libemulator_arm32.so",
                here / "libemulator_arm32.so",
            ]
            for p in candidates:
                if p.exists():
                    cls._instance = cls(p.resolve())
                    return cls._instance
            raise FileNotFoundError(
                f"libemulator_arm32.so not found. Checked: {candidates}"
            )
        return cls._instance


class NativeARMCPU:
    """High-performance ARM32 (ARMv4T) CPU instance backed by libemulator_arm32.so."""

    def __init__(self) -> None:
        self._lib = _LibEmulatorARM.get()
        self._ptr = self._lib.cdll.arm32_create()
        if not self._ptr:
            raise MemoryError("Failed to allocate native ARM32 CPU")

    def __del__(self) -> None:
        if hasattr(self, "_ptr") and self._ptr:
            self._lib.cdll.arm32_free(self._ptr)
            self._ptr = None

    @property
    def pc(self) -> int:
        return int(self._lib.cdll.arm32_get_pc(self._ptr))

    @pc.setter
    def pc(self, val: int) -> None:
        self._lib.cdll.arm32_set_pc(self._ptr, ctypes.c_uint32(val))

    @property
    def cpsr(self) -> int:
        return int(self._lib.cdll.arm32_get_cpsr(self._ptr))

    @cpsr.setter
    def cpsr(self, val: int) -> None:
        self._lib.cdll.arm32_set_cpsr(self._ptr, ctypes.c_uint32(val))

    def get_reg(self, idx: int) -> int:
        return int(self._lib.cdll.arm32_get_reg(self._ptr, idx))

    def set_reg(self, idx: int, val: int) -> None:
        self._lib.cdll.arm32_set_reg(self._ptr, idx, ctypes.c_uint32(val))

    def mem_read(self, addr: int, size: int) -> int:
        return int(self._lib.cdll.arm32_mem_read(self._ptr, ctypes.c_uint32(addr), size))

    def mem_write(self, addr: int, size: int, val: int) -> None:
        self._lib.cdll.arm32_mem_write(self._ptr, ctypes.c_uint32(addr), size, ctypes.c_uint32(val))

    def load_binary_data(self, data: bytes) -> int:
        return int(self._lib.cdll.arm32_load_binary_data(self._ptr, data, len(data)))

    def step(self) -> int:
        return int(self._lib.cdll.arm32_step(self._ptr))

    def step_trace(self) -> tuple[str, str] | None:
        p_buf = ctypes.create_string_buffer(1024)
        t_buf = ctypes.create_string_buffer(256)
        ok = self._lib.cdll.arm32_step_trace(
            self._ptr, p_buf, len(p_buf), t_buf, len(t_buf)
        )
        if not ok:
            return None
        return p_buf.value.decode("utf-8"), t_buf.value.decode("utf-8")

    def decode_prompt(self) -> str | None:
        p_buf = ctypes.create_string_buffer(1024)
        m_buf = ctypes.create_string_buffer(32)
        ok = self._lib.cdll.arm32_decode_prompt(
            self._ptr, p_buf, len(p_buf), m_buf, len(m_buf)
        )
        if not ok:
            return None
        return p_buf.value.decode("utf-8")
