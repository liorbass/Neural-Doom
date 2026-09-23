# Architecture

End-to-end view of neural-doom: how the bare-metal port, the RV32I
execution engines, and the neural prediction models fit together, and the contract
that keeps them in sync. For the exact text and tensor formats the model sees, see
[data_format.md](data_format.md); for how the numbers are measured, see
[evaluation.md](evaluation.md).

## Pipeline at a glance

```
setup_env.sh
   │  1. fetch pinned RISC-V toolchain (xpack riscv-none-elf-gcc 15.2.0-1)
   │  2. clone doomgeneric @ dcb7a8d (GPLv2 Chocolate Doom sources)
   │  3. copy port/ files into the build dir
   │  4. download shareware doom1.wad (archive.org)
   │  5. make        -> doom_rv32i.elf / .bin  (rv32i, ilp32, freestanding)
   │  6. make wad    -> truncate .bin to 10 MiB, append doom1.wad @ 0x80A00000
   │  7. make symbols-> symbols.txt (guest global addresses)
   v
doomgeneric/doomgeneric/doom_rv32i.bin   (code + appended WAD, ~14 MiB)
   |
   +-------------------------------------+-----------------------------+
   v                                                                   v
 emulator/ (rv32i.c, emulator.c, libemulator.so)                     the model
 Native C RV32I engine (~26–240M steps/s)                             trained on (state -> next)
 - boots the .bin                                                      pairs produced by gen_synthetic
 - high-speed JSONL dataset tracer (--dataset)                         and emulator
 - binary checkpointing (--checkpoint-at)
 - interactive browser GUI (--gui)
 - ctypes bindings for model/libemulator.py
```

The native C execution engine implements the authoritative RV32I semantics and MMIO map. Through `libemulator.so` and Python `ctypes`, evaluation harnesses (`eval_playable.py`) interact directly with native CPU instances without maintaining a duplicate Python emulator.

## The core components

| Piece | Lives in | Role |
|---|---|---|
| Bare-metal port | `port/` + `setup_env.sh` | Compiles doomgeneric to pure RV32I (no OS, no drivers). MMIO keyboard/timer/console + VRAM at fixed addresses. Appends the WAD to the binary. |
| Native execution engine | `emulator/` + `model/libemulator.py` | Single authoritative RV32I CPU in C: standalone CLI (`emulator`), browser GUI, dataset tracer, and shared library (`libemulator.so`) with Python ctypes bindings. |
| ARM32 (ARMv4T) port | `port/Makefile.arm32` + `emulator/arm32.c` + `emulator/emulator_arm32.c` + `web/arm32.js` | Second bare-metal target: doomgeneric compiled to ARMv4T (`doom_arm32.bin`, WAD appended at the 10 MB mark) and an ARM32 interpreter core (native harness + WASM). The web player's architecture selector loads this binary onto the ARM32 core. Plain interpreter (~127 M steps/s native, ~115 MIPS WASM) with its own trained Mamba-Nano head (ARM block traces, strided across the level) plus a confidence-gated bimodal counter: ~98.6% live branch accuracy; Neural Mode selectable on ARM32 too. |
| In-browser neural engine | `web/neural_runner.js` + `web/index.html` + `emulator/rv32i.c` | Client-side neural execution engine with 3-tier acceleration: (1) Embedded C/WASM Mamba-Nano kernel (~15µs, 70k+ blk/s), (2) Dynamic Mega-Block Chaining (128 insts/dispatch, 10.25+ MIPS, 35.0 DOOM FPS), and (3) Asynchronous zero-delay `MessageChannel` loop with decoupled 60 Hz vsync `requestAnimationFrame` canvas presentation. Includes speculative branch prediction with verified retirement. |
| Model & evaluation | `model/` | Mamba-Nano Selective SSM and baseline char-level models, dataset generators, and eval harnesses. |

## The shared-semantics contract

The tracer and the emulator stay in lockstep because they share:

1. **The RV32I subset** - exactly the instructions in [ISA subset](#isa-subset).
2. **The memory map / MMIO protocol** - see [Memory map](#memory-map).
3. **`symbols.txt`** - guest addresses of `gametic`, `pagetic`, `gamestate`,
   `gameaction`, `itemOn`, `menuactive`, produced by `make symbols`
   (`riscv-none-elf-nm | awk`). Both engines read it next to the `.bin` and warn
   if it is stale or missing. The tracer keeps fallback addresses only for the
   reference build; nothing is hardcoded across a rebuild.

The model never sees raw bytes - it sees the *textual* state encoding produced
by this contract. The two eval harnesses parse the model's text output back into
`(NPC, W_REG, W_MEM)` and apply it to a real CPU (`apply_prediction` in
`eval_playable.py`).

## Boot flow (guest)

`start.s::_start`: set `gp`, set `sp = __stack_top` (`0x80F00000`), zero BSS,
`call main`. `main` (in `doomgeneric_rv32i.c`) runs
`doomgeneric_Create(4, {"doom","-iwad","doom1.wad","-nofullscreen"})` then loops
`doomgeneric_Tick()` forever. libc (printf/malloc/fopen) comes from newlib via
the syscall shims in `syscalls.c`:

- `_sbrk` bumps the heap up to `__heap_end` (`0x80A00000`).
- `_write` routes printf to the MMIO debug console (`0xFFFF0008`).
- `_open` / `_read` / `_lseek` serve the WAD as a flat blob starting at
  `0x80A00000` (fd 3); WAD size is read from its own infotable header.
- `_exit` spins `ecall`, which both engines treat as a halt.

`DG_Init` redirects DOOM's framebuffer pointer straight at the VRAM region
(`0x81000000`); `DG_DrawFrame` is a no-op (the engines read VRAM directly), and
`DG_GetTicksMs` reads the MMIO timer.

## ISA subset

Implemented identically in `rv32i.c`, `emulator.c`, the
decoders in `eval_playable.py` (`decode_prompt` / `parse_next`), and the
generators in `gen_synthetic.py`:

- **R-type**: `add sub sll slt sltu xor srl sra or and`
- **I-type ALU**: `addi slti sltiu xori ori andi slli srli srai`
- **Loads**: `lb lh lw lbu lhu`
- **Stores**: `sb sh sw`
- **Branches**: `beq bne blt bge bltu bgeu`
- **Jumps**: `jal jalr`
- **Upper-immediate**: `lui auipc`
- **System**: `ecall` (halt)

x0 is hardwired to 0. Shifts mask the shift amount to 5 bits. Signed
comparisons (`slt`, `blt`, `bge`, `sra`, `srai`, loads `lb/lh`) use
sign-extended 32-bit values; unsigned ones use the raw 32-bit value. There are no
M/A/F/C extensions - DOOM runs on the pure integer base ISA (`-march=rv32i`).

## Memory map

Authoritative source: `port/link.ld`, `port/syscalls.c`,
`port/doomgeneric_rv32i.c`, and the `wad`/`symbols` make targets.

| Address | Region | Notes |
|---|---|---|
| `0x80000000` | code / rodata / data / bss | `.text.init` first (entry `_start`) |
| up to `0x80A00000` | heap | `_sbrk` bumps up to `__heap_end = 0x80A00000` |
| `0x80A00000` | `doom1.wad` | appended by `make wad` (binary truncated to 10 MiB first) |
| `0x80F00000` | stack top | grows down (`__stack_top`) |
| `0x81000000` | VRAM | 320x200 RGBA32 (256 KiB); `DG_ScreenBuffer` points here |
| `0xFFFF0000` | MMIO keyboard | write: `bit8 = pressed \| doomkey`; read: last event, read-to-clear |
| `0xFFFF0004` | MMIO timer | millisecond counter; +1 every 10 000 retired instructions |
| `0xFFFF0008` | MMIO console | `printf` chars (`_write`) |

The linker describes a flat 16 MiB RAM region (`0x80000000`-`0x81000000`); the
two engines back `0x80000000`-`0x81100000` (17 MiB) so VRAM at `0x81000000` is
reachable by a simple offset into their backing array.

## The virtual clock

The guest reads wall time from `DG_GetTicksMs()` -> `IO_TIMER_PORT`. Both engines
advance that timer deterministically: +1 ms per 10 000 retired instructions
(`TIMER_STEPS_PER_MS = 10000` in the tracer; `steps / 10000` in the emulator).
DOOM's menu navigation and the scripted `--input` schedule are therefore
expressed in retired-instruction counts - the `--input-start` and
`--input-every` timing knobs are step counts, not wall-clock ms.

## Keyed input

`DG_GetKey` reads the keyboard MMIO port and returns `0` when empty. The tracer
queues scheduled events from `--input 'key:down|up,...'` (event *i* fires at
`--input-start + i * --input-every`) and delivers one only when its step has
been reached (and, optionally, a `gs<N>` gamestate condition holds - e.g.
`UP:down:gs0`). The native emulator keeps the same schedule hardcoded. Reads
are read-to-clear, and the guest pumps the port until it reads `0` each tic,
so no queued event is lost.

## In-Browser Neural Engine & 3-Tier Optimization Pipeline

To run DOOM in real time purely inside the browser, the execution engine combines an embedded Selective State Space Model (Mamba-Nano) with a 3-tier systems optimization pipeline:

### 1. Tier 1: In-Engine Embedded C/WASM Kernel (`rv32i.c` / `rv32i.wasm`)
Rather than passing tensors between JavaScript and WebAssembly on every block, Mamba-Nano's forward pass is implemented as a hand-vectorized C kernel compiled directly into `rv32i.wasm` (`rv32i_step_superblock_neural`):
- **Zero FFI Boundary Overhead**: Reads CPU registers and guest RAM directly from WebAssembly linear memory.
- **Zero GC Spikes**: All recurrent state vectors ($h_t \in \mathbb{R}^{D \times 16}$) and scratch buffers are pre-allocated in linear memory.
- **Ultra-Low Latency**: ~15 µs per basic block evaluation (~70,000 blocks/sec).

### 2. Tier 2: Dynamic Mega-Block Chaining
Programs exhibit long sequences of deterministic arithmetic, memory loads/stores, and unrolled loops. The engine chains consecutive deterministic instructions up to **128 instructions per dispatch**:
- Execution halts only when reaching a conditional branch instruction (`BEQ`, `BNE`, `BLT`, `BGE`, `BLTU`, `BGEU`).
- The neural network is consulted for branch prediction (`branch_prob > 0.5` -> TAKEN).
- Pushes effective instruction throughput to **10.25+ MIPS**, enabling DOOM to run at full native **35.0 FPS**.

### 3. Tier 3: Asynchronous MessageChannel & Decoupled 60 Hz Vsync
- **Zero-Delay Macro-Task Loop**: Uses a browser `MessageChannel` (`port1.onmessage` -> `port2.postMessage(null)`) for continuous execution without `setTimeout` or `setInterval` clamping.
- **Decoupled Vsync Presentation**: Canvas rendering (`ctx.putImageData`) is called exclusively within `requestAnimationFrame(mainLoop)` at native 60 Hz vsync. This eliminates browser compositor starvation and frame drops.
- **Smoothed Telemetry**: Decoupled DOOM game engine tic rate (`curGametic` delta, 35.0 FPS) from physical screen refresh rate (60 Hz), filtered through a 1.0-second rolling sliding window and Exponential Moving Average (EMA $\alpha = 0.25$) to eliminate quantization jitter.

### 4. Speculative Branch Retirement
The neural network predicts branch direction (`TAKEN` vs. `FALLTHROUGH`). To prevent cascading arithmetic divergence over millions of instructions, the CPU verifies the decision against ground-truth register values before committing state (modeled after modern out-of-order CPU branch prediction). This allows DOOM to run indefinitely while accurately measuring true neural accuracy (~80–95%).

## First-level-completion demo build

`make demo` builds a second binary (`doom_rv32i_demo.bin`) that proves the
*platform* completes E1M1: `port/wadadd.py` appends the recorded vanilla 1.9
playthrough `port/e1m1uv.lmp` to the WAD as lump `E1M1DEMO`, and
`doomgeneric_rv32i.c` is compiled with `-DPLAY_DEMO`, adding
`-playdemo e1m1demo` to the guest argv. `-playdemo` first tries to open
`e1m1demo.lmp` as a file; the guest filesystem (`syscalls.c`) already has its
single fd on the WAD, so the open fails and chocolate-doom falls back to the
WAD-lump lookup (the same trick that makes vanilla `-playdemo demo1` work).
`-playdemo` sets `singledemo`, which makes key events unable to skip the demo
and quits the guest (`ecall` spin) when it ends. Both engines print
`[GS] gamestate=1` at the exit (gametic 378, matching the recorded 0:10.80).
The default binary is byte-compatible in behavior with the training
pipeline; see the README's "First-level completion" section for the exact
run command.

## File-level index

| Path | What it is |
|---|---|
| `port/start.s` | Reset entry: gp/sp setup, BSS zero, `call main`, halt loop on `ecall` |
| `port/link.ld` | Memory regions, heap/stack symbols, section placement |
| `port/syscalls.c` | Newlib syscall layer (`_sbrk`, `_write`, WAD-as-fd-3 file ops, `_exit`) |
| `port/doomgeneric_rv32i.c` | doomgeneric platform port: `DG_*` impls + `main`; `-DPLAY_DEMO` selects the demo-build args |
| `port/Makefile.rv32i` | rv32i/ilp32 build; `all`, `wad`, `symbols`, `demo`, `clean` targets |
| `port/wadadd.py` | Appends a lump to a WAD; `make demo` uses it to embed `e1m1uv.lmp` |
| `port/e1m1uv.lmp` | Recorded vanilla 1.9 E1M1 playthrough (0:10.80, Laszlo Vecsei, DSDA archive) |
| `emulator/rv32i.h` | RV32I core structure, constants, and function declarations |
| `emulator/rv32i.c` | Native RV32I CPU core implementation, execution, tracing, checkpointing |
| `emulator/emulator.c` | Standalone emulator CLI + threaded browser GUI |
| `model/schema.py` | Pydantic data schemas (`Pair`, `Transition`, `RegWrite`, `MemWrite`) |
| `model/libemulator.py` | Python `ctypes` bridge to `libemulator.so` (`NativeCPU`) |
| `model/gen_synthetic.py` | Randomized legal RV32I transition generator |
| `model/train.py` | Baseline char-level neural model + tokenizer + training loop |
| `model/train_mamba_nano.py` | 2-layer pure Selective SSM trainer & ONNX export |
| `model/block_schema.py` | Pydantic basic block transition schema |
| `model/eval_semantics.py` | Per-mnemonic exact-match accuracy on a heldout set |
| `model/eval_playable.py` | Model-in-the-loop DOOM (open/closed/free-run modes) |
| `model/plot_loss.py` | Regenerate `docs/training_loss.png` from `logs/train.log` |
| `web/neural_runner.js` | In-browser neural execution engine (WASM SIMD + WebGPU, speculative branch prediction, superblock chunking) |
| `web/mamba_nano.onnx` | Exported 35K parameter Mamba-Nano ONNX model |
| `web/index.html` | Interactive DOOM web interface with real-time neural telemetry and block inspector |
