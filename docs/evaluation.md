# Evaluation

How the headline numbers in the README are measured and reproduced. See
[data_format.md](data_format.md) for the text format these harnesses read and
write.

## Two evaluators

| Script | Measures | Reference |
|---|---|---|
| `model/eval_semantics.py` | per-instruction-class *exact-match* accuracy on a held-out synthetic set | ISA semantics in isolation |
| `model/eval_playable.py` | model-in-the-loop DOOM: how long the model can drive the CPU before diverging from truth | end-to-end closed-loop behavior |

Both load the trained checkpoint (`models/neural_doom.pt`) via
`ModelDriver`, which rebuilds the architecture from the saved `args`/`vocab` and
runs fp16 greedy decode on GPU (the `generate_graph` CUDA-graph path) or the
eager `generate` path on CPU.

## ISA semantics (`eval_semantics.py`)

For each held-out pair it:

1. takes the `state` prompt,
2. greedy-decodes the `next` with the model,
3. parses both prediction and ground-truth into `(NPC, W_REG, W_MEM)` via
   `parse_next` (regex over `[NPC]` / `[W_REG <reg>]` / `[W_MEM <addr>]`),
4. scores a pair **exact** iff all three fields match.

Accuracy is bucketed by mnemonic (extracted from the `[INST]` line). Defaults:
2 000 pairs from `data/dataset_synthetic_heldout.jsonl`. The held-out set is a
separately generated synthetic corpus with a **seed different from training** and
never mixed into the training split - regenerate it with, for example:

```bash
uv run python model/gen_synthetic.py 200000 data/dataset_synthetic_heldout.jsonl 11
```

This is what backs the per-class table in the README Results (16 classes at 100%,
`slt` 98.2%, `sltu` 98.0%, `sub` 96.3%, `bltu` 96.2%, `add` 92.3%, overall **88.10%** exact match).

Full 2,000 held-out breakdown:
```
overall exact-match: 1762/2000 (88.10%)
  lw        97/   97  100.00%
  lbu       92/   92  100.00%
  lb        90/   90  100.00%
  lhu       84/   84  100.00%
  lh        58/   58  100.00%
  xor       55/   55  100.00%
  or        53/   53  100.00%
  sll       51/   51  100.00%
  sra       51/   51  100.00%
  srli      49/   49  100.00%
  srai      43/   43  100.00%
  slli      42/   42  100.00%
  sltiu     42/   42  100.00%
  slti      40/   40  100.00%
  srl       40/   40  100.00%
  and       38/   38  100.00%
  slt       55/   56   98.21%
  sltu      48/   49   97.96%
  sub       52/   54   96.30%
  bltu      50/   52   96.15%
  add       48/   52   92.31%
  ori       51/   57   89.47%
  andi      58/   66   87.88%
  jalr      48/   55   87.27%
  bne       44/   51   86.27%
  blt       30/   36   83.33%
  addi      29/   35   82.86%
  beq       42/   51   82.35%
  bgeu      39/   48   81.25%
  sb        56/   70   80.00%
  sw        32/   40   80.00%
  xori      39/   49   79.59%
  sh        66/   83   79.52%
  bge       36/   48   75.00%
  jal       14/   54   25.93%
  auipc      0/   37    0.00%
  lui        0/   32    0.00%
```

## Model-in-the-loop DOOM (`eval_playable.py`)

Both CPUs are restored from the **same** mid-gameplay checkpoint
(`data/checkpoint.bin`, produced by `emulator/emulator --checkpoint-at`). `cpu_true` is
stepped by the native C RV32I engine (ground truth); `cpu_model` is driven by the model:
each step the current instruction is decoded into the text prompt **without
executing** (`decode_prompt`), the model predicts the transition, and the parsed
`(NPC, W_REG, W_MEM)` is applied to native CPU state (`apply_prediction`). The store
size (`sb`/`sh`/`sw`) is taken from the decoded mnemonic so memory writes land at
the right width.

Three modes:

- **`open-loop`** - per-step accuracy. The model predicts from the *true* state
  each step; `cpu_model` is resynced to truth after every step.
  Measured results (500 steps):
  - Exact match: **99.60%** (498 / 500)
  - NPC match: **100.00%** (500 / 500)
  - W_REG match: **99.60%** (498 / 500)
  - W_MEM match: **100.00%** (500 / 500)
  - Gen latency: **48 ms/step** (~21 steps/s)
- **`closed-loop`** - the model's own predictions feed the next state. It runs
  until the first step where *any* of NPC / W_REG / W_MEM differs from truth.
  Measured results:
  - **87 consecutive steps survived in 100% perfect sync**
  - Diverged at step 87 on `add x14, x14, x16` high carry nibble
    (`truth: a4fee2c6`, `model: a5fee2c6`)
  - Gen latency: ~75–90 ms/step
- **`free-run`** - like closed-loop but with no truth comparison: it just applies
  predictions and watches for a crash (PC runs outside `0x80000000`-`0x81100000`,
  or a zero-word fetch).
  Measured results (500 steps):
  - **Survived all 500 steps** without crashing (100.00% NPC match)
  - Gen latency: **45 ms/step** (22.1 steps/s)

All modes report greedy-decode wall time per step (about 45–75 ms/step on GPU),
i.e. the model-driven step rate - the reason "playing DOOM with the model today
takes ~2 days per frame". The bottleneck is the predictor itself (one neural
forward pass per step), not branch resolution.

## Reproduce

```bash
# checkpoint: scripted playthrough, save full CPU+memory state at step 90M (takes <1 second)
emulator/emulator --bin doomgeneric/doomgeneric/doom_rv32i.bin \
    --steps 90000000 --checkpoint-at 90000000 --checkpoint-file data/checkpoint.bin

uv run python model/eval_semantics.py --model models/neural_doom.pt \
    --data data/dataset_synthetic_heldout.jsonl

uv run python model/eval_playable.py --mode closed-loop --checkpoint data/checkpoint.bin --model models/neural_doom.pt
uv run python model/eval_playable.py --mode open-loop   --checkpoint data/checkpoint.bin --model models/neural_doom.pt
uv run python model/eval_playable.py --mode free-run    --checkpoint data/checkpoint.bin --model models/neural_doom.pt
```

`eval_semantics.py` reports `overall exact-match %` and a per-mnemonic table;
`eval_playable.py` reports `exact-match steps`, per-field match %, decode ms/step,
and (for closed-loop) either `DIVERGED at step N` with the differing triple or
`SURVIVED all N steps in perfect sync!`.

## First-level completion (platform, not model)

The E1M1-completion claim is about the bare-metal port, measured by replaying
the pinned vanilla demo build:

```bash
emulator/emulator --bin doomgeneric/doomgeneric/doom_rv32i_demo.bin --no-input \
    --steps 500000000
```

Success criterion: the emulator's gamestate probe prints
`[GS] gamestate=1 gametic=378` (GS_INTERMISSION after exiting E1M1 at the
recorded tic), and the last rendered frame is the E1M1 intermission screen
(`docs/doom_e1m1_finished.png`). `--no-input` suppresses the scripted-input
schedule (menu navigation is meaningless for the demo build). The model is
*not* in the loop here; closed-loop model behavior is what the two evaluators
above measure.

---

## In-Browser Speculative Neural Evaluation (`web/neural_runner.js` & `web/index.html`)

In addition to offline Python evaluators, the repository features live in-browser neural evaluation inside the WebAssembly DOOM engine:

### 3-Tier Execution & Performance Metrics:
1. **Tier 1 (Embedded C/WASM Kernel)**: **15 µs / block** latency (~70,000 blocks/sec), executing hand-vectorized Mamba-Nano inference directly inside WebAssembly linear memory without FFI overhead.
2. **Tier 2 (Dynamic Mega-Block Chaining)**: Groups deterministic ALU/memory instructions up to 128 instructions per dispatch, yielding **10.25+ MIPS** and running DOOM at **35.0 FPS (native speed)**.
3. **Tier 3 (Zero-Delay MessageChannel & Vsync Presentation)**: Asynchronous macro-task loop decoupled from 60 Hz `requestAnimationFrame` canvas blitting, ensuring smooth screen rendering with zero compositor starvation.
4. **ONNX Runtime Web Providers**:
   - **WASM SIMD (CPU)**: **0.398 ms / forward pass**, executing **~2,500 blocks/sec** (~37,500 insts/sec).
   - **WebGPU (Direct Hardware)**: Direct WebGPU hardware execution via ONNX Runtime Web.

### Metrics Measured Live:
1. **Branch Prediction Accuracy**: Compares the model's prediction (`prob > 0.5`) against real-time ground truth evaluated directly from live RISC-V registers (`BEQ`, `BNE`, `BLT`, `BGE`, `BLTU`, `BGEU`). Typical accuracy on live DOOM gameplay: **75%–85%**.
2. **Decoupled Game Engine FPS vs. Screen Refresh Hz**:
   - **Game Engine FPS**: Tracks DOOM's internal engine tic progression (`curGametic` delta, native 35.0 FPS).
   - **Screen Refresh Rate**: Tracks physical HTML5 canvas redraws synchronized to display vsync (60 Hz).
   - **Rolling Window & EMA Smoothing**: Evaluated over a 1.0-second sliding history window with Exponential Moving Average ($\alpha = 0.25$), eliminating quantization drops and display jitter.
3. **Branch Distribution**: Real-time counter of `Fallthrough` vs. `Taken` branches.
4. **Speculative Retirement**: Mimics modern CPU hardware branch prediction: the neural network speculates control flow, while the CPU retires ground-truth targets to prevent cascading divergence while continuously benchmarking neural accuracy.
