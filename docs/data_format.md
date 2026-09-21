# Data format

The literal interface between the CPU and the neural predictor: the text encoding of
one RV32I transition. This is what `emulator/emulator` and `model/gen_synthetic.py`
emit as JSONL, and what `model/train.py` tokenizes. See
[architecture.md](architecture.md) for how the pieces connect.

## One example

A JSONL line is `{"state": <prompt>, "next": <target>}`. The trainer concatenates
`state + "\n=>\n" + next` and trains with the loss **masked to the `next` part
only** (plus the trailing `<eos>`), so the model spends no capacity copying the
prompt.

A register-register `add`:

```
[CMD] STEP
[PC] 80000100
[INST] add x5, x1, x2
[REG x1] 00000003
[REG x2] 00000004
=>
[NPC] 80000104
[W_REG x5] 00000007
```

A load (`lw`) - note the `[MEM addr] raw` line, which makes the transition a pure
function of the prompt (the model never has to guess memory):

```
[CMD] STEP
[PC] 80000200
[INST] lw x6, 12(x1)
[REG x1] 80a0000c
[MEM 80a00018] deadbeef
=>
[NPC] 80000204
[W_REG x6] deadbeef
```

A store (`sw`) produces a memory write instead of a register write:

```
[CMD] STEP
[PC] 80000300
[INST] sw x2, 8(x1)
[REG x1] 800b0010
[REG x2] 000000ff
=>
[NPC] 80000308
[W_MEM 800b0018] 000000ff
```

A branch emits only `[NPC]` (taken -> `pc+imm`, not taken -> `pc+4`), no writes:

```
[CMD] STEP
[PC] 80000400
[INST] beq x1, x2, -16
[REG x1] 00000001
[REG x2] 00000001
=>
[NPC] 80000400
```

## Line schema

| Line | Always? | Meaning |
|---|---|---|
| `[CMD] STEP` | yes | transition kind (also `INPUT:<hex>` / `IDLE` in the tracer) |
| `[PC] <8-hex>` | yes | program counter before execution |
| `[INST] <mnem> <operands>` | yes | disassembly; operand format depends on type (see below) |
| `[REG <reg>] <8-hex>` | per operand | pre-execution value of each *source* register read |
| `[MEM <addr>] <8-hex>` | loads only | the raw bytes at the load address (so loads are self-contained) |
| `=>` | yes | prompt / target separator (reconstructed by the trainer, not stored in either field) |
| `[NPC] <8-hex>` | yes | next PC |
| `[W_REG <reg>] <8-hex>` | if `rd != 0` written | register writeback |
| `[W_MEM <addr>] <8-hex>` | stores only | memory write |

Registers are named `x0..x31` with the aliases `x0 -> zero`, `x1 -> ra`,
`x2 -> sp`, `x8 -> s0`. All register/memory/PC values are **8 lowercase hex
digits, zero-padded, 32-bit**. Immediates are **decimal, signed** (e.g. `-16`,
`12`); the shift amount for `slli`/`srli`/`srai` is a decimal `0..31` and occupies
the immediate slot.

> The `[MEM addr] raw` line for loads is what lets `gen_synthetic.py` emit
> self-contained examples without running anything - the model predicts `[W_REG]`
> straight from the prompt's `[MEM]` value. Stores' `[W_MEM]` value is just the
> (truncated) source register, already in the prompt as `[REG rs2]`.

## Operand formats per instruction class

- **R-type** `add x5, x1, x2` - `mnem rd, rs1, rs2`; two `[REG]` lines.
- **I-ALU** `addi x5, x1, 12` - `mnem rd, rs1, imm`; one `[REG]` line. Shifts use the shamt as the immediate.
- **Load** `lw x6, 12(x1)` - `mnem rd, imm(rs1)`; one `[REG]` + one `[MEM]` line.
- **Store** `sw x2, 8(x1)` - `mnem rs2, imm(rs1)`; two `[REG]` lines; output is `[W_MEM]`.
- **Branch** `beq x1, x2, -16` - `mnem rs1, rs2, imm`; two `[REG]` lines; output is `[NPC]` only.
- **JAL** `jal x1, 1024` - `mnem rd, imm`; no `[REG]`; output `[NPC]` + `[W_REG]` (link address `pc+4`).
- **JALR** `jalr x1, x2, 0` - `mnem rd, rs1, imm`; one `[REG]`; output `[NPC]` + `[W_REG]`.
- **LUI / AUIPC** `lui x5, 00400000` - `mnem rd, imm_hex`; no `[REG]`; output `[NPC]` + `[W_REG]`.

Stores write only memory (no `[W_REG]`); branches and loads to `x0` write nothing.
A halted state is emitted as `[STATE] HALTED` with `[NPC]` equal to the current
PC.

## Tokenization

`train.py::CharTokenizer` scans every training text and builds

```
vocab = ["<pad>", "<bos>", "<eos>"] + sorted(set(chars in all texts))
```

Each example is encoded as `<bos> + state + "\n=>\n" + next + <eos>`, right-padded
to `max_len`. Loss is computed over the whole sequence but masked so only tokens
from the `next` portion (and the trailing `<eos>`) contribute - i.e. the model is
trained to *predict* the next state given the prompt, not to copy it. Examples
whose concatenated length exceeds `max_len` are dropped at load time (counted as
`skipped` and reported by the trainer). The documented run yields a vocabulary of
57 over this alphabet.

## Two corpora

- **DOOM traces** (`emulator/emulator --dataset`) - real `(state -> next)` pairs sampled
  every `--sample-every` steps during a scripted E1M1 playthrough, within
  `--dataset-range`. These are the in-distribution eval set; on their own they
  lead to memorization rather than learned ISA semantics.
- **Synthetic ISA transitions** (`gen_synthetic.py`) - randomized *legal* RV32I
  steps: random PC (`0x80000000 + random<<2`), random registers, randomized
  operands (including sign boundaries and DOOM-ish pointer ranges), and a concrete
  `[MEM]` value for every load. This is the corpus that actually generalizes to
  the heldout set. Class mix is biased toward R/I/load/branch and away from
  jumps/lui/auipc, roughly matching DOOM's instruction frequency.

Both corpora use the identical text format above, so a model trained on synthetic
data is directly evaluable on real DOOM traces.

The repository also tracks a third, small file: `data/sample_dataset.jsonl`,
450 real DOOM-trace pairs in exactly this format. It is a format sample for
inspection and quick smoke tests (it is *not* a training corpus); regenerate
real traces with `emulator/emulator --bin ... --dataset ...` if you need more.

---

## Basic Block Transition Format (`model/block_schema.py`)

To support multi-instruction block prediction and higher throughput, the basic block format records execution deltas across entire straight-line basic blocks:

```json
{
  "start_pc": "800275ac",
  "end_pc": "800275e0",
  "next_pc": "800275ac",
  "instruction_count": 14,
  "regs_in": [0, 2147483648, ...],
  "reg_writes": {"10": 2147483652, "14": 2768167622},
  "mem_writes": [[2164260864, 255, 4]],
  "halted": false
}
```

- **`start_pc` / `end_pc`**: The entry and terminating instruction addresses of the basic block.
- **`next_pc`**: The control flow destination (e.g. branch target if TAKEN, sequential fallthrough if not).
- **`regs_in`**: Snapshot of all 32 general-purpose registers upon block entry.
- **`reg_writes`**: Sparse map `{register_index: final_written_value}` of registers modified during the block.
- **`mem_writes`**: List of `[address, value, size]` writes committed to the memory bus.

---

## Mamba-Nano Tensor Interface (`model/train_mamba_nano.py`)

For high-speed in-browser neural execution (15 µs in WASM / 0.4 ms in ONNX Runtime Web), inputs and outputs are structured as dense float32 tensors:

### Input Tensors
| Tensor | Shape | Type | Description |
|---|---|---|---|
| `pc_bits` | `(1, 1, 32)` | `float32` | Binary one-hot bit decomposition of the guest program counter `(PC >> i) & 1`. |
| `regs_norm` | `(1, 1, 32)` | `float32` | Non-linear normalized registers: $\ln(1 + R) / 22.2$. Preserves integer precision. |
| `s0`, `s1` | `(1, 64, 16)` | `float32` | Recurrent Mamba SSM hidden state vectors ($d_{\text{model}}=64, d_{\text{state}}=16$). |

### Output Tensors
| Tensor | Shape | Type | Description |
|---|---|---|---|
| `branch_taken_logits` | `(1, 1, 1)` | `float32` | Raw logit for branch condition. $P(\text{TAKEN}) = \sigma(\text{logit})$. |
| `next_pc_offset` | `(1, 1, 1)` | `float32` | Relative PC offset for predicted branch target. |
| `s0_out`, `s1_out` | `(1, 64, 16)` | `float32` | Updated recurrent SSM hidden states committed to memory for the next block. |

