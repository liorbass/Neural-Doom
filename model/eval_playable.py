#!/usr/bin/env python3
"""
Can neural-doom drive DOOM? Model-in-the-loop experiment.

Two CPUs restored from the same mid-gameplay checkpoint:
  cpu_true  - native C emulator steps it (ground truth)
  cpu_model - the MODEL drives it: each step the model sees the decoded
              state (no execution) and greedily generates the transition;
              we parse NPC/W_REG/W_MEM and apply them to native CPU state.

Modes:
  --open-loop   per-step accuracy: model predicts from the TRUE state each
                step; state is resynced to truth after every step.
  --closed-loop the model's own predictions feed the next state; we measure
                how many steps it survives before diverging from truth.
  --free-run    the model runs freely without truth comparison until crash.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import Transition, RegWrite, MemWrite, hex_word, REG_NAMES, REG_IDX  # noqa: E402
from libemulator import NativeCPU  # noqa: E402
from train import DoomTransformer, SEP, ModelCheckpoint  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def parse_next(text: str) -> Transition:
    """Parse a transition block (model output or truth) into a Transition."""
    return Transition.parse(text)


class ModelDriver:
    def __init__(self, ckpt_path: str | Path, device: str = "cpu") -> None:
        ck: ModelCheckpoint = torch.load(ckpt_path, map_location=device,
                                         weights_only=False)
        a = ck["args"]
        self.itos: list[str] = ck["vocab"]
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(self.itos)}
        self.eos = self.stoi["<eos>"]
        self.max_len = a["max_len"]
        self.model = DoomTransformer(
            len(self.itos), a["max_len"], a["d_model"], a["heads"],
            a["layers"], a["d_model"] * a["ffn_mult"])
        self.model.load_state_dict(ck["model"])
        self.model.to(device)
        if device == "cuda":
            self.model.half()          # fp16 inference on GPU
        self.model.eval()
        self.device = device
        if device == "cuda":           # warm up kernels/CUDA context
            self.predict("[CMD] STEP")

    def predict(self, prompt: str) -> str:
        text = prompt + SEP
        ids = [self.stoi["<bos>"]] + [self.stoi.get(c, 0) for c in text]
        ids_t = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            if self.device == "cuda":
                out = self.model.generate_graph(ids_t[0].tolist(),
                                                self.max_len, self.eos)
            else:
                out = self.model.generate_cached(ids_t[0].tolist(),
                                                 self.max_len, self.eos)
        gen = "".join(self.itos[i] for i in out[len(ids):]
                      if self.itos[i] not in ("<pad>", "<bos>"))
        return gen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "data" / "checkpoint.bin"
                                                if (ROOT / "data" / "checkpoint.bin").exists()
                                                else ROOT / "data" / "checkpoint.pkl"))
    ap.add_argument("--model", default=str(ROOT / "models" / "neural_doom.pt"
                                           if (ROOT / "models" / "neural_doom.pt").exists()
                                           else ROOT / "models" / "doom_transformer.pt"))
    ap.add_argument("--mode", choices=["open-loop", "closed-loop", "free-run"],
                    required=True)
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()

    cpu_true = NativeCPU()
    cpu_true.load_checkpoint(args.checkpoint)
    print(f"[CKPT] steps={cpu_true.steps} pc={cpu_true.pc:08x} "
          f"timer={cpu_true.timer_ms}ms", flush=True)

    cpu_model = cpu_true.clone()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEV] model on {device}", flush=True)
    driver = ModelDriver(args.model, device=device)

    n_exact = n_npc = n_wreg = n_wmem = 0
    misses_printed = 0
    gen_ms_total = 0.0
    div_step: int | None = None
    div_detail: tuple[str, str, str] | None = None

    step = -1
    for step in range(args.steps):
        dec = cpu_model.decode_prompt()
        if dec is None:
            print(f"[HALT] model cpu halted at step {step}")
            break
        prompt, mnemonic = dec

        truth = cpu_true.step()
        assert truth is not None
        _, _, t_target = truth.partition("\n=>\n")
        t_trans = parse_next(t_target)

        t0 = time.time()
        gen = driver.predict(prompt)
        gen_ms_total += (time.time() - t0) * 1000
        p_trans = parse_next(gen)

        npc_ok = p_trans.npc == t_trans.npc
        wreg_ok = p_trans.w_reg == t_trans.w_reg
        wmem_ok = p_trans.w_mem == t_trans.w_mem
        exact = npc_ok and wreg_ok and wmem_ok
        n_exact += exact
        n_npc += npc_ok
        n_wreg += wreg_ok
        n_wmem += wmem_ok

        if args.mode == "closed-loop":
            if not exact:
                div_step = step
                div_detail = (prompt, gen, t_target.rstrip("\n"))
                break
            cpu_model.apply_prediction(p_trans, mnemonic)
            if step % 25 == 0:
                print(f"[LOOP] step {step}: pc={cpu_model.pc:08x} "
                      f"({gen_ms_total/(step+1):.0f} ms/step)", flush=True)
        elif args.mode == "free-run":
            cpu_model.apply_prediction(p_trans, mnemonic)
            if not (0x80000000 <= cpu_model.pc < 0x81100000):
                print(f"[CRASH] step {step}: pc ran away to "
                       f"{cpu_model.pc:08x}", flush=True)
                break
            if cpu_model.decode_prompt() is None:
                print(f"[CRASH] step {step}: fetched invalid instruction at "
                      f"{cpu_model.pc:08x}", flush=True)
                break
            if step % 25 == 0:
                print(f"[FREE] step {step}: pc={cpu_model.pc:08x}",
                      flush=True)
        else:  # open-loop: resync model cpu to truth every step
            cpu_model = cpu_true.clone()
            if not exact and misses_printed < 8:
                misses_printed += 1
                print(f"[MISS] step {step}\nprompt: {prompt!r}\n"
                      f"gen: {gen!r}\ntruth: {t_target.rstrip()!r}",
                      flush=True)

    done = step + 1 if div_step is None else div_step
    ms = gen_ms_total / max(done, 1)
    print(f"\n===== {args.mode.upper()} RESULTS =====", flush=True)
    print(f"steps evaluated     : {done}")
    print(f"exact-match steps   : {n_exact} ({100*n_exact/max(done,1):.2f}%)")
    print(f"NPC matches         : {n_npc} ({100*n_npc/max(done,1):.2f}%)")
    print(f"W_REG matches       : {n_wreg} ({100*n_wreg/max(done,1):.2f}%)")
    print(f"W_MEM matches       : {n_wmem} ({100*n_wmem/max(done,1):.2f}%)")
    print(f"gen time per step   : {ms:.0f} ms "
          f"(= {1000/ms:.1f} model-driven steps/s)")
    if args.mode == "closed-loop":
        if div_step is None:
            print(f"SURVIVED all {args.steps} steps in perfect sync!")
        else:
            print(f"DIVERGED at step {div_step}")
            assert div_detail is not None
            p, g, t = div_detail
            print(f"  prompt: {p!r}")
            print(f"  model : {g!r}")
            print(f"  truth : {t!r}")


if __name__ == "__main__":
    main()
