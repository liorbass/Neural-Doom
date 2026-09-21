#!/usr/bin/env python3
"""
neural-doom trainer.

Learns CPU next-state prediction from JSONL pairs:
  {"state": "[CMD] STEP\n[PC] ...\n[INST] ...\n[REG ...] ...",
   "next":  "[NPC] ...\n[W_REG ...] ...\n[W_MEM ...] ..."}

Char-level decoder-only transformer over `state + "\n=>\n" + next`,
with loss masked so we only train on predicting the `next` part.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, TypedDict, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import ValidationError

from schema import Pair

ROOT = Path(__file__).resolve().parent.parent
SEP = "\n=>\n"


class TrainConfig(TypedDict):
    """Hyperparameters persisted into the model checkpoint."""
    data: str
    max_len: int
    batch: int
    steps: int
    lr: float
    d_model: int
    heads: int
    layers: int
    ffn_mult: int
    seed: int
    val_frac: float
    save: str
    log_every: int
    val_every: int
    limit: int | None
    threads: int
    samples: int


class ModelCheckpoint(TypedDict):
    """What torch.save writes to --save."""
    model: dict[str, torch.Tensor]
    vocab: list[str]
    args: TrainConfig
    val_loss: float


class CharTokenizer:
    def __init__(self) -> None:
        self.itos: list[str] = []
        self.stoi: dict[str, int] = {}

    def fit(self, texts: Sequence[str]) -> CharTokenizer:
        chars = sorted({c for t in texts for c in t})
        self.itos = ["<pad>", "<bos>", "<eos>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        return self

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi["<bos>"]]
        for c in text[: max_len - 2]:
            ids.append(self.stoi.get(c, 0))
        ids.append(self.stoi["<eos>"])
        if len(ids) < max_len:
            ids += [self.stoi["<pad>"]] * (max_len - len(ids))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        out: list[str] = []
        for i in ids:
            c = self.itos[i]
            if c == "<eos>":
                break
            if c not in ("<pad>", "<bos>"):
                out.append(c)
        return "".join(out)


class Example(NamedTuple):
    ids: list[int]
    loss_start: int


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x


KVCache = tuple[torch.Tensor, torch.Tensor]


@dataclass
class GraphState:
    """Persistent device buffers + captured CUDA graphs for generate_graph."""
    max_len: int
    K: list[torch.Tensor]
    V: list[torch.Tensor]
    ids_buf: torch.Tensor
    P_buf: torch.Tensor
    in_tok: torch.Tensor
    in_pos: torch.Tensor
    out_nxt: torch.Tensor
    out_ids: torch.Tensor
    alive: torch.Tensor
    pos_buf: torch.Tensor
    pad_buf: torch.Tensor
    eos_t: torch.Tensor
    graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.cuda.CUDAGraph]] = field(default_factory=dict)
    apc: Callable[[torch.Tensor, int], torch.Tensor] | None = None
    cdc: Callable[[int], None] | None = None


class DoomTransformer(nn.Module):
    def __init__(self, vocab: int, max_len: int, d_model: int, heads: int,
                 layers: int, ffn: int) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, heads, ffn) for _ in range(layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)
        self._gg: GraphState | None = None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        bsz, L = ids.shape
        pos = torch.arange(L, device=ids.device)
        x = self.tok(ids) + self.pos(pos)
        mask = torch.triu(
            torch.ones(L, L, device=ids.device, dtype=torch.bool), diagonal=1
        )
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln(x))

    @torch.no_grad()
    def generate_cached(self, prompt_ids: Sequence[int], max_len: int,
                        eos_id: int) -> list[int]:
        """Greedy decode: one full prefill forward, then KV-cached single-token
        steps. Same math as generate(), ~seq-len cheaper per token."""
        dev = next(self.parameters()).device
        d = self.blocks[0].attn.embed_dim
        hd = self.blocks[0].attn.head_dim
        H = d // hd
        ids = list(prompt_ids)
        P = len(ids)
        x = self.tok(torch.tensor([ids], device=dev)) + self.pos(
            torch.arange(P, device=dev))
        caches: list[list[torch.Tensor]] = []
        for blk in self.blocks:
            h = blk.ln1(x)
            W, b = blk.attn.in_proj_weight, blk.attn.in_proj_bias
            q, k, v = (h @ W.t() + b).split(d, dim=-1)
            q = q.view(1, P, H, hd).transpose(1, 2)
            k = k.view(1, P, H, hd).transpose(1, 2)
            v = v.view(1, P, H, hd).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + blk.attn.out_proj(a.transpose(1, 2).reshape(1, P, d))
            caches.append([k, v])
            x = x + blk.ffn(blk.ln2(x))
        nxt = int(self.head(self.ln(x))[0, -1].argmax())
        out: list[int] = []
        pos = P
        while nxt != eos_id and pos < max_len:
            out.append(nxt)
            x = self.tok(torch.tensor([[nxt]], device=dev)) + self.pos(
                torch.tensor([pos], device=dev))
            for blk, cv in zip(self.blocks, caches):
                h = blk.ln1(x)
                W, b = blk.attn.in_proj_weight, blk.attn.in_proj_bias
                q, k, v = (h @ W.t() + b).split(d, dim=-1)
                q = q.view(1, 1, H, hd).transpose(1, 2)
                k = torch.cat([cv[0], k.view(1, 1, H, hd).transpose(1, 2)], 2)
                v = torch.cat([cv[1], v.view(1, 1, H, hd).transpose(1, 2)], 2)
                cv[0], cv[1] = k, v
                a = F.scaled_dot_product_attention(q, k, v)
                x = x + blk.attn.out_proj(a.transpose(1, 2).reshape(1, 1, d))
                x = x + blk.ffn(blk.ln2(x))
            pos += 1
            nxt = int(self.head(self.ln(x))[0, -1].argmax())
        return ids + out

    @torch.no_grad()
    def generate_static(self, prompt_ids: Sequence[int], max_len: int,
                        eos_id: int, compiled: bool = True) -> list[int]:
        """KV-cached greedy decode with FIXED shapes so torch.compile can fuse
        the per-token step into a handful of kernels. Returns ids like generate()."""
        dev = next(self.parameters()).device
        d = self.blocks[0].attn.embed_dim
        hd = self.blocks[0].attn.head_dim
        H = d // hd
        ids = list(prompt_ids)
        P = len(ids)
        # ---- prefill (once, dynamic length) ----
        x = self.tok(torch.tensor([ids], device=dev)) + self.pos(
            torch.arange(P, device=dev))
        caches: list[KVCache] = []
        for blk in self.blocks:
            h = blk.ln1(x)
            W, b = blk.attn.in_proj_weight, blk.attn.in_proj_bias
            q, k, v = (h @ W.t() + b).split(d, dim=-1)
            q = q.view(1, P, H, hd).transpose(1, 2)
            k = k.view(1, P, H, hd).transpose(1, 2)
            v = v.view(1, P, H, hd).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + blk.attn.out_proj(a.transpose(1, 2).reshape(1, P, d))
            K = torch.zeros(1, H, max_len, hd, device=dev, dtype=x.dtype)
            V = torch.zeros_like(K)
            K[:, :, :P] = k
            V[:, :, :P] = v
            caches.append((K, V))
            x = x + blk.ffn(blk.ln2(x))
        nxt = self.head(self.ln(x))[0, -1].argmax().reshape(1)
        pos_buf = torch.arange(max_len, device=dev)

        def cell(tok: torch.Tensor, pos: torch.Tensor,
                 caches: list[KVCache]) -> tuple[torch.Tensor, list[KVCache]]:
            x = self.tok(tok.view(1, 1)) + self.pos(pos.view(1))
            new_caches: list[KVCache] = []
            for blk, (K, V) in zip(self.blocks, caches):
                h = blk.ln1(x)
                W, b = blk.attn.in_proj_weight, blk.attn.in_proj_bias
                q, k, v = (h @ W.t() + b).split(d, dim=-1)
                q = q.view(1, 1, H, hd).transpose(1, 2)   # (1,H,1,hd)
                k = k.view(1, 1, H, hd).transpose(1, 2)
                v = v.view(1, 1, H, hd).transpose(1, 2)
                K = K.index_copy(2, pos.view(1), k)
                V = V.index_copy(2, pos.view(1), v)
                new_caches.append((K, V))
                scores = (q @ K.transpose(-1, -2)) / (hd ** 0.5)
                scores = scores.masked_fill(
                    ~(pos_buf <= pos).view(1, 1, 1, max_len), float("-inf"))
                a = torch.softmax(scores, dim=-1) @ V
                x = x + blk.attn.out_proj(a.transpose(1, 2).reshape(1, 1, d))
                x = x + blk.ffn(blk.ln2(x))
            return self.head(self.ln(x))[0, -1].argmax().reshape(1), new_caches

        cell_fn: Callable[[torch.Tensor, torch.Tensor, list[KVCache]],
                          tuple[torch.Tensor, list[KVCache]]] = cell
        if compiled:
            try:
                cell_fn = torch.compile(cell, fullgraph=True)
            except Exception:
                pass
        out: list[int] = []
        pos = P
        while int(nxt) != eos_id and pos < max_len:
            out.append(int(nxt))
            nxt, caches = cell_fn(nxt, torch.tensor(pos, device=dev), caches)
            pos += 1
        return ids + out

    @torch.no_grad()
    def generate_graph(self, prompt_ids: Sequence[int], max_len: int,
                       eos_id: int) -> list[int]:
        """Whole-step CUDA-graph decode. Prefill (padded to a bucket length)
        and the full unrolled decode loop are each captured as ONE graph, so a
        step costs ~8 async host launches + 1 sync. Graphs cached per bucket."""
        dev = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        d = self.blocks[0].attn.embed_dim
        hd = self.blocks[0].attn.head_dim
        H = d // hd
        ids = list(prompt_ids)
        P = len(ids)
        PMAX = 128
        if P > PMAX:                      # rare: fall back to eager-cached
            return cast(list[int], self.generate_cached(ids, max_len, eos_id))
        pr = min(PMAX, ((P + 7) // 8) * 8)
        st = self._gg
        if st is None or st.max_len != max_len:
            st = GraphState(
                max_len=max_len,
                K=[torch.zeros(1, H, max_len, hd, device=dev, dtype=dtype)
                   for _ in self.blocks],
                V=[torch.zeros(1, H, max_len, hd, device=dev, dtype=dtype)
                   for _ in self.blocks],
                ids_buf=torch.zeros(PMAX, dtype=torch.long, device=dev),
                P_buf=torch.zeros(1, dtype=torch.long, device=dev),
                in_tok=torch.zeros(1, dtype=torch.long, device=dev),
                in_pos=torch.zeros(1, dtype=torch.long, device=dev),
                out_nxt=torch.zeros(1, dtype=torch.long, device=dev),
                out_ids=torch.zeros(max_len, dtype=torch.long, device=dev),
                alive=torch.ones(1, dtype=torch.long, device=dev),
                pos_buf=torch.arange(max_len, device=dev),
                pad_buf=torch.zeros(1, dtype=torch.long, device=dev),
                eos_t=torch.tensor([eos_id], device=dev),
            )
            self._gg = st
        Ks, Vs = st.K, st.V
        ids_buf, P_buf = st.ids_buf, st.P_buf
        in_tok, in_pos = st.in_tok, st.in_pos
        out_nxt, out_ids, alive = st.out_nxt, st.out_ids, st.alive
        pos_buf, eos_t = st.pos_buf, st.eos_t
        pad_buf = st.pad_buf
        neg = float("-inf")

        def attn_pre(x: torch.Tensor, pr: int) -> torch.Tensor:
            for blk, K, V in zip(self.blocks, Ks, Vs):
                h = blk.ln1(x)
                W, b = blk.attn.in_proj_weight, blk.attn.in_proj_bias
                q, k, v = (h @ W.t() + b).split(d, dim=-1)
                q = q.view(1, pr, H, hd).transpose(1, 2)
                k = k.view(1, pr, H, hd).transpose(1, 2)
                v = v.view(1, pr, H, hd).transpose(1, 2)
                qpos = pos_buf[:pr].view(pr, 1)
                kpos = pos_buf[:pr].view(1, pr)
                allow = torch.where(qpos < P_buf,
                                    (kpos < P_buf) & (kpos <= qpos),
                                    (kpos < P_buf) | (kpos <= qpos))
                scores = (q @ k.transpose(-1, -2)) / (hd ** 0.5)
                scores = scores + torch.where(allow, 0.0, neg).to(scores.dtype)
                a = torch.softmax(scores, dim=-1) @ v
                x = x + blk.attn.out_proj(a.transpose(1, 2).reshape(1, pr, d))
                K[:, :, :pr] = k
                V[:, :, :pr] = v
                x = x + blk.ffn(blk.ln2(x))
            return x

        def cell_dec(pr: int) -> None:
            good = ((in_tok != eos_t) & (in_tok != 0)).to(torch.long) * alive
            out_ids.index_copy_(0, in_pos - pr, good * in_tok)
            alive.mul_(good)
            x = self.tok(in_tok.view(1, 1)) + self.pos(in_pos - pad_buf)
            for blk, K, V in zip(self.blocks, Ks, Vs):
                h = blk.ln1(x)
                W, b = blk.attn.in_proj_weight, blk.attn.in_proj_bias
                q, k, v = (h @ W.t() + b).split(d, dim=-1)
                q = q.view(1, 1, H, hd).transpose(1, 2)
                k = k.view(1, 1, H, hd).transpose(1, 2)
                v = v.view(1, 1, H, hd).transpose(1, 2)
                K.index_copy_(2, in_pos, k)
                V.index_copy_(2, in_pos, v)
                kpos = pos_buf.view(1, 1, 1, max_len)
                allow = (kpos < P_buf) | ((kpos >= pr) & (kpos <= in_pos))
                scores = (q @ K.transpose(-1, -2)) / (hd ** 0.5)
                scores = scores + torch.where(allow, 0.0, neg).to(scores.dtype)
                a = torch.softmax(scores, dim=-1) @ V
                x = x + blk.attn.out_proj(a.transpose(1, 2).reshape(1, 1, d))
                x = x + blk.ffn(blk.ln2(x))
            out_nxt.copy_(self.head(self.ln(x))[0, -1].argmax().reshape(1))
            in_tok.copy_(out_nxt)
            in_pos.add_(1)

        apc, cdc = st.apc, st.cdc
        if apc is None or cdc is None:
            try:
                apc = torch.compile(attn_pre, fullgraph=True)
                cdc = torch.compile(cell_dec, fullgraph=True)
            except Exception:
                apc, cdc = attn_pre, cell_dec
            st.apc, st.cdc = apc, cdc

        g = st.graphs.get(pr)
        if g is None:
            ids_buf.fill_(0)
            ids_buf[:P] = torch.tensor(ids, device=dev)
            P_buf.fill_(P)
            pad_buf.fill_(pr - P)
            in_tok.fill_(0)
            in_pos.fill_(pr)
            alive.fill_(1)
            x = self.tok(ids_buf.view(1, PMAX)[:, :pr]) + self.pos(
                pos_buf[:pr].view(1, pr))
            apc(x, pr)
            for _ in range(max_len - pr):
                cdc(pr)
            torch.cuda.synchronize()
            gpre = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gpre):
                x = self.tok(ids_buf.view(1, PMAX)[:, :pr]) + self.pos(
                    pos_buf[:pr].view(1, pr))
                x = apc(x, pr)
                out_nxt.copy_(
                    self.head(self.ln(x))[0, P_buf - 1].argmax().reshape(1))
            gdec = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gdec):
                for _ in range(max_len - pr):
                    cdc(pr)
            g = (gpre, gdec)
            st.graphs[pr] = g
        gpre, gdec = g

        ids_buf.fill_(0)
        ids_buf[:P] = torch.tensor(ids, device=dev)
        P_buf.fill_(P)
        pad_buf.fill_(pr - P)
        out_ids.fill_(0)
        alive.fill_(1)
        in_pos.fill_(pr)
        gpre.replay()
        in_tok.copy_(out_nxt)
        gdec.replay()

        torch.cuda.synchronize()
        out: list[int] = []
        for t in out_ids.tolist():
            if t == eos_id or t == 0:
                break
            out.append(t)
        return ids + out

    @torch.no_grad()
    def generate(self, prompt_ids: Sequence[int], max_len: int, eos_id: int,
                 device: str) -> list[int]:
        ids = torch.tensor([prompt_ids], device=device)
        while ids.size(1) < max_len:
            logits = self.forward(ids)
            nxt = int(logits[0, -1].argmax())
            if nxt == eos_id:
                break
            ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
        return cast(list[int], ids[0].tolist())


def build_examples(path: str | Path, tok: CharTokenizer, max_len: int,
                   limit: int | None = None) -> tuple[list[Example], int]:
    """Returns list of Example(token_ids, loss_start_idx) plus skip count."""
    ex: list[Example] = []
    skipped = 0
    with open(path) as f:
        for line in f:
            if limit is not None and len(ex) >= limit:
                break
            try:
                pair = Pair.model_validate_json(line)
            except ValidationError:
                skipped += 1
                continue
            prompt = pair.state + SEP
            full = prompt + pair.next
            if len(full) + 2 > max_len:
                skipped += 1
                continue
            ids = tok.encode(full, max_len)
            loss_start = len(prompt)  # token idx of first 'next' char (+BOS-1 cancels)
            ex.append(Example(ids=ids, loss_start=loss_start))
    return ex, skipped


def collate(batch: Sequence[Example], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor([b.ids for b in batch], dtype=torch.long)
    starts = torch.tensor([b.loss_start for b in batch], dtype=torch.long)
    return ids, starts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "dataset_dense.jsonl"))
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--ffn-mult", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--save", default=str(ROOT / "models" / "neural_doom.pt"))
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of training+val examples")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--samples", type=int, default=5,
                    help="greedy-decode sanity samples at the end")
    args = ap.parse_args()
    cfg = cast(TrainConfig, vars(args))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[LOAD] reading {args.data} ...", flush=True)
    t0 = time.time()
    texts: list[str] = []
    with open(args.data) as f:
        for line in f:
            try:
                pair = Pair.model_validate_json(line)
            except ValidationError:
                continue
            texts.append(pair.state + SEP + pair.next)
    if args.limit:
        texts = texts[: args.limit]
    tok = CharTokenizer().fit(texts)
    print(f"[LOAD] {len(texts)} pairs, vocab={len(tok.itos)}, "
          f"read in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    ex, skipped = build_examples(args.data, tok, args.max_len, args.limit)
    random.shuffle(ex)
    n_val = max(1, int(len(ex) * args.val_frac))
    val, train = ex[:n_val], ex[n_val:]
    print(f"[DATA] train={len(train)} val={n_val} skipped={skipped} "
          f"(tokenized in {time.time()-t0:.1f}s)", flush=True)

    model = DoomTransformer(
        len(tok.itos), args.max_len, args.d_model, args.heads,
        args.layers, args.d_model * args.ffn_mult,
    ).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {nparams/1e6:.2f}M params  d={args.d_model} "
          f"h={args.heads} L={args.layers}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    eos = tok.stoi["<eos>"]
    pad = tok.stoi["<pad>"]

    def run_batch(batch_ids: torch.Tensor, batch_starts: torch.Tensor) -> torch.Tensor:
        inp = batch_ids[:, :-1].to(device)
        tgt = batch_ids[:, 1:].to(device)
        logits = model(inp)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
            reduction="none",
        ).reshape(tgt.shape)
        # mask: only learn to predict the 'next' portion (and EOS after it)
        pos = torch.arange(tgt.size(1), device=device)[None, :]
        mask = pos >= (batch_starts[:, None].to(device) - 1)
        mask &= (tgt != pad)
        return (loss * mask).sum() / mask.sum().clamp(min=1)

    best_val = float("inf")
    t_start = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        batch = random.sample(train, args.batch)
        ids, starts = collate(batch, pad)
        loss = run_batch(ids, starts)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.log_every == 0 or step == 1:
            el = time.time() - t_start
            print(f"[TRAIN] step {step}/{args.steps} loss={loss.item():.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} "
                  f"({step/el:.1f} it/s)", flush=True)

        if step % args.val_every == 0 or step == args.steps:
            model.eval()
            tot, cnt = 0.0, 0
            with torch.no_grad():
                for i in range(0, min(len(val), args.batch * 20), args.batch):
                    ids_v, starts_v = collate(val[i:i + args.batch], pad)
                    tot += run_batch(ids_v, starts_v).item()
                    cnt += 1
            val_loss = tot / max(cnt, 1)
            ppl = math.exp(min(val_loss, 20))
            print(f"[VAL] step {step} loss={val_loss:.4f} ppl={ppl:.2f}",
                  flush=True)
            if val_loss < best_val:
                best_val = val_loss
                Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                ckpt: ModelCheckpoint = {
                    "model": model.state_dict(),
                    "vocab": tok.itos,
                    "args": cfg,
                    "val_loss": val_loss,
                }
                torch.save(ckpt, args.save)
                print(f"[SAVE] best model -> {args.save} "
                      f"(val_loss={val_loss:.4f})", flush=True)

    # ---- greedy sanity samples ----
    print(f"\n[DECODE] {args.samples} greedy samples from val:", flush=True)
    for i in range(min(args.samples, len(val))):
        ex_ids, start = val[i]
        full_text = tok.decode(ex_ids)
        prompt_text, _, truth = full_text.partition(SEP)
        prompt_ids = ex_ids[: start + 1]  # BOS + state + "\n=>\n"
        gen_ids = model.generate(prompt_ids, args.max_len, eos, device)
        gen = tok.decode(gen_ids[len(prompt_ids):])
        ok = "OK " if gen.strip() == truth.strip() else "DIFF"
        print(f"--- sample {i} [{ok}] ---")
        print(f"prompt: {prompt_text!r}")
        print(f"truth : {truth!r}")
        print(f"gen   : {gen!r}")

    print(f"\n[DONE] best val loss={best_val:.4f} "
          f"total time {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
