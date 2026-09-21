#!/usr/bin/env python3
"""Per-instruction-class semantic accuracy on a held-out synthetic set."""

from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict
from dataclasses import dataclass

import torch
from pydantic import ValidationError

from schema import Pair
from eval_playable import ModelDriver, parse_next, ROOT


@dataclass
class MnemStats:
    exact: int = 0
    total: int = 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "neural_doom.pt"
                                           if (ROOT / "models" / "neural_doom.pt").exists()
                                           else ROOT / "models" / "doom_transformer.pt"))
    ap.add_argument("--data", default=str(ROOT / "data" / "dataset_synthetic_heldout.jsonl"))
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    driver = ModelDriver(args.model,
                         device="cuda" if torch.cuda.is_available() else "cpu")
    pairs: list[Pair] = []
    with open(args.data) as f:
        for line in f:
            if len(pairs) >= args.n:
                break
            try:
                pairs.append(Pair.model_validate_json(line))
            except ValidationError:
                continue

    stats: dict[str, MnemStats] = defaultdict(MnemStats)
    t0 = time.time()
    for p in pairs:
        m = re.search(r"\[INST\] (\w+)", p.state)
        mnem = m.group(1) if m else "?"
        gen = driver.predict(p.state)
        g = parse_next(gen)
        t = parse_next(p.next)
        ok = (g.npc == t.npc and g.w_reg == t.w_reg and g.w_mem == t.w_mem)
        s = stats[mnem]
        s.exact += 1 if ok else 0
        s.total += 1

    dt = time.time() - t0
    tot_ok = sum(s.exact for s in stats.values())
    print(f"\n===== SEMANTIC GENERALIZATION ({len(pairs)} held-out pairs, "
          f"{dt:.0f}s) =====")
    print(f"overall exact-match: {tot_ok}/{len(pairs)} "
          f"({100*tot_ok/len(pairs):.2f}%)")
    for mnem in sorted(stats, key=lambda k: -stats[k].total):
        s = stats[mnem]
        print(f"  {mnem:6s} {s.exact:5d}/{s.total:5d}  {100*s.exact/s.total:6.2f}%")


if __name__ == "__main__":
    main()
