"""V6, добавка: частота окон W=300 с g<theta в стационарном R0 без триггера при rho=0.95.

Нужна, чтобы отличить редкие стационарные флуктуации у насыщения (c=8, rho=0.95) от
'памяти' о триггере. 20 прогонов по 100000 (после warm-up 2000), seeds 90000+.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import Config
from src.model import SUCCESS, simulate

out = {}
for rho in (0.9, 0.95):
    cfg = Config(R="R0")
    n_w = 0
    lt5 = lt7 = 0
    for s in range(90000, 90020):
        res = simulate(cfg, [(100300.0, rho)], 100300.0, s)
        edges = np.arange(2000.0, 100000.0 + 1, 300.0)
        idx = np.searchsorted(res.req_t, edges)
        succ = np.concatenate(([0], np.cumsum(res.req_status == SUCCESS)))
        g = (succ[idx[1:]] - succ[idx[:-1]]) / np.diff(idx)
        n_w += len(g)
        lt5 += int(np.sum(g < 0.5))
        lt7 += int(np.sum(g < 0.7))
    out[str(rho)] = {"windows": n_w, "frac_g_lt_0.5": lt5 / n_w, "frac_g_lt_0.7": lt7 / n_w}
print(json.dumps(out, indent=1))
(ROOT / "reports" / "validation_data" / "v6_stationary.json").write_text(json.dumps(out, indent=1))
