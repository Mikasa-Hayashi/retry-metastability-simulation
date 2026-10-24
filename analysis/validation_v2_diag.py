"""Диагностика V2 (c=1, rho=0.5): парная разность LIFO-FIFO по среднему ожиданию и по средней длине очереди.

Seeds 80000+ (диапазон pilot). 200 реплик, горизонт 60000, warm-up 2000.
"""
from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from src.config import Config
from src.model import simulate


def one(a):
    disc, rho, seed, t_stop, warm = a
    cfg = Config(c=1, K=5000, T=math.inf, R="R0", Q=disc, D="off", S="Exp")
    res = simulate(cfg, [(t_stop, rho)], t_stop, seed, stats_start=warm)
    n = len(res.att_t)
    st = np.full(n, np.nan)
    sid = res.starts[:, 0].astype(np.int64)
    st[sid] = res.starts[:, 1]
    w = st - res.att_t
    m = (res.att_t >= warm) & (res.att_t < t_stop - 300)
    return float(np.nanmean(w[m])), res.area_q / (t_stop - warm), float(np.mean(res.starts[:, 2]))


if __name__ == "__main__":
    rho, nrep = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5, 200
    seeds = [80000 + int(rho * 100) * 200 + i for i in range(nrep)]
    out = {}
    with ProcessPoolExecutor(max_workers=7) as ex:
        r = {d: np.array(list(ex.map(one, [(d, rho, s, 60000.0, 2000.0) for s in seeds]))) for d in ("FIFO", "LIFO")}
    for name, j in (("mean_wait", 0), ("mean_Nq_timeavg", 1), ("mean_service_dur", 2)):
        d = r["LIFO"][:, j] - r["FIFO"][:, j]
        t = stats.ttest_1samp(d, 0.0)
        out[name] = {"FIFO": float(r["FIFO"][:, j].mean()), "LIFO": float(r["LIFO"][:, j].mean()),
                     "diff": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(nrep)), "p": float(t.pvalue)}
    out["ref_wq"] = rho / (1 - rho)
    out["ref_Nq"] = rho * rho / (1 - rho)
    print(json.dumps(out, indent=1))
    (ROOT / "reports" / "validation_data" / f"v2_diag_rho{rho}.json").write_text(json.dumps(out, indent=1))
