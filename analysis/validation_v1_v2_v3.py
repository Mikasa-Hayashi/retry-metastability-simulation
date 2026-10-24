"""Независимая проверка V1, V2, V3 (MODEL_SPEC §7). Собственный код валидации.

Критерий (задан до прогона): для каждой проверяемой величины 95 % t-интервал по независимым
репликам (alpha = 0.05, двусторонний, без поправки на множественность) должен содержать эталон.
Правило при промахе: повтор с новым блоком seed (--block 1, вдвое больше реплик); FAIL, если
промах повторяется. Seeds из диапазона pilot 0-99999: V1: 10000+, V2: 20000+, V3: 30000+;
блок 1 сдвигает seed на +50000.
Эталоны считаются здесь (формула Эрланга C, M/M/1, уравнения равновесия V3 решаются численно),
независимо от analysis/theory.py; только дисперсия FIFO/LIFO в M/M/1 берётся из theory.py (Finch).
Запуск: .venv\\Scripts\\python analysis\\validation_v1_v2_v3.py [--block 0|1]
"""

from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from src.config import Config, RetryParams
from src.model import simulate

OUT = ROOT / "reports" / "validation_data"


# ---------- независимые эталоны ----------
def erlang_c(c: int, a: float) -> float:
    s = sum(a**k / math.factorial(k) for k in range(c))
    top = a**c / math.factorial(c) / (1 - a / c)
    return top / (s + top)


def mmc_ref(c: int, rho: float) -> dict[str, float]:
    a = rho * c
    cc = erlang_c(c, a)
    wq = cc / (c - a)
    return {"pwait": cc, "wq": wq, "w": wq + 1.0}


def retrial_ref(lam: float, nu: float, n_max: int = 1500) -> tuple[float, float]:
    """Уравнения равновесия классической системы (mu = 1) на усечённом пространстве.

    Состояния (n, b): n на орбите, b = 1 сервер занят. Возвращает (P(busy), E[N]).
    """

    def idx(n: int, b: int) -> int:
        return 2 * n + b

    size = 2 * (n_max + 1)
    q = lil_matrix((size, size))
    for n in range(n_max + 1):
        q[idx(n, 0), idx(n, 1)] += lam
        if n > 0:
            q[idx(n, 0), idx(n - 1, 1)] += n * nu
        if n < n_max:
            q[idx(n, 1), idx(n + 1, 1)] += lam
        q[idx(n, 1), idx(n, 0)] += 1.0
    q = q.tocsr()
    d = np.asarray(q.sum(axis=1)).ravel()
    g = (q - diags(d)).T.tolil()
    g[0, :] = 1.0
    rhs = np.zeros(size)
    rhs[0] = 1.0
    p = spsolve(g.tocsr(), rhs)
    pb = sum(p[idx(n, 1)] for n in range(n_max + 1))
    en = sum(n * (p[idx(n, 0)] + p[idx(n, 1)]) for n in range(n_max + 1))
    return float(pb), float(en)


# ---------- прогоны ----------
def run_wait(args: tuple) -> dict[str, float]:
    c, rho, disc, kcap, t_stop, warm, seed = args
    cfg = Config(c=c, K=kcap, T=math.inf, R="R0", Q=disc, D="off", S="Exp")
    res = simulate(cfg, [(t_stop, rho)], t_stop, seed)
    n_att = len(res.att_t)
    start = np.full(n_att, np.nan)
    dur = np.full(n_att, np.nan)
    sid = res.starts[:, 0].astype(np.int64)
    start[sid] = res.starts[:, 1]
    dur[sid] = res.starts[:, 2]
    wait = start - res.att_t
    m = (res.att_t >= warm) & (res.att_t < t_stop - 300.0) & ~np.isnan(wait)
    w = wait[m]
    return {
        "wq": float(w.mean()),
        "pwait": float(np.mean(w > 0)),
        "w": float((w + dur[m]).mean()),
        "var": float(w.var(ddof=1)),
        "n": int(m.sum()),
        "rej": int(np.sum((res.att_state == 5) & (res.att_t >= warm))),
    }


def run_orbit(args: tuple) -> dict[str, float]:
    lam, nu, t_stop, warm, seed = args
    cfg = Config(c=1, K=0, T=math.inf, R="ORBIT", Q="FIFO", D="off", S="Exp",
                 retry=RetryParams(orbit_rate=nu))
    res = simulate(cfg, [(t_stop, lam)], t_stop, seed, stats_start=warm)
    span = t_stop - warm
    return {"busy": res.area_busy / span, "orbit": res.area_orbit / span}


def tci(x: list[float], truth: float) -> dict[str, float | bool]:
    a = np.asarray(x, dtype=float)
    n = len(a)
    m = float(a.mean())
    h = float(stats.t.ppf(0.975, n - 1) * a.std(ddof=1) / math.sqrt(n))
    return {"mean": m, "lo": m - h, "hi": m + h, "ref": truth, "n_rep": n,
            "rel_dev": (m - truth) / truth if truth else float("nan"),
            "in_ci": bool(m - h <= truth <= m + h)}


def main() -> None:
    block = int(sys.argv[sys.argv.index("--block") + 1]) if "--block" in sys.argv else 0
    off = 50000 * block
    mult = 1 + block
    out: dict = {"block": block, "V1": {}, "V2": {}, "V3": {}}
    with ProcessPoolExecutor(max_workers=7) as ex:
        # ---- V1 ----
        for c, t_stop, warm, nrep in [(1, 100000.0, 5000.0, 40), (8, 20000.0, 1000.0, 40)]:
            for rho in (0.5, 0.7, 0.9):
                seeds = [10000 + off + (c * 10 + int(round(rho * 10))) * 100 + i
                         for i in range(nrep * mult)]
                rs = list(ex.map(run_wait, [(c, rho, "FIFO", 5000, t_stop, warm, s) for s in seeds]))
                ref = mmc_ref(c, rho)
                key = f"c{c}_rho{rho}"
                out["V1"][key] = {k: tci([r[k] for r in rs], ref[k]) for k in ("wq", "pwait", "w")}
                out["V1"][key]["rejections"] = sum(r["rej"] for r in rs)
                print("V1", key, {k: (round(v["mean"], 4), round(v["ref"], 4), v["in_ci"])
                                  for k, v in out["V1"][key].items() if isinstance(v, dict)}, flush=True)
        # ---- V2 ----
        for c, rho, t_stop, warm, nrep in [(1, 0.5, 60000.0, 2000.0, 40), (1, 0.7, 60000.0, 3000.0, 40),
                                           (8, 0.9, 20000.0, 1000.0, 40)]:
            seeds = [20000 + off + (c * 10 + int(round(rho * 10))) * 100 + i for i in range(nrep * mult)]
            by_q = {}
            for disc in ("FIFO", "LIFO", "ALIFO"):
                by_q[disc] = list(ex.map(run_wait, [(c, rho, disc, 5000, t_stop, warm, s) for s in seeds]))
            ref = mmc_ref(c, rho)
            d: dict = {}
            for disc in ("FIFO", "LIFO", "ALIFO"):
                d[f"wq_{disc}"] = tci([r["wq"] for r in by_q[disc]], ref["wq"])
            for disc in ("LIFO", "ALIFO"):
                diff = [a["wq"] - b["wq"] for a, b in zip(by_q[disc], by_q["FIFO"])]
                t = tci(diff, 0.0)
                t["in_ci"] = bool(t["lo"] <= 0.0 <= t["hi"])
                d[f"paired_diff_{disc}-FIFO"] = t
            vf = np.array([r["var"] for r in by_q["FIFO"]])
            vl = np.array([r["var"] for r in by_q["LIFO"]])
            tt = stats.ttest_rel(vl, vf)
            d["var_FIFO_mean"] = float(vf.mean())
            d["var_LIFO_mean"] = float(vl.mean())
            d["var_LIFO_gt_FIFO_p_one_sided"] = float(tt.pvalue / 2 if tt.statistic > 0 else 1 - tt.pvalue / 2)
            d["var_LIFO_gt_FIFO_all_reps"] = bool(np.all(vl > vf))
            if c == 1:
                import theory as th  # справочно: Finch

                mf = th.mm1_wait_moments(rho, 1.0, lifo=False)
                ml = th.mm1_wait_moments(rho, 1.0, lifo=True)
                d["var_FIFO_vs_theory"] = tci(list(vf), mf.var)
                d["var_LIFO_vs_theory"] = tci(list(vl), ml.var)
            key = f"c{c}_rho{rho}"
            out["V2"][key] = d
            print("V2", key, {k: (round(v["mean"], 4), v.get("in_ci")) for k, v in d.items() if isinstance(v, dict)},
                  "varF/L", round(float(vf.mean()), 3), round(float(vl.mean()), 3), flush=True)
        # ---- V3 ----
        for lam, nu, t_stop, warm, nrep in [(0.5, 1.0, 100000.0, 2000.0, 40), (0.7, 0.5, 200000.0, 5000.0, 40),
                                            (0.5, 0.25, 100000.0, 4000.0, 40), (0.9, 2.0, 400000.0, 10000.0, 40)]:
            seeds = [30000 + off + (int(round(lam * 10)) * 10 + int(round(nu * 4))) * 100 + i
                     for i in range(nrep * mult)]
            rs = list(ex.map(run_orbit, [(lam, nu, t_stop, warm, s) for s in seeds]))
            pb, en = retrial_ref(lam, nu)
            en_formula = lam * (lam + lam / nu) / (1 - lam)
            key = f"lam{lam}_nu{nu}"
            out["V3"][key] = {"busy": tci([r["busy"] for r in rs], lam),
                              "orbit": tci([r["orbit"] for r in rs], en),
                              "orbit_vs_closed_form": tci([r["orbit"] for r in rs], en_formula),
                              "ref_numeric_busy": pb, "ref_numeric_orbit": en,
                              "ref_formula_orbit": en_formula}
            print("V3", key, out["V3"][key]["busy"]["in_ci"], round(out["V3"][key]["orbit"]["mean"], 4),
                  round(en, 4), round(en_formula, 4), out["V3"][key]["orbit"]["in_ci"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"v1_v2_v3_block{block}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
