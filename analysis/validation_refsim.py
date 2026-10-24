"""Независимая минимальная реализация (R1, Exp, FIFO/LIFO, D off/on) и сравнение с src.

Это страховка от общего бага ядра src.model: другая структура (объекты-попытки, потоки случайных
чисел по событию, генерация приходов последовательно по сегментам, а не преобразованием времени).
Критерий (задан до прогона): для каждой метрики (g_final, усиление, доля впустую, доля failed)
сравнение src и референса на N=100 независимых репликах: Welch t-тест (метрики) и точный
критерий Фишера (failed), alpha = 0.01 на сравнение. PASS: ни одного p < 0.001 и не более
двух p < 0.01 из всех сравнений. Seeds src: 70000+, референс: 75000+.
Запуск: .venv\\Scripts\\python analysis\\validation_refsim.py
"""

from __future__ import annotations

import heapq
import json
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.protocols import run_protocol

C, K, T, MAXR, DLY = 8, 128, 10.0, 3, 1.0


def ref_run(proto: str, rho: float, lifo: bool, drop: bool, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    if proto == "A":
        segs = [(200.0, rho), (250.0, 4 * rho), (1750.0, rho)]
    else:
        segs = [(400.0, 1.05), (1900.0, rho)]
    t_end = segs[-1][0]
    # вход продолжается за t_end на длину хвоста (как D-008), чтобы окно не разгружалось
    segs[-1] = (t_end + 50.0, segs[-1][1])
    ev: list = []
    n = 0
    # приходы: последовательно по сегментам
    t, seg_start = 0.0, 0.0
    primaries: list[float] = []
    for end, r in segs:
        t = seg_start
        lam = r * C
        while True:
            t += rng.exponential(1.0 / lam)
            if t >= end:
                break
            primaries.append(t)
        seg_start = end
    for i, tp in enumerate(primaries):
        heapq.heappush(ev, (tp, n, "ARR", i, 0))
        n += 1
    ok = [False] * len(primaries)
    natt = [0] * len(primaries)
    busy = 0
    q: deque = deque()
    wasted = served = 0.0
    lo = t_end - 300.0
    atts: list = []  # [req, dl, finished]

    def new_attempt(now: float, req: int, k: int) -> None:
        nonlocal busy, n
        natt[req] += 1
        a = [req, now + T, False, k]
        atts.append(a)
        heapq.heappush(ev, (now + T, n, "TO", len(atts) - 1, 0))
        n += 1
        if busy < C:
            start(now, a)
        elif len(q) < K:
            q.append(a)
        else:
            a[2] = True  # немедленный отказ
            fail(now, a)

    def start(now: float, a: list) -> None:
        nonlocal busy, n
        busy += 1
        s = rng.exponential(1.0)
        heapq.heappush(ev, (now + s, n, "DONE", id(a), s))
        live[id(a)] = a
        n += 1

    def fail(now: float, a: list) -> None:
        nonlocal n
        if a[3] < MAXR:
            heapq.heappush(ev, (now + DLY, n, "RETRY", a[0], a[3] + 1))
            n += 1

    live: dict = {}
    while ev:
        now, _, kind, x, s = heapq.heappop(ev)
        if kind == "ARR":
            new_attempt(now, x, 0)
        elif kind == "RETRY":
            new_attempt(now, x, s)
        elif kind == "TO":
            a = atts[x]
            if not a[2]:
                a[2] = True  # клиент ушёл
                fail(now, a)
        else:  # DONE
            a = live.pop(x)
            busy -= 1
            if lo <= now < t_end:
                served += s
            if not a[2]:
                a[2] = True
                ok[a[0]] = True
            else:
                if lo <= now < t_end:
                    wasted += s
            while q:
                b = q.pop() if lifo else q.popleft()
                if drop and now >= b[1]:
                    continue
                start(now, b)
                break
    idx = [i for i, tp in enumerate(primaries) if lo <= tp < t_end]
    g = float(np.mean([ok[i] for i in idx]))
    amp = float(np.mean([natt[i] for i in idx]))
    return {"g": g, "amp": amp, "wasted": wasted / served if served > 0 else float("nan"),
            "failed": int(g < 0.5)}


def src_run(proto: str, rho: float, lifo: bool, drop: bool, seed: int) -> dict:
    cfg = Config(R="R1", Q="LIFO" if lifo else "FIFO", D="on" if drop else "off", S="Exp")
    r = run_protocol(cfg, proto, rho, seed, check=True)
    return {"g": r["g_final"], "amp": r["amplification"], "wasted": r["wasted_fraction"], "failed": r["failed"]}


def job(a: tuple) -> tuple:
    which, proto, rho, lifo, drop, seed = a
    f = ref_run if which == "ref" else src_run
    return which, f(proto, rho, lifo, drop, seed)


def main() -> None:
    nrep = 100
    scen = []
    for lifo in (False, True):
        for drop in (False, True):
            for proto, rho in (("A", 0.5), ("A", 0.9), ("B", 0.4), ("B", 0.9)):
                scen.append((proto, rho, lifo, drop))
    scen += [("A", 0.35, False, False), ("B", 0.33, False, False), ("B", 0.30, False, False)]
    out = []
    with ProcessPoolExecutor(max_workers=7) as ex:
        for si, (proto, rho, lifo, drop) in enumerate(scen):
            jobs = [("src", proto, rho, lifo, drop, 70000 + si * 100 + i) for i in range(nrep)]
            jobs += [("ref", proto, rho, lifo, drop, 75000 + si * 100 + i) for i in range(nrep)]
            res = list(ex.map(job, jobs, chunksize=5))
            s = [r for w, r in res if w == "src"]
            f = [r for w, r in res if w == "ref"]
            row = {"proto": proto, "rho": rho, "Q": "LIFO" if lifo else "FIFO", "D": "on" if drop else "off"}
            for m in ("g", "amp", "wasted"):
                xs = np.array([r[m] for r in s])
                xf = np.array([r[m] for r in f])
                if np.nanstd(xs) == 0 and np.nanstd(xf) == 0:
                    p = 1.0 if np.nanmean(xs) == np.nanmean(xf) else 0.0
                else:
                    p = float(stats.ttest_ind(xs, xf, equal_var=False, nan_policy="omit").pvalue)
                row[m] = {"src": float(np.nanmean(xs)), "ref": float(np.nanmean(xf)), "p": p}
            ks = sum(r["failed"] for r in s)
            kf = sum(r["failed"] for r in f)
            row["failed"] = {"src": ks, "ref": kf, "p": float(stats.fisher_exact([[ks, nrep - ks], [kf, nrep - kf]])[1])}
            out.append(row)
            print(row, flush=True)
    ps = [r[m]["p"] for r in out for m in ("g", "amp", "wasted", "failed")]
    summ = {"n_comparisons": len(ps), "n_p_lt_0.01": int(sum(p < 0.01 for p in ps)),
            "n_p_lt_0.001": int(sum(p < 0.001 for p in ps)), "min_p": float(min(ps))}
    print(summ)
    (ROOT / "reports" / "validation_data" / "refsim.json").write_text(
        json.dumps({"summary": summ, "rows": out}, indent=1, default=float))


if __name__ == "__main__":
    main()
