"""V5 воспроизводимость (независимая проверка) + время одного прогона (MODEL_SPEC §9).

1. Один seed дважды в одном процессе: массивы SimResult побайтово равны.
2. Один seed в двух отдельных процессах с разным PYTHONHASHSEED через CLI src.run: строки parquet
   равны (кроме runtime).
3. Результат не зависит от того, какие прогоны выполнялись до него.
4. Разные seed (50 штук) дают попарно разные потоки приходов и разные g_final/n_primary.
5. Общие числа входа не зависят от политики (D-009): req_t для R0 и R1 при одном seed совпадают.
6. Глобальный random не используется (поиск по src).
7. Время одного прогона: A и B, опорная и все 72 конфигурации.
Запуск: .venv\\Scripts\\python analysis\\validation_v5.py
"""

from __future__ import annotations

import itertools
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.model import SimResult, simulate
from src.protocols import plan_for, run_protocol

PY = sys.executable


def res_bytes(r: SimResult) -> bytes:
    parts = []
    for f in fields(r):
        v = getattr(r, f.name)
        parts.append(v.tobytes() if isinstance(v, np.ndarray) else repr(v).encode())
    return b"|".join(parts)


def main() -> None:
    out: dict = {}
    cfg = Config(R="R3", Q="ALIFO", D="on", S="LogN")
    plan = plan_for(cfg, "A", 0.8)
    a = res_bytes(simulate(cfg, plan.schedule, plan.t_stop, 50001))
    # посторонние прогоны между
    for s in range(3):
        run_protocol(Config(R="R1"), "B", 0.7, 777 + s)
    b = res_bytes(simulate(cfg, plan.schedule, plan.t_stop, 50001))
    out["same_seed_same_process_bytes_equal"] = a == b
    out["same_seed_nbytes"] = len(a)
    c = res_bytes(simulate(cfg, plan.schedule, plan.t_stop, 50002))
    out["diff_seed_bytes_differ"] = a != c

    # отдельные процессы
    cfgs = ["configs/base_r1_fifo_off_exp.yaml", "configs/r0_fifo_off_exp.yaml"]
    proc_equal = {}
    tmp = ROOT / "reports" / "validation_data"
    tmp.mkdir(parents=True, exist_ok=True)
    for cf in cfgs:
        dfs = []
        for hs in ("0", "123"):
            f = tmp / f"_v5_{Path(cf).stem}_{hs}.parquet"
            if f.exists():
                f.unlink()
            env = dict(os.environ, PYTHONHASHSEED=hs)
            subprocess.run([PY, "-m", "src.run", "--config", cf, "--seed", "50010", "--n-seeds", "3",
                            "--rho", "0.5", "0.9", "--protocol", "both", "--out", str(f)],
                           cwd=ROOT, env=env, check=True, capture_output=True)
            dfs.append(pd.read_parquet(f).drop(columns=["runtime"]))
            f.unlink()
        proc_equal[cf] = bool(dfs[0].equals(dfs[1])) and len(dfs[0]) == 12
    out["cli_two_processes_equal_except_runtime"] = proc_equal

    # разные seed: 50 штук
    cfg1 = Config(R="R1")
    plan = plan_for(cfg1, "A", 0.6)
    sigs, gs, ns = set(), [], []
    for s in range(50100, 50150):
        r = simulate(cfg1, plan.schedule, plan.t_stop, s)
        sigs.add(r.req_t[:50].tobytes())
        ns.append(len(r.req_t))
    out["diff_seeds_distinct_arrival_streams"] = len(sigs) == 50
    out["diff_seeds_distinct_counts"] = len(set(ns)) > 20
    # общие числа входа
    r0 = simulate(Config(R="R0"), plan.schedule, plan.t_stop, 50200)
    r1 = simulate(Config(R="R1"), plan.schedule, plan.t_stop, 50200)
    r2 = simulate(Config(R="R2", Q="LIFO"), plan.schedule, plan.t_stop, 50200)
    out["arrivals_independent_of_policy"] = bool(
        np.array_equal(r0.req_t, r1.req_t) and np.array_equal(r0.req_t, r2.req_t))
    # грепом: глобальный random
    hits = []
    for p in (ROOT / "src").glob("*.py"):
        txt = p.read_text(encoding="utf-8")
        for m in re.finditer(r"^\s*(import random|from random import|.*np\.random\.seed|.*random\.random\()", txt, re.MULTILINE):
            hits.append(f"{p.name}: {m.group(0).strip()}")
    out["global_random_hits"] = hits

    # время одного прогона
    levels = dict(R=["R0", "R1", "R2", "R3"], Q=["FIFO", "LIFO", "ALIFO"], D=["off", "on"],
                  S=["Exp", "LogN", "Pareto"])
    times = []
    worst = (0.0, "")
    for r, q, d, s in itertools.product(*levels.values()):
        c = Config(R=r, Q=q, D=d, S=s)
        for proto, rho in (("A", 0.9), ("B", 0.9), ("A", 0.6)):
            t0 = time.perf_counter()
            run_protocol(c, proto, rho, 50300, check=True)
            dt = time.perf_counter() - t0
            times.append(dt)
            if dt > worst[0]:
                worst = (dt, f"{r}/{q}/{d}/{s} {proto} rho={rho}")
    tt = np.array(times)
    out["timing_all72_sec"] = {"n": len(tt), "mean": float(tt.mean()), "median": float(np.median(tt)),
                               "p95": float(np.percentile(tt, 95)), "max": float(tt.max()), "worst": worst[1]}
    base = []
    for sd in range(50400, 50420):
        for proto in ("A", "B"):
            t0 = time.perf_counter()
            run_protocol(Config(R="R1"), proto, 0.9, sd, check=True)
            base.append(time.perf_counter() - t0)
    base = np.array(base)
    out["timing_reference_R1_FIFO_off_Exp_sec"] = {"n": len(base), "mean": float(base.mean()),
                                                   "max": float(base.max())}
    out["cpu_count"] = os.cpu_count()
    (tmp / "v5_timing.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
