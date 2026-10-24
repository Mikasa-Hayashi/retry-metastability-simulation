"""V6: контроль отсутствия метастабильности (R0). Независимая проверка валидации.

R0 (нет повторов), FIFO, D=off, Exp: протоколы A (триггер x4 на 50, как в спецификации) и B
для rho = 0.40..0.95 шаг 0.05, 30 реплик на точку (seeds 60000+). Ожидание: failed = 0 везде,
оценка Delta (src.thresholds) = 0 (оба порога цензурированы на 1.0). Дополнительно:
- чувствительность: theta = 0.7, удвоенный горизонт H_rec = 3000 для rho = 0.95;
- положительный контроль (мощность проверки): R1 при тех же протоколах даёт отказы при rho < 1.
Конфигурации: фактические из configs/ + изменение только указанных полей. Pilot/main не запускаются.
Запуск: .venv\\Scripts\\python analysis\\validation_v6.py
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config, ProtoParams
from src.protocols import run_protocol
from src.thresholds import estimate_delta

RHOS = [round(0.40 + 0.05 * i, 2) for i in range(12)]  # 0.40 .. 0.95


def job(a: tuple) -> dict:
    r, proto, rho, seed, theta, h_rec = a
    cfg = Config(R=r, proto=ProtoParams(theta=theta, h_rec=h_rec))
    row = run_protocol(cfg, proto, rho, seed, check=True)
    return {k: row[k] for k in ("failed", "g_final", "recovery_time", "recovery_censored",
                                "rho_test", "protocol", "wasted_fraction", "qlen_max_final")}


def run_grid(ex, r, rhos, nrep, base_seed, theta=0.5, h_rec=1500.0):
    jobs, key = [], []
    for proto in ("A", "B"):
        for i, rho in enumerate(rhos):
            for k in range(nrep):
                jobs.append((r, proto, rho, base_seed + (0 if proto == "A" else 5000) + i * 100 + k, theta, h_rec))
    return list(ex.map(job, jobs, chunksize=4))


def summarize(rows):
    out = {}
    for proto in ("A", "B"):
        for rho in sorted({x["rho_test"] for x in rows}):
            sel = [x for x in rows if x["protocol"] == proto and x["rho_test"] == rho]
            out[f"{proto}_{rho}"] = {
                "n": len(sel), "p_failed": float(np.mean([x["failed"] for x in sel])),
                "g_min": float(np.min([x["g_final"] for x in sel])),
                "g_mean": float(np.mean([x["g_final"] for x in sel])),
                "rec_mean": float(np.mean([x["recovery_time"] for x in sel])),
                "rec_max": float(np.max([x["recovery_time"] for x in sel])),
                "rec_censored": int(np.sum([x["recovery_censored"] for x in sel])),
            }
    return out


def delta_of(rows):
    ra = np.array([x["rho_test"] for x in rows if x["protocol"] == "A"])
    fa = np.array([x["failed"] for x in rows if x["protocol"] == "A"])
    rb = np.array([x["rho_test"] for x in rows if x["protocol"] == "B"])
    fb = np.array([x["failed"] for x in rows if x["protocol"] == "B"])
    d = estimate_delta(ra, fa, rb, fb, n_boot=2000, seed=1)
    return {"delta": d.delta, "lo": d.lo, "hi": d.hi, "fold": d.fold.value, "fold_cens": d.fold.censored,
            "release": d.release.value, "release_cens": d.release.censored}


def main() -> None:
    out: dict = {}
    with ProcessPoolExecutor(max_workers=7) as ex:
        rows = run_grid(ex, "R0", RHOS, 30, 60000)
        out["R0_base"] = summarize(rows)
        out["R0_delta"] = delta_of(rows)
        out["R0_total_failed"] = int(sum(x["failed"] for x in rows))
        rows7 = run_grid(ex, "R0", [0.9, 0.95], 30, 62000, theta=0.7)
        out["R0_theta0.7"] = summarize(rows7)
        rows_h = run_grid(ex, "R0", [0.9, 0.95], 30, 63000, h_rec=3000.0)
        out["R0_Hrec3000"] = summarize(rows_h)
        # положительный контроль: R1
        ctl = run_grid(ex, "R1", [0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 30, 64000)
        out["R1_positive_control"] = summarize(ctl)
    (ROOT / "reports" / "validation_data" / "v6.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "R0_base"}, indent=1, default=float))
    print("R0 base: p_failed max =", max(v["p_failed"] for v in out["R0_base"].values()),
          "g_min =", min(v["g_min"] for v in out["R0_base"].values()),
          "rec_max =", max(v["rec_max"] for v in out["R0_base"].values()),
          "censored =", sum(v["rec_censored"] for v in out["R0_base"].values()))


if __name__ == "__main__":
    main()
