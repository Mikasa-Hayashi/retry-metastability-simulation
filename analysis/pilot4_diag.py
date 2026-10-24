"""Pilot, часть 4, п. 5: диагностика механизма у LIFO/ALIFO (CALIBRATION_PLAN_4.md).

python analysis/pilot4_diag.py
Повторное воспроизведение прогонов по сохранённым seed'ам (общий блок, реплики 0…9; V5, новых seed'ов не расходуется).
Конфигурации: Q ∈ {LIFO, ALIFO(0,5)} × шесть вариантов (опорная R1/off/Exp; D = on; R2; R3; LogN; Pareto), ρ ∈ {0,9; 1,0; 1,1;
1,5; 2,0}, протоколы A и B. Метрики итогового окна (последние 300): g, доля просроченных стартов, очередь, впустую
потраченная работа, доля отклонённых и пропущенных (D = on) попыток, усиление, пропускная способность.
Вывод: results/pilot/pilot4_diag.csv|txt. Выводы эксплораторные (описание, не причинность).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot4_common import CRN_LO, OUT, cfg_meta

from src.config import Config
from src.metrics import amplification, goodput, queue_stats, wasted_fraction
from src.model import DROPPED, QUEUED, REJECTED, simulate
from src.protocols import plan_for
from src.runner import load_tasks, stage_dir

NREP = 10
RHOS = (0.9, 1.0, 1.1, 1.5, 2.0)
VARIANTS = {
    "ref(R1/off/Exp)": ("R1", "off", "Exp"),
    "D=on": ("R1", "on", "Exp"),
    "R2": ("R2", "off", "Exp"),
    "R3": ("R3", "off", "Exp"),
    "LogN": ("R1", "off", "LogN"),
    "Pareto": ("R1", "off", "Pareto"),
}


def one(args: tuple[str, str, str, float, int, str]) -> dict[str, Any]:
    js, label, pr, rho, rep, q = args
    cfg = Config.model_validate(json.loads(js))
    plan = plan_for(cfg, pr, rho)  # type: ignore[arg-type]
    sr = simulate(cfg, plan.schedule, plan.t_stop, CRN_LO + rep)
    lo, hi = plan.t_end - cfg.proto.w_last, plan.t_end
    age = sr.starts[:, 1] - sr.att_t[sr.starts[:, 0].astype(int)]
    ms = (sr.starts[:, 1] >= lo) & (sr.starts[:, 1] < hi)
    ma = (sr.att_t >= lo) & (sr.att_t < hi)
    md = (sr.done[:, 1] >= lo) & (sr.done[:, 1] < hi)
    q_mean, q_max = queue_stats(sr, lo, hi)
    n_ok_done = int((md & (sr.done[:, 3] == 0)).sum()) if sr.done.shape[1] > 3 else np.nan
    return {
        "Q": q, "variant": label, "protocol": pr, "rho": rho, "rep": rep,
        "g": goodput(sr, lo, hi), "frac_start_expired": float((age[ms] > cfg.T).mean()) if ms.any() else np.nan,
        "mean_age_start": float(age[ms].mean()) if ms.any() else np.nan,
        "qlen_mean": q_mean, "qlen_max": q_max, "q_end": sr.q_end,
        "stale_q_end": int(((sr.att_state == QUEUED) & (sr.att_dl < sr.t_stop)).sum()),
        "wasted": wasted_fraction(sr, lo, hi), "ampl": amplification(sr, lo, hi),
        "rej_share": float((sr.att_state[ma] == REJECTED).mean()), "drop_share": float((sr.att_state[ma] == DROPPED).mean()),
        "throughput": float(md.sum() / cfg.proto.w_last), "useful_throughput": n_ok_done / cfg.proto.w_last,
        "budget_denied": sr.budget_denied, "n_att_window": int(ma.sum()),
    }


def main() -> int:
    meta = cfg_meta()
    tk = load_tasks(stage_dir("pilot"))
    tk = tk[tk["tag"] == "p4_scr"].drop_duplicates("config_hash").set_index("config_hash")
    base = meta[(meta.m == 3) & (meta["T"] == 10.0) & (meta.theta_q == 0.5) & (meta.K == 128) & (meta.theta == 0.5) & (meta.h_rec == 1500.0)]
    jobs = []
    for q in ("LIFO", "ALIFO", "FIFO"):
        for label, (r, d, s) in VARIANTS.items():
            row = base[(base.Q == q) & (base.R == r) & (base.D == d) & (base.S == s)]
            if row.empty:
                continue
            js = tk.loc[row.config_hash.iloc[0], "cfg_json"]
            for pr in ("A", "B"):
                for rho in RHOS:
                    for rep in range(NREP):
                        jobs.append((js, label, pr, rho, rep, q))
    with mp.Pool(max(1, (mp.cpu_count() or 2) - 1)) as pool:
        rows = pool.map(one, jobs, chunksize=8)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pilot4_diag_runs.csv", index=False)
    agg = df.groupby(["Q", "variant", "protocol", "rho"]).mean(numeric_only=True).drop(columns=["rep"]).reset_index()
    agg.to_csv(OUT / "pilot4_diag.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 1000)
    lines = [f"# pilot4 diag: реплики 0…{NREP - 1} (общий блок), итоговое окно 300; среднее по репликам; Q = FIFO — для сравнения"]
    for q in ("LIFO", "ALIFO", "FIFO"):
        for pr in ("A", "B"):
            lines.append(f"\n## Q = {q}, протокол {pr}")
            s = agg[(agg.Q == q) & (agg.protocol == pr)].drop(columns=["Q", "protocol"])
            lines.append(s.round(3).to_string(index=False))
    out = "\n".join(lines)
    print(out)
    (OUT / "pilot4_diag.txt").write_text(out + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
