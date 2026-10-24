"""Pilot, часть 3, п. B: диагностика LIFO/ALIFO (CALIBRATION_PLAN_3.md).

python analysis/pilot3_diag.py
Таблица по результатам прогонов + траектории (повторное воспроизведение прогонов по сохранённым seed'ам,
детерминировано, V5; новые seed'ы не расходуются). Вывод: results/pilot/pilot3_diag.csv|txt, pilot3_traj.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.metrics import goodput_series
from src.model import QUEUED, REJECTED, simulate
from src.protocols import plan_for
from src.runner import ROOT, load_results, load_tasks, stage_dir
from src.thresholds import aggregate, point_threshold

OUT = ROOT / "results" / "pilot"
LINES: list[str] = []
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def emit(s: str = "") -> None:
    print(s)
    LINES.append(s)


def cfg_table(tk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, js in tk.drop_duplicates("config_hash")[["config_hash", "cfg_json"]].itertuples(index=False):
        c = json.loads(js)
        if (c["R"], c["D"], c["S"], c["c"], c["K"], c["T"]) != ("R1", "off", "Exp", 8, 128, 10.0):
            continue
        p = c["proto"]
        if (p["trigger_mult"], p["trigger_len"], p["theta"], p["h_rec"], p["overload_len"]) != (1.5, 25.0, 0.5, 1500.0, 400.0):
            continue
        rows.append({"config_hash": h, "Q": c["Q"], "theta_q": c["theta_q"]})
    df = pd.DataFrame(rows)
    df["label"] = np.where(df.Q == "ALIFO", "ALIFO(" + df.theta_q.astype(str) + ")", df.Q)
    return df


def main() -> int:
    d = stage_dir("pilot")
    res, tk = load_results(d), load_tasks(d)
    ct = cfg_table(tk)
    res = res.merge(ct, on="config_hash")
    res["rej_share"] = res["n_rejected"] / res["n_attempts"]
    emit("# B. Диагностика LIFO/ALIFO: R1, D = off, Exp, триггер (1,5; 25); все имеющиеся реплики")
    agg = (res.groupby(["label", "protocol", "rho_test"])
           .agg(n=("failed", "size"), P_fail=("failed", "mean"), g_final=("g_final", "mean"), g_min=("g_final", "min"),
                qlen_mean=("qlen_mean_final", "mean"), qlen_max=("qlen_max_final", "max"),
                wasted=("wasted_fraction", "mean"), ampl=("amplification", "mean"), rej_share=("rej_share", "mean"),
                p99=("p99", "mean"))
           .reset_index())
    agg["inv_rho"] = 1.0 / agg["rho_test"]
    sel = agg[agg.rho_test.isin([0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5, 2.0])]
    for lab in ["FIFO", "LIFO", "ALIFO(0.5)", "ALIFO(0.25)", "ALIFO(0.75)"]:
        for pr in ("A", "B"):
            s = sel[(sel.label == lab) & (sel.protocol == pr)]
            if len(s):
                emit(f"\n{lab}, протокол {pr}:")
                emit(s.drop(columns=["label", "protocol"]).round(3).to_string(index=False))
    emit("\nρ* = наименьшее проверенное ρ с долей отказов ≥ 0,5 (по P_fail, без сглаживания; для A и B):")
    star = []
    for (lab, pr), g in agg.groupby(["label", "protocol"]):
        gg = g.sort_values("rho_test")
        hit = gg[gg.P_fail >= 0.5]
        star.append({"label": lab, "protocol": pr, "rho_min_tested": gg.rho_test.min(), "rho_max_tested": gg.rho_test.max(),
                     "rho_star": hit.rho_test.min() if len(hit) else np.nan,
                     "max_P_fail": gg.P_fail.max(), "min_g_final_mean": gg.g_final.min()})
    sd = pd.DataFrame(star)
    emit(sd.round(3).to_string(index=False))

    # пороги для ALIFO(0.25/0.75) по сетке ≤ 1.0 (грубо, 10 реплик)
    emit("\nПороги по сетке ρ ≤ 1,0 (скрининг-оценка, point_threshold, без ДИ):")
    thr_rows = []
    for lab in ["FIFO", "LIFO", "ALIFO(0.25)", "ALIFO(0.5)", "ALIFO(0.75)"]:
        r = {"label": lab}
        for pr in ("A", "B"):
            s = res[(res.label == lab) & (res.protocol == pr) & (res.rho_test <= 1.0 + 1e-9) & (res.replicate < 10)]
            x, n, k = aggregate(s.rho_test.to_numpy(), s.failed.to_numpy())
            v, cens, _ = point_threshold(x, n, k)
            r[f"thr_{pr}"], r[f"cens_{pr}"] = v, cens
        r["delta"] = r["thr_A"] - r["thr_B"]
        thr_rows.append(r)
    emit(pd.DataFrame(thr_rows).round(3).to_string(index=False))

    # post hoc (эксплораторно): доля прогонов с g_final < 0,95 (пересчёт флага; θ входит в модель только через failed)
    emit("\nPost hoc, эксплораторно: доля прогонов с g_final < 0,95 по ρ (не критерий отказа плана):")
    ph = res[res.label.isin(["LIFO", "ALIFO(0.5)", "ALIFO(0.75)", "FIFO"]) & res.rho_test.isin([0.7, 0.8, 0.85, 0.9, 0.95, 1.0])].copy()
    ph["low95"] = (ph.g_final < 0.95).astype(int)
    emit(ph.pivot_table(index=["label", "protocol"], columns="rho_test", values="low95", aggfunc="mean").round(2).to_string())
    emit("число прогонов в ячейке: " + str(ph.pivot_table(index=["label", "protocol"], columns="rho_test", values="low95", aggfunc="size").to_dict()))

    # траектории
    emit("\n# Траектории (повторное воспроизведение прогонов по seed'ам, ≤ 10 реплик на условие; бины по 100)")
    cases = [("FIFO", "A", 0.90), ("LIFO", "A", 0.90), ("ALIFO(0.5)", "A", 0.90), ("ALIFO(0.75)", "A", 0.90),
             ("FIFO", "A", 1.00), ("LIFO", "A", 1.00), ("ALIFO(0.5)", "A", 1.00),
             ("FIFO", "B", 0.60), ("LIFO", "B", 0.60), ("ALIFO(0.5)", "B", 0.60), ("ALIFO(0.75)", "B", 0.60),
             ("LIFO", "B", 1.05), ("ALIFO(0.5)", "B", 1.05), ("LIFO", "B", 1.10), ("LIFO", "B", 1.50)]
    tr_rows: list[dict[str, object]] = []
    fin_rows: list[dict[str, object]] = []
    tkd = tk.merge(ct, on="config_hash")
    for lab, pr, rho in cases:
        t = tkd[(tkd.label == lab) & (tkd.protocol == pr) & (np.isclose(tkd.rho_test, rho)) & (tkd.replicate < 10)]
        if t.empty:
            continue
        cfg = Config.model_validate(json.loads(t.cfg_json.iloc[0]))
        plan = plan_for(cfg, pr, rho)  # type: ignore[arg-type]
        acc: dict[int, list[list[float]]] = {}
        fin = []
        for seed in t.seed:
            sr = simulate(cfg, plan.schedule, plan.t_stop, int(seed))
            age = sr.starts[:, 1] - sr.att_t[sr.starts[:, 0].astype(int)]
            exp_start = age > cfg.T
            lo_w = plan.t_end - cfg.proto.w_last
            m_att = (sr.att_t >= lo_w) & (sr.att_t < plan.t_end)
            fin.append({
                "stale_q_end": int(((sr.att_state == QUEUED) & (sr.att_dl < sr.t_stop)).sum()), "q_end": sr.q_end,
                "frac_start_expired_final": float(exp_start[(sr.starts[:, 1] >= lo_w) & (sr.starts[:, 1] < plan.t_end)].mean()),
                "mean_age_start_final": float(age[(sr.starts[:, 1] >= lo_w) & (sr.starts[:, 1] < plan.t_end)].mean()),
                "rej_att_share_final": float((sr.att_state[m_att] == REJECTED).mean()),
                "throughput_final": float(((sr.done[:, 1] >= lo_w) & (sr.done[:, 1] < plan.t_end)).sum() / cfg.proto.w_last),
            })
            edges = np.arange(0.0, plan.t_end + 1e-9, 100.0)
            gser = goodput_series(sr, cfg.proto.window, edges[1:])
            for i in range(len(edges) - 1):
                m = (sr.samp_t >= edges[i]) & (sr.samp_t < edges[i + 1])
                ms = (sr.starts[:, 1] >= edges[i]) & (sr.starts[:, 1] < edges[i + 1])
                mr = (sr.att_t >= edges[i]) & (sr.att_t < edges[i + 1])
                acc.setdefault(i, []).append([
                    float(sr.samp_q[m].mean()), float(gser[i]),
                    float(exp_start[ms].mean()) if ms.any() else np.nan,
                    float((sr.att_state[mr] == REJECTED).mean()) if mr.any() else np.nan,
                ])
        fdf = pd.DataFrame(fin).mean()
        fin_rows.append({"cfg": lab, "protocol": pr, "rho": rho, "n": len(fin), **fdf.to_dict()})
        for i, v in acc.items():
            a = np.nanmean(np.array(v), axis=0)
            tr_rows.append({"cfg": lab, "protocol": pr, "rho": rho, "t_bin_start": i * 100, "n": len(v),
                            "qlen": a[0], "g_trailing": a[1], "frac_start_expired": a[2], "rej_att_share": a[3]})
    fr = pd.DataFrame(fin_rows)
    emit("Итоговое окно (последние 300), среднее по репликам:")
    emit(fr.round(3).to_string(index=False))
    tr = pd.DataFrame(tr_rows)
    tr.to_csv(OUT / "pilot3_traj.csv", index=False)
    for (lab, pr, rho), g in tr.groupby(["cfg", "protocol", "rho"], sort=False):
        emit(f"\n{lab}, {pr}, ρ = {rho}:")
        emit(g.drop(columns=["cfg", "protocol", "rho"]).round(3).T.to_string(header=False))
    agg.to_csv(OUT / "pilot3_diag.csv", index=False)
    sd.to_csv(OUT / "pilot3_diag_rhostar.csv", index=False)
    (OUT / "pilot3_diag.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
