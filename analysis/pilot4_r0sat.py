"""Pilot, часть 4, п. 7: насыщение или эффект повторов (R0 против R3, FIFO/off, ρ = 0,90…1,05 шаг 0,01).

python analysis/pilot4_r0sat.py
Общий блок, реплики 0…99 (парные: одинаковые случайные числа у R0 и R3). Для каждого S и протокола: доля отказов P(ρ) у R0 и R3,
разность P_R3 − P_R0 с парным bootstrap-ДИ, пороги (P = 0,5, окно 0,90…1,05; цензурирование справа на 1,05), g_final, очередь.
Критерий (заранее, см. план): насыщение — |P_R3 − P_R0| ≤ 0,1 при всех ρ и ДИ порогов перекрываются.
Эксплораторно: H_rec = 3000 и 6000 (R0, R3, Exp). Вывод: results/pilot/pilot4_r0sat.csv|txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot4_common import OUT, cfg_meta, load_crn

from src.thresholds import point_threshold

NB = 2000
NREP = 100
RHOS = [round(0.90 + 0.01 * i, 2) for i in range(16)]


def mats(d: pd.DataFrame, col: str) -> np.ndarray:
    pv = d.pivot_table(index="rho_test", columns="replicate", values=col, aggfunc="first").reindex(index=RHOS, columns=range(NREP))
    return pv.to_numpy()


def thr_boot(F: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Порог P = 0,5 по каждой бутстреп-выборке; F: [ρ, реплика] (0/1)."""
    x = np.array(RHOS)
    out = np.full(len(idx), np.nan)
    for b, ii in enumerate(idx):
        k = np.nansum(F[:, ii], axis=1)
        n = np.full(len(x), float(len(ii)))
        out[b] = point_threshold(x, n, k, 0.5, right_bound=1.05)[0]
    return out


def main() -> int:
    meta = cfg_meta()
    base = meta[(meta.Q == "FIFO") & (meta.D == "off") & (meta.m == 3) & (meta["T"] == 10.0) & (meta.K == 128) & (meta.theta_q == 0.5)
                & (meta.theta == 0.5) & (meta.trig == 1.5) & (meta.ol_len == 1000.0) & (meta.R.isin(["R0", "R3"]))]
    res = load_crn(["config_hash", "rho_test", "protocol", "failed", "g_final", "qlen_mean_final", "replicate", "wasted_fraction"])
    res["rho_test"] = res["rho_test"].round(2)
    res = res[res.rho_test.isin(RHOS) & (res.replicate < NREP)]
    rng = np.random.default_rng(np.random.SeedSequence(77))
    idx = rng.integers(0, NREP, (NB, NREP))
    L: list[str] = []
    rows = []
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 500)
    for hrec in (1500.0, 3000.0, 6000.0):
        b = base[base.h_rec == hrec]
        for s in ("Exp", "LogN", "Pareto"):
            h0 = b[(b.R == "R0") & (b.S == s)]
            h3 = b[(b.R == "R3") & (b.S == s)]
            if h0.empty or h3.empty:
                continue
            for pr in ("A", "B"):
                d0 = res[(res.config_hash == h0.config_hash.iloc[0]) & (res.protocol == pr)]
                d3 = res[(res.config_hash == h3.config_hash.iloc[0]) & (res.protocol == pr)]
                F0, F3 = mats(d0, "failed"), mats(d3, "failed")
                G0, G3 = mats(d0, "g_final"), mats(d3, "g_final")
                Q0, Q3 = mats(d0, "qlen_mean_final"), mats(d3, "qlen_mean_final")
                P0, P3 = np.nanmean(F0, 1), np.nanmean(F3, 1)
                diff_b = np.nanmean(F3[:, idx], axis=2) - np.nanmean(F0[:, idx], axis=2)  # [ρ, NB]
                lo, hi = np.percentile(diff_b, [2.5, 97.5], axis=1)
                t0 = point_threshold(np.array(RHOS), np.full(len(RHOS), float(NREP)), np.nansum(F0, 1), 0.5, 1.05)
                t3 = point_threshold(np.array(RHOS), np.full(len(RHOS), float(NREP)), np.nansum(F3, 1), 0.5, 1.05)
                b0, b3 = thr_boot(F0, idx[:1000]), thr_boot(F3, idx[:1000])
                dthr = b3 - b0
                n_obs = np.sum(~np.isnan(F0), axis=1).min()
                rows.append({"h_rec": hrec, "S": s, "protocol": pr, "n_rep": int(n_obs),
                             "thr_R0": t0[0], "thr_R0_cens": t0[1], "thr_R0_lo": np.percentile(b0, 2.5), "thr_R0_hi": np.percentile(b0, 97.5),
                             "thr_R3": t3[0], "thr_R3_cens": t3[1], "thr_R3_lo": np.percentile(b3, 2.5), "thr_R3_hi": np.percentile(b3, 97.5),
                             "dthr": t3[0] - t0[0], "dthr_lo": np.percentile(dthr, 2.5), "dthr_hi": np.percentile(dthr, 97.5),
                             "max_abs_dP": float(np.nanmax(np.abs(P3 - P0))),
                             "n_rho_dP_ci_excl0": int(np.sum((lo > 0) | (hi < 0)))})
                t = pd.DataFrame({"rho": RHOS, "P_R0": P0, "P_R3": P3, "dP": P3 - P0, "dP_lo": lo, "dP_hi": hi,
                                  "g_R0": np.nanmean(G0, 1), "g_R3": np.nanmean(G3, 1), "q_R0": np.nanmean(Q0, 1), "q_R3": np.nanmean(Q3, 1)})
                L.append(f"\n## H_rec = {hrec:g}, S = {s}, протокол {pr}, реплик {n_obs}: порог R0 {t0[0]:.3f} ({t0[1]}), R3 {t3[0]:.3f} ({t3[1]})")
                L.append(t.round(3).to_string(index=False))
    sm = pd.DataFrame(rows)
    sm.to_csv(OUT / "pilot4_r0sat.csv", index=False)
    head = ["# pilot4 r0sat: общий блок, реплики 0…99, FIFO/off, ρ = 0,90…1,05; парный bootstrap (общие индексы реплик)",
            "\n## Сводка (пороги P = 0,5, цензурирование справа на 1,05)", sm.round(3).to_string(index=False)]
    out = "\n".join(head + L)
    print(out)
    (OUT / "pilot4_r0sat.txt").write_text(out + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
