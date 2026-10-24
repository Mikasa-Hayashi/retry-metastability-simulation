"""Основной анализ, п. 4 плана: метрики-спутники по 72 конфигурациям и время восстановления (Каплан-Мейер).

python analysis/main_metrics.py
Метрики (PREREGISTRATION, «Критерий отказа. Метрики-спутники»), грубая сетка ρ = 0,20 … 1,00 (17 точек), трапеции:
  L_A, L_B = ∫ (min(1, 1/ρ) − ḡ(ρ)) dρ  (потеря goodput), H = ∫ |ḡ_B − ḡ_A| dρ (площадь петли гистерезиса);
  wasted, усиление при ρ ∈ {0,7; 0,9; 1,0} (протокол A и B); время восстановления.
Bootstrap (B = 2000): внутри каждой точки ρ реплики пересэмплируются независимо (seed'ы основного прогона независимы,
общих случайных чисел нет), ДИ — перцентильный 95 %.
Выход: t04_metrics72.csv (T04), t04_boot.npz, t05_points72.csv (T05), t06_km_curves.csv, t06_km_summary.csv (T06),
t04_pseudo_LA.npz (псевдореплики L_A для Cliff's delta).
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_common import BASE_TAGS, BOOT, GRID, OUT, base_meta, load_main, pct, rng_for  # noqa: E402

GRID_A = np.array(GRID)
GSTAR = np.minimum(1.0, 1.0 / GRID_A)
PTS = (0.7, 0.9, 1.0)
NPSEUDO = 100  # максимум реплик на точку (активные ячейки); у остальных 40, остальное NaN


def mean_draws(rng: np.random.Generator, v: np.ndarray, b: int) -> np.ndarray:
    return v[rng.integers(0, len(v), (b, len(v)))].mean(axis=1)


def work(args: tuple[str, dict]) -> dict:
    h, d = args
    out: dict = {"config_hash": h}
    draws: dict[str, np.ndarray] = {}
    gbar: dict[str, np.ndarray] = {}
    gdraw: dict[str, np.ndarray] = {}
    pseudo = {}
    for pi, pr in enumerate(("A", "B")):
        rng = rng_for(h, 31 + pi)
        gm = np.empty(len(GRID))
        gd = np.empty((BOOT, len(GRID)))
        nrep = len(d[(pr, GRID[0])]["g_final"])
        ps = np.empty((nrep, len(GRID)))
        for i, rho in enumerate(GRID):
            v = d[(pr, rho)]["g_final"]
            gm[i] = v.mean()
            gd[:, i] = mean_draws(rng, v, BOOT)
            ps[:, i] = v
        gbar[pr], gdraw[pr] = gm, gd
        pseudo[pr] = np.full(NPSEUDO, np.nan)
        pseudo[pr][:nrep] = np.trapezoid(GSTAR[None, :] - ps, GRID_A, axis=1)
        l_hat = float(np.trapezoid(GSTAR - gm, GRID_A))
        l_dr = np.trapezoid(GSTAR[None, :] - gd, GRID_A, axis=1)
        out[f"L_{pr}"] = l_hat
        out[f"L_{pr}_lo"], out[f"L_{pr}_hi"] = pct(l_dr)
        draws[f"L_{pr}"] = l_dr
    h_hat = float(np.trapezoid(np.abs(gbar["B"] - gbar["A"]), GRID_A))
    h_dr = np.trapezoid(np.abs(gdraw["B"] - gdraw["A"]), GRID_A, axis=1)
    out["H"] = h_hat
    out["H_lo"], out["H_hi"] = pct(h_dr)
    draws["H"] = h_dr
    # wasted, усиление при ρ = 0,9 (протокол A): с bootstrap (вторичные метрики H2′)
    for col, nm in (("wasted_fraction", "wasted"), ("amplification", "ampl")):
        rng = rng_for(h, 41 + (nm == "ampl"))
        v = d[("A", 0.9)][col]
        dr = mean_draws(rng, v, BOOT)
        out[f"{nm}_A09"] = float(v.mean())
        out[f"{nm}_A09_lo"], out[f"{nm}_A09_hi"] = pct(dr)
        draws[f"{nm}_A09"] = dr
    out["_draws"] = draws
    out["_pseudoA"] = pseudo["A"]
    out["_pseudoB"] = pseudo["B"]
    return out


def rmst_km(t: np.ndarray, ev: np.ndarray, horizon: float) -> tuple[float, float]:
    """Ограниченное среднее (площадь под кривой Каплана-Мейера до горизонта) и дисперсия по Гринвуду."""
    times = np.unique(t[(ev == 1) & (t <= horizon)])
    surv, prev, area = 1.0, 0.0, 0.0
    steps = []
    for u in times:
        n_i = float((t >= u).sum())
        d_i = float(((t == u) & (ev == 1)).sum())
        area += surv * (u - prev)
        surv *= 1.0 - d_i / n_i
        prev = u
        steps.append((u, n_i, d_i, surv))
    area += surv * (horizon - prev)
    # остаточная площадь после каждого шага
    var = 0.0
    for j, (u, n_i, d_i, _s) in enumerate(steps):
        tail = 0.0
        sv, pv = steps[j][3], u
        for (u2, _n2, _d2, s2) in steps[j + 1:]:
            tail += sv * (u2 - pv)
            sv, pv = s2, u2
        tail += sv * (horizon - pv)
        if n_i > d_i:
            var += tail**2 * d_i / (n_i * (n_i - d_i))
    return float(area), float(var)


def km_tables(res: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from lifelines import KaplanMeierFitter

    r = res.merge(meta[["config_hash", "R", "Q", "D", "S"]], on="config_hash")
    r = r[r["tag"].isin(BASE_TAGS) & r["rho_test"].isin([0.7, 0.9])]
    horizon = float(r["recovery_time"].max())
    curves, rows = [], []
    for (pr, rho, R, Q), g in r.groupby(["protocol", "rho_test", "R", "Q"]):
        t = g["recovery_time"].to_numpy()
        ev = (g["recovery_censored"].to_numpy() == 0).astype(int)
        kmf = KaplanMeierFitter().fit(t, ev, label=f"{R}/{Q}")
        sf = kmf.survival_function_.join(kmf.confidence_interval_)
        sf.columns = ["surv", "lo", "hi"]
        sf = sf.reset_index().rename(columns={"timeline": "t"})
        sf.insert(0, "Q", Q)
        sf.insert(0, "R", R)
        sf.insert(0, "rho", rho)
        sf.insert(0, "protocol", pr)
        curves.append(sf)
        rmst, var = rmst_km(t, ev, horizon)
        med = kmf.median_survival_time_
        rows.append({"protocol": pr, "rho": rho, "R": R, "Q": Q, "n": len(g), "n_censored": int((ev == 0).sum()),
                     "share_censored": float((ev == 0).mean()), "share_never_degraded": float((t == 0).mean()),
                     "km_median": float(med) if np.isfinite(med) else np.nan, "rmst": float(rmst),
                     "rmst_lo": float(rmst - 1.96 * np.sqrt(var)), "rmst_hi": float(rmst + 1.96 * np.sqrt(var)), "horizon": horizon})
    return pd.concat(curves, ignore_index=True), pd.DataFrame(rows)


def main() -> int:
    meta = base_meta()
    cols = ["config_hash", "rho_test", "protocol", "g_final", "wasted_fraction", "amplification", "p99", "recovery_time",
            "recovery_censored", "replicate", "tag"]
    res = load_main(cols, BASE_TAGS)
    res = res[res["config_hash"].isin(meta["config_hash"])]
    jobs = []
    sub = res[res["rho_test"].isin(GRID)]
    gr = {k: g for k, g in sub.groupby(["config_hash", "protocol", "rho_test"])}
    for h in meta["config_hash"]:
        d = {}
        for pr in ("A", "B"):
            for rho in GRID:
                g = gr[(h, pr, rho)].sort_values("replicate")
                d[(pr, rho)] = {c: g[c].to_numpy(float) for c in ("g_final", "wasted_fraction", "amplification")}
        jobs.append((h, d))
    with mp.Pool(7) as pool:
        rows = pool.map(work, jobs, chunksize=2)
    keys = rows[0]["_draws"].keys()
    np.savez_compressed(OUT / "t04_boot.npz", config_hash=np.array(meta["config_hash"]),
                        **{k: np.stack([r["_draws"][k] for r in rows]) for k in keys})
    np.savez_compressed(OUT / "t04_pseudo_LA.npz", config_hash=np.array(meta["config_hash"]),
                        A=np.stack([r["_pseudoA"] for r in rows]), B=np.stack([r["_pseudoB"] for r in rows]))
    tab = meta[["config_hash", "R", "Q", "D", "S"]].merge(pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]), on="config_hash")
    tab.to_csv(OUT / "t04_metrics72.csv", index=False, encoding="utf-8")
    print("T04:", len(tab))

    # точечные значения при ρ = 0,7; 0,9; 1,0 (протоколы A, B)
    pt = res[res["rho_test"].isin(PTS)].copy()
    pt["rec_bounded"] = pt["recovery_time"]
    agg = (pt.groupby(["config_hash", "protocol", "rho_test"])
           .agg(n=("g_final", "size"), g=("g_final", "mean"), wasted=("wasted_fraction", "mean"),
                ampl=("amplification", "mean"), p99=("p99", "mean"), rec_bounded_mean=("rec_bounded", "mean"),
                rec_censored_share=("recovery_censored", "mean")).reset_index())
    agg = meta[["config_hash", "R", "Q", "D", "S"]].merge(agg, on="config_hash")
    agg.to_csv(OUT / "t05_points72.csv", index=False, encoding="utf-8")

    cv, sm = km_tables(res, meta)
    cv.to_csv(OUT / "t06_km_curves.csv", index=False, encoding="utf-8")
    sm.to_csv(OUT / "t06_km_summary.csv", index=False, encoding="utf-8")
    print("KM групп:", len(sm))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
