"""Основной анализ, пп. 3-4 плана: классификация 72 конфигураций, факторная регрессия по активным ячейкам, парные сравнения
с поправкой Холма, размер эффекта по рангам (Cliff's delta).

python analysis/main_factorial.py     (после main_thresholds.py, main_metrics.py)
Регрессия: WLS (веса = 1 / дисперсия bootstrap Δ), только активные ячейки N1 (по плану); Q в них не варьируется (все FIFO), поэтому
эффект Q оценить в регрессии нельзя, он описан классификацией. ДИ коэффициентов — пересчёт регрессии по каждой из 2000 bootstrap-выборок Δ
(веса фиксированы). Парные сравнения: 24 заранее заданных различия Δ (R2−R1 по (D,S): 6; D on−off по (R,S): 6; пары S по (R,D): 12), p —
bootstrap (двусторонний, дискретность 1/2001), Холм по всем 24. Cliff's delta: по «псевдорепликам» L_A (индекс реплики
ставится в соответствие точкам ρ произвольно, т. к. seed'ы независимы; оценивает распределение L_A в целом); эксплораторно.
Выход: t30_classification.csv, t30_pairs_holm.csv, t31_regression_wls.csv, t31_marginal_contrasts.csv, t32_cliff.csv,
t33_heatmap_delta.csv, t34_marginal_LA.csv (эксплораторно), t35_LA_RxQ.csv (эксплораторно).
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_common import BOOT, DS, OUT, QS, RS, SS, ACTIVE12, boot_p, holm, pct, rng_for  # noqa: E402

CELL = list[tuple]


def load() -> dict:
    t01 = pd.read_csv(OUT / "t01_thresholds72.csv")
    b1 = np.load(OUT / "t01_boot.npz", allow_pickle=True)
    t02 = pd.read_csv(OUT / "t02_ext72.csv")
    t04 = pd.read_csv(OUT / "t04_metrics72.csv")
    b4 = np.load(OUT / "t04_boot.npz", allow_pickle=True)
    idx = {(r.R, r.Q, r.D, r.S): i for i, r in enumerate(t01.itertuples())}
    return {"t01": t01, "t02": t02, "t04": t04, "b4": b4, "idx": idx, "delta_dr": b1["fold"] - b1["rel"]}


def classification(c: dict) -> pd.DataFrame:
    t, e = c["t01"], c["t02"]
    rows = []
    t = t.assign(N1_ext=e.N1.to_numpy(), N2_ext=e.N2.to_numpy(), fold_cens_right=(t.fold_cens == "right"), rel_cens_right=(t.rel_cens == "right"),
                 bi=(t.cls == "метастабильность (N1)"))
    for fac in ("R", "Q", "D", "S"):
        for lv, g in t.groupby(fac):
            rows.append({"factor": fac, "level": lv, "n": len(g), "N1": int(g.N1.sum()), "N2": int(g.N2.sum()), "N1_ext": int(g.N1_ext.sum()),
                         "fold_censored": int(g.fold_cens_right.sum()), "release_censored": int(g.rel_cens_right.sum()),
                         "mean_delta": float(g.delta.mean()), "max_delta": float(g.delta.max())})
    for (r, q), g in t.groupby(["R", "Q"]):
        rows.append({"factor": "R×Q", "level": f"{r}/{q}", "n": len(g), "N1": int(g.N1.sum()), "N2": int(g.N2.sum()), "N1_ext": int(g.N1_ext.sum()),
                     "fold_censored": int(g.fold_cens_right.sum()), "release_censored": int(g.rel_cens_right.sum()),
                     "mean_delta": float(g.delta.mean()), "max_delta": float(g.delta.max())})
    rows.append({"factor": "все", "level": "72", "n": len(t), "N1": int(t.N1.sum()), "N2": int(t.N2.sum()), "N1_ext": int(t.N1_ext.sum()),
                 "fold_censored": int(t.fold_cens_right.sum()), "release_censored": int(t.rel_cens_right.sum()),
                 "mean_delta": float(t.delta.mean()), "max_delta": float(t.delta.max())})
    return pd.DataFrame(rows)


def dd(c: dict, r: str, q: str, d: str, s: str) -> np.ndarray:
    return c["delta_dr"][c["idx"][(r, q, d, s)]]


def dp(c: dict, r: str, q: str, d: str, s: str) -> float:
    return float(c["t01"].delta.iloc[c["idx"][(r, q, d, s)]])


def design(cells: list[tuple]) -> tuple[np.ndarray, list[str]]:
    """Матрица плана с кодированием «обработка» (база R1, D=off, S=Exp): главные эффекты и все двойные взаимодействия."""
    names = ["Intercept", "R2", "D=on", "S=LogN", "S=Pareto", "R2:D=on", "R2:S=LogN", "R2:S=Pareto", "D=on:S=LogN", "D=on:S=Pareto"]
    X = []
    for (r, _q, d, s) in cells:
        R2, Dn, L, P = float(r == "R2"), float(d == "on"), float(s == "LogN"), float(s == "Pareto")
        X.append([1, R2, Dn, L, P, R2 * Dn, R2 * L, R2 * P, Dn * L, Dn * P])
    return np.array(X), names


def regression(c: dict) -> pd.DataFrame:
    t = c["t01"]
    cells = [k for k in ACTIVE12 if bool(t.N1.iloc[c["idx"][k]])]
    if len(cells) != 12:
        print("ВНИМАНИЕ: активных ячеек не 12:", len(cells))
    y = np.array([dp(c, *k) for k in cells])
    Y = np.stack([dd(c, *k) for k in cells])
    w = 1.0 / np.var(Y, axis=1, ddof=1)
    rows = []
    X_full, names_full = design(cells)
    for model, cols in (("главные эффекты + все двойные взаимодействия (10 параметров, 2 ст. свободы остатка)", list(range(10))),
                        ("только главные эффекты (5 параметров, 7 ст. свободы остатка)", [0, 1, 2, 3, 4])):
        X = X_full[:, cols]
        W = np.diag(w)
        A = np.linalg.solve(X.T @ W @ X, X.T @ W)
        beta = A @ y
        bb = A @ Y  # [p, B]
        resid = y - X @ beta
        rss_w = float(np.sum(w * resid**2))
        ybar = np.sum(w * y) / np.sum(w)
        tss_w = float(np.sum(w * (y - ybar) ** 2))
        for j, ci in enumerate(cols):
            lo, hi = pct(bb[j])
            rows.append({"model": model, "term": names_full[ci], "coef": float(beta[j]), "lo": lo, "hi": hi,
                         "weighted_R2": 1 - rss_w / tss_w, "max_abs_resid": float(np.abs(resid).max()), "n_cells": len(cells)})
    return pd.DataFrame(rows)


def contrasts_marginal(c: dict) -> pd.DataFrame:
    cells = ACTIVE12
    rows = []

    def add(label: str, pos: list[tuple], neg: list[tuple], family: str) -> None:
        dr = np.mean([dd(c, *k) for k in pos], axis=0) - np.mean([dd(c, *k) for k in neg], axis=0)
        pt = np.mean([dp(c, *k) for k in pos]) - np.mean([dp(c, *k) for k in neg])
        lo, hi = pct(dr)
        rows.append({"family": family, "contrast": label, "effect": float(pt), "lo": lo, "hi": hi, "n_cells_each": len(pos)})

    add("R2 − R1 (D, S усреднены)", [k for k in cells if k[0] == "R2"], [k for k in cells if k[0] == "R1"], "маргинальные")
    add("D on − off (R, S усреднены)", [k for k in cells if k[2] == "on"], [k for k in cells if k[2] == "off"], "маргинальные")
    for a, b in (("LogN", "Exp"), ("Pareto", "Exp"), ("Pareto", "LogN")):
        add(f"S {a} − {b} (R, D усреднены)", [k for k in cells if k[3] == a], [k for k in cells if k[3] == b], "маргинальные")
    for s in SS:
        add(f"D on − off при S = {s} (R усреднены)", [k for k in cells if k[2] == "on" and k[3] == s], [k for k in cells if k[2] == "off" and k[3] == s], "условные")
    for d in DS:
        add(f"R2 − R1 при D = {d} (S усреднены)", [k for k in cells if k[0] == "R2" and k[2] == d], [k for k in cells if k[0] == "R1" and k[2] == d], "условные")
    return pd.DataFrame(rows)


def pairs_holm(c: dict) -> pd.DataFrame:
    rows = []

    def add(label: str, grp: str, k1: tuple, k2: tuple) -> None:
        dr = dd(c, *k1) - dd(c, *k2)
        pt = dp(c, *k1) - dp(c, *k2)
        lo, hi = pct(dr)
        rows.append({"group": grp, "comparison": label, "effect": pt, "lo": lo, "hi": hi, "p_boot": boot_p(dr)})

    for d, s in product(DS, SS):
        add(f"R2 − R1 | FIFO, D={d}, S={s}", "R2−R1", ("R2", "FIFO", d, s), ("R1", "FIFO", d, s))
    for r, s in product(("R1", "R2"), SS):
        add(f"D on − off | {r}/FIFO, S={s}", "D on−off", (r, "FIFO", "on", s), (r, "FIFO", "off", s))
    for r, d in product(("R1", "R2"), DS):
        for a, b in (("Exp", "LogN"), ("Exp", "Pareto"), ("LogN", "Pareto")):
            add(f"S {a} − {b} | {r}/FIFO, D={d}", "пары S", (r, "FIFO", d, a), (r, "FIFO", d, b))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_boot"].to_numpy())
    out["significant_holm_0.05"] = out["p_holm"] < 0.05
    return out


def cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    return float((np.sign(x[:, None] - y[None, :])).mean())


def cliff_table(c: dict) -> pd.DataFrame:
    ps = np.load(OUT / "t04_pseudo_LA.npz", allow_pickle=True)
    pa = ps["A"]
    idx = c["idx"]
    key_pairs = [
        ("R2 vs R1 | FIFO", lambda d, s: (("R2", "FIFO", d, s), ("R1", "FIFO", d, s))),
        ("R3 vs R2 | FIFO", lambda d, s: (("R3", "FIFO", d, s), ("R2", "FIFO", d, s))),
        ("R3 vs R2 | ALIFO", lambda d, s: (("R3", "ALIFO", d, s), ("R2", "ALIFO", d, s))),
        ("ALIFO vs FIFO | R2", lambda d, s: (("R2", "ALIFO", d, s), ("R2", "FIFO", d, s))),
        ("LIFO vs FIFO | R2", lambda d, s: (("R2", "LIFO", d, s), ("R2", "FIFO", d, s))),
        ("ALIFO vs LIFO | R2", lambda d, s: (("R2", "ALIFO", d, s), ("R2", "LIFO", d, s))),
        ("R1 vs R0 | FIFO", lambda d, s: (("R1", "FIFO", d, s), ("R0", "FIFO", d, s))),
    ]
    rows = []
    for name, fn in key_pairs:
        deltas, cell_rows = [], []
        draws_all = []
        for d, s in product(DS, SS):
            k1, k2 = fn(d, s)
            x, y = pa[idx[k1]], pa[idx[k2]]
            x, y = x[~np.isnan(x)], y[~np.isnan(y)]
            rng = rng_for(c["t01"].config_hash.iloc[idx[k1]], 500 + idx[k2])
            bs = np.array([cliff_delta(rng.choice(x, len(x)), rng.choice(y, len(y))) for _ in range(BOOT)])
            deltas.append(cliff_delta(x, y))
            draws_all.append(bs)
            lo, hi = pct(bs)
            cell_rows.append({"pair": name, "level": f"D={d}, S={s}", "cliff_delta": deltas[-1], "lo": lo, "hi": hi, "n_first": len(x), "n_second": len(y)})
        lo, hi = pct(np.mean(draws_all, axis=0))
        rows.extend(cell_rows)
        rows.append({"pair": name, "level": "среднее по 6 ячейкам (D, S)", "cliff_delta": float(np.mean(deltas)), "lo": lo, "hi": hi,
                     "n_first": np.nan, "n_second": np.nan})
    # D on vs off в активных ячейках
    deltas, draws_all = [], []
    for r, s in product(("R1", "R2"), SS):
        k1, k2 = (r, "FIFO", "on", s), (r, "FIFO", "off", s)
        x, y = pa[idx[k1]], pa[idx[k2]]
        x, y = x[~np.isnan(x)], y[~np.isnan(y)]
        rng = rng_for(c["t01"].config_hash.iloc[idx[k1]], 700 + idx[k2])
        bs = np.array([cliff_delta(rng.choice(x, len(x)), rng.choice(y, len(y))) for _ in range(BOOT)])
        deltas.append(cliff_delta(x, y))
        draws_all.append(bs)
        lo, hi = pct(bs)
        rows.append({"pair": "D on vs off | FIFO, R1/R2", "level": f"{r}, S={s}", "cliff_delta": deltas[-1], "lo": lo, "hi": hi, "n_first": len(x), "n_second": len(y)})
    lo, hi = pct(np.mean(draws_all, axis=0))
    rows.append({"pair": "D on vs off | FIFO, R1/R2", "level": "среднее по 6 ячейкам (R, S)", "cliff_delta": float(np.mean(deltas)), "lo": lo, "hi": hi,
                 "n_first": np.nan, "n_second": np.nan})
    return pd.DataFrame(rows)


def heatmap_table(c: dict) -> pd.DataFrame:
    t = c["t01"][["R", "Q", "D", "S", "fold", "fold_cens", "rel", "rel_cens", "delta", "d_lo", "d_hi", "N1", "N2", "cls", "delta_status"]].copy()
    return t


def marginal_LA(c: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    t04, b4 = c["t04"], c["b4"]
    rows = []
    for key in ("L_A", "L_B", "H"):
        for fac in ("R", "Q", "D", "S"):
            levels = {"R": RS, "Q": QS, "D": DS, "S": SS}[fac]
            ref = levels[0]
            m_ref = (t04[fac] == ref).to_numpy()
            dr_ref = b4[key][m_ref].mean(axis=0)
            for lv in levels:
                m = (t04[fac] == lv).to_numpy()
                dr = b4[key][m].mean(axis=0)
                lo, hi = pct(dr)
                dlo, dhi = pct(dr - dr_ref)
                rows.append({"metric": key, "factor": fac, "level": lv, "n_cells": int(m.sum()), "mean": float(t04[key][m].mean()),
                             "lo": lo, "hi": hi, "diff_vs_first_level": float(t04[key][m].mean() - t04[key][m_ref].mean()), "diff_lo": dlo, "diff_hi": dhi})
    rq = []
    for r, q in product(RS, QS):
        m = ((t04.R == r) & (t04.Q == q)).to_numpy()
        dr = b4["L_A"][m].mean(axis=0)
        lo, hi = pct(dr)
        rq.append({"R": r, "Q": q, "L_A_mean": float(t04["L_A"][m].mean()), "lo": lo, "hi": hi})
    return pd.DataFrame(rows), pd.DataFrame(rq)


def main() -> int:
    c = load()
    t = classification(c)
    t.to_csv(OUT / "t30_classification.csv", index=False, encoding="utf-8")
    print(t.round(3).to_string())
    t = pairs_holm(c)
    t.to_csv(OUT / "t30_pairs_holm.csv", index=False, encoding="utf-8")
    print(t.round(4).to_string())
    t = regression(c)
    t.to_csv(OUT / "t31_regression_wls.csv", index=False, encoding="utf-8")
    print(t.round(3).to_string())
    t = contrasts_marginal(c)
    t.to_csv(OUT / "t31_marginal_contrasts.csv", index=False, encoding="utf-8")
    print(t.round(3).to_string())
    t = cliff_table(c)
    t.to_csv(OUT / "t32_cliff.csv", index=False, encoding="utf-8")
    print(t[t.level.str.startswith("среднее")].round(3).to_string())
    heatmap_table(c).to_csv(OUT / "t33_heatmap_delta.csv", index=False, encoding="utf-8")
    m, rq = marginal_LA(c)
    m.to_csv(OUT / "t34_marginal_LA.csv", index=False, encoding="utf-8")
    rq.to_csv(OUT / "t35_LA_RxQ.csv", index=False, encoding="utf-8")
    print(m[m.metric == "L_A"].round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
