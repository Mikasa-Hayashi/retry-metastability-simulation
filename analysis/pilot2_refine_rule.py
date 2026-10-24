"""Pilot, часть 2.4: сравнение правил уточнения сетки ρ на плотных данных опорной конфигурации
(CALIBRATION_PLAN_2.md п. 4). «Истинная» кривая P(ρ) = монотонная (PAVA) линейная интерполяция
эмпирических долей по всем имеющимся точкам; симуляция схем с биномиальными выборками по R реплик.

python analysis/pilot2_refine_rule.py [--reps 1000] [--R 40]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runner import load_results, stage_dir
from src.thresholds import aggregate, pava, point_threshold

REF = "148eb8d011551aa1"
COARSE = np.round(np.arange(0.20, 1.0001, 0.05), 6)
# схемы: список раундов (шаг, полуширина)
SCHEMES: dict[str, list[tuple[float, float]]] = {
    "S0 грубая 0,05": [],
    "S1 0,025 ±0,075": [(0.025, 0.075)],
    "S2 0,01 ±0,05": [(0.01, 0.05)],
    "S3 0,025 ±0,075 + 0,01 ±0,025": [(0.025, 0.075), (0.01, 0.025)],
    "S4 0,01 ±0,03": [(0.01, 0.03)],
}


def truth_curve(rho: np.ndarray, failed: np.ndarray):
    x, n, k = aggregate(rho, failed)
    f = pava(k / n, n)
    return x, f


def run_scheme(rounds, x_t, f_t, R, rng, lo, hi):
    def p_at(r: float) -> float:
        return float(np.interp(r, x_t, f_t))

    pts: dict[float, list[int]] = {}

    def add(r: float) -> None:
        r = round(r, 6)
        if r in pts:
            return
        k = int(rng.binomial(R, p_at(r)))
        pts[r] = [R, k]

    for r in COARSE:
        add(float(r))
    n_runs = R * len(COARSE)

    def est():
        xs = np.array(sorted(pts))
        n = np.array([pts[v][0] for v in xs], dtype=float)
        k = np.array([pts[v][1] for v in xs], dtype=float)
        return xs, n, k, point_threshold(xs, n, k, 0.5, right_bound=hi)

    for step, hw in rounds:
        xs, n, k, (thr, cens, _) = est()
        if cens == "right":
            a, b = hi - hw, hi
        elif cens == "left":
            a, b = lo, lo + hw
        else:
            a, b = thr - hw, thr + hw
        a, b = max(a, lo), min(b, hi)
        i0, i1 = int(np.ceil(a / step - 1e-9)), int(np.floor(b / step + 1e-9))
        for i in range(i0, i1 + 1):
            r = round(i * step, 6)
            if r not in pts:
                add(r)
                n_runs += R
    xs, n, k, (thr, cens, _) = est()
    return thr, n_runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--R", type=int, default=40)
    a = ap.parse_args()
    res = load_results(stage_dir("pilot"))
    out_rows = []
    for proto, name in (("A", "rho_fold"), ("B", "rho_release")):
        r = res[(res["config_hash"] == REF) & (res["protocol"] == proto)]
        x_t, f_t = truth_curve(r["rho_test"].to_numpy(), r["failed"].to_numpy())
        xs, ns, ks = aggregate(r["rho_test"].to_numpy(), r["failed"].to_numpy())
        dense_thr = point_threshold(xs, ns, ks, 0.5, right_bound=1.0)
        # порог «истинной» кривой: первое пересечение 0,5 линейной интерполяцией
        j = int(np.flatnonzero(f_t >= 0.5)[0])
        t_true = float(x_t[j - 1] + (0.5 - f_t[j - 1]) / (f_t[j] - f_t[j - 1]) * (x_t[j] - x_t[j - 1]))
        print(f"{name}: точек {len(xs)}, реплик на точку min/med/max "
              f"{int(ns.min())}/{int(np.median(ns))}/{int(ns.max())}; порог по всем данным (логистика) "
              f"{dense_thr[0]:.4f}; порог PAVA-кривой {t_true:.4f}")
        for sname, rounds in SCHEMES.items():
            rng = np.random.default_rng(np.random.SeedSequence([4, 1 if proto == "A" else 2]))
            vals, runs = [], []
            for _ in range(a.reps):
                v, nr = run_scheme(rounds, x_t, f_t, a.R, rng, 0.20, 1.0)
                vals.append(v)
                runs.append(nr)
            vals_a = np.array(vals)
            out_rows.append(
                {
                    "порог": name, "схема": sname, "SD": vals_a.std(ddof=1),
                    "смещение_к_PAVA": vals_a.mean() - t_true,
                    "смещение_к_логистике": vals_a.mean() - dense_thr[0],
                    "RMSE_к_PAVA": float(np.sqrt(((vals_a - t_true) ** 2).mean())),
                    "прогонов/порог": float(np.mean(runs)),
                }
            )
    df = pd.DataFrame(out_rows)
    s2 = df[df["схема"].str.startswith("S2")].set_index("порог")["SD"]
    df["SD/SD(S2)"] = [r.SD / s2[r.порог] for r in df.itertuples()]
    pd.set_option("display.width", 250)
    print(df.round(4).to_string())
    df.to_csv(stage_dir("pilot") / "pilot2_refine_rule.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
