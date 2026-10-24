"""Pilot, часть 4: проверка выполнимости H1 и H2 на данных скрининга (reports/pilot_runs_4.md).

python analysis/pilot4_h12.py
Вход: results/pilot/pilot4_screen_main.csv (Δ по ρ ≤ 1 и Δ_ext). H1: τ-b Кендалла между векторами Δ (Exp) и Δ (LogN) по ячейкам
R×Q×D (18 без R0 и 24 с R0), bootstrap по ячейкам (эксплораторный). H2: контраст взаимодействия по Δ. Вывод: pilot4_h12.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot4_common import OUT

lines: list[str] = []


def emit(s: str = "") -> None:
    print(s)
    lines.append(s)


def tau(x: np.ndarray, y: np.ndarray) -> float:
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    return float(kendalltau(x, y).statistic)


def tau_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> str:
    t0 = tau(x, y)
    bs = [v for v in (tau(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(2000))) if not np.isnan(v)]
    if not bs:
        return f"τ-b = {t0:.3f}, ДИ не определён"
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"τ-b = {t0:.3f}, bootstrap по ячейкам [{lo:.3f}; {hi:.3f}] (валидных {len(bs)}/2000)"


def main() -> int:
    df = pd.read_csv(OUT / "pilot4_screen_main.csv")
    rng = np.random.default_rng(np.random.SeedSequence(5))
    for pre, name in (("le1", "Δ(ρ ≤ 1)"), ("ext", "Δ_ext (ρ ≤ 2,5)")):
        d = df.copy()
        d["cell"] = d.R + "/" + d.Q + "/" + d.D
        pv = d.pivot(index="cell", columns="S", values=f"{pre}_delta")
        n1 = d.pivot(index="cell", columns="S", values=f"{pre}_N1")
        n2 = d.pivot(index="cell", columns="S", values=f"{pre}_N2")
        nz = d.assign(nz=d[f"{pre}_delta"].abs() > 0.1).pivot(index="cell", columns="S", values="nz")
        emit(f"\n# H1, {name}: ячеек всего {len(pv)} (из них без R0 {len(pv) - 6})")
        emit(f"N1 при Exp {int(n1['Exp'].sum())}, LogN {int(n1['LogN'].sum())}, Pareto {int(n1['Pareto'].sum())}; Exp и LogN одновременно "
             f"{int((n1['Exp'] & n1['LogN']).sum())}; хотя бы в одном {int((n1['Exp'] | n1['LogN']).sum())}")
        emit(f"N2 (Δ ≥ 0,1) при Exp {int(n2['Exp'].sum())}, LogN {int(n2['LogN'].sum())}; одновременно {int((n2['Exp'] & n2['LogN']).sum())}")
        noR0 = pv[~pv.index.str.startswith("R0")]
        x, y = noR0["Exp"].to_numpy(), noR0["LogN"].to_numpy()
        emit(f"(а) 18 ячеек без R0: {tau_ci(x, y, rng)}")
        x24, y24 = pv["Exp"].to_numpy(), pv["LogN"].to_numpy()
        emit(f"(б) 24 ячейки с R0: {tau_ci(x24, y24, rng)}")
        sel = ((n1['Exp'] | n1['LogN']).reindex(noR0.index)).to_numpy()
        emit(f"(в) невырожденные хотя бы в одном S ({int(sel.sum())}): " + (tau_ci(x[sel], y[sel], rng) if sel.sum() >= 3 else "мало"))
        emit(f"Доля пар ячеек со связкой (ties) в Δ(Exp): {1 - np.unique(x).size / len(x):.2f}; число уникальных значений Δ(Exp) {np.unique(np.round(x, 2)).size} из 18")
        emit(pv.round(3).assign(N1_Exp=n1["Exp"], N1_LogN=n1["LogN"]).to_string())
        # H2
        ix = d.set_index(["R", "Q", "D", "S"])
        rows = []
        for dd in ("off", "on"):
            for ss in ("Exp", "LogN", "Pareto"):
                g = lambda R, Q, dd=dd, ss=ss: ix.loc[(R, Q, dd, ss)]
                a, b, c_, e = g("R3", "ALIFO"), g("R3", "FIFO"), g("R2", "ALIFO"), g("R2", "FIFO")
                dl = f"{pre}_delta"
                rows.append({"D": dd, "S": ss, "d_R3ALIFO": a[dl], "d_R3FIFO": b[dl], "d_R2ALIFO": c_[dl], "d_R2FIFO": e[dl],
                             "contrast": a[dl] - b[dl] - c_[dl] + e[dl]})
        h2 = pd.DataFrame(rows)
        emit(f"\n# H2, {name}: контраст C = Δ(R3,ALIFO) − Δ(R3,FIFO) − Δ(R2,ALIFO) + Δ(R2,FIFO)")
        emit(h2.round(3).to_string(index=False))
        emit(f"C < 0 в {int((h2.contrast < 0).sum())} из 6; min {h2.contrast.min():.3f}, max {h2.contrast.max():.3f}")
        # Δ в ячейках, где одиночные меры дают Δ ≈ 0
        emit(f"Одиночные меры (R3/FIFO и R2/ALIFO) дают |Δ| < 0,1: {int(((h2.d_R3FIFO.abs() < 0.1) & (h2.d_R2ALIFO.abs() < 0.1)).sum())} из 6")
    (OUT / "pilot4_h12.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
