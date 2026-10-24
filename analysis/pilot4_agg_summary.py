"""Pilot, часть 4, п. 4: сводка по факторам агрессивности клиента (из pilot4_screen_agg.csv).

python analysis/pilot4_agg_summary.py
Какие уровни (m, T, θ_q) дают информативные Δ (N1: ρ_fold < верхней границы и нижняя граница ДИ Δ > 0; N2: Δ ≥ 0,1) в ячейках
LIFO, ALIFO, R3. Пороги по ρ ≤ 1,5 (Δ_ext) и по ρ ≤ 1 (Δ(ρ ≤ 1)). Вывод: results/pilot/pilot4_agg_summary.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot4_common import OUT

lines: list[str] = []


def emit(s: str = "") -> None:
    print(s)
    lines.append(s)


def main() -> int:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 600)
    df = pd.read_csv(OUT / "pilot4_screen_agg.csv")
    df["cls_all"] = df.R + "/" + df.Q
    emit(f"# pilot4 agg: конфигураций {len(df)}; реплик в точке (min) {df.n_min.min()}")
    for pre, nm in (("le1", "Δ(ρ ≤ 1)"), ("ext", "Δ_ext (ρ ≤ 1,5)")):
        emit(f"\n## {nm}: число N1 / N2 по (m, T) × Q")
        t = df.groupby(["m", "T", "Q"]).agg(n=("config_hash", "size"), N1=(f"{pre}_N1", "sum"), N2=(f"{pre}_N2", "sum")).reset_index()
        emit(t.to_string(index=False))
        emit(f"\n## {nm}: по (m, T) × R")
        t = df.groupby(["m", "T", "R"]).agg(n=("config_hash", "size"), N1=(f"{pre}_N1", "sum"), N2=(f"{pre}_N2", "sum")).reset_index()
        emit(t.to_string(index=False))
        al = df[df.Q == "ALIFO"]
        emit(f"\n## {nm}: ALIFO по θ_q × (m, T)")
        t = al.groupby(["theta_q", "m", "T"]).agg(n=("config_hash", "size"), N1=(f"{pre}_N1", "sum"), N2=(f"{pre}_N2", "sum"),
                                                   delta_mean=(f"{pre}_delta", "mean"), delta_max=(f"{pre}_delta", "max")).reset_index()
        emit(t.round(3).to_string(index=False))
        emit(f"\n## {nm}: информативные (N1) ячейки среди LIFO, ALIFO, R3")
        sel = df[(df[f"{pre}_N1"]) & ((df.Q.isin(["LIFO", "ALIFO"])) | (df.R == "R3"))]
        cols = ["R", "Q", "D", "S", "m", "T", "theta_q", f"{pre}_fold", f"{pre}_rel", f"{pre}_delta", f"{pre}_d_lo", f"{pre}_d_hi"]
        emit(sel[cols].round(3).to_string(index=False) if len(sel) else "нет")
    emit("\n## Полная таблица по конфигурациям (Δ(ρ ≤ 1), Δ_ext, класс)")
    cols = ["R", "Q", "D", "S", "m", "T", "theta_q", "le1_fold", "le1_rel", "le1_delta", "le1_d_lo", "le1_cls", "ext_delta", "ext_cls", "rho_star_A", "rho_star_B", "maxPA", "maxPB"]
    emit(df.sort_values(["Q", "R", "D", "S", "m", "T", "theta_q"])[cols].round(3).to_string(index=False))
    (OUT / "pilot4_agg_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
