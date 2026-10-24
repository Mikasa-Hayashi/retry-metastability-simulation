"""Рисунки дополнения (эксплораторно): F14 гистограммы g_final, F15 чувствительность Δ и H1′ к определению отказа.

python analysis/main_addendum_figures.py     (после main_addendum.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_common import OUT, SS, save_fig  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})
SLAB = {"Exp": "Exp (cv = 1)", "LogN": "LogN (cv = 2)", "Pareto": "Pareto (α = 2,5)"}


def comma(ax: plt.Axes, axis: str = "both") -> None:
    from matplotlib.ticker import FuncFormatter

    f = FuncFormatter(lambda v, _p: f"{v:g}".replace(".", ","))
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(f)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(f)


def f14() -> None:
    h = pd.read_csv(OUT / "t40b_hist.csv")
    fig, axs = plt.subplots(4, 3, figsize=(12, 11), sharex=True)
    for i, (r, d) in enumerate((("R1", "off"), ("R1", "on"), ("R2", "off"), ("R2", "on"))):
        for j, s in enumerate(SS):
            ax = axs[i, j]
            g = h[(h.R == r) & (h.D == d) & (h.S == s)]
            ax.bar((g.bin_lo + g.bin_hi) / 2, g["count"].clip(lower=0.5), width=0.025, color="#bdbdbd", edgecolor="black", lw=0.4)
            ax.set_yscale("log")
            ax.set_ylim(0.5, 3e3)
            for th, ls in ((0.3, ":"), (0.5, "--"), (0.7, "-.")):
                ax.axvline(th, color="black", ls=ls, lw=1.0, label=f"θ = {th}".replace(".", ","))
            ax.axvline(g.valley_cut.iloc[0], color="#D55E00", lw=2.0, label="порог-долина")
            ax.set_title(f"{r}, FIFO, срок {d}, {SLAB[s]}", fontsize=9.5)
            if i == 3:
                ax.set_xlabel("g_final (доля успешных первичных запросов), 0-1")
            if j == 0:
                ax.set_ylabel("Число прогонов (лог. шкала)")
            comma(ax, "x")
    axs[0, 0].legend(fontsize=7.5, loc="upper center")
    fig.suptitle("Эксплораторно: распределение g_final по 5000 прогонам ячейки (протоколы A и B, ρ ≤ 1). Столбцы шириной 0,025; всюду два режима с провалом между ними.\n"
                 "При срок on режим отказа лежит около g = 0,2-0,4, то есть рядом с θ = 0,3-0,5", fontsize=10, y=1.0)
    fig.tight_layout()
    save_fig(fig, "F14_gistogrammy_g_final_eksploratorno", h)
    plt.close(fig)


def f15() -> None:
    t = pd.read_csv(OUT / "t41_theta_delta.csv")
    g = pd.read_csv(OUT / "t42_gap_delta.csv")
    hp = pd.read_csv(OUT / "t41b_theta_h1p.csv")
    gh = pd.read_csv(OUT / "t42b_gap_h1p.csv").iloc[0]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    cells = [(r, s) for r in ("R1", "R2") for s in SS]
    meas = [("θ = 0,3", t[t.theta == 0.3], "v"), ("θ = 0,5 (план)", t[t.theta == 0.5], "o"), ("θ = 0,7", t[t.theta == 0.7], "^"), ("порог-долина", g, "s")]
    rows = []
    for k, (lab, df, mk) in enumerate(meas):
        for ci, (r, s) in enumerate(cells):
            x = df[(df.R == r) & (df.D == "on") & (df.S == s)].iloc[0]
            v = x.delta if "delta" in x else x.delta_gap
            lo, hi = x.d_lo, x.d_hi
            ax.errorbar(ci + (k - 1.5) * 0.15, v, yerr=[[max(0, v - lo)], [max(0, hi - v)]], marker=mk, color=["#999999", "black", "#0072B2", "#D55E00"][k], capsize=2, ms=6,
                        ls="none", label=lab if ci == 0 else None)
            rows.append({"R": r, "S": s, "D": "on", "measure": lab, "delta": v, "lo": lo, "hi": hi})
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"{r}\n{SLAB[s]}" for r, s in cells], fontsize=8.5)
    ax.set_ylabel("Δ при срок on (доли ρ), 95 % ДИ")
    ax.set_title("Δ при разных определениях отказа (срок on, FIFO)", fontsize=10)
    ax.legend(fontsize=8.5)
    comma(ax, "y")
    ax = axs[1]
    labs = ["θ = 0,3", "θ = 0,5\n(план)", "θ = 0,7", "порог-\nдолина"]
    vals = list(zip(hp["diff"], hp.diff_lo, hp.diff_hi, strict=True)) + [(gh["diff"], gh.diff_lo, gh.diff_hi)]
    for k, (v, lo, hi) in enumerate(vals):
        ax.errorbar(k, v, yerr=[[v - lo], [hi - v]], marker="s", color="black", capsize=4, ms=7)
        rows.append({"R": "R1+R2", "S": "H1p", "D": "", "measure": labs[k].replace("\n", " "), "delta": v, "lo": lo, "hi": hi})
    ax.axhline(0, color="gray", lw=0.9)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labs)
    ax.set_ylim(-0.02, 0.4)
    ax.set_ylabel("r(Exp) − r(LogN), r = Δ(срок on)/Δ(срок off)")
    ax.set_title("H1′ при разных определениях отказа (95 % ДИ)", fontsize=10)
    comma(ax, "y")
    fig.suptitle("Эксплораторно: θ = 0,3 лежит внутри режима отказа при срок on, поэтому Δ занижается (LogN: нет отказов); при θ = 0,5-0,7 и по провалу между режимами Δ устойчива", fontsize=9.5, y=1.0)
    fig.tight_layout()
    save_fig(fig, "F15_chuvstvitelnost_k_opredeleniyu_otkaza_eksploratorno", pd.DataFrame(rows))
    plt.close(fig)


if __name__ == "__main__":
    f14()
    f15()
    print("F14, F15")
