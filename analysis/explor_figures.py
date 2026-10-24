"""Рисунки F16-F21 (эксплораторно) по таблицам results/analysis/explor_*.csv.

python analysis/explor_figures.py     (после explor_k2.py, explor_trg.py, explor_sens.py, explor_ol.py)
Рядом с каждым рисунком лежат png, svg и csv с данными (figures/).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explor_common import HRECS, K2_CELLS, OUT, RU_CELL, save_fig  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})
CB = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73", "black": "#000000", "purple": "#CC79A7", "grey": "#7f7f7f"}
NOTE = "Эксплораторно (вне препрегистрации). "


def comma(ax: plt.Axes, axis: str = "both") -> None:
    f = FuncFormatter(lambda v, _p: f"{v:g}".replace(".", ","))
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(f)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(f)


def hticks(ax: plt.Axes) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(HRECS)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{int(v)}"))
    ax.minorticks_off()
    ax.tick_params(axis="x", labelsize=8.5)


def f16() -> None:
    t = pd.read_csv(OUT / "explor_k2_thresholds.csv")
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, (r, q, d, s) in zip(axs, K2_CELLS, strict=True):
        g = t[(t.R == r) & (t.D == d) & (t.S == s)].sort_values("H_rec")
        x = g.H_rec.to_numpy(float)
        for col, lo, hi, lab, mk, ls, c in (("fold", "fold_lo", "fold_hi", "ρ_fold (порог срыва, A)", "s", "--", CB["orange"]),
                                           ("rel", "rel_lo", "rel_hi", "ρ_release (порог восстановления, B)", "o", "-", CB["blue"]),
                                           ("delta", "d_lo", "d_hi", "Δ = ρ_fold − ρ_release", "^", ":", CB["black"])):
            y = g[col].to_numpy(float)
            ax.errorbar(x, y, yerr=[y - g[lo].to_numpy(float), g[hi].to_numpy(float) - y], marker=mk, ls=ls, color=c, capsize=3, label=lab, ms=6)
        for col, c in (("rel", CB["blue"]), ("delta", CB["black"])):
            xx = np.log2(x / 1500.0)
            sl, ic = np.polyfit(xx, g[col].to_numpy(float), 1)
            ax.plot(x, ic + sl * xx, color=c, lw=0.8, alpha=0.6)
        hticks(ax)
        ax.set_xlabel("Горизонт восстановления H_rec, ед. E[S]")
        ax.set_title(RU_CELL[(r, q, d, s)])
        ax.axhline(0.1, color=CB["grey"], lw=0.8, ls="-.")
        comma(ax, "y")
    axs[0].set_ylabel("Порог нагрузки ρ и ширина Δ (безразм.)")
    axs[1].legend(fontsize=8, loc="center left")
    fig.suptitle(NOTE + "Пороги и Δ в зависимости от H_rec (ось log2): у R1/FIFO/выкл/Exp ρ_release растёт примерно на 0,016 за удвоение H; "
                 "у R1/вкл/LogN окно Δ закрывается.\nТонкие линии: МНК по log2 H. Штрихпунктир: Δ = 0,1. Усы: 95 % ДИ bootstrap по репликам. "
                 "H_rec = 1500, 3000, 6000 из основного прогона (R = 40)", fontsize=9, y=1.07)
    fig.tight_layout()
    save_fig(fig, "F16_porogi_i_delta_ot_H_rec_eksploratorno", t)
    plt.close(fig)


def f17() -> None:
    h = pd.read_csv(OUT / "explor_k2_h3.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = h.H_rec.to_numpy(float)
    y = h.dev.to_numpy(float)
    lo, hi = h.dev_lo.to_numpy(float), h.dev_hi.to_numpy(float)
    ax.errorbar(x[:3], y[:3], yerr=[y[:3] - lo[:3], hi[:3] - y[:3]], fmt="o", color=CB["blue"], capsize=3, ms=7, label="основной прогон (R = 40)")
    ax.errorbar(x[3:], y[3:], yerr=[y[3:] - lo[3:], hi[3:] - y[3:]], fmt="D", color=CB["orange"], capsize=3, ms=7, label="дозапуск explor (R = 40)")
    last = h.iloc[-1]
    xs = 2 ** np.linspace(np.log2(1500), np.log2(48000), 50)
    ax.plot(xs, last.intercept_1500 + last.slope_per_doubling * np.log2(xs / 1500.0), color=CB["black"], lw=1, ls="--", label="МНК по log2 H (5 точек)")
    ax.axhline(0.10, color=CB["black"], lw=1.2, ls="-.", label="критерий H3: |sim − MF| ≤ 0,10")
    ax.axvspan(last.H_star_lo, last.H_star_hi, color=CB["grey"], alpha=0.25,
               label=f"H* = {last.H_star_dev010:.0f} [{last.H_star_lo:.0f}; {last.H_star_hi:.0f}]".replace(".", ","))
    ax.set_xscale("log", base=2)
    ax.set_xticks([1500, 3000, 6000, 12000, 24000, 48000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{int(v)}"))
    ax.minorticks_off()
    ax.set_xlabel("Горизонт восстановления H_rec, ед. E[S]")
    ax.set_ylabel("ρ_release (имитация) − ρ_release (mean-field), безразм.")
    ax.set_title("Эксплораторно: отклонение от mean-field, R1/FIFO/выкл/Exp\n(MF ρ_release = 0,277; ДИ bootstrap по репликам)", fontsize=10)
    comma(ax, "y")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    save_fig(fig, "F17_otklonenie_H3_ot_H_rec_eksploratorno", h)
    plt.close(fig)


def f18() -> None:
    sv = pd.read_csv(OUT / "explor_k2_survival.csv")
    lf = pd.read_csv(OUT / "explor_k2_lifetime.csv")
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axs[0]
    g = sv[(sv.R == "R1") & (sv.D == "off")]
    hs = [1500, 3000, 6000, 12000, 24000]
    for rho, mk, ls, c in ((0.35, "o", "-", CB["blue"]), (0.40, "s", "--", CB["orange"]), (0.45, "^", ":", CB["black"])):
        r = g[np.isclose(g.rho, rho)].iloc[0]
        y = np.array([r[f"S_{h}"] for h in hs])
        lo = np.array([r[f"S_{h}_lo"] for h in hs])
        hi = np.array([r[f"S_{h}_hi"] for h in hs])
        ax.errorbar(hs, y, yerr=[y - lo, hi - y], marker=mk, ls=ls, color=c, capsize=3, label=f"ρ = {rho}".replace(".", ","))
    hticks(ax)
    ax.set_xlabel("Время t после перегрузки, ед. E[S]")
    ax.set_ylabel("Доля прогонов, не вернувшихся к g ≥ 0,9")
    ax.set_title("R1/FIFO/выкл/Exp: оценка Каплана-Майера\n(протокол B, H_rec = 24000)", fontsize=9.5)
    comma(ax, "y")
    ax.legend(fontsize=8.5)
    ax = axs[1]
    for (r, q, d, s), mk, c in zip(K2_CELLS, ("o", "s", "^"), (CB["blue"], CB["orange"], CB["black"]), strict=True):
        pts = lf[(lf.R == r) & (lf.D == d) & (lf.S == s) & (lf.kind == "точка")]
        fit = lf[(lf.R == r) & (lf.D == d) & (lf.S == s) & (lf.kind == "подгонка")].iloc[0]
        ok = pts.tau_hat.notna()
        ax.plot(pts.rho[ok], np.log(pts.tau_hat[ok]), marker=mk, ls="none", color=c, ms=7,
                label=f"{RU_CELL[(r, q, d, s)]}: наклон {fit.slope_ln_tau_per_rho:.0f}")
        sl, ic = np.polyfit(pts.rho[ok], np.log(pts.tau_hat[ok]), 1)
        xx = np.linspace(pts.rho.min() - 0.02, pts.rho.max() + 0.02, 10)
        ax.plot(xx, ic + sl * xx, color=c, lw=1)
    ax.axhline(np.log(24000), color=CB["grey"], ls="-.", lw=1, label="горизонт 24000 (цензурирование)")
    ax.set_xlabel("Нагрузка ρ после перегрузки (безразм.)")
    ax.set_ylabel("ln τ, τ - оценка среднего времени жизни отказа")
    ax.set_title("Время жизни отказа растёт экспоненциально по ρ", fontsize=9.5)
    comma(ax, "x")
    ax.legend(fontsize=8)
    fig.suptitle(NOTE + "τ = Σ min(t, H) / число выходов (экспоненциальная модель с цензурированием). "
                 "Сдвиг ρ_release на удвоение H оценён как ln 2 / наклон.", fontsize=9, y=1.03)
    fig.tight_layout()
    save_fig(fig, "F18_vremya_zhizni_otkaza_eksploratorno", pd.concat([sv.assign(table="survival"), lf.assign(table="lifetime")], ignore_index=True))
    plt.close(fig)


def f19() -> None:
    t = pd.read_csv(OUT / "explor_trg.csv")
    cells = t[["R", "Q", "D", "S"]].drop_duplicates().to_records(index=False).tolist()
    trigs = [1.5, 3.0, 4.0]

    def pick(col: str) -> np.ndarray:
        return np.array([[t[(t.R == r) & (t.Q == q) & (t.D == d) & (t.S == s) & (t.trig_mult == m)][col].iloc[0] for m in trigs]
                         for (r, q, d, s) in cells])

    pa, dl = pick("P_A_1.0"), pick("delta")
    cens = pick("fold_cens") == "right"
    fig, axs = plt.subplots(1, 2, figsize=(11, 6.4), sharey=True)
    labels = [f"{r}/{q}/{'вкл' if d == 'on' else 'выкл'}/{s}" for (r, q, d, s) in cells]
    xl = ["(1,5; 25)\nбаза", "(3; 50)", "(4; 100)"]
    for ax, mat, ttl, vmax, isd in ((axs[0], pa, "Доля отказов P_A при ρ = 1,0", 1.0, False),
                                    (axs[1], dl, "Δ = ρ_fold − ρ_release (— : ρ_fold > 1,0, не оценён)", 0.2, True)):
        ax.grid(False)
        ax.imshow(mat, cmap="Greys", vmin=0, vmax=vmax, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                dash = isd and cens[i, j]
                txt = "—" if dash else f"{mat[i, j] + 0.0:.2f}".replace("-0.00", "0.00").replace(".", ",")
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white" if (mat[i, j] > 0.55 * vmax and not dash) else "black")
        ax.set_xticks(range(3))
        ax.set_xticklabels(xl)
        ax.set_xlabel("Триггер (m_s; D_s), D_s в ед. E[S]")
        ax.set_title(ttl, fontsize=9.5)
    axs[0].set_yticks(range(len(cells)))
    axs[0].set_yticklabels(labels, fontsize=8.5)
    fig.suptitle(NOTE + "12 «нулевых» ячеек: сильные триггеры не создают гистерезис (Δ ≥ 0,1 нет ни в одной).\n"
                 "R3/FIFO/выкл срывается около ρ ≈ 0,96 без удержания отказа (Δ ≈ 0). ρ_release взят из базы (протокол B от триггера не зависит). R = 40",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    save_fig(fig, "F19_silnyj_trigger_nulevye_yacheiki_eksploratorno", t)
    plt.close(fig)


def f20() -> None:
    s = pd.read_csv(OUT / "explor_sens.csv")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.2))
    cmap = {("off", "Exp"): (CB["blue"], "o", "-"), ("off", "LogN"): (CB["orange"], "s", "--"), ("off", "Pareto"): (CB["green"], "^", ":"),
            ("on", "Exp"): (CB["black"], "D", "-"), ("on", "LogN"): (CB["purple"], "v", "--"), ("on", "Pareto"): (CB["grey"], "P", ":")}
    for row, (r, vals, xl, ttl) in enumerate((("R2", [0.5, 1.0, 2.0], "Базовая пауза повтора retry.base (ед. E[S])", "R2: паузы короче и длиннее"),
                                              ("R3", [0.05, 0.10, 0.20], "Бюджет повторов (доля от первичных)", "R3: бюджет повторов"))):
        for col, (qty, yl) in enumerate((("fold", "ρ_fold (порог срыва, протокол A)"), ("L_A", "L_A (потеря goodput по ρ, протокол A)"))):
            ax = axs[row, col]
            for (d, sg), (c, mk, ls) in cmap.items():
                g = s[(s.R == r) & (s.D == d) & (s.S == sg)].sort_values("value")
                y = g[qty].to_numpy(float)
                ax.errorbar(g.value, y, yerr=[y - g[f"{qty}_lo"].to_numpy(float), g[f"{qty}_hi"].to_numpy(float) - y], marker=mk, ls=ls, color=c,
                            capsize=2, ms=5, label=f"{'вкл' if d == 'on' else 'выкл'}/{sg}")
            ax.set_xscale("log")
            ax.set_xticks(vals)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}".replace(".", ",")))
            ax.minorticks_off()
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(ttl, fontsize=10)
            comma(ax, "y")
    axs[0, 0].legend(fontsize=7.5, ncol=2, title="срок/распределение", title_fontsize=8)
    fig.suptitle(NOTE + "Чувствительность R2 и R3 (протокол A, R = 40, FIFO). Значения 1,0 и 0,10 - основной прогон. ρ_release вариантов не измерен.\n"
                 "ρ_fold и L_A почти не зависят от параметра; R3 отличается от R2 по L_A и по ρ_fold на 0,08-0,19", fontsize=9, y=1.02)
    fig.tight_layout()
    save_fig(fig, "F20_chuvstvitelnost_R2_R3_eksploratorno", s)
    plt.close(fig)


def f21() -> None:
    o = pd.read_csv(OUT / "explor_ol.csv")
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    cmap = {("R1", "off"): (CB["blue"], "-"), ("R1", "on"): (CB["orange"], "--"), ("R2", "off"): (CB["green"], "-."), ("R2", "on"): (CB["black"], ":")}
    mk = {"Exp": "o", "LogN": "s", "Pareto": "^"}
    for ax, (col, lo, hi, yl) in zip(axs, (("rel", "rel_lo", "rel_hi", "ρ_release (порог восстановления, B)"),
                                           ("delta", "d_lo", "d_hi", "Δ = ρ_fold − ρ_release")), strict=True):
        for (r, d), (c, ls) in cmap.items():
            for sg in ("Exp", "LogN", "Pareto"):
                g = o[(o.R == r) & (o.D == d) & (o.S == sg)].sort_values("ol_len")
                y = g[col].to_numpy(float)
                ax.errorbar(g.ol_len, y, yerr=[y - g[lo].to_numpy(float), g[hi].to_numpy(float) - y], marker=mk[sg], ls=ls, color=c, capsize=2, ms=5,
                            label=f"{r}/{'вкл' if d == 'on' else 'выкл'}/{sg}")
        ax.set_xscale("log")
        ax.set_xticks([400, 1000, 2000])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{int(v)}"))
        ax.minorticks_off()
        ax.set_xlabel("Длина перегрузки в протоколе B, ед. E[S]")
        ax.set_ylabel(yl)
        comma(ax, "y")
    axs[1].legend(fontsize=7, ncol=2, loc="upper right")
    fig.suptitle(NOTE + "Длина перегрузки (400 / 1000 / 2000) почти не сдвигает ρ_release и Δ (максимум 0,012). Протокол B, FIFO, 12 активных ячеек, R = 40",
                 fontsize=9, y=1.0)
    fig.tight_layout()
    save_fig(fig, "F21_dlina_peregruzki_eksploratorno", o)
    plt.close(fig)


def main() -> int:
    for f in (f16, f17, f18, f19, f20, f21):
        f()
        print(f.__name__, "ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
