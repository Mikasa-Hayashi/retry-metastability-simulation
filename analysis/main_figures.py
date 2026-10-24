"""Графики основного анализа (figures/: png + svg + csv с данными рядом). Подписи на русском, различимы в ч/б (маркеры, штриховки, стили линий).

python analysis/main_figures.py     (после main_thresholds, main_metrics, main_sensitivity, main_hypotheses, main_factorial)
Рисунки: F01-F02 кривые P_A, P_B по политикам; F03 дисциплины очереди; F04 «петля гистерезиса» goodput; F05 столбцы Δ с ДИ;
F06 heatmap R×Q по Δ для каждого S; F07 рейтинг по S (H1) и эффект передачи срока (H1′); F08 теория против симуляции (H3);
F09 кривые восстановления (Каплан-Мейер); F10 чувствительность; F11 H2 и H2′; F12 область ρ > 1 (Δ_ext); F13 L_A, R×Q (эксплораторно).
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
from main_common import (  # noqa: E402
    BASE_TAGS, DS, EXT_TAGS, OUT, QS, RS, SS, VARIANTS, base_meta, load_main, norm_failed, save_fig,
)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "axes.titlesize": 10.5,
                     "legend.fontsize": 8.5, "figure.dpi": 100, "axes.axisbelow": True})
COL = {"R0": "#000000", "R1": "#0072B2", "R2": "#D55E00", "R3": "#009E73"}
MK = {"R0": "o", "R1": "s", "R2": "^", "R3": "D"}
LS = {"R0": "-", "R1": "--", "R2": "-.", "R3": ":"}
QCOL = {"FIFO": "#000000", "LIFO": "#CC79A7", "ALIFO": "#E69F00"}
QMK = {"FIFO": "o", "LIFO": "v", "ALIFO": "P"}
SCOL = {"Exp": "#bdbdbd", "LogN": "#636363", "Pareto": "#ffffff"}
SHATCH = {"Exp": "", "LogN": "///", "Pareto": "xx"}
RULAB = {"R0": "R0 (без повторов)", "R1": "R1 (фикс. задержка)", "R2": "R2 (выдержка + джиттер)", "R3": "R3 (R2 + бюджет)"}
SLAB = {"Exp": "Exp (cv = 1)", "LogN": "LogN (cv = 2)", "Pareto": "Pareto (α = 2,5)"}
XR = "Нагрузка ρ (безразм.)"


def comma(ax: plt.Axes, axis: str = "both") -> None:
    from matplotlib.ticker import FuncFormatter

    f = FuncFormatter(lambda v, _p: f"{v:g}".replace(".", ","))
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(f)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(f)


def load() -> dict:
    return {"cur": pd.read_csv(OUT / "t03_curves_base.csv"), "t01": pd.read_csv(OUT / "t01_thresholds72.csv"),
            "t02": pd.read_csv(OUT / "t02_ext72.csv"), "t04": pd.read_csv(OUT / "t04_metrics72.csv")}


def f01_f02(c: dict) -> None:
    cur = c["cur"]
    for fig_id, d in (("F01_PA_PB_po_politikam_D_off", "off"), ("F02_PA_PB_po_politikam_D_on", "on")):
        fig, axs = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True, sharey=True)
        rows = []
        for j, s in enumerate(SS):
            for i, pr in enumerate(("A", "B")):
                ax = axs[i, j]
                for r in RS:
                    g = cur[(cur.R == r) & (cur.Q == "FIFO") & (cur.D == d) & (cur.S == s) & (cur.protocol == pr)].sort_values("rho_test")
                    ax.errorbar(g.rho_test, g.P, yerr=1.96 * g.P_se, color=COL[r], marker=MK[r], ls=LS[r], ms=3.5, lw=1.3, capsize=1.5,
                                label=RULAB[r], elinewidth=0.7)
                    rows.append(g.assign(figure=fig_id))
                ax.axhline(0.5, color="gray", lw=0.8, ls=(0, (1, 3)))
                ax.set_title(f"{SLAB[s]}, протокол {pr}")
                if i == 1:
                    ax.set_xlabel(XR)
                if j == 0:
                    ax.set_ylabel("Доля отказов P (по репликам), 0-1")
                ax.set_xlim(0.2, 1.0)
                comma(ax)
        axs[0, 0].legend(loc="upper left")
        fig.suptitle(f"Доля отказов в протоколах A (путь «здоровый → отказ») и B (путь «перегруженный → снижение»), Q = FIFO, передача срока {d}\n"
                     "Планки: 95 % биномиальные; пунктир: уровень P = 0,5", y=1.0, fontsize=10.5)
        fig.tight_layout()
        save_fig(fig, fig_id, pd.concat(rows))
        plt.close(fig)


def f03(c: dict) -> None:
    cur = c["cur"]
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True, sharey=True)
    rows = []
    for j, r in enumerate(("R1", "R2", "R3")):
        for i, pr in enumerate(("A", "B")):
            ax = axs[i, j]
            for q in QS:
                g = cur[(cur.R == r) & (cur.Q == q) & (cur.D == "off") & (cur.S == "Exp") & (cur.protocol == pr)].sort_values("rho_test")
                ax.plot(g.rho_test, g.P, color=QCOL[q], marker=QMK[q], ms=4, lw=1.3, label=q, ls="-" if q == "FIFO" else "--")
                rows.append(g)
            ax.set_title(f"{RULAB[r]}, протокол {pr}")
            ax.set_xlim(0.2, 1.0)
            if i == 1:
                ax.set_xlabel(XR)
            if j == 0:
                ax.set_ylabel("Доля отказов P (по репликам), 0-1")
            comma(ax)
    axs[0, 0].legend(title="Дисциплина очереди", loc="upper left")
    fig.suptitle("Влияние дисциплины очереди на долю отказов (S = Exp, передача срока off)", y=1.0, fontsize=10.5)
    fig.tight_layout()
    save_fig(fig, "F03_dolya_otkazov_po_distsiplinam", pd.concat(rows))
    plt.close(fig)


def f04(c: dict) -> None:
    cur = c["cur"]
    fig, axs = plt.subplots(4, 3, figsize=(12, 13), sharex=True, sharey=True)
    rows = []
    for k, (r, d) in enumerate((("R1", "off"), ("R1", "on"), ("R2", "off"), ("R2", "on"))):
        for j, s in enumerate(SS):
            ax = axs[k, j]
            ga = cur[(cur.R == r) & (cur.Q == "FIFO") & (cur.D == d) & (cur.S == s) & (cur.protocol == "A")].sort_values("rho_test")
            gb = cur[(cur.R == r) & (cur.Q == "FIFO") & (cur.D == d) & (cur.S == s) & (cur.protocol == "B")].sort_values("rho_test")
            for g, lab, ls, mk, col in ((ga, "A: после всплеска", "-", "o", "#000000"), (gb, "B: после перегрузки", "--", "s", "#0072B2")):
                se = 1.96 * g.g_sd / np.sqrt(g.n)
                ax.plot(g.rho_test, g.g, ls=ls, marker=mk, ms=3, color=col, lw=1.3, label=lab)
                ax.fill_between(g.rho_test, g.g - se, g.g + se, color=col, alpha=0.15)
                rows.append(g.assign(panel=f"{r}/{d}/{s}"))
            ax.axhline(0.5, color="gray", lw=0.8, ls=(0, (1, 3)))
            ax.set_title(f"{r}, FIFO, срок {d}, {SLAB[s]}")
            ax.set_xlim(0.2, 1.0)
            ax.set_ylim(0, 1.03)
            if k == 3:
                ax.set_xlabel(XR)
            comma(ax)
    fig.supylabel("Доля успешных первичных запросов g (goodput), 0-1")
    axs[0, 0].legend(loc="lower left")
    fig.suptitle("«Петля гистерезиса»: goodput в конце наблюдения при подходе из здорового (A) и перегруженного (B) состояния\n"
                 "Лента: 95 % по репликам; пунктир: порог отказа θ = 0,5", y=1.0, fontsize=10.5)
    fig.tight_layout()
    save_fig(fig, "F04_petlya_gisterezisa_goodput", pd.concat(rows))
    plt.close(fig)


def f05(c: dict) -> None:
    t = c["t01"]
    groups = [(r, d) for r in RS for d in DS]
    fig, ax = plt.subplots(figsize=(12, 5))
    w = 0.26
    rows = []
    for gi, (r, d) in enumerate(groups):
        for si, s in enumerate(SS):
            x = t[(t.R == r) & (t.Q == "FIFO") & (t.D == d) & (t.S == s)].iloc[0]
            pos = gi + (si - 1) * w
            ax.bar(pos, x.delta, w * 0.92, color=SCOL[s], edgecolor="black", hatch=SHATCH[s], lw=0.8, label=SLAB[s] if gi == 0 else None)
            ax.errorbar(pos, x.delta, yerr=[[max(0, x.delta - x.d_lo)], [max(0, x.d_hi - x.delta)]], color="black", capsize=2, lw=0.9)
            rows.append({"R": r, "Q": "FIFO", "D": d, "S": s, "delta": x.delta, "d_lo": x.d_lo, "d_hi": x.d_hi, "cls": x.cls, "N1": x.N1})
            if x.cls == "нет отказов на сетке":
                ax.text(pos, 0.012, "0*", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{r}\nсрок {d}" for r, d in groups])
    ax.set_ylabel("Ширина области бистабильности Δ = ρ_fold − ρ_release")
    ax.set_xlabel("Политика повторов и передача срока (Q = FIFO)")
    ax.set_ylim(-0.02, 0.62)
    ax.legend(loc="upper right")
    comma(ax, "y")
    ax.set_title("Δ по конфигурациям Q = FIFO с 95 % ДИ (bootstrap по репликам)", fontsize=10.5)
    fig.text(0.01, 0.005, "0* — отказов на сетке ρ ≤ 1 нет (оба порога цензурированы справа, Δ = 0 по соглашению, не оценка). "
             "Все 48 конфигураций LIFO и ALIFO: Δ = 0*.\nR3, срок off: пороги ≈ 0,96 без гистерезиса.", fontsize=8.5, va="bottom", ha="left")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    save_fig(fig, "F05_stolbtsy_delta_po_konfiguratsiyam", pd.DataFrame(rows))
    plt.close(fig)


def f06(c: dict) -> None:
    t = c["t01"]
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 6.6))
    rows = []
    for i, d in enumerate(DS):
        for j, s in enumerate(SS):
            ax = axs[i, j]
            m = np.zeros((4, 3))
            for a, r in enumerate(RS):
                for b, q in enumerate(QS):
                    x = t[(t.R == r) & (t.Q == q) & (t.D == d) & (t.S == s)].iloc[0]
                    m[a, b] = x.delta
                    rows.append({"R": r, "Q": q, "D": d, "S": s, "delta": x.delta, "cls": x.cls})
                    txt = "0*" if x.cls == "нет отказов на сетке" else f"{x.delta:.2f}".replace(".", ",")
                    ax.text(b, a, txt, ha="center", va="center", color="white" if x.delta > 0.3 else "black", fontsize=10)
            im = ax.imshow(m, cmap="Greys", vmin=0, vmax=0.7, aspect="auto")
            ax.set_xticks(range(3))
            ax.set_xticklabels(QS)
            ax.set_yticks(range(4))
            ax.set_yticklabels(RS)
            ax.grid(False)
            ax.set_title(f"{SLAB[s]}, срок {d}")
            if i == 1:
                ax.set_xlabel("Дисциплина очереди Q")
            if j == 0:
                ax.set_ylabel("Политика повторов R")
    fig.colorbar(im, ax=axs, shrink=0.8, label="Δ = ρ_fold − ρ_release (доли нагрузки ρ)")
    fig.suptitle("Δ по сетке R × Q для каждого распределения времени обслуживания S и передачи срока D. 0* — отказов нет при ρ ≤ 1", y=0.99, fontsize=10.5)
    save_fig(fig, "F06_heatmap_RxQ_po_delta", pd.DataFrame(rows))
    plt.close(fig)


def f07(c: dict) -> None:
    t, t4 = c["t01"], c["t04"]
    h1 = pd.read_csv(OUT / "t20_h1.csv")
    hp = pd.read_csv(OUT / "t21_h1prime.csv")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    rows = []
    for ax, (key, tab, lab) in zip(axs[:2], (("delta", t, "Δ"), ("L_A", t4, "L_A")), strict=True):
        e = tab[tab.S == "Exp"].set_index(["R", "Q", "D"])[key]
        l = tab[tab.S == "LogN"].set_index(["R", "Q", "D"])[key]
        for r in RS:
            idx = [i for i in e.index if i[0] == r]
            ax.scatter(e[idx], l[idx], marker=MK[r], s=42, facecolors="none" if r == "R0" else COL[r], edgecolors="black" if r != "R0" else COL[r],
                       label=RULAB[r], alpha=0.85, linewidths=1.0)
            for i in idx:
                rows.append({"panel": lab, "R": i[0], "Q": i[1], "D": i[2], "Exp": e[i], "LogN": l[i]})
        if key == "L_A":
            ax.set_xscale("log")
            ax.set_yscale("log")
            lim = (4e-4, 0.4)
            ax.plot(lim, lim, color="gray", lw=0.8)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xlabel("L_A при S = Exp (потеря goodput, доли)")
            ax.set_ylabel("L_A при S = LogN")
        else:
            ax.plot((-0.02, 0.6), (-0.02, 0.6), color="gray", lw=0.8)
            ax.set_xlim(-0.03, 0.62)
            ax.set_ylim(-0.03, 0.62)
            ax.set_xlabel("Δ при S = Exp (доли нагрузки ρ)")
            ax.set_ylabel("Δ при S = LogN")
        r_ = h1[(h1.metric == ("Δ" if key == "delta" else "L_A")) & h1["set"].str.startswith("18")].iloc[0]
        ax.set_title(f"{lab}: τ-b = {r_.tau_b:.2f}".replace(".", ",") + f" (ДИ по репликам [{r_.rep_lo:.2f}; {r_.rep_hi:.2f}])".replace(".", ","), fontsize=9.5)
        ax.legend(loc="upper left" if key == "delta" else "lower right", fontsize=7.5)
        if key == "delta":
            ax.annotate("14 из 18 ячеек\nв точке (0; 0)", xy=(0, 0), xytext=(0.12, 0.05), arrowprops={"arrowstyle": "->"}, fontsize=8)
    ax = axs[2]
    pr = hp[(hp.scope.str.startswith("R1 и R2")) & (hp.comparison.str.startswith("r(")) & (hp.comparison.str.contains("Δ_on"))]
    for k, s in enumerate(SS):
        x = pr[pr.comparison.str.startswith(f"r({s})")].iloc[0]
        ax.bar(k, x.r_first, 0.6, color=SCOL[s], edgecolor="black", hatch=SHATCH[s], lw=0.8)
        ax.errorbar(k, x.r_first, yerr=[[x.r_first - x.lo], [x.hi - x.r_first]], color="black", capsize=3)
        ax.text(k, x.hi + 0.02, f"{x.r_first:.2f}".replace(".", ","), ha="center", fontsize=9)
        rows.append({"panel": "r(S)", "S": s, "r": x.r_first, "lo": x.lo, "hi": x.hi})
    ax.set_xticks(range(3))
    ax.set_xticklabels([SLAB[s] for s in SS])
    ax.set_ylabel("Остаточное Δ: r(S) = Δ(срок on) / Δ(срок off)")
    ax.set_ylim(0, 1)
    ax.set_title("H1′: эффект передачи срока зависит от S (R1, R2 усреднены, 95 % ДИ)", fontsize=9.5)
    comma(ax, "y")
    fig.tight_layout()
    save_fig(fig, "F07_reiting_po_S_H1_i_H1prime", pd.DataFrame(rows))
    plt.close(fig)


def f08(c: dict) -> None:
    cur = c["cur"]
    h3 = pd.read_csv(OUT / "t24_h3.csv")
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    rows = []
    mf_rel, mf_fold = float(h3.mf_release.iloc[0]), float(h3.mf_fold.iloc[0])
    for pr, ls, mk, col in (("A", "-", "o", "#000000"), ("B", "--", "s", "#0072B2")):
        g = cur[(cur.R == "R1") & (cur.Q == "FIFO") & (cur.D == "off") & (cur.S == "Exp") & (cur.protocol == pr)].sort_values("rho_test")
        ax.errorbar(g.rho_test, g.P, yerr=1.96 * g.P_se, ls=ls, marker=mk, ms=3.5, color=col, capsize=1.5, lw=1.2, elinewidth=0.7,
                    label=f"симуляция, протокол {pr}")
        rows.append(g)
    ax.axvline(mf_rel, color="#D55E00", lw=1.6, ls="-.", label=f"mean-field ρ_release = {mf_rel:.3f}".replace(".", ","))
    ax.axvline(mf_fold, color="#009E73", lw=1.6, ls=":", label=f"mean-field ρ_fold = {mf_fold:.3f}".replace(".", ","))
    ax.axvspan(mf_rel - 0.10, mf_rel + 0.10, color="#D55E00", alpha=0.10, label="допуск H3: ±0,10")
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel(XR)
    ax.set_ylabel("Доля отказов P (по репликам), 0-1")
    ax.set_title("R1, FIFO, срок off, Exp: симуляция и спинодали mean-field модели")
    ax.legend(loc="center left", fontsize=8)
    comma(ax)
    ax = axs[1]
    sel = h3[h3.H_rec.isin([1500, 3000, 6000]) & ~h3.case.str.startswith("справочно") & (h3.reps.isin([100, 40]))].copy()
    sel = sel[~sel.case.str.contains("base40")]
    for k, r in enumerate(sel.itertuples()):
        ax.errorbar(k, r.dev, yerr=[[r.dev - r.dev_lo], [r.dev_hi - r.dev]], marker="s", color="#D55E00", capsize=4, ms=7,
                    label="ρ_release: симуляция − теория" if k == 0 else None)
        ax.errorbar(k + 0.0, r.fold_dev, yerr=[[r.fold_dev - r.fold_dev_lo], [r.fold_dev_hi - r.fold_dev]], marker="o", color="#009E73", capsize=4, ms=7,
                    ls="none", label="ρ_fold: симуляция − теория (H3b)" if k == 0 else None)
        rows.append({"case": r.case, "H_rec": r.H_rec, "dev_release": r.dev, "lo": r.dev_lo, "hi": r.dev_hi, "dev_fold": r.fold_dev,
                     "fold_lo": r.fold_dev_lo, "fold_hi": r.fold_dev_hi})
    ax.axhspan(-0.10, 0.10, color="gray", alpha=0.15, label="допуск H3: ±0,10")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(sel)))
    ax.set_xticklabels([f"H_rec = {int(h)}\n(R = {int(n)})" for h, n in zip(sel.H_rec, sel.reps, strict=True)])
    ax.set_ylabel("Отклонение симуляции от mean-field (доли ρ)")
    ax.set_title("H3 и H3b при разных окнах восстановления H_rec (95 % ДИ)")
    ax.legend(loc="lower left", fontsize=8)
    comma(ax, "y")
    fig.tight_layout()
    data = pd.concat([r if isinstance(r, pd.DataFrame) else pd.DataFrame([r]) for r in rows], ignore_index=True)
    save_fig(fig, "F08_teoriya_protiv_simulyatsii_H3", data)
    plt.close(fig)


def f09() -> None:
    km = pd.read_csv(OUT / "t06_km_curves.csv")
    fig, axs = plt.subplots(3, 2, figsize=(11.5, 10), sharex=True, sharey=True)
    for i, q in enumerate(QS):
        for j, pr in enumerate(("A", "B")):
            ax = axs[i, j]
            for r in RS:
                g = km[(km.protocol == pr) & (km.rho == 0.9) & (km.R == r) & (km.Q == q)]
                ax.step(g.t, g.surv, where="post", color=COL[r], ls=LS[r], lw=1.6, label=RULAB[r])
                ax.fill_between(g.t, g.lo, g.hi, step="post", color=COL[r], alpha=0.10)
            ax.set_xscale("symlog", linthresh=20)
            ax.set_ylim(0, 1.03)
            ax.set_title(f"Q = {q}, протокол {pr}, ρ = 0,9")
            if i == 2:
                ax.set_xlabel("Время восстановления t (ед. E[S] = 1), шкала симлог")
            comma(ax, "y")
    fig.supylabel("Доля ещё не восстановившихся S(t), 0-1")
    axs[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Кривые восстановления (Каплан-Мейер; цензура горизонтом H_rec = 1500; D и S объединены внутри R × Q)\nЛента: 95 % ДИ", y=1.0, fontsize=10.5)
    fig.text(0.01, 0.005, "Прогоны, не деградировавшие после триггера, входят как t = 0. Из-за скользящего окна W = 100 время восстановления при наличии отказов не меньше ≈ 100.\n"
             "Единица времени: E[S] = 1 (параметры не калиброваны на реальную систему).", fontsize=8, va="bottom", ha="left")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save_fig(fig, "F09_krivye_vosstanovleniya", km[(km.rho == 0.9)])
    plt.close(fig)


def f10() -> None:
    s = pd.read_csv(OUT / "t10_sens.csv")
    order = ["base40", "theta03", "theta07", "T5", "T20", "K64", "K256", "hrec3000", "hrec6000", "trg125", "trg175"]
    lab = ["база\n(R=40)", "θ=0,3", "θ=0,7", "T=5", "T=20", "K=64", "K=256", "H_rec\n3000", "H_rec\n6000", "m_s\n1,25", "m_s\n1,75"]
    cells = [(r, d, sx) for r in ("R1", "R2") for d in DS for sx in SS]
    m = np.array([[float(s[(s.variant == v) & (s.R == r) & (s.D == d) & (s.S == sx)].delta.iloc[0]) for v in order] for r, d, sx in cells])
    # «нет эффекта»: ρ_fold цензурирован справа (отказов на сетке нет), Δ = 0 по соглашению; «оценённый ноль» — порог найден, Δ ≈ 0
    cens = np.array([[s[(s.variant == v) & (s.R == r) & (s.D == d) & (s.S == sx)].fold_cens.iloc[0] == "right" for v in order] for r, d, sx in cells])
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    im = ax.imshow(np.where(cens, np.nan, m), cmap="Greys", vmin=0, vmax=0.7, aspect="auto")
    ax.grid(False)
    from matplotlib.patches import Rectangle

    for a in range(m.shape[0]):
        for b in range(m.shape[1]):
            if cens[a, b]:
                ax.add_patch(Rectangle((b - 0.5, a - 0.5), 1, 1, facecolor="white", edgecolor="black", hatch="////", lw=0.6))
                ax.text(b, a, "нет", ha="center", va="center", color="black", fontsize=8.5, fontweight="bold",
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
            else:
                ax.text(b, a, f"{m[a, b]:.2f}".replace(".", ","), ha="center", va="center", color="white" if m[a, b] > 0.35 else "black", fontsize=8.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(lab)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([f"{r}, срок {d}, {sx}" for r, d, sx in cells])
    fig.colorbar(im, label="Δ = ρ_fold − ρ_release (доли ρ)")
    ax.set_title("Чувствительность Δ на 12 активных ячейках (FIFO, R = 40 реплик): одна переменная за раз")
    fig.text(0.01, 0.005, "Штриховка «нет» — эффекта нет: отказов на сетке ρ ≤ 1 не возникло (ρ_fold цензурирован справа), Δ = 0 по соглашению, а не оценка.\n"
             "Числа — оценённые Δ (в том числе близкие к нулю).", fontsize=8.5, va="bottom", ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    rows = [{"R": r, "D": d, "S": sx, **{v: m[i, k] for k, v in enumerate(order)}, **{f"{v}_no_effect": bool(cens[i, k]) for k, v in enumerate(order)}}
            for i, (r, d, sx) in enumerate(cells)]
    save_fig(fig, "F10_chuvstvitelnost_delta", pd.DataFrame(rows))
    plt.close(fig)

    h = pd.read_csv(OUT / "t11_sens_h1p.csv")
    fig, ax = plt.subplots(figsize=(10, 4.4))
    hv = h.dropna(subset=["diff_lo"])
    for k, r in enumerate(hv.itertuples()):
        ax.errorbar(k, r.diff_Exp_minus_LogN, yerr=[[r.diff_Exp_minus_LogN - r.diff_lo], [r.diff_hi - r.diff_Exp_minus_LogN]], marker="s", color="black", capsize=4)
    ax.axhline(0, color="gray", lw=0.9)
    ax.set_xticks(range(len(hv)))
    ax.set_xticklabels([lab[order.index(v)] for v in hv.variant])
    ax.set_ylabel("r(Exp) − r(LogN), где r = Δ(срок on)/Δ(срок off)")
    ax.set_title("H1′ в вариантах чувствительности (95 % ДИ). T = 20 и K = 64: метастабильности нет, r не определено")
    comma(ax, "y")
    fig.tight_layout()
    save_fig(fig, "F10b_chuvstvitelnost_H1prime", h)
    plt.close(fig)


def f11(c: dict) -> None:
    t = c["t01"]
    h2p = pd.read_csv(OUT / "t23_h2prime.csv")
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axs[0]
    combos = [("R2", "FIFO"), ("R2", "ALIFO"), ("R3", "FIFO"), ("R3", "ALIFO")]
    rows = []
    w = 0.26
    for gi, (r, q) in enumerate(combos):
        for si, s in enumerate(SS):
            sub = t[(t.R == r) & (t.Q == q) & (t.S == s)]
            v = sub.delta.mean()
            ax.bar(gi + (si - 1) * w, v, w * 0.92, color=SCOL[s], edgecolor="black", hatch=SHATCH[s], lw=0.8, label=SLAB[s] if gi == 0 else None)
            rows.append({"panel": "H2: Δ", "R": r, "Q": q, "S": s, "delta_mean_over_D": v})
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{r} + {q}" for r, q in combos])
    ax.set_ylabel("Δ (среднее по D), доли ρ")
    ax.set_title("H2: Δ у R2, R3 при FIFO и ALIFO. Комбинация R3 + ALIFO: Δ = 0, как и у ALIFO без R3")
    ax.legend(fontsize=8)
    comma(ax, "y")
    ax = axs[1]
    sub = h2p[h2p.metric.str.startswith("L_A")]
    labels = []
    for k, r in enumerate(sub.itertuples()):
        ax.errorbar(k, r.diff_R3_minus_R2, yerr=[[r.diff_R3_minus_R2 - r.lo], [r.hi - r.diff_R3_minus_R2]], marker="s" if r.Q == "FIFO" else "o",
                    color="black" if r.Q == "FIFO" else "#D55E00", capsize=3, ms=6)
        labels.append(f"{r.Q}\n{r.scope.replace('D и S усреднены', 'все').replace('S = ', '')}")
        rows.append({"panel": "H2′: L_A", "Q": r.Q, "scope": r.scope, "diff": r.diff_R3_minus_R2, "lo": r.lo, "hi": r.hi})
    ax.axhline(-0.01, color="#009E73", ls="--", lw=1.2, label="порог практической значимости −0,01")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_yscale("symlog", linthresh=0.003)
    ax.set_ylabel("L_A(R3) − L_A(R2) (доли), шкала симлог")
    ax.set_title("H2′: разность потери goodput (95 % ДИ). Ниже порога: FIFO.\nALIFO: |эффект| ≈ 0,0005 — практически ноль, хотя ДИ выше 0", fontsize=9.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "F11_H2_i_H2prime", pd.DataFrame(rows))
    plt.close(fig)


def f12() -> None:
    meta = base_meta()
    pick = [("R1", "FIFO", "off", "Exp"), ("R3", "FIFO", "on", "LogN"), ("R3", "FIFO", "off", "Exp"), ("R0", "FIFO", "off", "Exp")]
    hs = {p: meta[(meta.R == p[0]) & (meta.Q == p[1]) & (meta.D == p[2]) & (meta.S == p[3])].config_hash.iloc[0] for p in pick}
    res = load_main(["config_hash", "rho_test", "protocol", "g_final", "tag"], BASE_TAGS + EXT_TAGS)
    res = res[res.config_hash.isin(hs.values())].copy()
    res["fail"] = norm_failed(res.g_final.to_numpy(), res.rho_test.to_numpy())
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7), sharex=True, sharey=True)
    rows = []
    for ax, p in zip(axs.ravel(), pick, strict=True):
        for pr, ls, mk, col in (("A", "-", "o", "#000000"), ("B", "--", "s", "#0072B2")):
            g = res[(res.config_hash == hs[p]) & (res.protocol == pr)].groupby("rho_test").fail.mean()
            ax.plot(g.index, g.values, ls=ls, marker=mk, ms=3, color=col, label=f"протокол {pr}")
            rows.append(pd.DataFrame({"config": "/".join(p), "protocol": pr, "rho": g.index, "P_norm": g.values}))
        ax.axhline(0.5, color="gray", lw=0.8, ls=(0, (1, 3)))
        ax.axvline(1.0, color="gray", lw=0.8)
        ax.set_title("/".join(p))
        ax.set_xlim(0.2, 2.5)
        comma(ax)
    for ax in axs[1]:
        ax.set_xlabel(XR)
    fig.supylabel("Доля отказов P, 0-1 (критерий g < θ·min(1, 1/ρ))", fontsize=9.5)
    axs[0, 0].legend()
    fig.suptitle("Исследовательская иллюстрация: область перегрузки ρ > 1 (вертикаль ρ = 1; реплик 16 против 40-100 при ρ ≤ 1)", y=1.0, fontsize=10.5)
    fig.text(0.01, 0.005, "Δ_ext при ρ > 1 физически не определена: протокол B перестаёт быть снижением нагрузки, шаг сетки 0,1 больше различий кривых. "
             "Рисунок показывает лишь совпадение кривых A и B при перегрузке.", fontsize=8, va="bottom", ha="left", wrap=True)
    fig.tight_layout(rect=(0.02, 0.03, 1, 1))
    save_fig(fig, "F12_oblast_peregruzki_delta_ext", pd.concat(rows))
    plt.close(fig)


def f13(c: dict) -> None:
    t4 = c["t04"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    rows = []
    for ax, s in zip(axs, SS, strict=True):
        m = np.zeros((4, 3))
        for a, r in enumerate(RS):
            for b, q in enumerate(QS):
                x = t4[(t4.R == r) & (t4.Q == q) & (t4.S == s)]
                m[a, b] = x.L_A.mean()
                rows.append({"S": s, "R": r, "Q": q, "L_A_mean_over_D": m[a, b]})
        im = ax.imshow(np.log10(m), cmap="Greys", vmin=-3, vmax=-0.5, aspect="auto")
        ax.grid(False)
        for a in range(4):
            for b in range(3):
                ax.text(b, a, f"{m[a, b]:.3f}".replace(".", ","), ha="center", va="center", color="white" if np.log10(m[a, b]) > -1.6 else "black", fontsize=9)
        ax.set_xticks(range(3))
        ax.set_xticklabels(QS)
        ax.set_yticks(range(4))
        ax.set_yticklabels(RS)
        ax.set_title(SLAB[s])
    fig.colorbar(im, ax=axs, shrink=0.85, label="lg L_A (потеря goodput при ρ ≤ 1, среднее по D)")
    fig.suptitle("Эксплораторно: потеря goodput L_A по R × Q (непрерывная метрика, различает конфигурации с Δ = 0).\n"
                 "Пол L_A ≈ 0,003 (шум конечного окна); различия 0,001-0,007 в этих клетках не интерпретируются", y=1.06, fontsize=10)
    save_fig(fig, "F13_L_A_RxQ_eksploratorno", pd.DataFrame(rows))
    plt.close(fig)


def main() -> int:
    c = load()
    f01_f02(c)
    f03(c)
    f04(c)
    f05(c)
    f06(c)
    f07(c)
    f08(c)
    f09()
    f10()
    f11(c)
    f12()
    f13(c)
    print("рисунки: figures/F*.png|svg|csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
