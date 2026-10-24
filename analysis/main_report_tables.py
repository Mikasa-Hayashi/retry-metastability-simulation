"""Сборка таблиц в формате Markdown для reports/analysis.md: reports/analysis_tables.md (автоматически; числа из results/analysis/*.csv).

python analysis/main_report_tables.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_common import ROOT, OUT, ci_str, fmt, md_table  # noqa: E402

LINES: list[str] = []


def add(title: str, src: str, df: pd.DataFrame, **kw) -> None:
    LINES.append(f"### {title}\n\nИсточник: `results/analysis/{src}`.\n\n{md_table(df, **kw)}\n")


def main() -> int:
    t = pd.read_csv(OUT / "t01_thresholds72.csv")
    act = t[t.N1].copy()
    d = pd.DataFrame({"R": act.R, "Q": act.Q, "D": act.D, "S": act.S, "реплик": act.reps_max,
                      "ρ_fold [ДИ]": [ci_str(a, b, c) for a, b, c in zip(act.fold, act.fold_lo, act.fold_hi)],
                      "ρ_release [ДИ]": [ci_str(a, b, c) for a, b, c in zip(act.rel, act.rel_lo, act.rel_hi)],
                      "Δ [ДИ]": [ci_str(a, b, c) for a, b, c in zip(act.delta, act.d_lo, act.d_hi)], "N2": act.N2.map({True: "да", False: "нет"})})
    add("Таблица A1. Активные ячейки (N1): пороги и Δ, ρ ≤ 1 (T01)", "t01_thresholds72.csv", d)
    o = t[(~t.N1) & (t.cls != "нет отказов на сетке")]
    d = pd.DataFrame({"R": o.R, "Q": o.Q, "D": o.D, "S": o.S, "ρ_fold": o.fold, "цензура_fold": o.fold_cens, "ρ_release": o.rel,
                      "цензура_release": o.rel_cens, "Δ [ДИ]": [ci_str(a, b, c) for a, b, c in zip(o.delta, o.d_lo, o.d_hi)], "класс": o.cls})
    add("Таблица A2. Неактивные ячейки с нетривиальными порогами: R3 × FIFO, R0 × Pareto (T01)", "t01_thresholds72.csv", d)
    c = pd.read_csv(OUT / "t30_classification.csv")
    c = c[c.factor.isin(["R", "Q", "D", "S", "все"])]
    c = c.rename(columns={"N1_ext": "N1_ext (исследовательский, не вывод)"})
    add("Таблица A3. Классификация по уровням факторов (T30): число ячеек N1, N2 и цензурированных порогов. Столбец N1_ext (область ρ > 1) исследовательский: физически не определён, в выводы не входит. fold_censored = 57 и release_censored = 56 в строке «все»: R0/FIFO/off/Pareto имеет нецензурированный ρ_release = 0,999 (табл. A2)", "t30_classification.csv", c[["factor", "level", "n", "N1", "N2", "N1_ext (исследовательский, не вывод)", "fold_censored", "release_censored", "mean_delta"]])
    r0 = pd.read_csv(OUT / "t03b_r0_control.csv")
    s = pd.DataFrame([{"конфигураций R0": len(r0), "Δ min": r0.delta.min(), "Δ max": r0.delta.max(), "max верхней границы ДИ": r0.d_hi.max(),
                       "нижняя граница ДИ > 0": int((r0.d_lo > 0).sum()), "ρ_fold цензурирован": int((r0.fold_cens == "right").sum())}])
    add("Таблица A4. Контроль R0, ρ ≤ 1 (T03b)", "t03b_r0_control.csv", s, default_nd=4)
    e = pd.read_csv(OUT / "t02_ext72.csv")
    e0 = e[e.R == "R0"]
    s = pd.DataFrame([{"конфигураций R0": len(e0), "Δ_ext min": e0.delta.min(), "Δ_ext max": e0.delta.max(), "max верхней границы ДИ": e0.d_hi.max(),
                       "нижняя граница ДИ > 0": int((e0.d_lo > 0).sum()), "ρ_fold цензурирован справа (2,5)": int((e0.fold_cens == "right").sum())}])
    add("Таблица A5 (исследовательская). Контроль R0 в области ρ ≤ 2,5 (T02); Δ_ext физически не определена при ρ > 1", "t02_ext72.csv", s, default_nd=4)
    en = e[e.N1 & (e.R != "R1") & (e.R != "R2")]
    d = pd.DataFrame({"R": en.R, "Q": en.Q, "D": en.D, "S": en.S, "ρ_fold_ext": en.fold, "ρ_release_ext": en.rel,
                      "Δ_ext [ДИ]": [ci_str(a, b, c) for a, b, c in zip(en.delta, en.d_lo, en.d_hi)]})
    add("Таблица A6 (исследовательская, в выводы не входит). Δ_ext в области ρ ≤ 2,5 (кроме R1, R2) (T02). Разность 0,040 меньше шага сетки 0,1, реплик 16, множественность не учтена; протокол B при ρ > 1 не является снижением нагрузки", "t02_ext72.csv", d)
    h1 = pd.read_csv(OUT / "t20_h1.csv")
    add("Таблица H1. Кендалл τ-b между рейтингами при Exp и LogN (T20). Строка «4 активные ячейки» (ДИ [−1; 1]) в текст не переносится", "t20_h1.csv", h1)
    r = pd.read_csv(OUT / "t43_h1_rules.csv")
    add("Таблица H1v. H1 по двум правилам отвержения («нижняя граница 95 % ДИ τ ≥ 0,8»): ДИ по ячейкам и ДИ по репликам (T43). 14 из 18 ячеек вырождены, поэтому по Δ вердикт не содержателен", "t43_h1_rules.csv", r)
    h1p = pd.read_csv(OUT / "t21_h1prime.csv")
    add("Таблица H1′. Остаточное Δ при передаче срока, r(S) = Δ_on/Δ_off (T21)", "t21_h1prime.csv", h1p[h1p.scope.str.startswith("R1 и R2")][["comparison", "r_first", "r_second", "diff", "lo", "hi", "primary"]])
    add("Таблица H2. Контраст взаимодействия C по Δ (T22)", "t22_h2.csv", pd.read_csv(OUT / "t22_h2.csv"))
    add("Таблица H2′. R3 − R2 внутри Q (T23)", "t23_h2prime.csv", pd.read_csv(OUT / "t23_h2prime.csv"), default_nd=4)
    h3 = pd.read_csv(OUT / "t24_h3.csv")
    add("Таблица H3. Mean-field и симуляция, R1/FIFO/off (T24)", "t24_h3.csv", h3[["case", "H_rec", "reps", "mf_release", "sim_release", "dev", "dev_lo", "dev_hi", "H3", "mf_fold", "sim_fold", "fold_dev", "fold_dev_lo", "fold_dev_hi", "H3b"]])
    add("Таблица F1. Маргинальные контрасты Δ по активным ячейкам (T31)", "t31_marginal_contrasts.csv", pd.read_csv(OUT / "t31_marginal_contrasts.csv"))
    add("Таблица F2. WLS-регрессия Δ по активным ячейкам, кодирование «обработка» (T31)", "t31_regression_wls.csv", pd.read_csv(OUT / "t31_regression_wls.csv").drop(columns=["model"]).iloc[:10])
    add("Таблица F3. Парные сравнения Δ, поправка Холма по 24 сравнениям (T30)", "t30_pairs_holm.csv", pd.read_csv(OUT / "t30_pairs_holm.csv"), default_nd=3)
    cl = pd.read_csv(OUT / "t32_cliff.csv")
    add("Таблица F4. Cliff's delta по псевдорепликам L_A (эксплораторно) (T32)", "t32_cliff.csv", cl[cl.level.str.startswith("среднее")])
    s10 = pd.read_csv(OUT / "t10_sens.csv")
    pv = s10.pivot_table(index=["R", "D", "S"], columns="variant", values="delta").reset_index().rename(columns=lambda c: f"θ = {c}".replace(".", ",") if isinstance(c, float) else c)
    pv = pv[["R", "D", "S", "base40", "theta03", "theta07", "T5", "T20", "K64", "K256", "hrec3000", "hrec6000", "trg125", "trg175"]]
    add("Таблица S1. Чувствительность: Δ по вариантам (T10; ДИ в файле)", "t10_sens.csv", pv)
    add("Таблица S2. Чувствительность H1′: r(S) и разность r(Exp) − r(LogN) (T11)", "t11_sens_h1p.csv", pd.read_csv(OUT / "t11_sens_h1p.csv"))
    km = pd.read_csv(OUT / "t06_km_summary.csv")
    add("Таблица K1. Время восстановления: Каплан-Мейер, ρ = 0,9 (T06). rmst — ограниченное среднее до горизонта 1500", "t06_km_summary.csv", km[km.rho == 0.9][["protocol", "R", "Q", "n", "share_censored", "share_never_degraded", "km_median", "rmst", "rmst_lo", "rmst_hi"]], default_nd=2)
    m = pd.read_csv(OUT / "t34_marginal_LA.csv")
    add("Таблица E1 (эксплораторно). Маргинальные средние L_A, L_B, H по уровням факторов (T34)", "t34_marginal_LA.csv", m, default_nd=4)
    add("Таблица X1 (эксплораторно). Диагностика бимодальности g_final, 12 активных ячеек, 5000 прогонов в ячейке (T40). valley_ratio = плотность в провале / меньший из двух пиков; ashman_D > 2 — разделение режимов; share_0.3_0.7_* — доля g_final в [0,3; 0,7]", "t40_bimodality.csv",
        pd.read_csv(OUT / "t40_bimodality.csv")[["R", "D", "S", "valley_cut", "peak_low", "valley_ratio", "share_low_mode", "ashman_D", "share_0.3_0.7_A", "share_0.3_0.7_B", "share_0.3_0.7_window", "window_valley_ratio", "two_modes"]])
    g72 = pd.read_csv(OUT / "t40c_gap_share72.csv")
    add("Таблица X2 (эксплораторно). Доля g_final в [0,3; 0,7] по 72 ячейкам (T40c): максимум и ячейки выше 0,02", "t40c_gap_share72.csv", g72[(g72["share_0.3_0.7_A"] > 0.02) | (g72["share_0.3_0.7_B"] > 0.02)][["R", "Q", "D", "S", "share_0.3_0.7_A", "share_0.3_0.7_B", "share_fail_A", "share_fail_B"]])
    th = pd.read_csv(OUT / "t41_theta_delta.csv")
    thp = th.assign(cell=th.R + "/" + th.D + "/" + th.S).pivot_table(index="cell", columns="theta", values="delta").reset_index().rename(columns=lambda c: f"θ = {c}".replace(".", ",") if isinstance(c, float) else c)
    add("Таблица X3 (эксплораторно). Δ при θ = 0,3 / 0,5 / 0,7, пересчёт g_final по основным данным (R = 100; ДИ в файле) (T41). θ = 0,5 воспроизводит T01. У R1, R2 / on / LogN при θ = 0,3 Δ = 0: режим отказа лежит выше 0,3", "t41_theta_delta.csv", thp)
    add("Таблица X4 (эксплораторно). H1′ при каждом θ (T41b) и по порогу-долине (T42b)", "t41b_theta_h1p.csv", pd.read_csv(OUT / "t41b_theta_h1p.csv")[["theta", "r_Exp", "r_LogN", "r_Pareto", "diff", "diff_lo", "diff_hi"]])
    gp = pd.read_csv(OUT / "t42_gap_delta.csv")
    add("Таблица X5 (эксплораторно). Δ_gap: отказ = g_final ниже порога-долины ячейки, без θ (T42); win_share_low_A/B — доля прогонов в низком режиме внутри окна (ρ_release_gap; ρ_fold_gap) для протоколов A и B", "t42_gap_delta.csv",
        gp[["R", "D", "S", "valley_cut", "two_modes", "win_share_low_A", "win_share_low_B", "fold_gap", "rel_gap", "delta_gap", "d_lo", "d_hi", "delta_theta05"]])
    add("Таблица X6 (эксплораторно). H1′ на Δ_gap (T42b)", "t42b_gap_h1p.csv", pd.read_csv(OUT / "t42b_gap_h1p.csv")[["measure", "r_Exp", "r_LogN", "r_Pareto", "diff", "diff_lo", "diff_hi"]])
    m = pd.read_csv(OUT / "t44_methods.csv")
    a12 = m[m.apply(lambda z: z.R in ("R1", "R2") and z.Q == "FIFO", axis=1)]
    add("Таблица M1. ρ_fold, ρ_release и Δ тремя методами для 12 активных ячеек (T44): по плану (логистика, изотоника при разделении), изотоника, линейная интерполяция сырых P. «вне ДИ» — оценка интерполяции вне 95 % ДИ плана", "t44_methods.csv",
        a12[["R", "D", "S", "fold_plan", "fold_iso", "fold_lin", "fold_lin_in_ci", "rel_plan", "rel_iso", "rel_lin", "rel_lin_in_ci", "delta_plan", "delta_iso", "delta_lin"]])
    add("Таблица M2. Эффекты и H1′ при разных методах порога (T44c)", "t44c_method_effects.csv", pd.read_csv(OUT / "t44c_method_effects.csv"))
    add("Таблица M3. Сырые P_A вокруг порога R2/FIFO/on/Exp (T44b): рост P_A замедляется на 0,85-0,87 (≈ 0,41), затем скачок до 0,61", "t44b_r2_on_exp.csv", pd.read_csv(OUT / "t44b_r2_on_exp.csv"))
    (ROOT / "reports" / "analysis_tables.md").write_text("# Таблицы основного анализа (автоматически из results/analysis)\n\nСборка: `analysis/main_report_tables.py`.\n\n" + "\n".join(LINES), encoding="utf-8")
    print("reports/analysis_tables.md:", len(LINES), "таблиц")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
