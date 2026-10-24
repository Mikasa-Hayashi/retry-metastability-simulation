"""Основной анализ, п. 2 плана: проверка гипотез H1, H1′, H2, H2′, H3, H3b.

python analysis/main_hypotheses.py     (после main_thresholds.py, main_metrics.py, main_sensitivity.py)
Критерии (PREREGISTRATION, «Гипотезы»); решения, принятые при реализации, где план допускает толкование, помечены «[реализация]».
  H1: τ-b(Δ_Exp, Δ_LogN) по 18 ячейкам R×Q×D (R1-R3); H1 отвергается, если нижняя граница 95 % ДИ ≥ 0,8.
      [реализация] ДИ двух видов: (а) по репликам — τ по бутстреп-выборкам порогов (оценки конфигураций независимы), (б) по ячейкам
      (пересэмплирование 18 ячеек, как в pilot). Вывод «H1 отвергнута» только если нижняя граница ≥ 0,8 в обоих; иначе «не отвергнута» либо «неопределённо».
  H1′: r(S) = Δ_on/Δ_off на 12 активных ячейках (R1, R2 усреднены); разность r(Exp) − r(LogN), подтверждается, если ДИ не содержит 0.
  H2: контраст C = Δ(R3,ALIFO) − Δ(R3,FIFO) − Δ(R2,ALIFO) + Δ(R2,FIFO); подтверждается, если ДИ ниже 0 при Exp и знак сохраняется при LogN.
  H2′: L_A(R3) − L_A(R2) внутри ALIFO и внутри FIFO; практически значимо при верхней границе ДИ ≤ −0,01.
  H3: |ρ_release^sim − ρ_release^MF| ≤ 0,10 (R1/FIFO/off/Exp). [реализация] подтверждена, если верхняя граница ДИ |отклонения| ≤ 0,10;
      опровергнута, если нижняя граница > 0,10; иначе неопределённо. Отклонение при каждом H_rec (1500: основной прогон, 100 реплик;
      3000 и 6000: чувствительность, 40 реплик).
  H3b: ρ_fold^sim − ρ_fold^MF; отвергается, если нижняя граница ДИ > 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
import theory as th  # noqa: E402
from main_common import BOOT, OUT, QS, RS, SS, DS, fmt, holm, pct  # noqa: E402

RNG_TAU = 7


def load() -> dict:
    t01 = pd.read_csv(OUT / "t01_thresholds72.csv")
    b1 = np.load(OUT / "t01_boot.npz", allow_pickle=True)
    assert list(b1["config_hash"]) == list(t01["config_hash"])
    t04 = pd.read_csv(OUT / "t04_metrics72.csv")
    b4 = np.load(OUT / "t04_boot.npz", allow_pickle=True)
    assert list(b4["config_hash"]) == list(t04["config_hash"])
    idx = {(r.R, r.Q, r.D, r.S): i for i, r in enumerate(t01.itertuples())}
    delta_dr = b1["fold"] - b1["rel"]
    return {"t01": t01, "t04": t04, "idx": idx, "delta_dr": delta_dr, "b4": b4, "fold": b1["fold"], "rel": b1["rel"]}


def tau_b(x: np.ndarray, y: np.ndarray) -> float:
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(kendalltau(x, y, variant="b").statistic)


def tau_block(name: str, cells: list[tuple], pt_x: np.ndarray, pt_y: np.ndarray, dr_x: np.ndarray, dr_y: np.ndarray) -> dict:
    """pt_*: [n_cells]; dr_*: [n_cells, B] бутстреп-выборки."""
    t0 = tau_b(pt_x, pt_y)
    nested = np.array([tau_b(dr_x[:, b], dr_y[:, b]) for b in range(dr_x.shape[1])])
    nv = int((~np.isnan(nested)).sum())
    n_lo, n_hi = pct(nested[~np.isnan(nested)])
    rng = np.random.default_rng(RNG_TAU)
    cb = []
    for _ in range(BOOT):
        i = rng.integers(0, len(pt_x), len(pt_x))
        v = tau_b(pt_x[i], pt_y[i])
        if not np.isnan(v):
            cb.append(v)
    c_lo, c_hi = pct(np.array(cb))
    r1 = np.round(pt_x, 2)
    ties = float(np.mean([r1[i] == r1[j] for i in range(len(r1)) for j in range(i + 1, len(r1))]))
    return {"set": name, "n_cells": len(cells), "tau_b": t0, "rep_lo": n_lo, "rep_hi": n_hi, "rep_valid": nv,
            "cell_lo": c_lo, "cell_hi": c_hi, "cell_valid": len(cb), "tie_share_Exp": ties}


def h1(c: dict) -> pd.DataFrame:
    t01, t04, idx, b4 = c["t01"], c["t04"], c["idx"], c["b4"]
    rows = []
    for nm, rs in (("18 ячеек R1-R3 × Q × D", RS[1:]), ("24 ячейки с R0 (справочно)", RS)):
        cells = [(r, q, d) for r in rs for q in QS for d in DS]
        ie = [idx[(r, q, d, "Exp")] for r, q, d in cells]
        il = [idx[(r, q, d, "LogN")] for r, q, d in cells]
        for metric in ("Δ", "L_A", "H"):
            if metric == "Δ":
                px, py = t01.delta.to_numpy()[ie], t01.delta.to_numpy()[il]
                dx, dy = c["delta_dr"][ie], c["delta_dr"][il]
            else:
                key = {"L_A": "L_A", "H": "H"}[metric]
                px, py = t04[key].to_numpy()[ie], t04[key].to_numpy()[il]
                dx, dy = b4[key][ie], b4[key][il]
            r = tau_block(nm, cells, px, py, dx, dy)
            r["metric"] = metric
            rows.append(r)
    # эксплораторно: ячейки, активные хотя бы при одном S (4 ячейки R1/R2 × FIFO × D)
    cells = [(r, "FIFO", d) for r in ("R1", "R2") for d in DS]
    ie = [idx[(r, q, d, "Exp")] for r, q, d in cells]
    il = [idx[(r, q, d, "LogN")] for r, q, d in cells]
    r = tau_block("4 активные ячейки (эксплораторно)", cells, t01.delta.to_numpy()[ie], t01.delta.to_numpy()[il], c["delta_dr"][ie], c["delta_dr"][il])
    r["metric"] = "Δ"
    rows.append(r)
    out = pd.DataFrame(rows)
    # степень вырождения 18 ячеек
    n18 = [(r, q, d) for r in RS[1:] for q in QS for d in DS]
    degen = sum(1 for (r, q, d) in n18 if not t01.N1.to_numpy()[idx[(r, q, d, "Exp")]] and not t01.N1.to_numpy()[idx[(r, q, d, "LogN")]])
    out["degenerate_cells_18"] = degen

    def verdict(row: pd.Series) -> str:
        if row.metric != "Δ" or not row["set"].startswith("18"):
            return "справочно"
        ok_rep, ok_cell = row.rep_lo >= 0.8, row.cell_lo >= 0.8
        if ok_rep and ok_cell:
            return "H1 отвергнута (рейтинги согласованы)"
        if (not ok_rep) and (not ok_cell):
            return "H1 не отвергнута по критерию (нижняя граница < 0,8 в обоих ДИ)"
        return "неопределённо (критерий выполнен только в одном из двух ДИ)"

    out["verdict"] = out.apply(verdict, axis=1)
    return out[["metric", "set", "n_cells", "tau_b", "rep_lo", "rep_hi", "rep_valid", "cell_lo", "cell_hi", "cell_valid",
                "tie_share_Exp", "degenerate_cells_18", "verdict"]]


def d_dr(c: dict, r: str, q: str, d: str, s: str) -> np.ndarray:
    return c["delta_dr"][c["idx"][(r, q, d, s)]]


def d_pt(c: dict, r: str, q: str, d: str, s: str) -> float:
    return float(c["t01"].delta.iloc[c["idx"][(r, q, d, s)]])


def h1prime(c: dict) -> pd.DataFrame:
    rows = []
    for scope, rl in (("R1 и R2 усреднены (план)", ("R1", "R2")), ("только R1", ("R1",)), ("только R2", ("R2",))):
        rr, pp = {}, {}
        for s in SS:
            on = sum(d_dr(c, r, "FIFO", "on", s) for r in rl)
            off = sum(d_dr(c, r, "FIFO", "off", s) for r in rl)
            rr[s] = on / off
            pp[s] = sum(d_pt(c, r, "FIFO", "on", s) for r in rl) / sum(d_pt(c, r, "FIFO", "off", s) for r in rl)
        for a, b in (("Exp", "LogN"), ("Exp", "Pareto"), ("LogN", "Pareto")):
            dd = rr[a] - rr[b]
            lo, hi = pct(dd)
            rows.append({"scope": scope, "comparison": f"r({a}) − r({b})", "r_first": pp[a], "r_second": pp[b], "diff": pp[a] - pp[b],
                         "lo": lo, "hi": hi, "primary": scope.startswith("R1 и R2") and (a, b) == ("Exp", "LogN"),
                         "verdict": ("ДИ не содержит 0" if (lo > 0 or hi < 0) else "ДИ содержит 0")})
        for s in SS:
            lo, hi = pct(rr[s])
            rows.append({"scope": scope, "comparison": f"r({s}) = Δ_on/Δ_off", "r_first": pp[s], "r_second": np.nan, "diff": np.nan,
                         "lo": lo, "hi": hi, "primary": False, "verdict": ""})
    return pd.DataFrame(rows)


def h2(c: dict) -> pd.DataFrame:
    rows = []
    cd: dict[tuple, np.ndarray] = {}
    for d in DS:
        for s in SS:
            g = lambda r, q: d_dr(c, r, q, d, s)  # noqa: E731
            cd[(d, s)] = g("R3", "ALIFO") - g("R3", "FIFO") - g("R2", "ALIFO") + g("R2", "FIFO")
            gp = lambda r, q: d_pt(c, r, q, d, s)  # noqa: E731
            pt = gp("R3", "ALIFO") - gp("R3", "FIFO") - gp("R2", "ALIFO") + gp("R2", "FIFO")
            lo, hi = pct(cd[(d, s)])
            rows.append({"level": f"D={d}, S={s}", "C": pt, "lo": lo, "hi": hi, "D_R3_ALIFO": gp("R3", "ALIFO"), "D_R3_FIFO": gp("R3", "FIFO"),
                         "D_R2_ALIFO": gp("R2", "ALIFO"), "D_R2_FIFO": gp("R2", "FIFO")})
    for s in SS:
        dr = np.mean([cd[(d, s)] for d in DS], axis=0)
        pt = float(np.mean([r["C"] for r in rows if r["level"].endswith(f"S={s}")]))
        lo, hi = pct(dr)
        rows.append({"level": f"S={s} (D усреднено)", "C": pt, "lo": lo, "hi": hi})
    dr = np.mean([cd[k] for k in cd], axis=0)
    pt = float(np.mean([r["C"] for r in rows[:6]]))
    lo, hi = pct(dr)
    rows.append({"level": "все (D и S усреднены)", "C": pt, "lo": lo, "hi": hi})
    out = pd.DataFrame(rows)
    sub = out.set_index("level")
    exp_ok = sub.loc["S=Exp (D усреднено)", "hi"] < 0
    logn_sign = sub.loc["S=LogN (D усреднено)", "C"] < 0
    out["verdict"] = ""
    out.loc[out.level == "все (D и S усреднены)", "verdict"] = (
        "H2 подтверждена" if (exp_ok and logn_sign) else "H2 не подтверждена: ДИ при Exp не ниже 0" if not exp_ok else "H2 не подтверждена: знак при LogN не сохраняется")
    return out


def h2prime(c: dict) -> pd.DataFrame:
    t04, b4, idx = c["t04"], c["b4"], c["idx"]
    rows = []
    for metric, key, thr in (("L_A (потеря goodput, ρ ≤ 1)", "L_A", -0.01), ("усиление при ρ = 0,9 (A)", "ampl_A09", None),
                             ("wasted при ρ = 0,9 (A)", "wasted_A09", None)):
        for q in ("ALIFO", "FIFO"):
            for scope, ss in (("D и S усреднены", SS), ("S = Exp", ["Exp"]), ("S = LogN", ["LogN"]), ("S = Pareto", ["Pareto"])):
                cells = [(d, s) for d in DS for s in ss]
                dr = np.mean([b4[key][idx[("R3", q, d, s)]] - b4[key][idx[("R2", q, d, s)]] for d, s in cells], axis=0)
                pt = float(np.mean([t04[key].iloc[idx[("R3", q, d, s)]] - t04[key].iloc[idx[("R2", q, d, s)]] for d, s in cells]))
                lo, hi = pct(dr)
                row = {"metric": metric, "Q": q, "scope": scope, "diff_R3_minus_R2": pt, "lo": lo, "hi": hi, "n_cells": len(cells)}
                if thr is not None:
                    row["p_one_sided"] = float((np.sum(dr >= thr) + 1) / (len(dr) + 1))  # H0: эффект ≥ −0,01
                    row["practical"] = bool(hi <= thr)
                rows.append(row)
    out = pd.DataFrame(rows)
    m = out["metric"].str.startswith("L_A")
    out.loc[m, "p_holm"] = holm(out.loc[m, "p_one_sided"].to_numpy())
    return out


def mf_spinodals(K: int, T: float, theta: float = 0.5) -> tuple[float, float]:
    p = th.MFParams(c=8, k_queue=K, t_out=T)

    def has_h(r: float) -> bool:
        return any(x.stable and th.goodput_ratio(x.q, 3) >= theta for x in th.fixed_points(r, p, 801))

    def has_f(r: float) -> bool:
        return any(x.stable and th.goodput_ratio(x.q, 3) < theta for x in th.fixed_points(r, p, 801))

    fold = th.refine_threshold(has_h, 0.3, 1.2, 1e-3) if (has_h(0.3) and not has_h(1.2)) else float("nan")
    rel = th.refine_threshold(has_f, 0.05, 1.2, 1e-3) if ((not has_f(0.05)) and has_f(1.2)) else float("nan")
    return rel, fold


def h3(c: dict) -> pd.DataFrame:
    mf_rel, mf_fold = mf_spinodals(128, 10.0)
    s10 = pd.read_csv(OUT / "t10_sens.csv")
    b10 = np.load(OUT / "t10_boot.npz", allow_pickle=True)
    k10 = {k: i for i, k in enumerate(b10["key"])}
    rows = []

    def add(label: str, hrec: str, reps: int, rel_d: np.ndarray, rel_v: float, fold_d: np.ndarray, fold_v: float, rel_c: str, fold_c: str, crit: bool) -> None:
        dev = rel_v - mf_rel
        dd = rel_d - mf_rel
        lo, hi = pct(dd)
        alo, ahi = pct(np.abs(dd))
        fdev = fold_v - mf_fold
        flo, fhi = pct(fold_d - mf_fold)
        if crit:
            v3 = ("H3 подтверждена" if ahi <= 0.10 else "H3 опровергнута" if alo > 0.10 else "H3 неопределённо")
            v3b = ("H3b отвергнута" if flo > 0 else "H3b не отвергнута")
        else:
            v3 = v3b = "справочно"
        rows.append({"case": label, "H_rec": hrec, "reps": reps, "mf_release": mf_rel, "sim_release": rel_v, "release_cens": rel_c,
                     "dev": dev, "dev_lo": lo, "dev_hi": hi, "abs_dev_lo": alo, "abs_dev_hi": ahi, "H3": v3,
                     "mf_fold": mf_fold, "sim_fold": fold_v, "fold_cens": fold_c, "fold_dev": fdev, "fold_dev_lo": flo, "fold_dev_hi": fhi,
                     "H3b": v3b})

    i = c["idx"][("R1", "FIFO", "off", "Exp")]
    t01 = c["t01"]
    add("основной прогон (R1/FIFO/off/Exp)", "1500", 100, c["rel"][i], float(t01.rel.iloc[i]), c["fold"][i], float(t01.fold.iloc[i]),
        t01.rel_cens.iloc[i], t01.fold_cens.iloc[i], True)
    for var, hr in (("base40", "1500"), ("hrec3000", "3000"), ("hrec6000", "6000")):
        j = k10[f"{var}|R1|FIFO|off|Exp"]
        r = s10[(s10.variant == var) & (s10.R == "R1") & (s10.D == "off") & (s10.S == "Exp")].iloc[0]
        add(f"чувствительность, {var}", hr, 40, b10["rel"][j], float(r.rel), b10["fold"][j], float(r.fold), r.rel_cens, r.fold_cens, var != "base40")
    # справочно: LogN и Pareto (MF к ним неприменима, порог не назначается)
    for s in ("LogN", "Pareto"):
        k = c["idx"][("R1", "FIFO", "off", s)]
        add(f"справочно: R1/FIFO/off/{s} (теория неприменима)", "1500", 100, c["rel"][k], float(t01.rel.iloc[k]), c["fold"][k], float(t01.fold.iloc[k]),
            t01.rel_cens.iloc[k], t01.fold_cens.iloc[k], False)
    return pd.DataFrame(rows)


def main() -> int:
    c = load()
    t = h1(c)
    t.to_csv(OUT / "t20_h1.csv", index=False, encoding="utf-8")
    print(t.round(3).to_string())
    t = h1prime(c)
    t.to_csv(OUT / "t21_h1prime.csv", index=False, encoding="utf-8")
    print(t[t.primary | t.comparison.str.startswith("r(")].round(3).to_string())
    t = h2(c)
    t.to_csv(OUT / "t22_h2.csv", index=False, encoding="utf-8")
    print(t.round(3).to_string())
    t = h2prime(c)
    t.to_csv(OUT / "t23_h2prime.csv", index=False, encoding="utf-8")
    print(t.round(4).to_string())
    t = h3(c)
    t.to_csv(OUT / "t24_h3.csv", index=False, encoding="utf-8")
    print(t.round(3).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
