"""Эксплораторно (В6): чувствительность R2 к длине пауз (retry.base 0,5 / 2,0 против 1,0) и R3 к бюджету (0,05 / 0,20 против 0,10).

python analysis/explor_sens.py
Данные: results/explor (теги x_sens_r2, x_sens_r3 и *_ref), только протокол A; база - results/raw (реплики < 40).
ВАЖНО: протокол B для вариантов не запускался, поэтому ρ_release варианта не измерен. Δ* = ρ_fold(вариант) - ρ_release(база той же ячейки)
- это оценка с допущением, что ρ_release не зависит от параметра; параметры повторов на неё влиять МОГУТ. Надёжные величины: ρ_fold и L_A.
Выход: results/analysis/explor_sens.csv (E20, по ячейкам и значениям параметра), explor_sens_rank.csv (E21, контрасты R2-R1, R3-R2,
влияние параметра), explor_sens_curves.csv (E22, P_A по ρ).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explor_common import OUT, base_cfg, base_data, explor_meta, l_a, load_explor, pct, thr_of  # noqa: E402

DS = [(d, s) for d in ("off", "on") for s in ("Exp", "LogN", "Pareto")]


def main() -> int:
    meta = explor_meta()
    ex = load_explor(["config_hash", "rho_test", "protocol", "failed", "g_final", "replicate"],
                     ["x_sens_r2", "x_sens_r2_ref", "x_sens_r3", "x_sens_r3_ref"])
    bd = base_data(["config_hash", "rho_test", "protocol", "failed", "g_final"])
    store: dict[tuple, dict] = {}
    rows, curves = [], []
    code = 2000
    for r in ("R0", "R1", "R2", "R3"):
        for (d, s) in DS:
            hb = base_cfg(r, "FIFO", d, s)
            db = bd[bd.config_hash == hb]
            code += 10
            cases = [("база", 1.0 if r in ("R1", "R2") else (0.10 if r == "R3" else np.nan), hb, db, "results/raw")]
            if r in ("R2", "R3"):
                for val in ((0.5, 2.0) if r == "R2" else (0.05, 0.20)):
                    sel = meta[(meta.R == r) & (meta.Q == "FIFO") & (meta.D == d) & (meta.S == s) &
                               ((meta.rbase == val) if r == "R2" else (meta.budget == val))]
                    sel = sel[(sel.trig == 1.5) & (sel.ol_len == 1000.0) & (sel.h_rec == 1500.0)]
                    if r == "R2":
                        sel = sel[sel.budget == 0.10]
                    else:
                        sel = sel[sel.rbase == 1.0]
                    if len(sel) != 1:
                        raise RuntimeError(f"не найдена ячейка {r} {d} {s} {val}")
                    h = sel.config_hash.iloc[0]
                    cases.append(("вариант", val, h, ex[ex.config_hash == h], "results/explor"))
            rel_b = thr_of(db, "B", hb, code + 2)
            for ci, (kind, val, h, dd, src) in enumerate(cases):
                fold = thr_of(dd, "A", h, code + 3 + ci)
                coarse = dd[dd["tag"].isin(["m_scr", "m_r100", "x_sens_r2", "x_sens_r3"])]
                la = l_a(coarse, h, code + 7 + ci)
                dr = fold["draws"] - rel_b["draws"]
                lo, hi = pct(dr)
                rows.append({"R": r, "Q": "FIFO", "D": d, "S": s, "param": ("retry.base" if r == "R2" else "budget_ratio" if r == "R3" else "-"),
                             "value": val, "kind": kind, "reps": int(dd["replicate"].max() + 1), "source": src,
                             "fold": fold["value"], "fold_lo": fold["lo"], "fold_hi": fold["hi"], "fold_cens": fold["cens"],
                             "rel_base": rel_b["value"], "rel_base_cens": rel_b["cens"],
                             "delta_star": fold["value"] - rel_b["value"], "dstar_lo": lo, "dstar_hi": hi,
                             "L_A": la["L"], "L_A_lo": la["L_lo"], "L_A_hi": la["L_hi"]})
                store[(r, d, s, val if kind == "вариант" else "база")] = {"fold": fold["draws"], "L": la["L_draws"], "rel": rel_b["draws"],
                                                                         "foldv": fold["value"], "Lv": la["L"], "cens": fold["cens"]}
                pa = dd[dd.protocol == "A"].groupby("rho_test").agg(n=("failed", "size"), P=("failed", "mean")).reset_index()
                pa.insert(0, "value", val); pa.insert(0, "S", s); pa.insert(0, "D", d); pa.insert(0, "R", r)
                curves.append(pa)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "explor_sens.csv", index=False, encoding="utf-8")
    pd.concat(curves).to_csv(OUT / "explor_sens_curves.csv", index=False, encoding="utf-8")
    pd.set_option("display.width", 250)
    print(tab[["R", "D", "S", "value", "kind", "reps", "fold", "fold_cens", "rel_base", "delta_star", "L_A", "L_A_lo", "L_A_hi"]].round(3).to_string())

    # контрасты
    out = []

    def con(label: str, a: tuple, b: tuple, d: str, s: str, extra: dict) -> None:
        A, B = store[a], store[b]
        row = {"контраст": label, "D": d, "S": s, **extra}
        for key, nm in (("fold", "fold"), ("L", "L_A")):
            dd = A[key] - B[key]
            lo, hi = pct(dd)
            row[f"{nm}_diff"] = float((A["foldv"] - B["foldv"]) if key == "fold" else (A["Lv"] - B["Lv"]))
            row[f"{nm}_diff_lo"], row[f"{nm}_diff_hi"] = lo, hi
        row["fold_cens_a"], row["fold_cens_b"] = A["cens"], B["cens"]
        out.append(row)

    for (d, s) in DS:
        for val in ("база", 0.5, 2.0):
            con("R2(base) - R1", ("R2", d, s, val), ("R1", d, s, "база"), d, s, {"R2_base": 1.0 if val == "база" else val})
        for rv in (("база", 0.05, 0.20)):
            pass
        for val in ("база", 0.05, 0.20):
            for bv in ("база", 0.5, 2.0):
                con("R3(бюджет) - R2(base)", ("R3", d, s, val), ("R2", d, s, bv), d, s,
                    {"R3_budget": 0.10 if val == "база" else val, "R2_base": 1.0 if bv == "база" else bv})
        for val in (0.05, 0.20):
            con("R3(бюджет) - R3(0,10)", ("R3", d, s, val), ("R3", d, s, "база"), d, s, {"R3_budget": val})
        for val in (0.5, 2.0):
            con("R2(вариант) - R2(1,0)", ("R2", d, s, val), ("R2", d, s, "база"), d, s, {"R2_base": val})
        for val in ("база", 0.05, 0.20):
            con("R3(бюджет) - R0", ("R3", d, s, val), ("R0", d, s, "база"), d, s, {"R3_budget": 0.10 if val == "база" else val})
    ct = pd.DataFrame(out)
    ct.to_csv(OUT / "explor_sens_rank.csv", index=False, encoding="utf-8")
    print(ct[["контраст", "D", "S", "R2_base", "R3_budget", "fold_diff", "fold_diff_lo", "fold_diff_hi", "L_A_diff", "L_A_diff_lo", "L_A_diff_hi"]].round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
