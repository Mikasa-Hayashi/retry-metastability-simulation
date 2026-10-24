"""Эксплораторно (В2): появляется ли гистерезис у 12 «нулевых» ячеек основного прогона при сильных триггерах.

python analysis/explor_trg.py
Ячейки: R3/FIFO и R1 с LIFO, ALIFO; D x S(Exp, LogN). Триггеры (m_s; D_s): базовый (1,5; 25) из results/raw (реплики < 40),
(3; 50) и (4; 100) из results/explor (теги x_trg, x_trg_ref), протокол A. Протокол B от триггера не зависит (DECISIONS D-015),
поэтому ρ_release берётся из базы той же ячейки (results/raw, реплики < 40); Δ = ρ_fold(триггер) - ρ_release(база).
Выход: results/analysis/explor_trg.csv (E10), explor_trg_curves.csv (E11: P_A(ρ) по ячейкам и триггерам).
Цензурирование: если порог срыва выше сетки (ρ_fold = 1,0 - граница), Δ не оценивается, класс «A не отказывает».
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explor_common import OUT, base_cfg, base_data, delta_row, explor_meta, l_a, load_explor, thr_of  # noqa: E402

CELLS = [(r, q, d, s) for (r, q) in (("R3", "FIFO"), ("R1", "LIFO"), ("R1", "ALIFO")) for d in ("off", "on") for s in ("Exp", "LogN")]
TRIG = [(1.5, 25.0), (3.0, 50.0), (4.0, 100.0)]


def classify(row: dict) -> str:
    if row["fold_cens"] == "right":
        return "A не отказывает до ρ = 1"
    if row["rel_cens"] == "right":
        return "A отказывает, B не отказывает (нет удержания отказа)"
    if row["N1"] and row["delta"] >= 0.1:
        return "метастабильность (Δ ≥ 0,1, ДИ > 0)"
    if row["N1"]:
        return "малый гистерезис (0 < Δ < 0,1, ДИ > 0)"
    return "нет гистерезиса"


def main() -> int:
    meta = explor_meta()
    ex = load_explor(["config_hash", "rho_test", "protocol", "failed", "g_final", "replicate"], ["x_trg", "x_trg_ref"])
    bd = base_data(["config_hash", "rho_test", "protocol", "failed", "g_final"])
    rows, curves = [], []
    for ci, (r, q, d, s) in enumerate(CELLS):
        hb = base_cfg(r, q, d, s)
        db = bd[bd.config_hash == hb]
        rel_b = thr_of(db, "B", hb, 1000 + ci * 10 + 2)
        for ti, (m, ln) in enumerate(TRIG):
            if (m, ln) == (1.5, 25.0):
                h, dd, src = hb, db, "results/raw (база, реплики < 40)"
                fold = thr_of(dd, "A", h, 1000 + ci * 10 + 1)
            else:
                h = meta[(meta.R == r) & (meta.Q == q) & (meta.D == d) & (meta.S == s) & (meta.trig == m)].config_hash
                if len(h) != 1:
                    raise RuntimeError("не найдена ячейка")
                h = h.iloc[0]
                dd = ex[ex.config_hash == h]
                src = "results/explor (x_trg)"
                fold = thr_of(dd, "A", h, 1000 + ci * 10 + 3 + ti)
            row = delta_row(fold, rel_b)
            row.update({"R": r, "Q": q, "D": d, "S": s, "trig_mult": m, "trig_len": ln, "source": src,
                        "reps": int(dd["replicate"].max() + 1)})
            pa = dd[dd.protocol == "A"].groupby("rho_test").agg(n=("failed", "size"), P=("failed", "mean"), g=("g_final", "mean"))
            row["P_A_max"] = float(pa.P.max())
            row["rho_at_P_A_max"] = float(pa.P.idxmax())
            for rho in (0.8, 0.9, 1.0):
                row[f"P_A_{rho}"] = float(pa.P.loc[round(rho, 6)]) if round(rho, 6) in pa.index else np.nan
            row["class"] = classify(row)
            # L_A по грубой сетке
            coarse = dd[dd["tag"].isin(["m_scr", "m_r100", "x_trg"])]
            la = l_a(coarse, h, 77 + ti)
            row["L_A"], row["L_A_lo"], row["L_A_hi"] = la["L"], la["L_lo"], la["L_hi"]
            rows.append(row)
            cv = pa.reset_index()
            cv.insert(0, "trig_mult", m)
            cv.insert(0, "S", s); cv.insert(0, "D", d); cv.insert(0, "Q", q); cv.insert(0, "R", r)
            curves.append(cv)
    tab = pd.DataFrame(rows)
    cols = ["R", "Q", "D", "S", "trig_mult", "trig_len", "reps", "source", "fold", "fold_lo", "fold_hi", "fold_cens", "rel", "rel_lo", "rel_hi",
            "rel_cens", "delta", "d_lo", "d_hi", "N1", "N2", "class", "P_A_max", "rho_at_P_A_max", "P_A_0.8", "P_A_0.9", "P_A_1.0",
            "L_A", "L_A_lo", "L_A_hi"]
    tab[cols].to_csv(OUT / "explor_trg.csv", index=False, encoding="utf-8")
    pd.concat(curves).to_csv(OUT / "explor_trg_curves.csv", index=False, encoding="utf-8")
    pd.set_option("display.width", 250)
    print(tab[["R", "Q", "D", "S", "trig_mult", "reps", "fold", "fold_cens", "rel", "rel_cens", "delta", "d_lo", "P_A_max", "rho_at_P_A_max", "P_A_1.0", "L_A", "class"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
