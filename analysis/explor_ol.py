"""Эксплораторно (В9): чувствительность ρ_release и Δ к длине перегрузки протокола B (400, 1000 (база), 2000).

python analysis/explor_ol.py
12 активных ячеек (R1, R2 x FIFO x D x S). Данные: results/explor (теги x_ol, x_ol_ref), протокол B; база 1000 - results/raw (реплики < 40).
Протокол A от длины перегрузки не зависит, поэтому ρ_fold берётся из базы той же ячейки: Δ = ρ_fold(база) - ρ_release(длина).
Выход: results/analysis/explor_ol.csv (E30), explor_ol_curves.csv (E31: P_B по ρ).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explor_common import OUT, base_cfg, base_data, delta_row, explor_meta, load_explor, pct, thr_of  # noqa: E402

CELLS = [(r, "FIFO", d, s) for r in ("R1", "R2") for d in ("off", "on") for s in ("Exp", "LogN", "Pareto")]


def main() -> int:
    meta = explor_meta()
    ex = load_explor(["config_hash", "rho_test", "protocol", "failed", "g_final", "replicate"], ["x_ol", "x_ol_ref"])
    bd = base_data(["config_hash", "rho_test", "protocol", "failed", "g_final"])
    rows, curves, draws = [], [], {}
    for ci, (r, q, d, s) in enumerate(CELLS):
        hb = base_cfg(r, q, d, s)
        db = bd[bd.config_hash == hb]
        fold = thr_of(db, "A", hb, 3000 + ci * 10 + 1)
        for oi, ol in enumerate((400.0, 1000.0, 2000.0)):
            if ol == 1000.0:
                h, dd, src = hb, db, "results/raw (база, реплики < 40)"
                rel = thr_of(dd, "B", h, 3000 + ci * 10 + 2)
            else:
                h = meta[(meta.R == r) & (meta.Q == q) & (meta.D == d) & (meta.S == s) & (meta.ol_len == ol) & (meta.trig == 1.5)
                         & (meta.rbase == 1.0) & (meta.budget == 0.10) & (meta.h_rec == 1500.0)].config_hash
                if len(h) != 1:
                    raise RuntimeError("не найдена ячейка")
                h = h.iloc[0]
                dd = ex[ex.config_hash == h]
                src = "results/explor (x_ol)"
                rel = thr_of(dd, "B", h, 3000 + ci * 10 + 3 + oi)
            row = delta_row(fold, rel)
            row.update({"R": r, "Q": q, "D": d, "S": s, "ol_len": ol, "reps": int(dd["replicate"].max() + 1), "source": src})
            rows.append(row)
            draws[(r, d, s, ol)] = rel["draws"]
            pb = dd[dd.protocol == "B"].groupby("rho_test").agg(n=("failed", "size"), P=("failed", "mean")).reset_index()
            pb.insert(0, "ol_len", ol); pb.insert(0, "S", s); pb.insert(0, "D", d); pb.insert(0, "R", r)
            curves.append(pb)
    tab = pd.DataFrame(rows)
    # разность с базой (1000) по ρ_release и по Δ
    for col, f in (("rel_diff_vs_1000", lambda a, b: a), ("delta_diff_vs_1000", None)):
        pass
    diffs = []
    for _, rw in tab.iterrows():
        b = tab[(tab.R == rw.R) & (tab.D == rw.D) & (tab.S == rw.S) & (tab.ol_len == 1000.0)].iloc[0]
        dr = draws[(rw.R, rw.D, rw.S, rw.ol_len)] - draws[(rw.R, rw.D, rw.S, 1000.0)]
        lo, hi = pct(dr)
        diffs.append((float(rw.rel - b.rel), lo, hi, float(rw.delta - b.delta)))
    tab["rel_diff_vs_1000"], tab["rel_diff_lo"], tab["rel_diff_hi"], tab["delta_diff_vs_1000"] = zip(*diffs, strict=True)
    cols = ["R", "Q", "D", "S", "ol_len", "reps", "source", "fold", "fold_cens", "rel", "rel_lo", "rel_hi", "rel_cens", "delta", "d_lo", "d_hi",
            "N1", "N2", "rel_diff_vs_1000", "rel_diff_lo", "rel_diff_hi", "delta_diff_vs_1000"]
    tab[cols].to_csv(OUT / "explor_ol.csv", index=False, encoding="utf-8")
    pd.concat(curves).to_csv(OUT / "explor_ol_curves.csv", index=False, encoding="utf-8")
    pd.set_option("display.width", 250)
    print(tab[["R", "D", "S", "ol_len", "reps", "fold", "rel", "rel_lo", "rel_hi", "rel_cens", "delta", "rel_diff_vs_1000", "rel_diff_lo", "rel_diff_hi"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
