"""Pilot, часть 4, п. 2б (эксплораторно): протокол B' (перегрузка ρ = 2,6) против B (ρ = 1,05).

python analysis/pilot4_bp.py
A берётся из основной сетки (теги p4_scr, p4_ext, p4_ref1, p4_ref2n), B и B' — по тем же ρ. Для каждой из 36 конфигураций:
порог по B', порог по B, Δ' = ρ_fold(A) − ρ_release(B') (ρ ≤ 1 и ρ ≤ 2,5), максимум |P_B' − P_B| по общим узлам.
Вывод: results/pilot/pilot4_bp.csv|txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot4_common import OUT, cfg_meta, delta_block, load_crn

NB = 1000


def main() -> int:
    meta = cfg_meta()
    res = load_crn(["config_hash", "rho_test", "protocol", "failed", "replicate"], ["p4_scr", "p4_ext", "p4_ref1", "p4_ref2n", "p4_bp"])
    res = res[res.replicate < 40]
    res["rho_test"] = res["rho_test"].round(6)
    bp = meta[(meta.ol_rho == 2.6)]
    base = meta[(meta.ol_rho == 1.05) & (meta.m == 3) & (meta["T"] == 10.0) & (meta.theta_q == 0.5) & (meta.K == 128) & (meta.theta == 0.5)
                & (meta.trig == 1.5) & (meta.h_rec == 1500.0)]
    rows = []
    for r in bp.itertuples():
        b = base[(base.R == r.R) & (base.Q == r.Q) & (base.D == r.D) & (base.S == r.S)].iloc[0]
        a = res[(res.config_hash == b.config_hash) & (res.protocol == "A")]
        b1 = res[(res.config_hash == b.config_hash) & (res.protocol == "B")]
        b2 = res[(res.config_hash == r.config_hash) & (res.protocol == "B")]
        row: dict[str, object] = {"R": r.R, "Q": r.Q, "D": r.D, "S": r.S, "n_rho_Bp": b2.rho_test.nunique()}
        for pre, hi in (("le1", 1.0), ("ext", 2.5)):
            for nm, bb in (("Bp", b2), ("B", b1)):
                sa, sb = a[a.rho_test <= hi + 1e-9], bb[bb.rho_test <= hi + 1e-9]
                blk = delta_block(sa.rho_test.to_numpy(), sa.failed.to_numpy(), sb.rho_test.to_numpy(), sb.failed.to_numpy(), hi, NB, 5)
                for k in ("fold", "rel", "delta", "d_lo", "d_hi", "rel_cens"):
                    row[f"{pre}_{nm}_{k}"] = blk[k]
        p1 = b1.groupby("rho_test").failed.mean()
        p2 = b2.groupby("rho_test").failed.mean()
        com = p1.index.intersection(p2.index)
        row["max_abs_dP_Bp_B"] = float((p2[com] - p1[com]).abs().max())
        row["max_abs_dP_gt1"] = float((p2[com[com > 1.0]] - p1[com[com > 1.0]]).abs().max())
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["R", "Q", "D", "S"])
    df.to_csv(OUT / "pilot4_bp.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 100)
    out = ("# pilot4 bp (эксплораторно): B' (перегрузка 2,6) против B (1,05); A общий\n"
           + df[["R", "Q", "D", "S", "le1_B_rel", "le1_Bp_rel", "le1_B_delta", "le1_Bp_delta", "ext_B_rel", "ext_Bp_rel", "ext_B_delta",
                 "ext_Bp_delta", "ext_Bp_d_lo", "ext_Bp_d_hi", "max_abs_dP_Bp_B", "max_abs_dP_gt1"]].round(3).to_string(index=False))
    print(out)
    (OUT / "pilot4_bp.txt").write_text(out + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
