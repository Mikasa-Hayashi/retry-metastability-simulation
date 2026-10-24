"""Pilot, часть 4, п. 6: чувствительность опорной конфигурации прямыми прогонами (CALIBRATION_PLAN_4.md).

python analysis/pilot4_sens.py
Опорная R1/FIFO/off/Exp, триггер (1,5; 25), перегрузка B 1000. Варианты: θ ∈ {0,3; 0,5; 0,7}; T ∈ {5; 10; 20}; K ∈ {64; 128; 256};
m_s ∈ {1,25; 1,5; 1,75} (только A; B из базы). Данные: общий блок, реплики 0…39, ρ ≤ 1,5, все имеющиеся точки (включая уточнение).
Δ(ρ ≤ 1) и Δ_ext (ρ ≤ 1,5), mean-field (analysis/theory.py). Проверка идентичности траекторий при смене θ (g_final совпадает).
Вывод: results/pilot/pilot4_sens.csv|txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import theory as th
from pilot4_common import OUT, cfg_meta, delta_block, load_crn

NB = 2000
RHO_MAX = 1.5


def mf_spinodals(K: int, T: float, theta: float = 0.5) -> tuple[float, float]:
    p = th.MFParams(c=8, k_queue=K, t_out=T)

    def has_h(r: float) -> bool:
        return any(x.stable and th.goodput_ratio(x.q, 3) >= theta for x in th.fixed_points(r, p, 801))

    def has_f(r: float) -> bool:
        return any(x.stable and th.goodput_ratio(x.q, 3) < theta for x in th.fixed_points(r, p, 801))

    fold = th.refine_threshold(has_h, 0.3, 1.2, 1e-3) if (has_h(0.3) and not has_h(1.2)) else float("nan")
    rel = th.refine_threshold(has_f, 0.05, 1.2, 1e-3) if ((not has_f(0.05)) and has_f(1.2)) else float("nan")
    return rel, fold


def main() -> int:
    meta = cfg_meta()
    ref = meta[(meta.R == "R1") & (meta.Q == "FIFO") & (meta.D == "off") & (meta.S == "Exp") & (meta.theta_q == 0.5)
               & (meta.m == 3) & (meta.c == 8) & (meta.h_rec == 1500.0) & (meta.ol_len == 1000.0)]
    res = load_crn(["config_hash", "rho_test", "protocol", "failed", "g_final", "replicate"])
    res = res[(res.replicate < 40) & (res.rho_test <= RHO_MAX + 1e-9)]

    def h_of(T: float = 10.0, K: int = 128, trig: float = 1.5, theta: float = 0.5) -> str:
        r = ref[(ref["T"] == T) & (ref.K == K) & (ref.trig == trig) & (ref.theta == theta)]
        return str(r.config_hash.iloc[0])

    base = h_of()
    variants = [("база", "T=10,K=128,θ=0,5,m_s=1,5", base, base)]
    for t in (5.0, 20.0):
        variants.append(("T", f"{t:g}", h_of(T=t), h_of(T=t)))
    for k in (64, 256):
        variants.append(("K", f"{k}", h_of(K=k), h_of(K=k)))
    for th_ in (0.3, 0.7):
        variants.append(("θ (прямой прогон)", f"{th_:g}", h_of(theta=th_), h_of(theta=th_)))
    for m in (1.25, 1.75):
        variants.append(("сила триггера m_s", f"{m:g}", h_of(trig=m), base))
    mf_cache: dict[tuple[int, float, float], tuple[float, float]] = {}
    rows = []
    for var, lev, ha, hb in variants:
        ra = res[(res.config_hash == ha) & (res.protocol == "A")]
        rb = res[(res.config_hash == hb) & (res.protocol == "B")]
        row: dict[str, object] = {"var": var, "level": lev, "nA": len(ra), "nB": len(rb),
                                  "rhoA": ra.rho_test.nunique(), "rhoB": rb.rho_test.nunique(), "n_min": int(min(ra.groupby("rho_test").size().min(), rb.groupby("rho_test").size().min()))}
        for pre, hi in (("le1", 1.0), ("ext", RHO_MAX)):
            sa = ra[ra.rho_test <= hi + 1e-9]
            sb = rb[rb.rho_test <= hi + 1e-9]
            blk = delta_block(sa.rho_test.to_numpy(), sa.failed.to_numpy(), sb.rho_test.to_numpy(), sb.failed.to_numpy(), hi, NB, 21)
            for k, v in blk.items():
                row[f"{pre}_{k}"] = v
        c = ref[ref.config_hash == ha].iloc[0]
        key = (int(c.K), float(c["T"]), float(c.theta))
        if key not in mf_cache:
            mf_cache[key] = mf_spinodals(*key)
        row["mf_rel"], row["mf_fold"] = mf_cache[key]
        row["sim_minus_mf_rel"] = row["le1_rel"] - row["mf_rel"]
        row["sim_minus_mf_fold"] = row["le1_fold"] - row["mf_fold"]
        row["admissible"] = bool(row["le1_delta"] >= 0.1 and row["le1_fold_cens"] != "right")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pilot4_sens.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    L = [f"# pilot4 sens: общий блок, реплики 0…39, ρ ≤ {RHO_MAX}, bootstrap {NB}; база {base}"]
    cols = ["var", "level", "n_min", "rhoA", "rhoB", "le1_fold", "le1_fold_lo", "le1_fold_hi", "le1_fold_cens", "le1_rel", "le1_rel_lo",
            "le1_rel_hi", "le1_rel_cens", "le1_delta", "le1_d_lo", "le1_d_hi", "admissible"]
    L.append("\n## Δ(ρ ≤ 1)\n" + df[cols].round(4).to_string(index=False))
    cols2 = ["var", "level", "ext_fold", "ext_fold_cens", "ext_rel", "ext_rel_cens", "ext_delta", "ext_d_lo", "ext_d_hi", "ext_cls"]
    L.append(f"\n## Δ_ext (ρ ≤ {RHO_MAX}; область перегрузки)\n" + df[cols2].round(4).to_string(index=False))
    L.append("\n## Mean-field и разность sim − MF (ρ ≤ 1)\n" + df[["var", "level", "mf_rel", "mf_fold", "sim_minus_mf_rel", "sim_minus_mf_fold"]].round(4).to_string(index=False))
    # идентичность траекторий при смене θ
    g = res[res.config_hash.isin([base, h_of(theta=0.3), h_of(theta=0.7)])]
    pv = g.pivot_table(index=["protocol", "rho_test", "replicate"], columns="config_hash", values="g_final", aggfunc="first").dropna()
    d03 = float(np.abs(pv[h_of(theta=0.3)] - pv[base]).max())
    d07 = float(np.abs(pv[h_of(theta=0.7)] - pv[base]).max())
    L.append(f"\nИдентичность траекторий при смене θ (одинаковые seed'ы): max|g_final(θ) − g_final(0,5)| = {d03:.3g} (θ=0,3), {d07:.3g} (θ=0,7); пар {len(pv)}")
    # P(ρ) вблизи порогов для каждого варианта
    for var, lev, ha, hb in variants:
        for pr, hh in (("A", ha), ("B", hb)):
            s = res[(res.config_hash == hh) & (res.protocol == pr)].groupby("rho_test").failed.agg(["mean", "size"])
            thr = s[(s["mean"] > 0) & (s["mean"] < 1)]
            if len(thr):
                L.append(f"P_{pr} [{var} {lev}] (0<P<1): " + ", ".join(f"{i:.3f}:{r['mean']:.2f}" for i, r in thr.iterrows()))
    out = "\n".join(L)
    print(out)
    (OUT / "pilot4_sens.txt").write_text(out + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
