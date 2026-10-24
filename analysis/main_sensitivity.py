"""Основной анализ, п. 5 плана: чувствительность на 12 активных ячейках.

python analysis/main_sensitivity.py
Варианты (по одной переменной за раз, R = 40 реплик, грубая сетка + один раунд уточнения): θ = 0,3 / 0,7; T = 5 / 20;
K = 64 / 256; H_rec = 3000 / 6000; сила триггера m_s = 1,25 / 1,75 (масштабирование m_s − 1 на ×0,5 и ×1,5, DECISIONS D-020; протокол A,
ρ_release берётся из базы, т. к. протокол B не зависит от триггера, D-015). База для сравнения — реплики с номером < 40
базовой сетки (m_scr, m_ref1; точки грубые и уточнённые), чтобы объём данных совпадал.
Выход: t10_sens.csv (T10), t10_boot.npz, t11_sens_h1p.csv (T11: эффект D по S в каждом варианте), t12_mf_variants.csv (T12: mean-field
для вариантов T, K; эксплораторно).
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_common import (  # noqa: E402
    ACTIVE12, BASE_TAGS, BOOT, DELTA_MIN, OUT, SENS_TAGS, VARIANTS, cfg_meta, load_main, pct, thr_boot,
)


def work(args: tuple) -> dict:
    key, h_a, ra, fa, h_b, rb, fb, code = args
    a = thr_boot(ra, fa, 1.0, h_a, code + 1) if len(ra) else None
    b = thr_boot(rb, fb, 1.0, h_b, code + 2) if len(rb) else None
    return {"key": key, "a": a, "b": b}


def main() -> int:
    meta = cfg_meta()
    cols = ["config_hash", "rho_test", "protocol", "failed", "replicate", "tag"]
    res = load_main(cols, BASE_TAGS + SENS_TAGS)
    res = res[(res["rho_test"] <= 1.0 + 1e-9)]
    base_data = res[res["tag"].isin(BASE_TAGS) & (res["replicate"] < 40)]
    sens_data = res[res["tag"].isin(SENS_TAGS)]
    cells = {c: None for c in ACTIVE12}
    jobs = []
    keys = []
    code = 1000
    for var in ["base40"] + VARIANTS:
        for (r, q, d, s) in cells:
            code += 10
            if var == "base40":
                hs = meta[(meta.variant == "base") & (meta.R == r) & (meta.Q == q) & (meta.D == d) & (meta.S == s)].config_hash
                src = base_data
            else:
                hs = meta[(meta.variant == var) & (meta.R == r) & (meta.Q == q) & (meta.D == d) & (meta.S == s)].config_hash
                src = sens_data
            if len(hs) != 1:
                raise RuntimeError(f"не найдена конфигурация {var} {r}/{q}/{d}/{s}")
            h = hs.iloc[0]
            dd = src[src["config_hash"] == h]
            a, b = dd[dd["protocol"] == "A"], dd[dd["protocol"] == "B"]
            keys.append((var, r, q, d, s))
            # для протокола B вариантов триггера используем базу (код bootstrap совпадает с базой той же ячейки)
            jobs.append(((var, r, q, d, s), h, a["rho_test"].to_numpy(), a["failed"].to_numpy().astype(int), h,
                         b["rho_test"].to_numpy(), b["failed"].to_numpy().astype(int), code))
    with mp.Pool(7) as pool:
        out = pool.map(work, jobs, chunksize=2)
    res_map = {o["key"]: o for o in out}
    rows, fd, rd = [], [], []
    for key in keys:
        var, r, q, d, s = key
        o = res_map[key]
        a, b = o["a"], o["b"]
        if b is None:  # триггерные варианты: ρ_release из базы той же ячейки
            b = res_map[("base40", r, q, d, s)]["b"]
            rel_src = "база (B не зависит от триггера)"
        else:
            rel_src = "вариант"
        dr = a["draws"] - b["draws"]
        lo, hi = pct(dr)
        delta = a["value"] - b["value"]
        rows.append({"variant": var, "R": r, "Q": q, "D": d, "S": s, "reps": a["n_max"], "rel_source": rel_src,
                     "fold": a["value"], "fold_lo": a["lo"], "fold_hi": a["hi"], "fold_cens": a["cens"],
                     "rel": b["value"], "rel_lo": b["lo"], "rel_hi": b["hi"], "rel_cens": b["cens"],
                     "delta": delta, "d_lo": lo, "d_hi": hi,
                     "N1": bool(a["cens"] != "right" and lo > 0), "N2": bool(a["cens"] != "right" and delta >= DELTA_MIN)})
        fd.append(a["draws"])
        rd.append(b["draws"])
    tab = pd.DataFrame(rows)
    fd_a, rd_a = np.stack(fd), np.stack(rd)
    np.savez_compressed(OUT / "t10_boot.npz", key=np.array(["|".join(k) for k in keys]), fold=fd_a, rel=rd_a)
    # разность с базой
    idx = {k: i for i, k in enumerate(keys)}
    dd_lo, dd_hi, dd = [], [], []
    for key in keys:
        var, r, q, d, s = key
        i, j = idx[key], idx[("base40", r, q, d, s)]
        dv = (fd_a[i] - rd_a[i]) - (fd_a[j] - rd_a[j])
        lo, hi = pct(dv)
        dd.append(float(tab.delta[i] - tab.delta[j]))
        dd_lo.append(lo)
        dd_hi.append(hi)
    tab["diff_vs_base"], tab["diff_lo"], tab["diff_hi"] = dd, dd_lo, dd_hi
    tab.to_csv(OUT / "t10_sens.csv", index=False, encoding="utf-8")
    print("T10:", len(tab), "строк")

    # эффект D по S в каждом варианте: r(S) = Δ_on / Δ_off (R1, R2 усреднены); разность r(Exp) − r(LogN) (H1')
    h1 = []
    for var in ["base40"] + VARIANTS:
        def dr_of(r: str, d: str, s: str, var: str = var) -> np.ndarray:
            i = idx[(var, r, "FIFO", d, s)]
            return fd_a[i] - rd_a[i]

        def ratio(s: str, var: str = var) -> np.ndarray:
            on = dr_of("R1", "on", s) + dr_of("R2", "on", s)
            off = dr_of("R1", "off", s) + dr_of("R2", "off", s)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(off > 1e-9, on / off, np.nan)

        rs = {s: ratio(s) for s in ("Exp", "LogN", "Pareto")}
        def pt(s: str, var: str = var) -> float:
            sub = tab[(tab.variant == var) & (tab.S == s)]
            on = sub[sub.D == "on"].delta.sum()
            off = sub[sub.D == "off"].delta.sum()
            return float(on / off) if off > 1e-9 else float("nan")

        diff = rs["Exp"] - rs["LogN"]
        ok = ~np.isnan(diff)
        lo, hi = pct(diff[ok]) if ok.sum() > 100 else (np.nan, np.nan)
        h1.append({"variant": var, "r_Exp": pt("Exp"), "r_LogN": pt("LogN"), "r_Pareto": pt("Pareto"),
                   "diff_Exp_minus_LogN": pt("Exp") - pt("LogN"), "diff_lo": lo, "diff_hi": hi, "valid_draws": int(ok.sum()),
                   "N1_cells": int(tab[tab.variant == var].N1.sum())})
    pd.DataFrame(h1).to_csv(OUT / "t11_sens_h1p.csv", index=False, encoding="utf-8")

    # mean-field для вариантов T, K (эксплораторно)
    import theory as th

    def mf_spin(K: int, T: float, theta: float = 0.5) -> tuple[float, float]:
        p = th.MFParams(c=8, k_queue=K, t_out=T)

        def has_h(r_: float) -> bool:
            return any(x.stable and th.goodput_ratio(x.q, 3) >= theta for x in th.fixed_points(r_, p, 801))

        def has_f(r_: float) -> bool:
            return any(x.stable and th.goodput_ratio(x.q, 3) < theta for x in th.fixed_points(r_, p, 801))

        fold = th.refine_threshold(has_h, 0.3, 1.2, 1e-3) if (has_h(0.3) and not has_h(1.2)) else float("nan")
        rel = th.refine_threshold(has_f, 0.05, 1.2, 1e-3) if ((not has_f(0.05)) and has_f(1.2)) else float("nan")
        return rel, fold

    mf = []
    for var, T, K in (("base", 10.0, 128), ("T5", 5.0, 128), ("T20", 20.0, 128), ("K64", 10.0, 64), ("K256", 10.0, 256)):
        rel, fold = mf_spin(K, T)
        mf.append({"variant": var, "T": T, "K": K, "mf_release": rel, "mf_fold": fold})
    pd.DataFrame(mf).to_csv(OUT / "t12_mf_variants.csv", index=False, encoding="utf-8")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
