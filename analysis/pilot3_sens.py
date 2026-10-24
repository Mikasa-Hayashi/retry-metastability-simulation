"""Pilot, часть 3, п. C: чувствительность опорной конфигурации (CALIBRATION_PLAN_3.md).

python analysis/pilot3_sens.py
θ: пересчёт флага failed по g_final базы (θ входит в модель только через failed); T, K, сила триггера: новые прогоны.
Mean-field для T, K (analysis/theory.py, конечное K). Вывод: results/pilot/pilot3_sens.csv|txt, pilot3_mf.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import theory as th

from src.config import Config
from src.protocols import run_protocol
from src.runner import ROOT, load_results, load_tasks, stage_dir
from src.thresholds import estimate_delta

OUT = ROOT / "results" / "pilot"
NB = 2000
MAX_REP = 40
LINES: list[str] = []
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def emit(s: str = "") -> None:
    print(s)
    LINES.append(s)


def base_variants(tk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, js in tk.drop_duplicates("config_hash")[["config_hash", "cfg_json"]].itertuples(index=False):
        c = json.loads(js)
        if (c["R"], c["Q"], c["D"], c["S"], c["c"], c["theta_q"]) != ("R1", "FIFO", "off", "Exp", 8, 0.5):
            continue
        p = c["proto"]
        if (p["theta"], p["h_rec"], p["overload_len"], p["t_warm"]) != (0.5, 1500.0, 400.0, 200.0):
            continue
        if p["trigger_len"] != 25.0:
            continue
        rows.append({"hash": h, "T": c["T"], "K": c["K"], "m_s": p["trigger_mult"]})
    return pd.DataFrame(rows)


def thr(res: pd.DataFrame, ha: str, hb: str, theta: float | None = None, seed: int = 21) -> dict[str, object]:
    g25 = np.abs(res.rho_test.to_numpy() * 40.0 - np.round(res.rho_test.to_numpy() * 40.0)) < 1e-6  # сетка 0,025 (как у вариаций)
    ra = res[(res.config_hash == ha) & (res.protocol == "A") & (res.replicate < MAX_REP) & g25]
    rb = res[(res.config_hash == hb) & (res.protocol == "B") & (res.replicate < MAX_REP) & g25]
    fa = ra.failed.to_numpy() if theta is None else (ra.g_final.to_numpy() < theta).astype(int)
    fb = rb.failed.to_numpy() if theta is None else (rb.g_final.to_numpy() < theta).astype(int)
    e = estimate_delta(ra.rho_test.to_numpy(), fa, rb.rho_test.to_numpy(), fb, NB, seed, right_bound=1.0)
    return {
        "nA": len(ra), "nB": len(rb), "rhoA": ra.rho_test.nunique(), "rhoB": rb.rho_test.nunique(),
        "fold": e.fold.value, "fold_lo": e.fold.lo, "fold_hi": e.fold.hi, "fold_cens": e.fold.censored,
        "rel": e.release.value, "rel_lo": e.release.lo, "rel_hi": e.release.hi, "rel_cens": e.release.censored,
        "delta": e.delta, "d_lo": e.lo, "d_hi": e.hi,
        "maxP_A": float(ra.groupby("rho_test").failed.mean().max()) if theta is None else float(pd.Series(fa, index=ra.index).groupby(ra.rho_test).mean().max()),
        "maxP_B": float(rb.groupby("rho_test").failed.mean().max()) if theta is None else float(pd.Series(fb, index=rb.index).groupby(rb.rho_test).mean().max()),
        "eligible": bool(e.delta >= 0.1 and e.fold.censored != "right"),
    }


def mf_spinodals(K: int, T: float, theta: float = 0.5) -> tuple[float, float]:
    p = th.MFParams(c=8, k_queue=K, t_out=T)

    def has_h(r: float) -> bool:
        return any(x.stable and th.goodput_ratio(x.q, 3) >= theta for x in th.fixed_points(r, p, 801))

    def has_f(r: float) -> bool:
        return any(x.stable and th.goodput_ratio(x.q, 3) < theta for x in th.fixed_points(r, p, 801))

    lo_h, hi_h = 0.3, 1.2
    if has_h(lo_h) and not has_h(hi_h):
        fold = th.refine_threshold(has_h, lo_h, hi_h, 1e-3)
    else:
        fold = float("nan")  # здоровая ветвь не исчезает до ρ = 1,2
    lo_f, hi_f = 0.05, 1.2
    if (not has_f(lo_f)) and has_f(hi_f):
        rel = th.refine_threshold(has_f, lo_f, hi_f, 1e-3)
    else:
        rel = float("nan")  # устойчивой отказной ветви нет до ρ = 1,2
    return rel, fold


def main() -> int:
    d = stage_dir("pilot")
    res, tk = load_results(d), load_tasks(d)
    bv = base_variants(tk)
    base = bv[(bv["T"] == 10.0) & (bv.K == 128) & (bv.m_s == 1.5)].hash.iloc[0]
    emit(f"# C. Чувствительность опорной (R1/FIFO/off/Exp), реплики < {MAX_REP}, только ρ, кратные 0,025, bootstrap {NB}; база hash {base}")
    rows = []

    def add(var: str, level: str, ha: str, hb: str, theta: float | None = None) -> None:
        rows.append({"var": var, "level": level, **thr(res, ha, hb, theta)})

    add("база", "T=10,K=128,θ=0,5,m_s=1,5", base, base)
    for t_ in (5.0, 20.0):
        h = bv[(bv["T"] == t_) & (bv.K == 128) & (bv.m_s == 1.5)].hash.iloc[0]
        add("T", f"{t_:g}", h, h)
    for k_ in (64, 256):
        h = bv[(bv["T"] == 10.0) & (bv.K == k_) & (bv.m_s == 1.5)].hash.iloc[0]
        add("K", f"{k_}", h, h)
    for th_ in (0.3, 0.7):
        add("θ", f"{th_:g}", base, base, th_)
    for m_ in (1.25, 1.75):
        h = bv[(bv["T"] == 10.0) & (bv.K == 128) & (bv.m_s == m_)].hash.iloc[0]
        add("сила триггера (m_s)", f"{m_:g} (×{(m_ - 1) / 0.5:g} по m_s−1)", h, base)
    df = pd.DataFrame(rows)

    # mean-field
    emit("\nMean-field (MFParams c=8, k_queue=K, t_out=T, m=3; спинодали бисекцией, допуск 1e-3):")
    mf = {}
    for var, level, K, T, theta in [("база", "", 128, 10.0, 0.5), ("T", "5", 128, 5.0, 0.5), ("T", "20", 128, 20.0, 0.5),
                                    ("K", "64", 64, 10.0, 0.5), ("K", "256", 256, 10.0, 0.5),
                                    ("θ", "0.3", 128, 10.0, 0.3), ("θ", "0.7", 128, 10.0, 0.7)]:
        rel, fold = mf_spinodals(K, T, theta)
        mf[(var, level)] = (rel, fold)
        emit(f"  {var} {level}: ρ_release^MF = {rel:.4f}, ρ_fold^MF = {fold:.4f}")
    def mf_for(var: str, level: str) -> tuple[float, float]:
        if var == "база":
            return mf[("база", "")]
        key = level.replace(",", ".")
        return mf.get((var, key), (np.nan, np.nan))

    df["mf_rel"] = [mf_for(r.var, r.level)[0] for r in df.itertuples()]
    df["mf_fold"] = [mf_for(r.var, r.level)[1] for r in df.itertuples()]
    df["sim_minus_mf_rel"] = df.rel - df.mf_rel
    df["sim_minus_mf_fold"] = df.fold - df.mf_fold
    emit("\nПороги и допустимость (Δ ≥ 0,1 и ρ_fold не цензурирован справа):")
    emit(df.drop(columns=["nA", "nB"]).round(3).to_string(index=False))
    emit("\nЧисло реплик/точек: " + "; ".join(f"{r.var} {r.level}: A {r.nA}/{r.rhoA}, B {r.nB}/{r.rhoB}" for r in df.itertuples()))
    emit("\nДопустимость по переменным (сохраняется при обоих уровнях):")
    for var in ("T", "K", "θ", "сила триггера (m_s)"):
        s = df[df["var"] == var]
        emit(f"  {var}: {'сохраняется' if s.eligible.all() else 'НЕ сохраняется'} ({', '.join(f'{r.level}: {r.eligible}' for r in s.itertuples())})")

    # эквивалентность θ (прямой запуск 24 прогонов с иным θ)
    emit("\nПроверка эквивалентности пересчёта θ (прямой запуск, без записи в этап):")
    t0 = tk[(tk.config_hash == base) & (tk.replicate < 6) & (tk.rho_test.isin([0.35, 0.85, 0.9]))].head(24)
    bad = 0
    for r in t0.itertuples():
        cfg0 = Config.model_validate(json.loads(r.cfg_json))
        proto = cfg0.proto.model_copy(update={"theta": 0.3})
        cfg1 = cfg0.model_copy(update={"proto": proto})
        row = run_protocol(cfg1, r.protocol, r.rho_test, int(r.seed))  # type: ignore[arg-type]
        old = res[(res.config_hash == base) & (res.seed == r.seed) & (res.protocol == r.protocol)].iloc[0]
        ok = (row["g_final"] == old["g_final"]) and (row["failed"] == int(old["g_final"] < 0.3))
        bad += 0 if ok else 1
    emit(f"  прогонов {len(t0)}, расхождений g_final или failed: {bad}")

    df.to_csv(OUT / "pilot3_sens.csv", index=False)
    (OUT / "pilot3_sens.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
