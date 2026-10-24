"""Pilot, часть 4, п. 6: покрытие bootstrap-ДИ порогов (CALIBRATION_PLAN_4.md).

python analysis/pilot4_cov.py [--sub 1000] [--par 1000]
Опорная конфигурация R1/FIFO/off/Exp, триггер (1,5; 25), перегрузка B 1000. Окна ±0,05 вокруг порогов, шаг 0,01 (pilot4_cov_*.yaml).
«Эталон»: независимый блок T (400 реплик на точку); пул подвыборок: независимый блок P (2000 реплик на точку), блоки не пересекаются.
Подвыборки R = 40 (и 8, эксплораторно) без возвращения из пула (случайные, 1000 штук) и параметрические Bin(R, P̂_эталон).
Вывод: results/pilot/pilot4_coverage.csv|txt, pilot4_coverage_raw{suffix}.csv.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot4_common import OUT, load_indep

from src.runner import ROOT
from src.thresholds import estimate_delta

NB = 2000


def est(fa: dict[float, np.ndarray], fb: dict[float, np.ndarray], seed: int) -> dict[str, object]:
    ra = np.concatenate([np.full(len(v), r) for r, v in fa.items()])
    va = np.concatenate(list(fa.values()))
    rb = np.concatenate([np.full(len(v), r) for r, v in fb.items()])
    vb = np.concatenate(list(fb.values()))
    e = estimate_delta(ra, va, rb, vb, NB, seed, right_bound=1.0)
    return {"fold": e.fold.value, "fold_lo": e.fold.lo, "fold_hi": e.fold.hi, "fold_cens": e.fold.censored,
            "rel": e.release.value, "rel_lo": e.release.lo, "rel_hi": e.release.hi, "rel_cens": e.release.censored,
            "delta": e.delta, "d_lo": e.lo, "d_hi": e.hi}


def job(args: tuple[str, int, int, dict[float, np.ndarray], dict[float, np.ndarray]]) -> dict[str, object]:
    kind, r, i, fa, fb = args
    out = est(fa, fb, 1000 + i)
    out.update({"kind": kind, "R": r, "i": i})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=1000)
    ap.add_argument("--par", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--truth-all", action="store_true",
                    help="эксплораторно (после просмотра): эталон = блоки T и P вместе (2400 реплик), подвыборки из того же объединения")
    a = ap.parse_args()
    suffix = "_all" if a.truth_all else ""
    specs = {p: yaml.safe_load(open(ROOT / "experiments" / "grids" / f"pilot4_cov_T_{p}.yaml", encoding="utf-8")) for p in ("A", "B")}
    rho = {p: specs[p]["rho_by_protocol"][p] for p in ("A", "B")}
    res = load_indep(["config_hash", "rho_test", "protocol", "failed", "replicate"], ["p4_cov_T_A", "p4_cov_T_B", "p4_cov_P_A", "p4_cov_P_B"])
    res["rho_test"] = res["rho_test"].round(6)
    L: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        L.append(s)

    def block(pr: str, kind: str) -> dict[float, np.ndarray]:
        d = res[(res.tag == f"p4_cov_{kind}_{pr}") & (res.protocol == pr)]
        return {r: d[np.isclose(d.rho_test, r)].sort_values("replicate").failed.to_numpy() for r in rho[pr]}

    ta, tb = block("A", "T"), block("B", "T")
    pa, pb = block("A", "P"), block("B", "P")
    if a.truth_all:
        ta = {k: np.concatenate([ta[k], pa[k]]) for k in ta}
        tb = {k: np.concatenate([tb[k], pb[k]]) for k in tb}
        pa, pb = ta, tb
    emit(f"# Покрытие bootstrap-ДИ; bootstrap {NB}; окна A {rho['A'][0]}…{rho['A'][-1]}, B {rho['B'][0]}…{rho['B'][-1]} шаг 0,01")
    emit(f"Реплик на точку: эталон A {sorted({len(v) for v in ta.values()})}, B {sorted({len(v) for v in tb.values()})}; "
         f"пул A {sorted({len(v) for v in pa.values()})}, B {sorted({len(v) for v in pb.values()})}")
    emit("P_A (эталон): " + str({k: round(float(v.mean()), 3) for k, v in ta.items()}))
    emit("P_B (эталон): " + str({k: round(float(v.mean()), 3) for k, v in tb.items()}))
    truth = est(ta, tb, 7)
    tf, tr, td = float(truth["fold"]), float(truth["rel"]), float(truth["delta"])
    emit(f"\nЭталон: ρ_fold = {tf:.4f} [{truth['fold_lo']:.4f}; {truth['fold_hi']:.4f}], ρ_release = {tr:.4f} "
         f"[{truth['rel_lo']:.4f}; {truth['rel_hi']:.4f}], Δ = {td:.4f} [{truth['d_lo']:.4f}; {truth['d_hi']:.4f}]; цензурирование {truth['fold_cens']}/{truth['rel_cens']}")
    pool = est(pa, pb, 8)
    emit(f"Оценка по всему пулу: ρ_fold = {pool['fold']:.4f}, ρ_release = {pool['rel']:.4f}, Δ = {pool['delta']:.4f}")
    rng = np.random.default_rng(np.random.SeedSequence(2027))
    jobs = []
    for r in (40, 8):
        for i in range(a.sub):
            fa = {k: rng.choice(v, size=r, replace=False) for k, v in pa.items()}
            fb = {k: rng.choice(v, size=r, replace=False) for k, v in pb.items()}
            jobs.append(("real", r, i, fa, fb))
    for r in (40, 8):
        if a.truth_all:
            break
        for i in range(a.par):
            fa = {k: (rng.random(r) < v.mean()).astype(np.int64) for k, v in ta.items()}
            fb = {k: (rng.random(r) < v.mean()).astype(np.int64) for k, v in tb.items()}
            jobs.append(("param", r, i, fa, fb))
    emit(f"\nПодвыборок: {len(jobs)}")
    with mp.get_context("spawn").Pool(a.workers) as p_:
        out = p_.map(job, jobs, chunksize=4)
    df = pd.DataFrame(out)
    df.to_csv(OUT / f"pilot4_coverage_raw{suffix}.csv", index=False)
    rows = []
    for (kind, r), g in df.groupby(["kind", "R"]):
        for nm, e_, lo, hi, tv in (("ρ_fold", "fold", "fold_lo", "fold_hi", tf), ("ρ_release", "rel", "rel_lo", "rel_hi", tr), ("Δ", "delta", "d_lo", "d_hi", td)):
            cov = int(((g[lo] <= tv) & (tv <= g[hi])).sum())
            wl = binomtest(cov, len(g)).proportion_ci(method="wilson")
            rows.append({"kind": kind, "R": r, "threshold": nm, "n_sub": len(g), "covered": cov, "coverage": cov / len(g),
                         "wilson_lo": wl.low, "wilson_hi": wl.high, "below": int((tv < g[lo]).sum()), "above": int((tv > g[hi]).sum()),
                         "mean_est_minus_truth": float((g[e_] - tv).mean()), "sd_est": float(g[e_].std()),
                         "mean_half_width": float(((g[hi] - g[lo]) / 2).mean())})
    cv = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    emit("\nПокрытие (доля ДИ, содержащих эталон):")
    emit(cv.round(4).to_string(index=False))
    cv.to_csv(OUT / f"pilot4_coverage{suffix}.csv", index=False)
    (OUT / f"pilot4_coverage{suffix}.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
