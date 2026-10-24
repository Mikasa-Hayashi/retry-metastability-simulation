"""Pilot, часть 3, п. D: покрытие bootstrap-ДИ порогов (CALIBRATION_PLAN_3.md).

python analysis/pilot3_coverage.py [--sub 200] [--par 1000]
«Истина» — оценка по репликам 100…399 (300 на точку); подвыборки — из реплик 0…99 (без возвращения).
Параметрическая версия: Bin(R, P̂_истина). Вывод: results/pilot/pilot3_coverage.csv|txt.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runner import ROOT, load_results, load_tasks, stage_dir
from src.thresholds import estimate_delta

OUT = ROOT / "results" / "pilot"
NB = 2000
RHO_A = [0.80, 0.82, 0.84, 0.86, 0.88, 0.90]
RHO_B = [0.29, 0.31, 0.33, 0.35, 0.37, 0.39]
LINES: list[str] = []


def emit(s: str = "") -> None:
    print(s)
    LINES.append(s)


def _est(fa: dict[float, np.ndarray], fb: dict[float, np.ndarray], seed: int, nb: int = NB) -> dict[str, object]:
    ra = np.concatenate([np.full(len(v), r) for r, v in fa.items()])
    va = np.concatenate(list(fa.values()))
    rb = np.concatenate([np.full(len(v), r) for r, v in fb.items()])
    vb = np.concatenate(list(fb.values()))
    e = estimate_delta(ra, va, rb, vb, nb, seed, right_bound=1.0)
    return {"fold": e.fold.value, "fold_lo": e.fold.lo, "fold_hi": e.fold.hi, "fold_cens": e.fold.censored,
            "rel": e.release.value, "rel_lo": e.release.lo, "rel_hi": e.release.hi, "rel_cens": e.release.censored,
            "delta": e.delta, "d_lo": e.lo, "d_hi": e.hi}


def _job(args: tuple[str, int, int, dict[float, np.ndarray], dict[float, np.ndarray]]) -> dict[str, object]:
    kind, R, i, fa, fb = args
    out = _est(fa, fb, 1000 + i)
    out.update({"kind": kind, "R": R, "i": i})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=200)
    ap.add_argument("--par", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    d = stage_dir("pilot")
    res, tk = load_results(d), load_tasks(d)
    h = tk[tk["tag"] == "p3_cov_A"]["config_hash"].iloc[0]
    r = res[res.config_hash == h]

    def pts(proto: str, rhos: list[float]) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray]]:
        u, t = {}, {}
        for rho in rhos:
            s = r[(r.protocol == proto) & np.isclose(r.rho_test, rho)].sort_values("replicate")
            assert s.replicate.is_unique
            u[rho] = s[s.replicate < 100].failed.to_numpy()
            t[rho] = s[(s.replicate >= 100) & (s.replicate < 400)].failed.to_numpy()
        return u, t

    ua, ta = pts("A", RHO_A)
    ub, tb = pts("B", RHO_B)
    emit(f"# D. Покрытие bootstrap-ДИ; конфигурация {h}; bootstrap {NB}")
    emit(f"Реплик на точку: подвыборочный пул A {[len(v) for v in ua.values()]}, B {[len(v) for v in ub.values()]}; "
         f"«истина» A {[len(v) for v in ta.values()]}, B {[len(v) for v in tb.values()]}")
    emit("P_A («истина»): " + str({k: round(float(v.mean()), 3) for k, v in ta.items()}))
    emit("P_B («истина»): " + str({k: round(float(v.mean()), 3) for k, v in tb.items()}))
    emit("P_A (пул 0…99): " + str({k: round(float(v.mean()), 3) for k, v in ua.items()}))
    emit("P_B (пул 0…99): " + str({k: round(float(v.mean()), 3) for k, v in ub.items()}))
    truth = _est(ta, tb, 7)
    tf, tr, td = float(truth["fold"]), float(truth["rel"]), float(truth["delta"])
    emit(f"\n«Истина» (300 реплик): ρ_fold = {tf:.4f} [{truth['fold_lo']:.4f}; {truth['fold_hi']:.4f}], "
         f"ρ_release = {tr:.4f} [{truth['rel_lo']:.4f}; {truth['rel_hi']:.4f}], Δ = {td:.4f} [{truth['d_lo']:.4f}; {truth['d_hi']:.4f}]; "
         f"цензурирование {truth['fold_cens']}/{truth['rel_cens']}")
    pool = _est(ua, ub, 8)
    emit(f"Оценка по всему пулу 0…99 (100 реплик): ρ_fold = {pool['fold']:.4f}, ρ_release = {pool['rel']:.4f}, Δ = {pool['delta']:.4f}")

    rng = np.random.default_rng(np.random.SeedSequence(2026))
    jobs = []
    for R in (40, 8):
        for i in range(a.sub):
            fa = {k: rng.choice(v, size=R, replace=False) for k, v in ua.items()}
            fb = {k: rng.choice(v, size=R, replace=False) for k, v in ub.items()}
            jobs.append(("real", R, i, fa, fb))
    for R in (40, 8):
        for i in range(a.par):
            fa = {k: (rng.random(R) < v.mean()).astype(np.int64) for k, v in ta.items()}
            fb = {k: (rng.random(R) < v.mean()).astype(np.int64) for k, v in tb.items()}
            jobs.append(("param", R, i, fa, fb))
    emit(f"\nЗадач подвыборок: {len(jobs)}")
    with mp.get_context("spawn").Pool(a.workers) as pool_:
        out = pool_.map(_job, jobs, chunksize=4)
    df = pd.DataFrame(out)
    df.to_csv(OUT / "pilot3_coverage_raw.csv", index=False)
    rows = []
    for (kind, R), g in df.groupby(["kind", "R"]):
        for nm, est, lo, hi, tv in (("ρ_fold", "fold", "fold_lo", "fold_hi", tf), ("ρ_release", "rel", "rel_lo", "rel_hi", tr),
                                    ("Δ", "delta", "d_lo", "d_hi", td)):
            cov = ((g[lo] <= tv) & (tv <= g[hi])).sum()
            ci = binomtest(int(cov), len(g)).proportion_ci(method="wilson")
            rows.append({"kind": kind, "R": R, "threshold": nm, "n_sub": len(g), "covered": int(cov),
                         "coverage": cov / len(g), "wilson_lo": ci.low, "wilson_hi": ci.high,
                         "below": int((tv < g[lo]).sum()), "above": int((tv > g[hi]).sum()),
                         "mean_est_minus_truth": float((g[est] - tv).mean()), "sd_est": float(g[est].std()),
                         "mean_half_width": float(((g[hi] - g[lo]) / 2).mean()),
                         "censored_right_share": float((g[{"fold": "fold_cens", "rel": "rel_cens", "delta": "fold_cens"}[est]] == "right").mean())})
    cv = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    emit("\nПокрытие (доля ДИ, содержащих «истину»):")
    emit(cv.round(4).to_string(index=False))
    cv.to_csv(OUT / "pilot3_coverage.csv", index=False)
    (OUT / "pilot3_coverage.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
