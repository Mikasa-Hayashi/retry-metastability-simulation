"""Дополнительно: (1) KS-тесты распределений обслуживания из прогонов src; (2) восстановление порога
оценщиком src.thresholds на синтетической логистической кривой с известным порогом.
Уровень значимости KS заранее: alpha = 0.01 на тест. Seeds 95000+.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import Config
from src.model import simulate
from src.thresholds import estimate_threshold

out: dict = {"ks": {}, "threshold_recovery": {}}
# service durations (D=off, R0, rho=0.5: все принятые попытки обслужены)
for dist, cdf in (("Exp", lambda x: 1 - np.exp(-x)),
                  ("LogN", lambda x: stats.norm.cdf((np.log(x) + 0.5 * math.log(5)) / math.sqrt(math.log(5)))),
                  ("Pareto", lambda x: np.where(x < 0.6, 0.0, 1 - (0.6 / np.maximum(x, 0.6)) ** 2.5))):
    cfg = Config(R="R0", S=dist, T=math.inf)
    res = simulate(cfg, [(40000.0, 0.5)], 40000.0, 95000)
    x = res.starts[:, 2]
    x = x[: 100000]
    ks = stats.kstest(x, cdf)
    out["ks"][dist] = {"n": len(x), "mean": float(x.mean()), "cv": float(x.std() / x.mean()),
                       "ks_stat": float(ks.statistic), "p": float(ks.pvalue), "pass_alpha_0.01": bool(ks.pvalue > 0.01)}
# threshold recovery: logistic P(rho) with known 0.5-crossing
rng = np.random.default_rng(95001)
for true_thr, slope in ((0.62, 40.0), (0.8, 25.0)):
    est = []
    cover = 0
    nsim = 60
    for i in range(nsim):
        rhos = np.repeat(np.round(np.arange(0.40, 1.0001, 0.05), 2), 30)
        p = 1 / (1 + np.exp(-slope * (rhos - true_thr)))
        failed = (rng.random(len(rhos)) < p).astype(int)
        t = estimate_threshold(rhos, failed, n_boot=300, seed=i)
        est.append(t.value)
        cover += int(t.lo <= true_thr <= t.hi)
    out["threshold_recovery"][f"thr{true_thr}_slope{slope}"] = {
        "mean_est": float(np.mean(est)), "sd_est": float(np.std(est)), "bias": float(np.mean(est) - true_thr),
        "ci_coverage": cover / nsim, "nsim": nsim}
print(json.dumps(out, indent=1))
(ROOT / "reports" / "validation_data" / "misc.json").write_text(json.dumps(out, indent=1))
