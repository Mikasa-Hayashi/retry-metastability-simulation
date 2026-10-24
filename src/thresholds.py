"""Оценка порогов ρ_fold, ρ_release и Δ (MODEL_SPEC §6, DECISIONS D-011)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

Censor = Literal["none", "right", "left"]


@dataclass(frozen=True)
class Threshold:
    value: float
    lo: float
    hi: float
    censored: Censor
    method: str  # "logistic" | "isotonic" | "constant"
    n_boot: int


@dataclass(frozen=True)
class DeltaEstimate:
    delta: float
    lo: float
    hi: float
    fold: Threshold
    release: Threshold


def aggregate(rho: np.ndarray, failed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает (x_i, n_i, k_i): уникальные ρ, число реплик, число отказов."""
    rho = np.round(np.asarray(rho, dtype=np.float64), 9)
    failed = np.asarray(failed, dtype=np.int64)
    if rho.shape != failed.shape or rho.ndim != 1:
        raise ValueError("rho и failed — одномерные массивы одной длины")
    if not np.all((failed == 0) | (failed == 1)):
        raise ValueError("failed должен быть 0/1")
    x, inv = np.unique(rho, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    k = np.bincount(inv, weights=failed).astype(np.float64)
    return x, n, k


def pava(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Изотоническая (неубывающая) регрессия, алгоритм PAVA с весами."""
    vals: list[float] = []
    wts: list[float] = []
    cnt: list[int] = []
    for pi, wi in zip(p.tolist(), w.tolist(), strict=True):
        vals.append(pi)
        wts.append(wi)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            wt = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / wt
            c = cnt[-2] + cnt[-1]
            vals[-2:] = [v]
            wts[-2:] = [wt]
            cnt[-2:] = [c]
    return np.repeat(np.asarray(vals), cnt)


def _logistic_fit(x: np.ndarray, n: np.ndarray, k: np.ndarray) -> tuple[float, float] | None:
    """MLE логистической регрессии logit p = a + b (x - xm); None при расходимости."""
    xm = float(np.average(x, weights=n))
    z = x - xm
    p0 = min(max(k.sum() / n.sum(), 1e-6), 1 - 1e-6)
    beta = np.array([math.log(p0 / (1 - p0)), 0.0])
    xmat = np.column_stack((np.ones_like(z), z))
    for _ in range(60):
        eta = np.clip(xmat @ beta, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        grad = xmat.T @ (k - n * p)
        w = n * p * (1.0 - p)
        hess = (xmat * w[:, None]).T @ xmat + 1e-10 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    else:
        return None
    a, b = float(beta[0]), float(beta[1])
    if not (math.isfinite(a) and math.isfinite(b)) or abs(b) * (x.max() - x.min()) > 200.0:
        return None  # почти полное разделение
    # b — наклон по z = x - xm; возвращаем порог через (a, b, xm)
    return a - b * xm, b


def _separated(x: np.ndarray, n: np.ndarray, k: np.ndarray) -> bool:
    """Полное/квазиполное разделение: все неотказы лежат не правее всех отказов."""
    has_ok = n - k > 0
    has_fail = k > 0
    if not has_ok.any() or not has_fail.any():
        return False
    return bool(x[has_ok].max() <= x[has_fail].min())


def _isotonic_threshold(
    x: np.ndarray, n: np.ndarray, k: np.ndarray, level: float
) -> tuple[float, Censor]:
    f = pava(k / n, n)
    above = np.flatnonzero(f >= level)
    if len(above) == 0:
        return math.nan, "right"
    j = int(above[0])
    if j == 0:
        return float(x[0]), "left"
    f0, f1 = float(f[j - 1]), float(f[j])
    return float(x[j - 1] + (level - f0) / (f1 - f0) * (x[j] - x[j - 1])), "none"


def point_threshold(
    x: np.ndarray, n: np.ndarray, k: np.ndarray, level: float = 0.5, right_bound: float = 1.0
) -> tuple[float, Censor, str]:
    """Порог P(ρ) = level по агрегированным данным (D-011)."""
    if k.sum() == 0:
        return right_bound, "right", "constant"
    if k.sum() == n.sum():
        return float(x[0]), "left", "constant"
    fit = None if _separated(x, n, k) else _logistic_fit(x, n, k)
    if fit is not None and fit[1] > 0.0:
        a, b = fit
        thr = (math.log(level / (1 - level)) - a) / b
        if thr > x[-1] + 1e-12:
            return right_bound, "right", "logistic"
        if thr < x[0] - 1e-12:
            return float(x[0]), "left", "logistic"
        return float(thr), "none", "logistic"
    thr, cens = _isotonic_threshold(x, n, k, level)
    if cens == "right":
        return right_bound, "right", "isotonic"
    return thr, cens, "isotonic"


def bootstrap_thresholds(
    x: np.ndarray,
    n: np.ndarray,
    k: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    level: float = 0.5,
    right_bound: float = 1.0,
) -> np.ndarray:
    """Бутстреп по репликам внутри каждой точки ρ: k* ~ Bin(n, k/n)."""
    p = k / n
    n_int = n.astype(np.int64)
    out = np.empty(n_boot)
    for b in range(n_boot):
        kb = rng.binomial(n_int, p).astype(np.float64)
        out[b] = point_threshold(x, n, kb, level, right_bound)[0]
    return out


def estimate_threshold(
    rho: np.ndarray,
    failed: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.5,
    right_bound: float = 1.0,
    alpha: float = 0.05,
) -> Threshold:
    x, n, k = aggregate(rho, failed)
    val, cens, method = point_threshold(x, n, k, level, right_bound)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    bs = bootstrap_thresholds(x, n, k, rng, n_boot, level, right_bound)
    lo, hi = np.percentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Threshold(val, float(lo), float(hi), cens, method, n_boot)


def estimate_delta(
    rho_a: np.ndarray,
    failed_a: np.ndarray,
    rho_b: np.ndarray,
    failed_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.5,
    right_bound: float = 1.0,
    alpha: float = 0.05,
) -> DeltaEstimate:
    """Δ = ρ_fold (протокол A) − ρ_release (протокол B), bootstrap ДИ."""
    xa, na, ka = aggregate(rho_a, failed_a)
    xb, nb, kb = aggregate(rho_b, failed_b)
    ss_a, ss_b = np.random.SeedSequence(seed).spawn(2)
    rng_a, rng_b = np.random.default_rng(ss_a), np.random.default_rng(ss_b)
    ba = bootstrap_thresholds(xa, na, ka, rng_a, n_boot, level, right_bound)
    bb = bootstrap_thresholds(xb, nb, kb, rng_b, n_boot, level, right_bound)
    fold_v, fold_c, fold_m = point_threshold(xa, na, ka, level, right_bound)
    rel_v, rel_c, rel_m = point_threshold(xb, nb, kb, level, right_bound)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    fa, fb = np.percentile(ba, q), np.percentile(bb, q)
    d_lo, d_hi = np.percentile(ba - bb, q)
    return DeltaEstimate(
        fold_v - rel_v,
        float(d_lo),
        float(d_hi),
        Threshold(fold_v, float(fa[0]), float(fa[1]), fold_c, fold_m, n_boot),
        Threshold(rel_v, float(fb[0]), float(fb[1]), rel_c, rel_m, n_boot),
    )
