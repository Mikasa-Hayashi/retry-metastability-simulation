"""Эталонная аналитика для валидации симулятора и mean-field модели R1.

Чистые функции, без зависимости от симулятора (src/). Время нормировано, но
функции принимают явные параметры (lam, mu).

Источники и пометки см. docs/THEORY.md. Кратко:
* Erlang C, W_q, W для M/M/c: https://en.wikipedia.org/wiki/M/M/c_queue
* Моменты времени ожидания FIFO/LIFO в M/G/1: Finch S., arXiv:2208.09980.
* Классическая система с повторными вызовами: Daduna, arXiv:2004.08816
  (формулы Фалина для стационарного распределения); замкнутая форма для
  среднего размера орбиты и mean-field модель: вывод теоретика, проверен
  численно (tests/test_theory.py).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammainc, gammaln

# --------------------------------------------------------------------------
# M/M/1, M/M/c
# --------------------------------------------------------------------------


def erlang_b(c: int, a: float) -> float:
    """Формула Эрланга B через устойчивую рекурсию."""
    b = 1.0
    for k in range(1, c + 1):
        b = a * b / (k + a * b)
    return b


def erlang_c(c: int, a: float) -> float:
    """P(ожидание) для M/M/c, a = lam/mu < c (Erlang C)."""
    if c < 1:
        raise ValueError("c >= 1")
    if a >= c:
        raise ValueError("нужна стабильность a < c")
    rho = a / c
    b = erlang_b(c, a)
    return b / (1.0 - rho * (1.0 - b))


def mmc_mean_wq(c: int, lam: float, mu: float) -> float:
    """Среднее время ожидания в очереди W_q = C / (c mu - lam)."""
    return erlang_c(c, lam / mu) / (c * mu - lam)


def mmc_mean_w(c: int, lam: float, mu: float) -> float:
    """Среднее время в системе W = W_q + 1/mu."""
    return mmc_mean_wq(c, lam, mu) + 1.0 / mu


def mmc_wq_tail(c: int, lam: float, mu: float, t: float) -> float:
    """P(W_q > t) для FIFO: C * exp(-(c mu - lam) t) (вывод теоретика)."""
    return erlang_c(c, lam / mu) * math.exp(-(c * mu - lam) * max(t, 0.0))


def mm1_mean_wq(lam: float, mu: float) -> float:
    """M/M/1: W_q = rho / (mu - lam)."""
    return (lam / mu) / (mu - lam)


def mm1_mean_w(lam: float, mu: float) -> float:
    """M/M/1: W = 1 / (mu - lam)."""
    return 1.0 / (mu - lam)


def mm1_prob_wait(lam: float, mu: float) -> float:
    """M/M/1: P(ожидание) = rho."""
    return lam / mu


# --------------------------------------------------------------------------
# FIFO / LIFO (невытесняющий) в M/G/1: моменты времени ожидания
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WaitMoments:
    mean: float
    var: float


def mg1_wait_moments_fifo(lam: float, es2: float, es3: float, es1: float) -> WaitMoments:
    """M/G/1 FIFO: mean = lam*E[S^2]/(2(1-rho)); var по Finch (arXiv:2208.09980).

    var = lam E[S^3]/(3(1-rho)) + (lam E[S^2])^2 / (4 (1-rho)^2).
    """
    rho = lam * es1
    mean = lam * es2 / (2.0 * (1.0 - rho))
    var = lam * es3 / (3.0 * (1.0 - rho)) + (lam * es2) ** 2 / (4.0 * (1.0 - rho) ** 2)
    return WaitMoments(mean, var)


def mg1_wait_moments_lifo(lam: float, es2: float, es3: float, es1: float) -> WaitMoments:
    """M/G/1 LIFO (невытесняющий): среднее то же, дисперсия по Finch.

    var = lam E[S^3]/(3(1-rho)^2) + (lam E[S^2])^2 (1+rho) / (4 (1-rho)^3).
    """
    rho = lam * es1
    mean = lam * es2 / (2.0 * (1.0 - rho))
    var = lam * es3 / (3.0 * (1.0 - rho) ** 2) + (lam * es2) ** 2 * (1.0 + rho) / (
        4.0 * (1.0 - rho) ** 3
    )
    return WaitMoments(mean, var)


def mm1_wait_moments(lam: float, mu: float, lifo: bool) -> WaitMoments:
    """M/M/1: E[S^k] = k!/mu^k, подстановка в формулы M/G/1."""
    es1, es2, es3 = 1.0 / mu, 2.0 / mu**2, 6.0 / mu**3
    f = mg1_wait_moments_lifo if lifo else mg1_wait_moments_fifo
    return f(lam, es2, es3, es1)


# --------------------------------------------------------------------------
# Классическая система с повторными вызовами M/M/1//(K=0)
# --------------------------------------------------------------------------


def retrial_busy_prob(lam: float, mu: float) -> float:
    """P(сервер занят) = rho = lam/mu (баланс потока, требуется rho < 1)."""
    return lam / mu


def retrial_mean_orbit(lam: float, mu: float, nu: float) -> float:
    """Среднее число на орбите: rho (rho + lam/nu) / (1 - rho).

    Вывод теоретика; проверен решением уравнений равновесия и суммированием
    формул Фалина (см. retrial_distribution_falin) в tests/test_theory.py.
    """
    rho = lam / mu
    if rho >= 1.0:
        raise ValueError("нужно rho < 1")
    return rho * (rho + lam / nu) / (1.0 - rho)


def retrial_numeric(lam: float, mu: float, nu: float, n_max: int = 600) -> tuple[float, float]:
    """Уравнения равновесия на усечённом пространстве (n, s), n <= n_max.

    Состояния: (n, 0) сервер свободен, (n, 1) занят; n = число на орбите.
    Переходы: (n,0)->(n,1) lam; (n,0)->(n-1,1) n*nu (успешный повтор);
    (n,1)->(n+1,1) lam; (n,1)->(n,0) mu. Возвращает (P(занят), E[N]).
    """
    size = 2 * (n_max + 1)

    def idx(n: int, s: int) -> int:
        return 2 * n + s

    q = np.zeros((size, size))
    for n in range(n_max + 1):
        q[idx(n, 0), idx(n, 1)] += lam
        if n >= 1:
            q[idx(n, 0), idx(n - 1, 1)] += n * nu
        if n < n_max:
            q[idx(n, 1), idx(n + 1, 1)] += lam
        q[idx(n, 1), idx(n, 0)] += mu
    np.fill_diagonal(q, -q.sum(axis=1))
    a = np.vstack([q.T, np.ones(size)])
    b = np.zeros(size + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(a, b, rcond=None)
    ns = np.repeat(np.arange(n_max + 1), 2)
    busy = pi[1::2].sum()
    return float(busy), float((pi * ns).sum())


def retrial_distribution_falin(lam: float, mu: float, nu: float, n_max: int = 600) -> np.ndarray:
    """P(n, занят) по формуле Фалина (Daduna, arXiv:2004.08816, (2.13)) с nu_k = k nu.

    x(n,b) = x0 (lam/mu)^(n+1) prod_{k=1..n} (lam + k nu)/(k nu); нормировка
    выполняется через условие P(занят) = rho (из баланса потока), см. THEORY.md.
    """
    rho = lam / mu
    logx = np.empty(n_max + 1)
    acc = 0.0
    for n in range(n_max + 1):
        if n > 0:
            acc += math.log((lam + n * nu) / (n * nu))
        logx[n] = (n + 1) * math.log(rho) + acc
    x = np.exp(logx - logx.max())
    return x / x.sum() * rho


# --------------------------------------------------------------------------
# M/M/c/K и вероятность успеха попытки P(W + S <= T), FIFO
# --------------------------------------------------------------------------


def mmck_state_probs(c: int, k_cap: int, lam: float, mu: float) -> np.ndarray:
    """Стационарное распределение числа в системе M/M/c/K (K — всего мест, K >= c).

    Здесь K — ёмкость очереди по MODEL_SPEC (мест в очереди), поэтому
    всего мест в системе N = c + K; параметр k_cap = N.
    """
    a = lam / mu
    n = np.arange(k_cap + 1)
    logp = np.where(
        n <= c,
        n * math.log(a) - gammaln(n + 1),
        c * math.log(a) - gammaln(c + 1) + (n - c) * math.log(a / c),
    )
    p = np.exp(logp - logp.max())
    return p / p.sum()


def _hypoexp_tail(rate1: float, rate2: float, t: float) -> float:
    """P(Exp(rate1) + Exp(rate2) > t)."""
    if abs(rate1 - rate2) < 1e-9 * max(rate1, rate2):
        r = 0.5 * (rate1 + rate2)
        return math.exp(-r * t) * (1.0 + r * t)
    return (rate2 * math.exp(-rate1 * t) - rate1 * math.exp(-rate2 * t)) / (rate2 - rate1)


def success_prob_infinite(c: int, lam: float, mu: float, t_out: float) -> float:
    """P(W + S <= T) для M/M/c с бесконечной очередью, FIFO, Exp-обслуживание.

    Вывод теоретика: W = 0 с вер. 1-C, иначе Exp(c mu - lam); S ~ Exp(mu).
    При lam >= c mu очередь нестабильна: возвращаем 0.
    """
    if lam >= c * mu:
        return 0.0
    cc = erlang_c(c, lam / mu)
    theta = c * mu - lam
    fail = (1.0 - cc) * math.exp(-mu * t_out) + cc * _hypoexp_tail(theta, mu, t_out)
    return 1.0 - fail


def _succ_given_n(c: int, mu: float, t_out: float, n_max_sys: int) -> np.ndarray:
    """s_n = P(W + S <= T | прибывшая принятая попытка застала n в системе).

    n < c: W = 0. n >= c: W ~ Erlang(j = n-c+1, c mu), S ~ Exp(mu).
    Вывод теоретика. Для c mu > mu:
      P(W+S<=T) = P(W<=T) - e^{-mu T} (r/(r-mu))^j * gammainc(j, (r-mu) T).
    """
    r = c * mu
    s = np.empty(n_max_sys)
    s[: min(c, n_max_sys)] = 1.0 - math.exp(-mu * t_out)
    for n in range(c, n_max_sys):
        j = n - c + 1
        if c == 1:
            s[n] = gammainc(j + 1, mu * t_out)
        else:
            pw = gammainc(j, r * t_out)
            corr = math.exp(-mu * t_out + j * math.log(r / (r - mu))) * gammainc(
                j, (r - mu) * t_out
            )
            s[n] = pw - corr
    return np.clip(s, 0.0, 1.0)


def success_prob_finite(c: int, k_queue: int, lam: float, mu: float, t_out: float) -> float:
    """Безусловная вероятность успеха попытки в M/M/c/(c+K), FIFO, D = off.

    Отклонённые (система полна) считаются неуспешными. Принятые обслуживаются
    полностью (в том числе осиротевшие). Точно при пуассоновском входе lam.
    """
    n_sys = c + k_queue
    pi = mmck_state_probs(c, n_sys, lam, mu)
    s = _succ_given_n(c, mu, t_out, n_sys)  # n = 0..n_sys-1 принимаются
    return float(np.dot(pi[:n_sys], s))


# --------------------------------------------------------------------------
# Mean-field для R1
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MFParams:
    c: int = 8
    k_queue: int | None = 128  # None: бесконечная очередь (вариант §8 спецификации)
    t_out: float = 10.0
    mu: float = 1.0
    m_retries: int = 3


def attempts_per_request(q: float, m: int) -> float:
    """sum_{k=0..m} q^k."""
    return sum(q**k for k in range(m + 1))


def mf_map(q: float, rho: float, p: MFParams) -> float:
    """F(q; rho): вероятность неуспеха попытки при вероятности неуспеха q."""
    lam = rho * p.c * p.mu
    lam_eff = lam * attempts_per_request(q, p.m_retries)
    if p.k_queue is None:
        s = success_prob_infinite(p.c, lam_eff, p.mu, p.t_out)
    else:
        s = success_prob_finite(p.c, p.k_queue, lam_eff, p.mu, p.t_out)
    return 1.0 - s


@dataclass(frozen=True)
class FixedPoint:
    q: float
    stable: bool
    slope: float


def fixed_points(
    rho: float, p: MFParams, n_grid: int = 2001, f: Callable[..., float] = mf_map
) -> list[FixedPoint]:
    """Все решения q = F(q; rho) на [0, 1]; устойчивость по F'(q*) < 1."""
    grid = np.linspace(0.0, 1.0, n_grid)
    g = np.array([f(float(x), rho, p) - x for x in grid])
    roots: list[float] = []
    for i in range(n_grid - 1):
        if g[i] == 0.0:
            roots.append(float(grid[i]))
        elif g[i] * g[i + 1] < 0.0:
            roots.append(
                brentq(lambda x: f(x, rho, p) - x, float(grid[i]), float(grid[i + 1]), xtol=1e-13)
            )
    if g[-1] == 0.0:
        roots.append(1.0)
    out: list[FixedPoint] = []
    h = 1e-6
    for r in roots:
        lo, hi = max(r - h, 0.0), min(r + h, 1.0)
        slope = (f(hi, rho, p) - f(lo, rho, p)) / (hi - lo)
        out.append(FixedPoint(r, bool(slope < 1.0), float(slope)))
    return out


def goodput_ratio(q: float, m: int) -> float:
    """Доля запросов, получивших успех за <= m+1 попыток: 1 - q^(m+1)."""
    return 1.0 - q ** (m + 1)


def branches(rhos: Sequence[float], p: MFParams) -> list[tuple[float, list[FixedPoint]]]:
    return [(float(r), fixed_points(float(r), p)) for r in rhos]


@dataclass(frozen=True)
class Spinodals:
    rho_fold: float | None  # верх устойчивой «здоровой» ветви (g >= theta)
    rho_release: float | None  # низ устойчивой «отказной» ветви (g < theta)


def spinodals(
    rhos: Sequence[float], p: MFParams, theta: float = 0.5
) -> tuple[Spinodals, list[tuple[float, list[FixedPoint]]]]:
    """Границы области бистабильности модели на заданной сетке rho.

    Здоровая ветвь: устойчивое решение с goodput >= theta; отказная: с < theta.
    rho_fold = макс rho со здоровой устойчивой; rho_release = мин rho с отказной.
    """
    br = branches(rhos, p)
    healthy = [
        r
        for r, fps in br
        if any(x.stable and goodput_ratio(x.q, p.m_retries) >= theta for x in fps)
    ]
    failed = [
        r for r, fps in br if any(x.stable and goodput_ratio(x.q, p.m_retries) < theta for x in fps)
    ]
    return (
        Spinodals(max(healthy) if healthy else None, min(failed) if failed else None),
        br,
    )


def refine_threshold(
    pred: Callable[[float], bool], lo: float, hi: float, tol: float = 1e-5
) -> float:
    """Бисекция границы: pred(lo) != pred(hi)."""
    plo = pred(lo)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if pred(mid) == plo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
