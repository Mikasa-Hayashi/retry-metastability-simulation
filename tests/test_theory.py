"""Тесты analysis/theory.py: известные значения и независимые численные проверки."""

from __future__ import annotations

import heapq
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import theory as th
import theory_check_sim as ts

# ---------------- M/M/1, M/M/c ----------------


def _erlang_c_direct(c: int, a: float) -> float:
    """Прямая формула (Wikipedia, M/M/c queue) без рекурсии Эрланга B."""
    rho = a / c
    top = a**c / math.factorial(c) / (1 - rho)
    bottom = sum(a**k / math.factorial(k) for k in range(c)) + top
    return top / bottom


@pytest.mark.parametrize("c", [1, 2, 8, 20])
@pytest.mark.parametrize("rho", [0.3, 0.5, 0.7, 0.9, 0.95])
def test_erlang_c_matches_direct_formula(c: int, rho: float) -> None:
    assert th.erlang_c(c, c * rho) == pytest.approx(_erlang_c_direct(c, c * rho), rel=1e-10)


def test_erlang_c_known_values() -> None:
    assert th.erlang_c(1, 0.7) == pytest.approx(0.7)  # M/M/1: C = rho
    assert th.erlang_c(2, 1.0) == pytest.approx(1.0 / 3.0)  # a^2/2/(1-.5) / (1+1+1) = 1/3


def test_mm1_known_values() -> None:
    assert th.mm1_mean_w(0.9, 1.0) == pytest.approx(10.0)
    assert th.mm1_mean_wq(0.9, 1.0) == pytest.approx(9.0)
    assert th.mmc_mean_wq(1, 0.5, 1.0) == pytest.approx(th.mm1_mean_wq(0.5, 1.0))
    assert th.mmc_mean_w(1, 0.7, 1.0) == pytest.approx(th.mm1_mean_w(0.7, 1.0))


@pytest.mark.parametrize("c,rho", [(1, 0.5), (8, 0.7), (8, 0.9)])
def test_wq_from_birth_death_and_tail_integral(c: int, rho: float) -> None:
    lam, mu = rho * c, 1.0
    n_sys = 3000
    pi = th.mmck_state_probs(c, n_sys, lam, mu)
    lq = float(np.sum(np.maximum(np.arange(n_sys + 1) - c, 0) * pi))
    assert th.mmc_mean_wq(c, lam, mu) == pytest.approx(lq / lam, rel=1e-6)  # Little
    tail_int, _ = integrate.quad(lambda t: th.mmc_wq_tail(c, lam, mu, t), 0, np.inf)
    assert tail_int == pytest.approx(th.mmc_mean_wq(c, lam, mu), rel=1e-7)
    assert th.mmc_wq_tail(c, lam, mu, 0.0) == pytest.approx(th.erlang_c(c, lam / mu))


# ---------------- LIFO / FIFO ----------------


def test_finch_md1_numbers() -> None:
    """Finch arXiv:2208.09980: M/D/1, lam=2, mu=3: mean 1/3, var LIFO 7/9, FIFO 5/27."""
    lam, a = 2.0, 1.0 / 3.0
    args = (lam, a**2, a**3, a)
    lifo, fifo = th.mg1_wait_moments_lifo(*args), th.mg1_wait_moments_fifo(*args)
    assert lifo.mean == pytest.approx(1 / 3)
    assert fifo.mean == pytest.approx(1 / 3)
    assert lifo.var == pytest.approx(7 / 9)
    assert fifo.var == pytest.approx(5 / 27)


def test_finch_mm1_numbers() -> None:
    """Finch: M/M/1 lam=2, mu=3: mean 2/3, var LIFO 32/9, FIFO 8/9."""
    lifo, fifo = th.mm1_wait_moments(2.0, 3.0, True), th.mm1_wait_moments(2.0, 3.0, False)
    assert lifo.mean == pytest.approx(2 / 3)
    assert fifo.mean == pytest.approx(2 / 3)
    assert lifo.var == pytest.approx(32 / 9)
    assert fifo.var == pytest.approx(8 / 9)


def _mm1_lifo_waits(lam: float, mu: float, n: int, seed: int, lifo: bool) -> np.ndarray:
    """M/M/1, невытесняющая дисциплина: LIFO (стек) или FIFO (куча по номеру)."""
    rng = np.random.default_rng(seed)
    arr = np.cumsum(rng.exponential(1 / lam, n))
    svc = rng.exponential(1 / mu, n)
    waiting: list[int] = []
    waits = np.empty(n)
    t_free, i, served = 0.0, 0, 0
    while served < n:
        if not waiting:
            t_free = max(t_free, arr[i])
        while i < n and arr[i] <= t_free:
            if lifo:
                waiting.append(i)
            else:
                heapq.heappush(waiting, i)
            i += 1
        j = waiting.pop() if lifo else heapq.heappop(waiting)
        waits[j] = t_free - arr[j]
        t_free += svc[j]
        served += 1
    return waits


def test_lifo_equals_fifo_mean_larger_variance_mm1() -> None:
    lam, mu = 0.5, 1.0
    n = 300_000
    w_f = _mm1_lifo_waits(lam, mu, n, 1, lifo=False)
    w_l = _mm1_lifo_waits(lam, mu, n, 2, lifo=True)
    th_f, th_l = th.mm1_wait_moments(lam, mu, False), th.mm1_wait_moments(lam, mu, True)
    assert w_f.mean() == pytest.approx(th_f.mean, rel=0.04)
    assert w_l.mean() == pytest.approx(th_l.mean, rel=0.04)
    assert w_l.var() > w_f.var()
    assert w_f.var() == pytest.approx(th_f.var, rel=0.08)
    assert w_l.var() == pytest.approx(th_l.var, rel=0.15)


# ---------------- Классическая система с повторными вызовами ----------------


@pytest.mark.parametrize("lam,mu,nu", [(0.5, 1.0, 1.0), (0.7, 1.0, 0.5), (0.9, 1.0, 2.0)])
def test_retrial_closed_form_vs_balance_equations(lam: float, mu: float, nu: float) -> None:
    busy, mean_n = th.retrial_numeric(lam, mu, nu, n_max=1200 if lam > 0.8 else 500)
    assert busy == pytest.approx(th.retrial_busy_prob(lam, mu), rel=1e-6)
    assert mean_n == pytest.approx(th.retrial_mean_orbit(lam, mu, nu), rel=1e-5)


def test_retrial_nu_to_infinity_is_mm1_queue() -> None:
    lam, mu = 0.6, 1.0
    rho = lam / mu
    assert th.retrial_mean_orbit(lam, mu, 1e9) == pytest.approx(rho**2 / (1 - rho), rel=1e-6)


def test_retrial_falin_product_form_matches_numerics() -> None:
    lam, mu, nu = 0.7, 1.0, 0.8
    n_max = 400
    # P(n, занят) из уравнений равновесия
    size = 2 * (n_max + 1)
    q = np.zeros((size, size))
    for n in range(n_max + 1):
        q[2 * n, 2 * n + 1] += lam
        if n:
            q[2 * n, 2 * (n - 1) + 1] += n * nu
        if n < n_max:
            q[2 * n + 1, 2 * (n + 1) + 1] += lam
        q[2 * n + 1, 2 * n] += mu
    np.fill_diagonal(q, -q.sum(axis=1))
    a = np.vstack([q.T, np.ones(size)])
    b = np.zeros(size + 1)
    b[-1] = 1
    pi = np.linalg.lstsq(a, b, rcond=None)[0]
    falin = th.retrial_distribution_falin(lam, mu, nu, n_max)
    np.testing.assert_allclose(pi[1::2][:60], falin[:60], rtol=1e-6, atol=1e-12)


# ---------------- M/M/c/K и вероятность успеха ----------------


def test_success_finite_tends_to_infinite() -> None:
    for rho in (0.5, 0.9):
        s_inf = th.success_prob_infinite(8, rho * 8, 1.0, 10.0)
        s_fin = th.success_prob_finite(8, 600, rho * 8, 1.0, 10.0)
        assert s_fin == pytest.approx(s_inf, abs=1e-9)


def test_success_single_server_matches_integration() -> None:
    """c=1: W+S ~ mixture; сверка с численным интегрированием плотности."""
    lam, mu, t_out = 0.8, 1.0, 5.0
    theta = mu - lam
    c0 = lam / mu

    def dens_w(w: float) -> float:
        return c0 * theta * math.exp(-theta * w)

    p_zero = 1 - c0
    direct = p_zero * (1 - math.exp(-mu * t_out))
    direct += integrate.quad(lambda w: dens_w(w) * (1 - math.exp(-mu * (t_out - w))), 0, t_out)[0]
    assert th.success_prob_infinite(1, lam, mu, t_out) == pytest.approx(direct, rel=1e-9)
    assert th.success_prob_finite(1, 2000, lam, mu, t_out) == pytest.approx(direct, rel=1e-6)


def test_success_prob_vs_independent_simulation() -> None:
    """Пуассоновский вход без повторов, rho=0.97, K=128: формула vs прямая симуляция."""
    p = ts.SimParams(m_retries=0)
    fails = []
    for seed in range(3):
        tm, ok, _ = ts.simulate([(12000.0, 0.97)], p, seed)
        fails.append(1 - ok[tm > 500].mean())
    theory_fail = 1 - th.success_prob_finite(8, 128, 0.97 * 8, 1.0, 10.0)
    assert np.mean(fails) == pytest.approx(theory_fail, abs=0.025)


# ---------------- Mean-field R1 ----------------


def test_fixed_points_structure_base_parameters() -> None:
    p = th.MFParams()
    low = th.fixed_points(0.20, p, n_grid=801)
    assert len(low) == 1 and low[0].stable and low[0].q < 1e-3
    mid = th.fixed_points(0.50, p, n_grid=801)
    assert [x.stable for x in mid] == [True, False, True]
    for x in mid:  # решение действительно неподвижная точка
        assert th.mf_map(x.q, 0.50, p) == pytest.approx(x.q, abs=1e-9)
    assert th.fixed_points(1.0, p, n_grid=801)[-1].q > 0.99


def test_spinodals_base_parameters() -> None:
    p = th.MFParams()

    def has_h(r: float) -> bool:
        return any(x.stable and th.goodput_ratio(x.q, 3) >= 0.5 for x in th.fixed_points(r, p, 801))

    def has_f(r: float) -> bool:
        return any(x.stable and th.goodput_ratio(x.q, 3) < 0.5 for x in th.fixed_points(r, p, 801))

    fold = th.refine_threshold(has_h, 0.85, 1.0, 1e-3)
    rel = th.refine_threshold(has_f, 0.2, 0.4, 1e-3)
    assert 0.92 < fold < 0.935
    assert 0.27 < rel < 0.285


def test_release_boundary_is_one_over_attempts_for_infinite_queue() -> None:
    """При q = 1 нагрузка (m+1) rho; отказная ветвь исчезает при rho = 1/(m+1)."""
    p = th.MFParams(k_queue=None)
    assert th.fixed_points(0.24, p, 801)[-1].q < 0.01  # только здоровая
    assert any(x.q > 0.99 and x.stable for x in th.fixed_points(0.30, p, 801))
