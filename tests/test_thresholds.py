"""Оценка порогов: логистика, изотоника, цензурирование, bootstrap, Δ."""

from __future__ import annotations

import numpy as np
import pytest

from src.thresholds import aggregate, estimate_delta, estimate_threshold, pava


def _logistic_data(
    x0: float, width: float, reps: int, seed: int, grid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    g = np.arange(0.40, 1.0001, 0.05) if grid is None else grid
    rho = np.repeat(g, reps)
    p = 1.0 / (1.0 + np.exp(-(rho - x0) / width))
    return rho, (rng.random(len(rho)) < p).astype(int)


def test_pava() -> None:
    f = pava(np.array([0.0, 0.5, 0.2, 0.8, 1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    assert f.tolist() == pytest.approx([0.0, 0.35, 0.35, 0.8, 1.0])
    w = pava(np.array([0.9, 0.1]), np.array([1.0, 3.0]))
    assert w.tolist() == pytest.approx([0.3, 0.3])


def test_aggregate() -> None:
    x, n, k = aggregate(np.array([0.5, 0.5, 0.6, 0.6, 0.6]), np.array([0, 1, 1, 1, 0]))
    assert x.tolist() == [0.5, 0.6] and n.tolist() == [2, 3] and k.tolist() == [1, 2]


def test_logistic_recovers_known_threshold() -> None:
    rho, y = _logistic_data(0.70, 0.04, reps=200, seed=1)
    est = estimate_threshold(rho, y, n_boot=300, seed=2)
    assert est.method == "logistic" and est.censored == "none"
    assert est.value == pytest.approx(0.70, abs=0.015)
    assert est.lo < est.value < est.hi
    assert est.hi - est.lo < 0.06


def test_bootstrap_ci_covers_truth_and_scales_with_replicates() -> None:
    w = []
    for reps in (10, 40):
        rho, y = _logistic_data(0.65, 0.05, reps=reps, seed=7)
        est = estimate_threshold(rho, y, n_boot=500, seed=3)
        w.append(est.hi - est.lo)
        assert est.lo <= 0.65 <= est.hi
    assert w[1] < w[0]


def test_separation_uses_isotonic() -> None:
    grid = np.arange(0.40, 1.0001, 0.05)
    rho = np.repeat(grid, 10)
    y = (rho > 0.70 + 1e-9).astype(int)  # полное разделение: отказ при ρ >= 0.75
    est = estimate_threshold(rho, y, n_boot=200, seed=0)
    assert est.method == "isotonic"
    assert est.value == pytest.approx(0.725, abs=1e-9)  # f(0.70)=0, f(0.75)=1 -> середина
    assert est.lo <= est.value <= est.hi


def test_right_censoring_when_never_fails() -> None:
    rho = np.repeat(np.arange(0.40, 1.0001, 0.05), 10)
    est = estimate_threshold(rho, np.zeros(len(rho), dtype=int), n_boot=50)
    assert est.value == 1.0 and est.censored == "right"
    assert est.lo == est.hi == 1.0


def test_left_censoring_when_always_fails() -> None:
    rho = np.repeat(np.arange(0.40, 1.0001, 0.05), 10)
    est = estimate_threshold(rho, np.ones(len(rho), dtype=int), n_boot=50)
    assert est.value == pytest.approx(0.40) and est.censored == "left"


def test_right_censoring_when_threshold_above_grid() -> None:
    rho, y = _logistic_data(1.4, 0.1, reps=30, seed=5)
    est = estimate_threshold(rho, y, n_boot=100, seed=1)
    assert est.censored == "right" and est.value == 1.0


def test_delta_positive_and_zero_control() -> None:
    rho_a, y_a = _logistic_data(0.85, 0.03, reps=60, seed=11)
    rho_b, y_b = _logistic_data(0.40, 0.03, reps=60, seed=12)
    d = estimate_delta(rho_a, y_a, rho_b, y_b, n_boot=300, seed=4)
    assert d.delta == pytest.approx(0.45, abs=0.03)
    assert d.lo > 0.3
    assert d.fold.value > d.release.value
    # контроль: одинаковые профили A и B -> Δ ≈ 0, ДИ содержит 0
    rho_c, y_c = _logistic_data(0.70, 0.03, reps=60, seed=13)
    rho_d, y_d = _logistic_data(0.70, 0.03, reps=60, seed=14)
    d0 = estimate_delta(rho_c, y_c, rho_d, y_d, n_boot=300, seed=4)
    assert abs(d0.delta) < 0.03
    assert d0.lo <= 0.0 <= d0.hi


def test_deterministic_given_seed() -> None:
    rho, y = _logistic_data(0.7, 0.05, reps=20, seed=3)
    a = estimate_threshold(rho, y, n_boot=100, seed=5)
    b = estimate_threshold(rho, y, n_boot=100, seed=5)
    assert a == b


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        aggregate(np.array([0.5, 0.6]), np.array([0, 2]))
    with pytest.raises(ValueError):
        aggregate(np.array([0.5]), np.array([0, 1]))
