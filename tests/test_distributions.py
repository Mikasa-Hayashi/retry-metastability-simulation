"""Распределения времени обслуживания: среднее и cv на большой выборке."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.distributions import Exp, LogN, Pareto, make_service

N = 4_000_000


def _stats(x: np.ndarray) -> tuple[float, float]:
    return float(x.mean()), float(x.std() / x.mean())


def test_exp_mean_cv() -> None:
    x = Exp().sample(np.random.default_rng(1), N)
    m, cv = _stats(x)
    assert m == pytest.approx(1.0, rel=0.005)
    assert cv == pytest.approx(1.0, rel=0.01)


@pytest.mark.parametrize("cv_target", [0.5, 1.0, 2.0])
def test_lognormal_mean_cv(cv_target: float) -> None:
    d = LogN(cv_target)
    assert d.cv == cv_target
    x = d.sample(np.random.default_rng(2), N)
    m, cv = _stats(x)
    assert m == pytest.approx(1.0, rel=0.02)
    assert cv == pytest.approx(cv_target, rel=0.06)


def test_lognormal_theoretical_moments() -> None:
    d = LogN(2.0)
    # E[X] = exp(mu + s2/2) = 1, Var = (exp(s2) - 1) exp(2 mu + s2) = exp(s2) - 1 = cv^2
    assert math.exp(d.mu + d.sigma2 / 2) == pytest.approx(1.0)
    assert math.exp(d.sigma2) - 1.0 == pytest.approx(4.0)


def test_pareto_mean_cv_and_tail() -> None:
    alpha = 2.5
    d = Pareto(alpha)
    assert d.xm == pytest.approx(0.6)
    assert d.cv == pytest.approx(1.0 / math.sqrt(alpha * (alpha - 2.0)))
    x = d.sample(np.random.default_rng(3), 8_000_000)
    assert x.min() >= d.xm
    assert x.mean() == pytest.approx(1.0, rel=0.01)
    # 4-й момент бесконечен: выборочный cv сходится медленно, допуск шире
    assert x.std() / x.mean() == pytest.approx(d.cv, rel=0.12)
    # хвост P(X > x) = (xm / x)^alpha проверяется точно
    for thr in (2.0, 5.0, 10.0):
        emp = float(np.mean(x > thr))
        assert emp == pytest.approx((d.xm / thr) ** alpha, rel=0.08)


def test_pareto_infinite_variance_flag() -> None:
    assert Pareto(1.8).cv == math.inf
    with pytest.raises(ValueError):
        Pareto(1.0)


def test_factory() -> None:
    assert make_service("Exp").cv == 1.0
    assert make_service("LogN", logn_cv=2.0).cv == 2.0
    assert isinstance(make_service("Pareto", pareto_alpha=2.5), Pareto)
    with pytest.raises(ValueError):
        make_service("Weibull")
