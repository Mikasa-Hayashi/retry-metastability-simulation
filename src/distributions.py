"""Распределения времени обслуживания со средним 1 (MODEL_SPEC §3, DECISIONS D-007)."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np


class ServiceDist(ABC):
    """Распределение времени обслуживания. Среднее всегда равно 1."""

    name: str

    @property
    def mean(self) -> float:
        return 1.0

    @property
    @abstractmethod
    def cv(self) -> float:
        """Теоретический коэффициент вариации (inf, если дисперсия бесконечна)."""

    @abstractmethod
    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """n независимых значений."""


class Exp(ServiceDist):
    name = "Exp"

    @property
    def cv(self) -> float:
        return 1.0

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.standard_exponential(n)


class LogN(ServiceDist):
    """Логнормальное со средним 1 и заданным cv: sigma^2 = ln(1+cv^2), mu = -sigma^2/2."""

    name = "LogN"

    def __init__(self, cv: float = 2.0) -> None:
        if cv <= 0:
            raise ValueError("cv > 0")
        self._cv = cv
        self.sigma2 = math.log(1.0 + cv * cv)
        self.mu = -0.5 * self.sigma2

    @property
    def cv(self) -> float:
        return self._cv

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.lognormal(self.mu, math.sqrt(self.sigma2), n)


class Pareto(ServiceDist):
    """Классическое Pareto (тип I) с индексом alpha > 1 и масштабом x_m = (alpha-1)/alpha.

    Среднее 1. Дисперсия конечна при alpha > 2: cv = 1/sqrt(alpha (alpha - 2)).
    """

    name = "Pareto"

    def __init__(self, alpha: float = 2.5) -> None:
        if alpha <= 1:
            raise ValueError("alpha > 1")
        self.alpha = alpha
        self.xm = (alpha - 1.0) / alpha

    @property
    def cv(self) -> float:
        if self.alpha <= 2:
            return math.inf
        return 1.0 / math.sqrt(self.alpha * (self.alpha - 2.0))

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        # rng.pareto — Lomax (сдвинутое Pareto с x_m = 1): X = x_m (1 + L)
        return self.xm * (1.0 + rng.pareto(self.alpha, n))


def make_service(name: str, logn_cv: float = 2.0, pareto_alpha: float = 2.5) -> ServiceDist:
    if name == "Exp":
        return Exp()
    if name == "LogN":
        return LogN(logn_cv)
    if name == "Pareto":
        return Pareto(pareto_alpha)
    raise ValueError(f"неизвестное распределение: {name}")
