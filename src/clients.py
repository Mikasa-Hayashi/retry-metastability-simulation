"""Клиентские политики повторов R0–R3 (+ORBIT для V3) и бюджет повторов (D-006, D-007)."""

from __future__ import annotations

import math
from collections import deque

from .config import Config


class RetryPolicy:
    """Расписание повторов. `delay(n, u)`: задержка n-го повтора (n >= 1), u ~ U[0,1)."""

    max_retries: int | None = 0  # None — не ограничено
    uses_jitter: bool = False

    def delay(self, retry_no: int, u: float) -> float:
        raise NotImplementedError

    def max_total_delay(self) -> float:
        """Верхняя оценка суммы задержек всех повторов (для длины хвоста наблюдения)."""
        return 0.0


class NoRetry(RetryPolicy):
    """R0."""

    max_retries = 0

    def delay(self, retry_no: int, u: float) -> float:
        raise RuntimeError("R0: повторов нет")


class FixedDelay(RetryPolicy):
    """R1: до max_retries повторов, постоянная задержка d."""

    def __init__(self, max_retries: int, d: float) -> None:
        self.max_retries = max_retries
        self.d = d

    def delay(self, retry_no: int, u: float) -> float:
        return self.d

    def max_total_delay(self) -> float:
        return float(self.max_retries) * self.d


class ExpBackoffFullJitter(RetryPolicy):
    """R2: задержка n-го повтора = u * min(cap, base * mult^(n-1))."""

    uses_jitter = True

    def __init__(self, max_retries: int, base: float, multiplier: float, cap: float) -> None:
        self.max_retries = max_retries
        self.base = base
        self.multiplier = multiplier
        self.cap = cap

    def upper(self, retry_no: int) -> float:
        return min(self.cap, self.base * self.multiplier ** (retry_no - 1))

    def delay(self, retry_no: int, u: float) -> float:
        return u * self.upper(retry_no)

    def max_total_delay(self) -> float:
        return sum(self.upper(n) for n in range(1, self.max_retries + 1))


class RetrialOrbit(RetryPolicy):
    """ORBIT (V3): неограниченные повторы, задержка Exp(nu)."""

    max_retries = None
    uses_jitter = True

    def __init__(self, nu: float) -> None:
        self.nu = nu

    def delay(self, retry_no: int, u: float) -> float:
        return -math.log1p(-u) / self.nu

    def max_total_delay(self) -> float:
        return math.inf


class RetryBudget:
    """Скользящее окно (D-006): повтор разрешён, если
    n_retries(t-W, t] + 1 <= reserve + ratio * n_primary(t-W, t]."""

    def __init__(self, ratio: float, window: float, reserve: float) -> None:
        self.ratio = ratio
        self.window = window
        self.reserve = reserve
        self._primary: deque[float] = deque()
        self._retries: deque[float] = deque()
        self.granted = 0
        self.denied = 0

    def on_primary(self, t: float) -> None:
        self._primary.append(t)

    def _expire(self, t: float) -> None:
        lo = t - self.window
        p, r = self._primary, self._retries
        while p and p[0] <= lo:
            p.popleft()
        while r and r[0] <= lo:
            r.popleft()

    def acquire(self, t: float) -> bool:
        self._expire(t)
        if len(self._retries) + 1 <= self.reserve + self.ratio * len(self._primary) + 1e-9:
            self._retries.append(t)
            self.granted += 1
            return True
        self.denied += 1
        return False


def make_retry(cfg: Config) -> tuple[RetryPolicy, RetryBudget | None]:
    rp = cfg.retry
    if cfg.R == "R0":
        return NoRetry(), None
    if cfg.R == "R1":
        return FixedDelay(rp.max_retries, rp.fixed_delay), None
    if cfg.R in ("R2", "R3"):
        pol = ExpBackoffFullJitter(rp.max_retries, rp.base, rp.multiplier, rp.cap)
        if cfg.R == "R3":
            return pol, RetryBudget(rp.budget_ratio, rp.budget_window, rp.budget_reserve)
        return pol, None
    if cfg.R == "ORBIT":
        return RetrialOrbit(rp.orbit_rate), None
    raise ValueError(f"неизвестная политика {cfg.R}")
