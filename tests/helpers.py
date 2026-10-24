"""Вспомогательные функции тестов: детерминированные сценарии для run_core."""

from __future__ import annotations

import numpy as np

from src.config import Config
from src.model import SimResult, run_core


def det_run(
    cfg: Config,
    arrivals: list[float],
    service: list[float],
    t_stop: float = 1000.0,
    jitter: list[float] | None = None,
) -> SimResult:
    """Прогон с заданными приходами и временами обслуживания (по номеру попытки)."""
    chunk = 8192

    def draw_service(n: int) -> np.ndarray:
        out = np.ones(n)
        out[: len(service)] = service
        return out

    def draw_uniform(n: int) -> np.ndarray:
        out = np.full(n, 0.5)
        if jitter is not None:
            out[: len(jitter)] = jitter
        return out

    assert chunk >= len(service)
    return run_core(cfg, np.asarray(arrivals, dtype=float), t_stop, draw_service, draw_uniform)


def start_order(res: SimResult) -> list[int]:
    return [int(x) for x in res.starts[:, 0]]
