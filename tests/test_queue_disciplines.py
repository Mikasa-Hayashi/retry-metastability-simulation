"""Порядок выбора из очереди FIFO / LIFO / ALIFO на детерминированных сценариях."""

from __future__ import annotations

from src.config import Config
from src.model import OK, REJECTED

from .helpers import det_run, start_order


def _cfg(q: str, k: int = 6) -> Config:
    # c = 1, повторов нет, клиент не уходит (T = inf)
    return Config(c=1, K=k, T=float("inf"), R="R0", Q=q, D="off")  # type: ignore[arg-type]


ARR = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
SVC = [100.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_fifo_order() -> None:
    res = det_run(_cfg("FIFO"), ARR, SVC)
    assert start_order(res) == [0, 1, 2, 3, 4, 5]


def test_lifo_order() -> None:
    res = det_run(_cfg("LIFO"), ARR, SVC)
    assert start_order(res) == [0, 5, 4, 3, 2, 1]


def test_alifo_order_threshold() -> None:
    # K = 6, порог 0.5*K = 3. В момент освобождения воркера в очереди 5 записей:
    # 5 > 3 -> LIFO (5); 4 > 3 -> LIFO (4); 3 -> не > 3 -> FIFO (1), затем 2, 3.
    res = det_run(_cfg("ALIFO"), ARR, SVC)
    assert start_order(res) == [0, 5, 4, 1, 2, 3]


def test_alifo_equals_fifo_when_queue_short() -> None:
    res = det_run(_cfg("ALIFO", k=100), ARR, SVC)  # порог 50 никогда не достигается
    assert start_order(res) == [0, 1, 2, 3, 4, 5]


def test_alifo_equals_lifo_when_threshold_tiny() -> None:
    cfg = Config(c=1, K=6, T=float("inf"), R="R0", Q="ALIFO", D="off", theta_q=1e-9)
    res = det_run(cfg, ARR, SVC)
    assert start_order(res) == [0, 5, 4, 3, 2, 1]


def test_queue_capacity_and_rejection() -> None:
    # K = 2: приходы 1,2 встают в очередь, 3,4,5 отклоняются сразу
    res = det_run(_cfg("FIFO", k=2), ARR, SVC)
    assert list(res.att_state[:6]) == [OK, OK, OK, REJECTED, REJECTED, REJECTED]
    assert res.n_rejected == 3
    assert start_order(res) == [0, 1, 2]


def test_work_conserving_no_idle_with_queue() -> None:
    res = det_run(_cfg("FIFO"), ARR, SVC)
    starts = res.starts
    # каждое следующее обслуживание начинается сразу в момент завершения предыдущего
    assert starts[1, 1] == 100.0
    assert list(starts[1:, 1]) == [100.0, 101.0, 102.0, 103.0, 104.0]
