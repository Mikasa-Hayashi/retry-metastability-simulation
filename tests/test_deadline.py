"""Deadline drop (D = on) и срок попытки."""

from __future__ import annotations

import pytest

from src.config import Config
from src.metrics import wasted_fraction
from src.model import DROPPED, FAIL_TIMEOUT, LATE, OK, REJECTED, SUCCESS

from .helpers import det_run, start_order

ARR = [0.0, 1.0, 18.0]
SVC = [20.0, 1.0, 1.0]


def _cfg(d: str, q: str = "FIFO", k: int = 10) -> Config:
    return Config(c=1, K=k, T=5.0, R="R0", Q=q, D=d)  # type: ignore[arg-type]


def test_drop_on_skips_expired_entry() -> None:
    res = det_run(_cfg("on"), ARR, SVC)
    assert list(res.att_state) == [LATE, DROPPED, OK]
    assert res.n_dropped == 1
    assert start_order(res) == [0, 2]
    # обслуживание пропущенной попытки времени не тратит: job 2 стартует сразу в t = 20
    assert list(res.starts[:, 1]) == [0.0, 20.0]
    assert wasted_fraction(res, 0.0, 100.0) == pytest.approx(20.0 / 21.0)


def test_drop_off_serves_orphans() -> None:
    res = det_run(_cfg("off"), ARR, SVC)
    assert list(res.att_state) == [LATE, LATE, OK]
    assert res.n_dropped == 0
    assert start_order(res) == [0, 1, 2]
    assert list(res.starts[:, 1]) == [0.0, 20.0, 21.0]
    assert wasted_fraction(res, 0.0, 100.0) == pytest.approx(21.0 / 22.0)


def test_client_view_independent_of_drop() -> None:
    # клиентские исходы: job 0 и job 1 — таймаут, job 2 — успех (обслужен до t + T = 23)
    for d in ("on", "off"):
        res = det_run(_cfg(d), ARR, SVC)
        assert list(res.req_status) == [FAIL_TIMEOUT, FAIL_TIMEOUT, SUCCESS]
        assert list(res.req_res_t) == [5.0, 6.0, 21.0 if d == "on" else 22.0]


def test_drop_with_lifo_takes_newest_first() -> None:
    res = det_run(_cfg("on", q="LIFO"), ARR, SVC)
    assert start_order(res) == [0, 2]
    # LIFO: job 2 взят первым (жив), затем job 1 — просрочен, пропущен в t = 21
    assert list(res.att_state) == [LATE, DROPPED, OK]


def test_expired_entry_still_occupies_slot_until_dequeued() -> None:
    # D-005: просроченная запись пропускается только при извлечении воркером
    res = det_run(_cfg("on", k=1), ARR, SVC)
    assert res.att_state[2] == REJECTED


def test_attempt_not_expired_at_dequeue_is_served_fully() -> None:
    # запись жива в момент старта, но обслуживание выйдет за срок: обслуживается впустую
    cfg = Config(c=1, K=10, T=5.0, R="R0", Q="FIFO", D="on")
    res = det_run(cfg, [0.0, 1.0], [3.0, 10.0])
    assert list(res.att_state) == [OK, LATE]
    assert res.n_dropped == 0


def test_second_job_finishing_after_deadline_is_late() -> None:
    cfg = Config(c=1, K=10, T=5.0, R="R0", D="off")
    res = det_run(cfg, [0.0, 0.0], [4.999, 4.0])
    # вторая стартует в 4.999, завершается в 8.999 > 5 -> поздно
    assert list(res.att_state) == [OK, LATE]
