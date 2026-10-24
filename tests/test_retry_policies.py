"""Политики повторов: расписание задержек, лимит повторов, бюджет."""

from __future__ import annotations

import math

import pytest

from src.clients import (
    ExpBackoffFullJitter,
    FixedDelay,
    NoRetry,
    RetrialOrbit,
    RetryBudget,
    make_retry,
)
from src.config import Config
from src.model import FAIL_REJECT, FAIL_TIMEOUT, SUCCESS

from .helpers import det_run


def test_factory_types() -> None:
    assert isinstance(make_retry(Config(R="R0"))[0], NoRetry)
    assert isinstance(make_retry(Config(R="R1"))[0], FixedDelay)
    p2, b2 = make_retry(Config(R="R2"))
    assert isinstance(p2, ExpBackoffFullJitter) and b2 is None
    p3, b3 = make_retry(Config(R="R3"))
    assert isinstance(p3, ExpBackoffFullJitter) and isinstance(b3, RetryBudget)
    assert isinstance(make_retry(Config(R="ORBIT", T=math.inf))[0], RetrialOrbit)


def test_r1_fixed_schedule() -> None:
    pol = FixedDelay(3, 1.0)
    assert [pol.delay(n, 0.37) for n in (1, 2, 3)] == [1.0, 1.0, 1.0]
    assert pol.max_retries == 3


def test_r2_full_jitter_bounds() -> None:
    pol = ExpBackoffFullJitter(3, 1.0, 2.0, 16.0)
    assert [pol.upper(n) for n in (1, 2, 3)] == [1.0, 2.0, 4.0]
    assert [pol.delay(n, 1.0) for n in (1, 2, 3)] == [1.0, 2.0, 4.0]
    assert [pol.delay(n, 0.5) for n in (1, 2, 3)] == [0.5, 1.0, 2.0]
    assert pol.delay(2, 0.0) == 0.0
    assert pol.max_total_delay() == 7.0


def test_r2_cap() -> None:
    pol = ExpBackoffFullJitter(8, 1.0, 2.0, 16.0)
    assert [pol.upper(n) for n in range(1, 8)] == [1, 2, 4, 8, 16, 16, 16]


def test_r0_no_retry_error() -> None:
    with pytest.raises(RuntimeError):
        NoRetry().delay(1, 0.5)


def test_retry_limit_and_fixed_times_in_simulation() -> None:
    # c = 1, K = 0: сервер занят первой заявкой (S = 1000), вторая отклоняется и
    # повторяется в t = 1, 2, 3, 4 (первичная + 3 повтора с d = 1) и сдаётся.
    cfg = Config(c=1, K=0, T=math.inf, R="R1", Q="FIFO", D="off")
    res = det_run(cfg, [0.0, 1.0], [1000.0], t_stop=100.0)
    assert list(res.att_req) == [0, 1, 1, 1, 1]
    assert list(res.att_no) == [0, 0, 1, 2, 3]
    assert list(res.att_t) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert res.req_status[1] == FAIL_REJECT
    assert res.req_res_t[1] == 4.0
    assert res.req_att[1] == 4


def test_retry_delay_counted_from_timeout() -> None:
    # c = 1, K = 5, T = 5: запрос 1 ждёт в очереди; таймаут в t = 1 + 5 = 6, повтор в 7
    cfg = Config(c=1, K=50, T=5.0, R="R1", Q="FIFO", D="off")
    res = det_run(cfg, [0.0, 1.0], [1000.0], t_stop=100.0)
    retry_times = res.att_t[(res.att_req == 1) & (res.att_no > 0)]
    assert retry_times[0] == 7.0
    assert retry_times[1] == 13.0  # таймаут 12 + 1
    assert res.req_status[1] == FAIL_TIMEOUT
    assert res.req_res_t[1] == 4 * 5.0 + 3 * 1.0 + 1.0  # 4-я попытка в 19, таймаут в 24


def test_r2_uses_jitter_stream() -> None:
    cfg = Config(c=1, K=0, T=10.0, R="R2", Q="FIFO", D="off")
    # id попытки 1 (первая попытка запроса 1) берёт jitter[1]: задержка 1-го повтора = u * 1
    res = det_run(cfg, [0.0, 1.0], [1000.0], t_stop=100.0, jitter=[0.0, 0.5, 0.25, 0.75, 0.5])
    t = res.att_t[res.att_req == 1]
    assert t[0] == 1.0
    assert t[1] == pytest.approx(1.0 + 0.5 * 1.0)  # u=jitter[1]=0.5, верх 1
    assert t[2] == pytest.approx(t[1] + 0.25 * 2.0)  # попытка id 2: u=0.25, верх 2
    assert t[3] == pytest.approx(t[2] + 0.75 * 4.0)  # id 3: u=0.75, верх 4


def test_success_cancels_retries() -> None:
    cfg = Config(c=1, K=5, T=5.0, R="R1", Q="FIFO", D="off")
    res = det_run(cfg, [0.0], [1.0], t_stop=100.0)
    assert res.req_status[0] == SUCCESS
    assert res.req_att[0] == 1


# ---------------- бюджет ----------------


def test_budget_reserve_only() -> None:
    b = RetryBudget(0.1, 100.0, 10.0)
    granted = sum(b.acquire(1.0) for _ in range(30))
    assert granted == 10
    assert b.denied == 20


def test_budget_ratio_added_by_primaries() -> None:
    b = RetryBudget(0.1, 100.0, 10.0)
    for i in range(100):
        b.on_primary(float(i) * 0.5)  # 100 первичных за 50 единиц
    granted = sum(b.acquire(50.0) for _ in range(40))
    assert granted == 20  # 10 + 0.1 * 100


def test_budget_window_expiry() -> None:
    b = RetryBudget(0.1, 100.0, 10.0)
    assert sum(b.acquire(0.0) for _ in range(10)) == 10
    assert not b.acquire(99.0)
    assert b.acquire(100.5)  # старые выпали из окна (t - 100, t]


def test_budget_fraction_cap_in_simulation() -> None:
    # перегруженная система: все попытки отклоняются. Повторов не больше 10 + 0.1 * первичных.
    cfg = Config(c=1, K=0, T=10.0, R="R3", Q="FIFO", D="off")
    arrivals = [1.0 + 0.1 * i for i in range(400)]  # 400 первичных за 40 единиц
    res = det_run(cfg, [0.0, *arrivals], [1e6], t_stop=1000.0)
    n_retries = int((res.att_no > 0).sum())
    assert n_retries == res.budget_granted
    assert res.budget_denied > 0
    assert n_retries <= 10 + 0.1 * (len(arrivals) + 1) + 1e-9


def test_r3_without_budget_pressure_equals_r2_attempts() -> None:
    cfg2 = Config(c=1, K=0, T=10.0, R="R2")
    cfg3 = Config(c=1, K=0, T=10.0, R="R3")
    a = det_run(cfg2, [0.0, 1.0], [1000.0], t_stop=100.0)
    b = det_run(cfg3, [0.0, 1.0], [1000.0], t_stop=100.0)
    assert list(a.att_t) == list(b.att_t)
    assert b.budget_denied == 0
