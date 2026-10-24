"""V4: баланс запросов, уникальные id, неотрицательные времена, монотонность времени."""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any

import numpy as np
import pytest

from src.config import Config, ProtoParams
from src.metrics import check_v4
from src.model import SUCCESS, InvariantError, run_core, simulate
from src.protocols import plan_for, run_protocol


def small_proto() -> ProtoParams:
    return ProtoParams(t_warm=40.0, trigger_len=20.0, h_rec=150.0, overload_len=60.0, w_last=60.0)


@pytest.mark.parametrize(
    "r,q,d,s",
    list(
        itertools.product(
            ["R0", "R1", "R2", "R3"],
            ["FIFO", "LIFO", "ALIFO"],
            ["off", "on"],
            ["Exp", "LogN", "Pareto"],
        )
    ),
)
def test_v4_across_full_grid(r: str, q: str, d: str, s: str) -> None:
    cfg: Any = Config(R=r, Q=q, D=d, S=s, K=32, proto=small_proto())
    plan = plan_for(cfg, "A", 0.9)
    res = simulate(cfg, plan.schedule, plan.t_stop, seed=11)
    cnt = check_v4(res)
    assert cnt["n_primary"] == (
        cnt["n_success"] + cnt["n_fail_timeout"] + cnt["n_fail_reject"] + cnt["n_inflight"]
    )
    assert cnt["n_primary"] > 100
    assert res.samp_busy.max() <= cfg.c
    assert res.samp_q.max() <= cfg.K


def test_run_protocol_asserts_v4_and_reports_counters() -> None:
    cfg = Config(proto=small_proto())
    for proto in ("A", "B"):
        row = run_protocol(cfg, proto, 0.8, seed=3)  # type: ignore[arg-type]
        assert row["n_primary"] == (
            row["n_success"] + row["n_fail_timeout"] + row["n_fail_reject"] + row["n_inflight"]
        )


def test_inflight_only_near_end() -> None:
    cfg = Config(proto=small_proto())
    plan = plan_for(cfg, "A", 0.9)
    res = simulate(cfg, plan.schedule, plan.t_stop, seed=5)
    pend = res.req_status == 0
    assert res.req_t[pend].min() >= plan.t_end  # всё до конца наблюдения разрешено


def test_unique_ids_and_attempt_numbering() -> None:
    cfg = Config(R="R2", proto=small_proto())
    plan = plan_for(cfg, "B", 0.9)
    res = simulate(cfg, plan.schedule, plan.t_stop, seed=6)
    n_att = len(res.att_req)
    assert n_att == int(res.req_att.sum())
    assert np.all(res.req_att <= cfg.retry.max_retries + 1)
    # id попыток плотны (индекс массива); попытки запроса идут с номерами 0, 1, 2, ...
    order = np.lexsort((res.att_no, res.att_req))
    sorted_no = res.att_no[order]
    sorted_req = res.att_req[order]
    same_req = np.diff(sorted_req) == 0
    assert np.all(np.diff(sorted_no)[same_req] == 1)
    assert len(np.unique(res.starts[:, 0])) == len(res.starts)


def test_tampered_result_is_detected() -> None:
    cfg = Config(proto=small_proto())
    plan = plan_for(cfg, "A", 0.6)
    res = simulate(cfg, plan.schedule, plan.t_stop, seed=1)
    check_v4(res)
    st = res.req_status.copy()
    i = int(np.flatnonzero(st == SUCCESS)[0])
    st[i] = 0
    with pytest.raises(InvariantError):
        check_v4(dataclasses.replace(res, req_status=st))
    lat = res.req_lat.copy()
    lat[i] = -1.0
    with pytest.raises(InvariantError):
        check_v4(dataclasses.replace(res, req_lat=lat))
    with pytest.raises(InvariantError):
        check_v4(dataclasses.replace(res, n_rejected=res.n_rejected + 1))


def test_time_monotonicity_enforced() -> None:
    cfg = Config(c=1, K=5, T=10.0, R="R0")

    def ones(n: int) -> np.ndarray:
        return np.ones(n)

    with pytest.raises(InvariantError):
        run_core(cfg, np.array([5.0, 3.0]), 100.0, ones, ones)


def test_times_nonnegative_everywhere() -> None:
    cfg = Config(R="R2", Q="ALIFO", D="on", S="LogN", proto=small_proto())
    plan = plan_for(cfg, "B", 0.7)
    res = simulate(cfg, plan.schedule, plan.t_stop, seed=9)
    assert np.all(res.att_t >= 0.0)
    assert np.all(res.starts[:, 1] >= 0.0)
    assert np.all(np.diff(res.samp_t) > 0.0)
    done_ok = res.done[res.done[:, 3] == 0]
    assert np.all(done_ok[:, 1] <= res.att_dl[done_ok[:, 0].astype(int)])
