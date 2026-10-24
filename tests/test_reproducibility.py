"""V5: одинаковый seed даёт одинаковый результат, разные seed — разные; независимость потоков."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from src.config import Config, ProtoParams
from src.model import make_streams, simulate
from src.protocols import plan_for, run_protocol
from src.run import write_parquet


def _proto() -> ProtoParams:
    return ProtoParams(t_warm=40.0, trigger_len=20.0, h_rec=150.0, overload_len=60.0, w_last=60.0)


def _strip(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "runtime"}


def _same_arrays(a: Any, b: Any) -> bool:
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray):
            if x.tobytes() != y.tobytes():
                return False
        elif x != y:
            return False
    return True


def test_same_seed_byte_identical_arrays() -> None:
    cfg = Config(R="R3", Q="ALIFO", D="on", S="LogN", proto=_proto())
    plan = plan_for(cfg, "A", 0.85)
    a = simulate(cfg, plan.schedule, plan.t_stop, seed=123)
    b = simulate(cfg, plan.schedule, plan.t_stop, seed=123)
    assert _same_arrays(a, b)


def test_same_seed_same_row_and_parquet_bytes(tmp_path: Path) -> None:
    cfg = Config(proto=_proto())
    r1 = _strip(run_protocol(cfg, "B", 0.7, 42))
    r2 = _strip(run_protocol(cfg, "B", 0.7, 42))
    assert str(r1) == str(r2)  # str: NaN == NaN
    p1, p2 = tmp_path / "a.parquet", tmp_path / "b.parquet"
    write_parquet([r1], p1)
    write_parquet([r2], p2)
    assert p1.read_bytes() == p2.read_bytes()


def test_different_seeds_differ() -> None:
    cfg = Config(proto=_proto())
    plan = plan_for(cfg, "A", 0.85)
    a = simulate(cfg, plan.schedule, plan.t_stop, seed=1)
    b = simulate(cfg, plan.schedule, plan.t_stop, seed=2)
    assert a.req_t.tobytes() != b.req_t.tobytes()
    assert a.starts.tobytes() != b.starts.tobytes()


def test_streams_are_independent_and_deterministic() -> None:
    a1, s1, j1 = make_streams(7)
    _, s2, j2 = make_streams(7)
    a1.random(1000)  # потребление одного потока не сдвигает другие
    assert s1.random(5).tolist() == s2.random(5).tolist()
    assert j1.random(5).tolist() == j2.random(5).tolist()
    _, s3, j3 = make_streams(8)
    assert s3.random(5).tolist() != make_streams(7)[1].random(5).tolist()
    assert j3.random(5).tolist() != make_streams(7)[2].random(5).tolist()


def test_factor_change_does_not_shift_input_stream() -> None:
    base = Config(proto=_proto())
    plan = plan_for(base, "A", 0.8)
    ref = simulate(base, plan.schedule, plan.t_stop, seed=77)
    assert len(ref.req_t) > 50
    for kw in ({"Q": "LIFO"}, {"D": "on"}, {"S": "LogN"}, {"R": "R2"}, {"R": "R0"}):
        cfg = Config(proto=_proto(), **kw)  # type: ignore[arg-type]
        res = simulate(cfg, plan.schedule, plan.t_stop, seed=77)
        # первичные запросы (вход) идентичны при любом факторе
        assert np.array_equal(ref.req_t, res.req_t)


def test_service_times_are_tied_to_attempt_id() -> None:
    base = Config(proto=_proto())
    plan = plan_for(base, "A", 0.6)
    ref = simulate(base, plan.schedule, plan.t_stop, seed=5)
    ref_s = {int(a): s for a, _, s in ref.starts}
    n_checked = 0
    for kw in ({"Q": "LIFO"}, {"D": "on"}):
        cfg = Config(proto=_proto(), **kw)  # type: ignore[arg-type]
        res = simulate(cfg, plan.schedule, plan.t_stop, seed=5)
        for a, _, s in res.starts[:200]:
            if int(a) in ref_s:
                assert ref_s[int(a)] == s
                n_checked += 1
    assert n_checked > 100


def test_rho_change_uses_common_random_numbers() -> None:
    cfg = Config(proto=_proto())
    lo = simulate(cfg, [(100.0, 0.5)], 100.0, seed=3)
    hi = simulate(cfg, [(100.0, 0.9)], 100.0, seed=3)
    # замена времени потока единичной интенсивности: при большей нагрузке приходы раньше
    n = min(len(lo.req_t), len(hi.req_t))
    assert np.all(hi.req_t[:n] <= lo.req_t[:n] + 1e-12)
