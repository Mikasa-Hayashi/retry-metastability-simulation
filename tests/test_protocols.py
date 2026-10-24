"""Протоколы A/B: профили нагрузки, структура строки результата, метрики, CLI, конфиги."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import metrics
from src.config import Config, ProtoParams, config_hash, load_config
from src.model import SUCCESS, simulate
from src.protocols import plan_for, run_protocol, tail_length, trigger_schedule
from src.run import main

ROOT = Path(__file__).resolve().parents[1]

ROW_KEYS = {
    "config_hash", "R", "Q", "D", "S", "rho_test", "protocol", "seed", "failed", "g_final",
    "retries_per_request", "wasted_fraction", "recovery_time", "p99", "runtime", "git_commit",
}  # fmt: skip


def test_plan_protocol_a() -> None:
    plan = plan_for(Config(), "A", 0.8)
    assert plan.schedule == [(200.0, 0.8), (250.0, pytest.approx(3.2)), (1750.0, 0.8)]
    assert plan.t_ref == 250.0
    assert plan.t_end == 1750.0


def test_plan_protocol_b() -> None:
    plan = plan_for(Config(), "B", 0.7)
    assert plan.schedule == [(400.0, 1.05), (1900.0, 0.7)]
    assert plan.t_ref == 400.0
    assert plan.t_end == 1900.0  # горизонт ≈ 1900 (спецификация)


def test_trigger_is_config_parameter() -> None:
    proto = ProtoParams(trigger_mult=2.5, trigger_len=30.0)
    plan = plan_for(Config(proto=proto), "A", 0.4)
    assert plan.schedule[1] == (230.0, pytest.approx(1.0))
    assert trigger_schedule(0.5, 10.0, 2.0, 5.0, 100.0) == [(10.0, 0.5), (15.0, 1.0), (100.0, 0.5)]


def test_tail_lengths_cover_resolution() -> None:
    assert tail_length(Config(R="R0")) == 11.0
    assert tail_length(Config(R="R1")) == 44.0
    assert tail_length(Config(R="R2")) == 48.0
    assert tail_length(Config(proto=ProtoParams(tail=77.0))) == 77.0


@pytest.mark.parametrize("protocol", ["A", "B"])
@pytest.mark.parametrize("r", ["R0", "R1", "R2", "R3"])
def test_protocols_run_and_structure(protocol: str, r: str) -> None:
    cfg = Config(R=r, proto=ProtoParams(h_rec=400.0, w_last=100.0))  # type: ignore[arg-type]
    row = run_protocol(cfg, protocol, 0.6, seed=4)  # type: ignore[arg-type]
    assert ROW_KEYS <= set(row)
    assert row["failed"] in (0, 1)
    assert 0.0 <= row["g_final"] <= 1.0
    assert row["failed"] == int(row["g_final"] < 0.5)
    assert row["retries_per_request"] >= 0.0
    assert row["retries_per_request"] <= (cfg.retry.max_retries if r != "R0" else 0)
    assert 0.0 <= row["wasted_fraction"] <= 1.0
    assert row["recovery_time"] >= 0.0
    assert row["runtime"] > 0.0
    assert row["protocol"] == protocol
    assert row["seed"] == 4
    assert row["config_hash"] == config_hash(cfg)


def test_a_low_load_r0_recovers() -> None:
    row = run_protocol(Config(R="R0"), "A", 0.4, seed=1)
    assert row["failed"] == 0 and row["g_final"] > 0.95
    assert row["recovery_censored"] == 0


def test_b_overload_state_is_failed_for_r0() -> None:
    # ρ_test = 1.05 равен фазе перегрузки: система остаётся в отказе (g << θ)
    row = run_protocol(Config(R="R0"), "B", 1.05, seed=1)
    assert row["failed"] == 1


def test_goodput_series_and_recovery_synthetic() -> None:
    cfg = Config(proto=ProtoParams(h_rec=100.0))
    res = simulate(cfg, [(10.0, 0.3)], 10.0, seed=1)
    t = np.arange(0.0, 400.0, 0.5)  # 2 запроса на единицу времени
    status = np.where(t < 150.0, 2, 1).astype(np.int8)  # неудачи до 150, затем успехи
    fake = dataclasses.replace(res, req_t=t, req_status=status)
    grid = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    g = metrics.goodput_series(fake, 100.0, grid)
    assert g[0] == 0.0 and g[1] == 0.0
    assert g[2] == pytest.approx(0.5)  # окно [100, 200): 50 единиц неудач, 50 успехов
    assert g[3] == 1.0 and g[4] == 1.0
    # окно [s-100, s) содержит неудачи до t = 150: доля неудач (250 - s)/100 <= 0.1 при s >= 240
    rec, cens = metrics.recovery_time(fake, s0=100.0, t_end=399.0, window=100.0)
    assert not cens
    assert rec == pytest.approx(240.0 - 100.0, abs=1.0)
    # никогда не восстанавливается -> цензурировано
    bad = dataclasses.replace(res, req_t=t, req_status=np.full(len(t), 2, dtype=np.int8))
    rec2, cens2 = metrics.recovery_time(bad, s0=100.0, t_end=399.0)
    assert cens2 and rec2 == pytest.approx(299.0)


def test_metrics_on_real_run_are_consistent() -> None:
    cfg = Config(R="R1", proto=ProtoParams(h_rec=300.0, w_last=100.0))
    plan = plan_for(cfg, "A", 0.5)
    res = simulate(cfg, plan.schedule, plan.t_stop, seed=2)
    m = (res.req_t >= plan.t_end - 100.0) & (res.req_t < plan.t_end)
    assert metrics.goodput(res, plan.t_end - 100.0, plan.t_end) == pytest.approx(
        float(np.mean(res.req_status[m] == SUCCESS))
    )
    amp = metrics.amplification(res, plan.t_end - 100.0, plan.t_end)
    assert 1.0 <= amp <= 4.0
    assert math.isnan(metrics.goodput(res, 1e9, 1e9 + 1))


def test_cli_writes_parquet(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("R: R1\nQ: LIFO\nD: on\nS: LogN\nproto:\n  h_rec: 300\n  w_last: 100\n")
    out = tmp_path / "r.parquet"
    rc = main(
        ["--config", str(cfg_path), "--seed", "10", "--n-seeds", "2", "--rho", "0.5", "0.6",
         "--protocol", "both", "--out", str(out)]
    )  # fmt: skip
    assert rc == 0
    df = pd.read_parquet(out)
    assert len(df) == 8
    assert set(df["protocol"]) == {"A", "B"}
    assert set(df["seed"]) == {10, 11}
    assert set(df["D"]) == {"on"} and set(df["Q"]) == {"LIFO"}
    assert ROW_KEYS <= set(df.columns)
    # дозапись
    main(["--config", str(cfg_path), "--seed", "12", "--rho", "0.5", "--out", str(out)])
    assert len(pd.read_parquet(out)) == 9
    assert "config_hash=" in capsys.readouterr().out


def test_yaml_configs_load_and_hash() -> None:
    base = load_config(ROOT / "configs" / "base_r1_fifo_off_exp.yaml")
    r0 = load_config(ROOT / "configs" / "r0_fifo_off_exp.yaml")
    assert base.D == "off" and base.R == "R1" and r0.R == "R0"
    assert base == Config()  # YAML совпадает со значениями по умолчанию (MODEL_SPEC §3)
    assert config_hash(base) == config_hash(Config())
    assert config_hash(base) != config_hash(r0)
    changed = base.model_copy(update={"proto": base.proto.model_copy(update={"trigger_mult": 2.0})})
    assert config_hash(changed) != config_hash(base)  # триггер входит в хэш


def test_yaml_bool_d_and_validation(tmp_path: Path) -> None:
    p = tmp_path / "x.yaml"
    p.write_text("D: on\n")  # YAML 1.1 -> True
    assert load_config(p).D == "on"
    p.write_text("unknown: 1\n")
    with pytest.raises(ValueError):
        load_config(p)
