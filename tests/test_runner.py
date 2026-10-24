"""Тесты раннера: задачи, возобновление, ошибки, диапазоны seed, независимость B от триггера."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import runner
from src.config import Config, ProtoParams
from src.protocols import run_protocol


def small_cfg(mult: float = 3.0, length: float = 5.0) -> Config:
    return Config(
        c=2,
        K=8,
        T=5.0,
        R="R1",
        proto=ProtoParams(
            t_warm=20.0,
            trigger_mult=mult,
            trigger_len=length,
            h_rec=60.0,
            overload_len=30.0,
            window=20.0,
            w_last=30.0,
        ),
    )


def test_unknown_stage_and_ranges() -> None:
    with pytest.raises(ValueError):
        runner.stage_spec("raw")
    assert runner.STAGES["pilot"].seed_hi == 999_999
    assert runner.STAGES["main"].seed_lo == 1_000_000


def test_make_tasks_idempotent_and_deterministic(tmp_path: Path) -> None:
    cfgs = [small_cfg(2.0), small_cfg(3.0)]
    t1 = runner.make_tasks("pilot", cfgs, [0.5, 0.6], ["A", "B"], 3, "x", root=tmp_path)
    assert len(t1) == 2 * 2 * 2 * 3
    assert sorted(t1["seed"]) == list(range(len(t1)))  # base 0 + индекс
    t2 = runner.make_tasks("pilot", cfgs, [0.5, 0.6], ["A", "B"], 3, "x", root=tmp_path)
    assert len(t2) == 0
    # докладывание реплик: новые seed продолжают нумерацию, replicate продолжается
    t3 = runner.make_tasks("pilot", cfgs[:1], [0.5], ["A"], 5, "y", root=tmp_path)
    assert list(t3["replicate"]) == [3, 4]
    assert list(t3["seed"]) == [len(t1), len(t1) + 1]
    all_t = runner.load_tasks(tmp_path / "pilot")
    assert all_t["seed"].is_unique


def test_main_seeds_not_in_pilot(tmp_path: Path) -> None:
    runner.make_tasks("main", [small_cfg()], [0.5], ["A"], 2, "x", root=tmp_path)
    t = runner.load_tasks(tmp_path / "main")
    assert t["seed"].min() >= 1_000_000
    runner.make_tasks("pilot", [small_cfg()], [0.5], ["A"], 2, "x", root=tmp_path)
    assert runner.load_tasks(tmp_path / "pilot")["seed"].max() < 100_000


def test_run_resume_and_complete(tmp_path: Path) -> None:
    runner.make_tasks("pilot", [small_cfg()], [0.5, 0.7], ["A", "B"], 3, "t", root=tmp_path)
    log = io.StringIO()
    info = runner.run_stage("pilot", batch_size=5, workers=1, limit=7, root=tmp_path, log=log)
    assert info["n_ok"] == 7
    d = tmp_path / "pilot"
    assert len(runner._parts(d, "part")) == 2  # 5 + 2
    rep = runner.check_complete("pilot", root=tmp_path)
    assert rep["n_missing"] == 5 and not rep["complete"]
    runner.run_stage("pilot", batch_size=5, workers=2, root=tmp_path, log=log)
    rep = runner.check_complete("pilot", root=tmp_path)
    assert rep["complete"] and rep["n_duplicates"] == 0 and rep["n_results"] == 12
    # повторный запуск ничего не делает
    info = runner.run_stage("pilot", workers=1, root=tmp_path, log=log)
    assert info["n_tasks"] == 0
    res = runner.load_results(d)
    assert {"failed", "g_final", "src_hash", "replicate", "tag", "git_commit"} <= set(res.columns)
    m = runner.write_manifest("pilot", root=tmp_path)
    assert "part-0000.parquet" in m.read_text(encoding="utf-8")


def test_results_equal_direct_run(tmp_path: Path) -> None:
    cfg = small_cfg()
    runner.make_tasks("pilot", [cfg], [0.6], ["A"], 1, "t", root=tmp_path)
    runner.run_stage("pilot", workers=1, root=tmp_path, log=io.StringIO())
    row = runner.load_results(tmp_path / "pilot").iloc[0]
    direct = run_protocol(cfg, "A", 0.6, int(row["seed"]))
    assert row["g_final"] == direct["g_final"] and row["failed"] == direct["failed"]


def test_error_logged_and_run_continues(tmp_path: Path) -> None:
    runner.make_tasks("pilot", [small_cfg()], [0.5], ["A", "Z"], 2, "t", root=tmp_path)
    runner.run_stage("pilot", batch_size=10, workers=1, root=tmp_path, log=io.StringIO())
    d = tmp_path / "pilot"
    err = runner.load_errors(d)
    assert len(err) == 2 and err["traceback"].str.contains("Traceback").all()
    assert len(runner.load_results(d)) == 2
    rep = runner.check_complete("pilot", root=tmp_path)
    assert rep["n_error_unresolved"] == 2 and not rep["complete"]
    # повтор: ошибка воспроизводится, записи ошибок накапливаются, ничего не удалено
    runner.run_stage("pilot", workers=1, root=tmp_path, log=io.StringIO())
    assert len(runner.load_errors(d)) == 4


def test_protocol_b_independent_of_trigger() -> None:
    a = run_protocol(small_cfg(2.0, 5.0), "B", 0.6, 123)
    b = run_protocol(small_cfg(4.0, 50.0), "B", 0.6, 123)
    assert a["g_final"] == b["g_final"] and a["failed"] == b["failed"]


def test_refine_rhos() -> None:
    rho = np.repeat(np.arange(0.3, 0.96, 0.05), 10)
    failed = (rho > 0.62).astype(int)
    new = runner.refine_rhos(rho, failed, 0.025, 0.075, 0.3, 0.95)
    assert new and all(abs(v / 0.025 - round(v / 0.025)) < 1e-6 for v in new)
    assert all(0.3 <= v <= 0.95 for v in new)
    assert all(abs(v - 0.625) <= 0.08 for v in new)
    assert not set(new) & {round(float(v), 6) for v in np.unique(np.round(rho, 6))}
    # цензурирование справа: окно у верхнего края
    new_r = runner.refine_rhos(rho, np.zeros_like(failed), 0.025, 0.075, 0.3, 0.95)
    assert new_r and min(new_r) >= 0.875 - 1e-9


def test_refine_tasks_uses_only_failed(tmp_path: Path) -> None:
    cfg = small_cfg()
    runner.make_tasks("pilot", [cfg], [0.4, 0.6, 0.8], ["A"], 4, "calib", root=tmp_path)
    runner.run_stage("pilot", workers=1, root=tmp_path, log=io.StringIO())
    new = runner.refine_tasks("pilot", "calib", "A", 0.1, 0.15, 0.3, 0.95, 4, "ref", root=tmp_path)
    assert isinstance(new, pd.DataFrame)
    assert set(new["tag"]) <= {"ref"}
