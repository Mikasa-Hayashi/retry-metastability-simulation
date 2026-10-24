"""Тесты V1, V2, V3, V6 (MODEL_SPEC §7) против analysis/theory.py.

Критерий V1/V2/V3: теоретическое значение лежит в t-доверительном интервале по независимым
репликациям. Уровень интервала задан заранее и общий для файла: alpha = 0.05 / FAMILY, где
FAMILY = 40 — верхняя оценка числа сравнений в этом файле (поправка Бонферрони), чтобы при
фиксированных seed регрессионный тест не давал ложных отказов. Уровень не подбирался под
результат. Итоговый вердикт V1–V6 выносит валидация (reports/validation.md), используя
собственные независимые прогоны.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import theory as th

from src.config import Config, ProtoParams, RetryParams
from src.model import SimResult, simulate
from src.protocols import run_protocol

BIG_K = (
    5000  # K = 128 при rho <= 0.9 даёт блокировку ~1e-6 (THEORY §1); берём "бесконечную" очередь
)


FAMILY = 40


def ci_contains(samples: list[float], truth: float, n_tests: int = 1, alpha: float = 0.05) -> bool:
    """Лежит ли truth в t-интервале уровня alpha / FAMILY (n_tests сохранён для читаемости)."""
    del n_tests
    x = np.asarray(samples)
    se = x.std(ddof=1) / math.sqrt(len(x))
    q = stats.t.ppf(1.0 - alpha / FAMILY / 2.0, df=len(x) - 1)
    return abs(float(x.mean()) - truth) <= q * se


def waits_and_sojourn(res: SimResult, burn: float, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Ожидание и время в системе попыток, пришедших в [burn, t_stop - margin]."""
    aid = res.starts[:, 0].astype(np.int64)
    arrival = res.att_t[aid]
    wait = res.starts[:, 1] - arrival
    soj = wait + res.starts[:, 2]
    m = (arrival >= burn) & (arrival <= res.t_stop - margin)
    return wait[m], soj[m]


def mm_cfg(c: int, q: str = "FIFO") -> Config:
    return Config(c=c, K=BIG_K, T=math.inf, R="R0", Q=q, D="off", S="Exp")  # type: ignore[arg-type]


# ---------------------------------------------------------------- V1


@pytest.mark.parametrize("c", [1, 8])
def test_v1_mmc_mean_delay_and_prob_wait(c: int) -> None:
    rhos = (0.5, 0.7, 0.9)
    n_tests = 2 * len(rhos)
    reps = 16
    t_stop = 30000.0 if c == 1 else 8000.0
    for rho in rhos:
        lam = rho * c
        cfg = mm_cfg(c)
        w_means, soj_means, p_wait = [], [], []
        for seed in range(reps):
            res = simulate(cfg, [(t_stop, rho)], t_stop, seed=1000 * c + seed)
            wait, soj = waits_and_sojourn(res, burn=500.0, margin=400.0)
            w_means.append(float(wait.mean()))
            soj_means.append(float(soj.mean()))
            p_wait.append(float(np.mean(wait > 0.0)))
        assert ci_contains(soj_means, th.mmc_mean_w(c, lam, 1.0), n_tests), (c, rho, "W")
        assert ci_contains(w_means, th.mmc_mean_wq(c, lam, 1.0), n_tests), (c, rho, "Wq")
        assert ci_contains(p_wait, th.erlang_c(c, lam), n_tests), (c, rho, "P(wait)")


def test_v1_time_average_busy_servers() -> None:
    # загрузка: среднее число занятых воркеров = rho * c (закон сохранения)
    rho, c, t_stop = 0.7, 8, 6000.0
    vals = []
    for seed in range(10):
        res = simulate(mm_cfg(c), [(t_stop, rho)], t_stop, seed=seed, stats_start=500.0)
        vals.append(res.area_busy / (t_stop - 500.0))
    assert ci_contains(vals, rho * c, 1)


# ---------------------------------------------------------------- V2


@pytest.mark.parametrize("c,rho", [(1, 0.7), (8, 0.9)])
def test_v2_lifo_mean_equals_fifo_variance_larger(c: int, rho: float) -> None:
    reps = 16
    t_stop = 30000.0 if c == 1 else 8000.0
    diffs, means_f, means_l = [], [], []
    var_ratio = []
    for seed in range(reps):
        wf, _ = waits_and_sojourn(
            simulate(mm_cfg(c, "FIFO"), [(t_stop, rho)], t_stop, seed=seed), 500.0, 400.0
        )
        wl, _ = waits_and_sojourn(
            simulate(mm_cfg(c, "LIFO"), [(t_stop, rho)], t_stop, seed=seed), 500.0, 400.0
        )
        means_f.append(float(wf.mean()))
        means_l.append(float(wl.mean()))
        diffs.append(float(wl.mean() - wf.mean()))
        var_ratio.append(float(wl.var() / wf.var()))
    wq = th.mmc_mean_wq(c, rho * c, 1.0)
    assert ci_contains(means_f, wq, 4), "FIFO"
    assert ci_contains(means_l, wq, 4), "LIFO"
    assert ci_contains(diffs, 0.0, 4), "разность средних LIFO - FIFO"
    assert np.mean(var_ratio) > 1.5
    assert min(var_ratio) > 1.0


def test_v2_variance_matches_finch_mm1() -> None:
    lam, reps, t_stop = 0.5, 20, 40000.0
    vf, vl = [], []
    for seed in range(reps):
        for q, out in (("FIFO", vf), ("LIFO", vl)):
            w, _ = waits_and_sojourn(
                simulate(mm_cfg(1, q), [(t_stop, lam)], t_stop, seed=seed), 500.0, 400.0
            )
            out.append(float(w.var()))
    assert ci_contains(vf, th.mm1_wait_moments(lam, 1.0, lifo=False).var, 2)
    assert ci_contains(vl, th.mm1_wait_moments(lam, 1.0, lifo=True).var, 2)


# ---------------------------------------------------------------- V3


@pytest.mark.parametrize("lam,nu", [(0.5, 1.0), (0.7, 0.5), (0.9, 2.0)])
def test_v3_classical_retrial(lam: float, nu: float) -> None:
    cfg = Config(
        c=1, K=0, T=math.inf, R="ORBIT", Q="FIFO", D="off", S="Exp",
        retry=RetryParams(orbit_rate=nu),
    )  # fmt: skip
    t_stop, burn, reps = 30000.0, 500.0, 14
    busy, orbit = [], []
    for seed in range(reps):
        res = simulate(cfg, [(t_stop, lam)], t_stop, seed=seed, stats_start=burn, sample_dt=10.0)
        busy.append(res.area_busy / (t_stop - burn))
        orbit.append(res.area_orbit / (t_stop - burn))
    assert ci_contains(busy, th.retrial_busy_prob(lam, 1.0), 2), "P(busy)"
    assert ci_contains(orbit, th.retrial_mean_orbit(lam, 1.0, nu), 2), "E[orbit]"


def test_v3_every_request_eventually_served_and_no_loss() -> None:
    cfg = Config(c=1, K=0, T=math.inf, R="ORBIT", retry=RetryParams(orbit_rate=1.0))
    res = simulate(cfg, [(2000.0, 0.6)], 2000.0, seed=3)
    # без таймаутов запрос завершается успехом; неудач нет
    assert int(np.sum(res.req_status == 2)) == 0 and int(np.sum(res.req_status == 3)) == 0
    assert res.n_rejected > 0


# ---------------------------------------------------------------- V6


@pytest.mark.parametrize("rho", [0.5, 0.8, 0.9, 0.95])
def test_v6_r0_recovers_after_trigger(rho: float) -> None:
    cfg = Config(R="R0")
    for seed in range(3):
        row = run_protocol(cfg, "A", rho, seed)
        assert row["failed"] == 0, (rho, seed, row["g_final"])
        assert row["g_final"] > 0.9


@pytest.mark.parametrize("rho", [0.5, 0.9, 0.95])
def test_v6_r0_recovers_from_overload_protocol_b(rho: float) -> None:
    for seed in range(2):
        row = run_protocol(Config(R="R0"), "B", rho, seed)
        assert row["failed"] == 0, (rho, seed, row["g_final"])


def test_v6_control_does_not_depend_on_trigger_strength() -> None:
    # ослабленный и усиленный триггер при R0, rho = 0.9: восстановление остаётся
    for mult, length in ((2.5, 30.0), (6.0, 50.0)):
        cfg = Config(R="R0", proto=ProtoParams(trigger_mult=mult, trigger_len=length))
        assert run_protocol(cfg, "A", 0.9, 0)["failed"] == 0


# ---------------------------------------------------------------- перекрёстная проверка с THEORY §4


def test_crosscheck_attempt_success_vs_mean_field_building_block() -> None:
    """R0 (без повторов), конечная очередь K = 128, T = 10, rho = 0.97: доля неудач попыток
    совпадает с success_prob_finite (THEORY: 0.0919)."""
    rho, t_stop = 0.97, 12000.0
    fails = []
    for seed in range(6):
        res = simulate(Config(R="R0"), [(t_stop, rho)], t_stop, seed=seed)
        m = (res.req_t > 500.0) & (res.req_t < t_stop - 100.0)
        fails.append(1.0 - float(np.mean(res.req_status[m] == 1)))
    truth = 1.0 - th.success_prob_finite(8, 128, rho * 8, 1.0, 10.0)
    assert ci_contains(fails, truth)
