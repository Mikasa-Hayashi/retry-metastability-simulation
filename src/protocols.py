"""Протоколы A и B (MODEL_SPEC §6), триггеры и сбор строки результата (§10)."""

from __future__ import annotations

import math
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from typing import Any, Literal

from . import metrics
from .clients import make_retry
from .config import Config, config_hash
from .model import SimResult, simulate

Protocol = Literal["A", "B"]


@dataclass(frozen=True)
class Plan:
    """Профиль нагрузки одного прогона."""

    schedule: list[tuple[float, float]]  # [(t_end, rho)], последний сегмент идёт до t_stop
    t_ref: float  # точка отсчёта времени восстановления
    t_end: float  # конец наблюдения
    t_stop: float  # конец симуляции (t_end + хвост)


def tail_length(cfg: Config) -> float:
    """Хвост, достаточный для завершения всех запросов окна (D-008)."""
    if cfg.proto.tail is not None:
        return cfg.proto.tail
    pol, _ = make_retry(cfg)
    mr = pol.max_retries if pol.max_retries is not None else 0
    return (mr + 1) * cfg.T + pol.max_total_delay() + 1.0


def trigger_schedule(
    rho: float, t0: float, mult: float, length: float, t_end: float
) -> list[tuple[float, float]]:
    """Нагрузка ρ, всплеск ρ·mult на [t0, t0+length), затем ρ до t_end."""
    return [(t0, rho), (t0 + length, rho * mult), (t_end, rho)]


def plan_for(cfg: Config, protocol: Protocol, rho: float) -> Plan:
    p = cfg.proto
    if protocol == "A":
        t_trig_end = p.t_warm + p.trigger_len
        t_end = t_trig_end + p.h_rec
        sched = trigger_schedule(rho, p.t_warm, p.trigger_mult, p.trigger_len, t_end)
        t_ref = t_trig_end
    elif protocol == "B":
        t_end = p.overload_len + p.h_rec
        sched = [(p.overload_len, p.overload_rho), (t_end, rho)]
        t_ref = p.overload_len
    else:
        raise ValueError(f"протокол {protocol}")
    return Plan(sched, t_ref, t_end, t_end + tail_length(cfg))


def summarize(cfg: Config, res: SimResult, plan: Plan) -> dict[str, Any]:
    p = cfg.proto
    lo, hi = plan.t_end - p.w_last, plan.t_end
    g_final = metrics.goodput(res, lo, hi)
    if math.isnan(g_final):
        raise ValueError("окно итогового наблюдения пусто (нет первичных запросов)")
    amp = metrics.amplification(res, lo, hi)
    rec, rec_cens = metrics.recovery_time(
        res, plan.t_ref, plan.t_end, p.window, p.rec_level, p.rec_hold
    )
    q_mean, q_max = metrics.queue_stats(res, lo, hi)
    return {
        "failed": int(g_final < p.theta),
        "g_final": g_final,
        "retries_per_request": amp - 1.0,
        "amplification": amp,
        "wasted_fraction": metrics.wasted_fraction(res, lo, hi),
        "recovery_time": rec,
        "recovery_censored": int(rec_cens),
        "p99": metrics.p99_latency(res, lo, hi),
        "qlen_mean_final": q_mean,
        "qlen_max_final": q_max,
        "n_primary_final": int(metrics.window_mask(res, lo, hi).sum()),
    }


@lru_cache(maxsize=1)
def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@lru_cache(maxsize=1)
def package_versions() -> str:
    names = ["numpy", "scipy", "pandas", "pyarrow", "pydantic", "pyyaml"]
    parts = [f"python={sys.version.split()[0]}"]
    for n in names:
        try:
            parts.append(f"{n}={metadata.version(n)}")
        except metadata.PackageNotFoundError:
            parts.append(f"{n}=?")
    return ";".join(parts)


def run_protocol(
    cfg: Config, protocol: Protocol, rho: float, seed: int, check: bool = True
) -> dict[str, Any]:
    """Один прогон протокола A или B. Возвращает строку результата (§10)."""
    t_start = time.perf_counter()
    plan = plan_for(cfg, protocol, rho)
    res = simulate(cfg, plan.schedule, plan.t_stop, seed)
    counters: dict[str, int] = metrics.check_v4(res) if check else {}
    row = summarize(cfg, res, plan)
    runtime = time.perf_counter() - t_start
    out: dict[str, Any] = {
        "config_hash": config_hash(cfg),
        "R": cfg.R,
        "Q": cfg.Q,
        "D": cfg.D,
        "S": cfg.S,
        "c": cfg.c,
        "K": cfg.K,
        "T": cfg.T,
        "theta": cfg.proto.theta,
        "trigger_mult": cfg.proto.trigger_mult,
        "trigger_len": cfg.proto.trigger_len,
        "rho_test": float(rho),
        "protocol": protocol,
        "seed": int(seed),
        **row,
        **counters,
        "n_rejected": res.n_rejected,
        "n_dropped": res.n_dropped,
        "budget_denied": res.budget_denied,
        "runtime": runtime,
        "git_commit": git_commit(),
        "versions": package_versions(),
    }
    return out
