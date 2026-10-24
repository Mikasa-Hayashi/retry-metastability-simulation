"""Дискретно-событийное ядро (heapq): сервис с c воркерами, очередь, дисциплины, deadline drop.

Попытки и запросы нумеруются целыми числами (уникальные id). Время обслуживания и джиттер
попытки берутся из потоков по номеру попытки (D-009).
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .clients import make_retry
from .config import Config
from .distributions import make_service

# состояния попытки
QUEUED, SERVING, OK, LATE, DROPPED, REJECTED = range(6)
# статус запроса
PENDING, SUCCESS, FAIL_TIMEOUT, FAIL_REJECT = range(4)

_COMPLETE, _RETRY, _TIMEOUT, _ARRIVAL, _SAMPLE = range(5)
_CHUNK = 8192

Draw = Callable[[int], np.ndarray]


class InvariantError(AssertionError):
    """Нарушение инварианта модели (V4 и монотонность времени)."""


@dataclass
class SimResult:
    """Сырой результат прогона (массивы numpy)."""

    t_stop: float
    stats_start: float
    # запросы (по порядку первичного прихода)
    req_t: np.ndarray
    req_att: np.ndarray
    req_status: np.ndarray
    req_res_t: np.ndarray
    req_lat: np.ndarray
    # попытки (индекс = id попытки)
    att_req: np.ndarray
    att_no: np.ndarray
    att_t: np.ndarray
    att_dl: np.ndarray
    att_state: np.ndarray
    # старты обслуживания: (id, время, длительность); завершения: (id, время, длительность, поздно)
    starts: np.ndarray
    done: np.ndarray
    # сэмплы раз в sample_dt
    samp_t: np.ndarray
    samp_q: np.ndarray
    samp_busy: np.ndarray
    samp_orbit: np.ndarray
    # интегралы по [stats_start, t_stop]
    area_busy: float
    area_q: float
    area_orbit: float
    # счётчики и состояние на конец
    n_rejected: int
    n_dropped: int
    busy_end: int
    q_end: int
    orbit_end: int
    budget_granted: int
    budget_denied: int


def poisson_arrivals(
    rng: np.random.Generator, schedule: Sequence[tuple[float, float]], t_stop: float, c: int
) -> np.ndarray:
    """Моменты первичных приходов для кусочно-постоянной нагрузки.

    schedule: [(t_end_i, rho_i)], последний сегмент продолжается до t_stop.
    lambda = rho * c / E[S], E[S] = 1. Преобразование времени потока единичной
    интенсивности (D-009).
    """
    edges = [0.0]
    rates: list[float] = []
    for t_end, rho in schedule:
        if rho <= 0:
            raise ValueError("rho > 0")
        if edges[-1] >= t_stop:
            break
        edges.append(min(t_end, t_stop))
        rates.append(rho * c)
    if edges[-1] < t_stop:
        edges.append(t_stop)
        rates.append(rates[-1])
    e = np.asarray(edges)
    lam_cum = np.concatenate(([0.0], np.cumsum(np.asarray(rates) * np.diff(e))))
    total = float(lam_cum[-1])
    n = int(total + 6.0 * math.sqrt(total) + 64)
    gam = np.cumsum(rng.standard_exponential(n))
    while gam[-1] < total:
        gam = np.concatenate((gam, gam[-1] + np.cumsum(rng.standard_exponential(n))))
    gam = gam[gam < total]
    return np.interp(gam, lam_cum, e)


def run_core(
    cfg: Config,
    arrivals: np.ndarray,
    t_stop: float,
    draw_service: Draw,
    draw_uniform: Draw,
    stats_start: float = 0.0,
    sample_dt: float = 1.0,
) -> SimResult:
    """Ядро. arrivals — отсортированные моменты первичных приходов (< t_stop)."""
    c, kcap, t_out = cfg.c, cfg.K, cfg.T
    finite_t = math.isfinite(t_out)
    mode = {"FIFO": 0, "LIFO": 1, "ALIFO": 2}[cfg.Q]
    alifo_thr = cfg.theta_q * kcap
    drop = cfg.D == "on"
    policy, budget = make_retry(cfg)
    max_r = policy.max_retries if policy.max_retries is not None else (1 << 60)
    pdelay = policy.delay
    uses_u = policy.uses_jitter
    inf = math.inf

    arr = arrivals.tolist()
    n_arr = len(arr)
    ia = 0
    next_arr = arr[0] if n_arr else inf
    next_samp = 0.0

    svc: list[float] = []
    jit: list[float] = []
    heap: list[tuple[float, int, int, int]] = []
    push = heapq.heappush
    pop = heapq.heappop
    q: deque[int] = deque()

    a_req: list[int] = []
    a_no: list[int] = []
    a_t: list[float] = []
    a_dl: list[float] = []
    a_state: list[int] = []
    r_t: list[float] = []
    r_att: list[int] = []
    r_status: list[int] = []
    r_res: list[float] = []
    r_lat: list[float] = []
    starts: list[tuple[int, float, float]] = []
    done: list[tuple[int, float, float, int]] = []
    samp_t: list[float] = []
    samp_q: list[int] = []
    samp_b: list[int] = []
    samp_o: list[int] = []

    busy = 0
    n_orbit = 0
    n_rej = 0
    n_drop = 0
    now = 0.0
    last = 0.0
    ss = stats_start
    ab = aq = ao = 0.0
    nan = math.nan

    while True:
        th = heap[0][0] if heap else inf
        if next_samp <= next_arr and next_samp <= th:
            t = next_samp
            kind = _SAMPLE
            a = b = 0
        elif th <= next_arr:
            t, kind, a, b = pop(heap)
        else:
            t = next_arr
            kind = _ARRIVAL
            a = b = 0
        if t > t_stop:
            break
        if t < now:
            raise InvariantError(f"время пошло назад: {t} < {now}")
        now = t
        if t > ss:
            dt = t - max(ss, last)
            ab += dt * busy
            aq += dt * len(q)
            ao += dt * n_orbit
        last = t

        failed = False
        reject = False
        rid = 0
        k = 0
        aid = 0

        if kind == _SAMPLE:
            samp_t.append(t)
            samp_q.append(len(q))
            samp_b.append(busy)
            samp_o.append(n_orbit)
            next_samp += sample_dt
            continue

        if kind == _COMPLETE:
            aid = a
            busy -= 1
            rid = a_req[aid]
            if t <= a_dl[aid]:
                a_state[aid] = OK
                r_status[rid] = SUCCESS
                r_res[rid] = t
                r_lat[rid] = t - r_t[rid]
                late = 0
            else:
                a_state[aid] = LATE
                late = 1
            done.append((aid, t, svc[aid], late))
            while q:
                if mode == 0:
                    x = q.popleft()
                elif mode == 1:
                    x = q.pop()
                else:
                    x = q.pop() if len(q) > alifo_thr else q.popleft()
                if drop and t >= a_dl[x]:
                    a_state[x] = DROPPED
                    n_drop += 1
                    continue
                a_state[x] = SERVING
                busy += 1
                while x >= len(svc):
                    svc.extend(draw_service(_CHUNK).tolist())
                s = svc[x]
                starts.append((x, t, s))
                push(heap, (t + s, _COMPLETE, x, 0))
                break
            continue

        if kind == _TIMEOUT:
            aid = a
            if a_state[aid] == OK:
                continue
            rid = a_req[aid]
            k = a_no[aid]
            failed = True
        else:
            if kind == _ARRIVAL:
                rid = len(r_t)
                r_t.append(t)
                r_att.append(1)
                r_status.append(PENDING)
                r_res.append(nan)
                r_lat.append(nan)
                if budget is not None:
                    budget.on_primary(t)
                ia += 1
                next_arr = arr[ia] if ia < n_arr else inf
                k = 0
            else:  # _RETRY
                rid = a
                k = b
                n_orbit -= 1
                r_att[rid] = k + 1
            aid = len(a_req)
            a_req.append(rid)
            a_no.append(k)
            a_t.append(t)
            dl = t + t_out
            a_dl.append(dl)
            if busy < c:
                busy += 1
                a_state.append(SERVING)
                while aid >= len(svc):
                    svc.extend(draw_service(_CHUNK).tolist())
                s = svc[aid]
                starts.append((aid, t, s))
                end = t + s
                push(heap, (end, _COMPLETE, aid, 0))
                if end > dl:
                    push(heap, (dl, _TIMEOUT, aid, 0))
            elif len(q) < kcap:
                a_state.append(QUEUED)
                q.append(aid)
                if finite_t:
                    push(heap, (dl, _TIMEOUT, aid, 0))
            else:
                a_state.append(REJECTED)
                n_rej += 1
                failed = True
                reject = True

        if failed:
            if k < max_r and (budget is None or budget.acquire(t)):
                if uses_u:
                    while aid >= len(jit):
                        jit.extend(draw_uniform(_CHUNK).tolist())
                    u = jit[aid]
                else:
                    u = 0.0
                push(heap, (t + pdelay(k + 1, u), _RETRY, rid, k + 1))
                n_orbit += 1
            else:
                r_status[rid] = FAIL_REJECT if reject else FAIL_TIMEOUT
                r_res[rid] = t

    if t_stop > ss:
        dt = t_stop - max(ss, last)
        ab += dt * busy
        aq += dt * len(q)
        ao += dt * n_orbit

    return SimResult(
        t_stop=t_stop,
        stats_start=ss,
        req_t=np.asarray(r_t, dtype=np.float64),
        req_att=np.asarray(r_att, dtype=np.int64),
        req_status=np.asarray(r_status, dtype=np.int8),
        req_res_t=np.asarray(r_res, dtype=np.float64),
        req_lat=np.asarray(r_lat, dtype=np.float64),
        att_req=np.asarray(a_req, dtype=np.int64),
        att_no=np.asarray(a_no, dtype=np.int64),
        att_t=np.asarray(a_t, dtype=np.float64),
        att_dl=np.asarray(a_dl, dtype=np.float64),
        att_state=np.asarray(a_state, dtype=np.int8),
        starts=np.asarray(starts, dtype=np.float64).reshape(-1, 3),
        done=np.asarray(done, dtype=np.float64).reshape(-1, 4),
        samp_t=np.asarray(samp_t, dtype=np.float64),
        samp_q=np.asarray(samp_q, dtype=np.int64),
        samp_busy=np.asarray(samp_b, dtype=np.int64),
        samp_orbit=np.asarray(samp_o, dtype=np.int64),
        area_busy=ab,
        area_q=aq,
        area_orbit=ao,
        n_rejected=n_rej,
        n_dropped=n_drop,
        busy_end=busy,
        q_end=len(q),
        orbit_end=n_orbit,
        budget_granted=budget.granted if budget else 0,
        budget_denied=budget.denied if budget else 0,
    )


def make_streams(seed: int) -> tuple[np.random.Generator, np.random.Generator, np.random.Generator]:
    """Независимые потоки: вход, обслуживание, джиттер (D-009)."""
    s_arr, s_svc, s_jit = np.random.SeedSequence(seed).spawn(3)
    return (
        np.random.default_rng(s_arr),
        np.random.default_rng(s_svc),
        np.random.default_rng(s_jit),
    )


def simulate(
    cfg: Config,
    schedule: Sequence[tuple[float, float]],
    t_stop: float,
    seed: int,
    stats_start: float = 0.0,
    sample_dt: float = 1.0,
) -> SimResult:
    """Один прогон: кусочно-постоянная нагрузка [(t_end, rho), ...], остановка в t_stop."""
    rng_arr, rng_svc, rng_jit = make_streams(seed)
    dist = make_service(cfg.S, cfg.logn_cv, cfg.pareto_alpha)
    arrivals = poisson_arrivals(rng_arr, schedule, t_stop, cfg.c)

    def draw_service(n: int) -> np.ndarray:
        return dist.sample(rng_svc, n)

    def draw_uniform(n: int) -> np.ndarray:
        return rng_jit.random(n)

    return run_core(cfg, arrivals, t_stop, draw_service, draw_uniform, stats_start, sample_dt)
