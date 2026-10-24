"""Независимый минимальный симулятор для проверки теории (не src/).

Только FIFO, Exp-обслуживание, D = off, политика R1 (фиксированные повторы).
Для FIFO время ухода известно в момент прихода (воркеры берутся по порядку
освобождения), поэтому достаточно кучи событий-попыток. Назначение —
проверка формул THEORY.md и допущений mean-field, а не замена src/.
Запуск:  python analysis/theory_check_sim.py
"""

from __future__ import annotations

import heapq
import sys
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimParams:
    c: int = 8
    k_queue: int = 128
    t_out: float = 10.0
    delay: float = 1.0
    m_retries: int = 3
    mu: float = 1.0


def simulate(
    schedule: list[tuple[float, float]],
    p: SimParams,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """schedule: [(t_end, rho), ...] — кусочно-постоянная нагрузка.

    Возвращает (время первичных запросов, успех 0/1, доля принятых попыток
    с успехом по всем попыткам для справки). Успех запроса: хотя бы одна
    попытка получила ответ <= t_out после своего начала.
    """
    rng = np.random.default_rng(seed)
    free: list[float] = [0.0] * p.c
    heapq.heapify(free)
    deps: list[float] = []
    events: list[tuple[float, int, int]] = []  # (время, id запроса, номер попытки)
    prim_t: list[float] = []
    prim_ok: list[int] = []

    # первичные заявки генерируем по сегментам заранее
    t0 = 0.0
    for t_end, rho in schedule:
        lam = rho * p.c * p.mu
        t = t0
        while True:
            t += rng.exponential(1.0 / lam)
            if t >= t_end:
                break
            rid = len(prim_t)
            prim_t.append(t)
            prim_ok.append(0)
            heapq.heappush(events, (t, rid, 0))
        t0 = t_end
    att_total = 0
    att_ok = 0
    cap = p.c + p.k_queue
    while events:
        t, rid, k = heapq.heappop(events)
        att_total += 1
        while deps and deps[0] <= t:
            heapq.heappop(deps)
        if len(deps) >= cap:
            fail_time = t  # отказ сразу
        else:
            start = max(t, heapq.heappop(free))
            dep = start + rng.exponential(p.mu)
            heapq.heappush(free, dep)
            heapq.heappush(deps, dep)
            if dep - t <= p.t_out:
                att_ok += 1
                prim_ok[rid] = 1
                continue
            fail_time = t + p.t_out
        if k < p.m_retries and prim_ok[rid] == 0:
            heapq.heappush(events, (fail_time + p.delay, rid, k + 1))
    return np.array(prim_t), np.array(prim_ok), att_ok / max(att_total, 1)


def g_in_window(times: np.ndarray, ok: np.ndarray, lo: float, hi: float) -> float:
    mask = (times >= lo) & (times < hi)
    return float(ok[mask].mean()) if mask.any() else float("nan")


def protocol_a(rho: float, seed: int, p: SimParams, trig: float = 4.0) -> float:
    d_s, t0, h_rec = 50.0, 200.0, 1500.0
    sched = [(t0, rho), (t0 + d_s, rho * trig), (t0 + d_s + h_rec, rho)]
    tm, ok, _ = simulate(sched, p, seed)
    end = sched[-1][0]
    return g_in_window(tm, ok, end - 300.0, end)


def protocol_b(rho: float, seed: int, p: SimParams) -> float:
    sched = [(400.0, 1.05), (400.0 + 1500.0, rho)]
    tm, ok, _ = simulate(sched, p, seed)
    end = sched[-1][0]
    return g_in_window(tm, ok, end - 300.0, end)


def sweep(
    rhos: list[float], reps: int, p: SimParams, trig: float, theta: float = 0.5
) -> list[tuple[float, float, float]]:
    """Возвращает [(rho, P_A, P_B)] — доли отказов g < theta по reps репликам."""
    out = []
    for rho in rhos:
        pa = float(np.mean([protocol_a(rho, 10_000 + i, p, trig) < theta for i in range(reps)]))
        pb = float(np.mean([protocol_b(rho, 20_000 + i, p) < theta for i in range(reps)]))
        out.append((rho, pa, pb))
    return out


def main(reps: int = 30) -> None:
    p = SimParams()
    rows = ["trigger,rho,P_A,P_B"]
    for trig, rhos in [
        (4.0, [0.20 + 0.025 * i for i in range(13)]),
        (2.0, [0.20 + 0.05 * i for i in range(16)]),
    ]:
        for rho, pa, pb in sweep(rhos, reps, p, trig):
            print(f"trigger x{trig} rho={rho:.3f} P_A={pa:.2f} P_B={pb:.2f}")
            rows.append(f"{trig},{rho:.3f},{pa:.3f},{pb:.3f}")
            sys.stdout.flush()
    with open("figures/theory_check_protocols.csv", "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
