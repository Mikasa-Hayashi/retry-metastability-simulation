"""V4 (независимый аудит инвариантов) по сырым массивам SimResult. Код валидации.

Не использует src.metrics.check_v4: все проверки пересчитаны здесь из массивов
att_*, req_*, starts, done, samp_*.
Проверяется: баланс запросов; <= c одновременных обслуживаний; интеграл занятости; длина очереди
(пересчёт на сэмплах, D=off); определение успеха/позднего ответа; цепочка повторов (задержки R1/R2,
максимум попыток, момент неудачи); порядок выбора из очереди FIFO/LIFO/ALIFO (перебор на коротком
прогоне); D=on (пропущенные не обслуживались, обслуженные начали до срока); бюджет R3;
неотрицательность времён; число первичных приходов в сегментах (Пуассон).
Запуск: .venv\\Scripts\\python analysis\\validation_v4.py
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.model import (
    DROPPED,
    FAIL_REJECT,
    FAIL_TIMEOUT,
    LATE,
    OK,
    PENDING,
    REJECTED,
    SERVING,
    SUCCESS,
    SimResult,
    simulate,
)
from src.protocols import plan_for, run_protocol

EPS = 1e-9


def audit(cfg: Config, res: SimResult, check_samples: bool = True) -> list[str]:
    bad: list[str] = []
    n_att = len(res.att_t)
    n_req = len(res.req_t)
    st = res.req_status
    # 1. баланс запросов (пересчёт: исход по попыткам)
    ok_att = res.att_state == OK
    succ_req = np.zeros(n_req, bool)
    succ_req[res.att_req[ok_att]] = True
    if not np.array_equal(succ_req, st == SUCCESS):
        bad.append("успех запроса не совпадает с наличием OK-попытки")
    if np.bincount(res.att_req[ok_att], minlength=n_req).max(initial=0) > 1:
        bad.append("у запроса более одной успешной попытки")
    nres = np.isnan(res.req_res_t)
    if not np.array_equal(nres, st == PENDING):
        bad.append("PENDING != NaN во времени исхода")
    if n_req != int(np.sum(st == SUCCESS) + np.sum(st == FAIL_TIMEOUT) + np.sum(st == FAIL_REJECT) + np.sum(st == PENDING)):
        bad.append("баланс запросов")
    # 2. времена
    if n_att and (res.att_t.min() < 0 or np.any(np.diff(res.req_t) < 0)):
        bad.append("отрицательное/немонотонное время прихода")
    sid = res.starts[:, 0].astype(np.int64)
    s0 = res.starts[:, 1]
    sd = res.starts[:, 2]
    if np.any(s0 < res.att_t[sid] - EPS) or np.any(sd <= 0):
        bad.append("старт раньше прихода или неположительное обслуживание")
    if len(np.unique(sid)) != len(sid):
        bad.append("двойной старт")
    ends = s0 + sd
    # 3. <= c одновременных обслуживаний (sweep)
    ev = np.concatenate((np.column_stack((s0, np.ones_like(s0))), np.column_stack((ends, -np.ones_like(ends)))))
    order = np.lexsort((ev[:, 1], ev[:, 0]))  # при равенстве времён -1 раньше +1
    conc = np.cumsum(ev[order, 1])
    if conc.max(initial=0) > cfg.c:
        bad.append(f"одновременных обслуживаний {conc.max()} > c={cfg.c}")
    # 4. интеграл занятости = сумма пересечений интервалов обслуживания с [ss, t_stop]
    lo, hi = res.stats_start, res.t_stop
    area = float(np.sum(np.clip(np.minimum(ends, hi) - np.maximum(s0, lo), 0, None)))
    if abs(area - res.area_busy) > 1e-6 * max(1.0, area):
        bad.append(f"area_busy {res.area_busy} != {area}")
    # 5. определение успеха/позднего ответа и состояния
    started = np.zeros(n_att, bool)
    started[sid] = True
    end_of = np.full(n_att, np.nan)
    end_of[sid] = ends
    fin = started & (end_of <= res.t_stop + EPS)
    exp_ok = fin & (end_of <= res.att_dl + EPS)
    exp_late = fin & (end_of > res.att_dl + EPS)
    if np.any(exp_ok != (res.att_state == OK)) or np.any(exp_late != (res.att_state == LATE)):
        bad.append("OK/LATE не совпадают с end<=deadline")
    if np.any(res.att_state[started & ~fin] != SERVING):
        bad.append("недообслуженные не в SERVING")
    if np.sum(res.att_state == SERVING) != res.busy_end:
        bad.append("busy_end")
    # D=on: DROPPED не стартовали; обслуженные начали до срока
    if np.any(started[res.att_state == DROPPED]):
        bad.append("DROPPED стартовала")
    if cfg.D == "on" and np.any(s0 >= res.att_dl[sid] - EPS):
        bad.append("D=on: обслуживание начато после срока")
    if cfg.D == "off" and (np.any(res.att_state == DROPPED)):
        bad.append("D=off: есть DROPPED")
    # 6. очередь на сэмплах (D=off: QUEUED = пришла, не начала, не отказана)
    if check_samples and cfg.D == "off" and len(res.samp_t):
        adm = res.att_state != REJECTED
        st_t = np.full(n_att, math.inf)
        st_t[sid] = s0
        # прямой расчёт на разреженной сетке сэмплов (до 400 точек)
        pick = np.linspace(0, len(res.samp_t) - 1, min(400, len(res.samp_t))).astype(int)
        waited = adm & (st_t > res.att_t + EPS)  # прошли через очередь
        for i in pick:
            t = res.samp_t[i]
            qlen = int(np.sum(waited & (res.att_t <= t) & (st_t > t)))
            if qlen != res.samp_q[i]:
                bad.append(f"очередь в t={t}: пересчёт {qlen} != {res.samp_q[i]}")
                break
    if len(res.samp_q) and res.samp_q.max() > cfg.K:
        bad.append("очередь > K")
    # 7. цепочка повторов
    order_a = np.lexsort((res.att_no, res.att_req))
    ar, an, at, ast = res.att_req[order_a], res.att_no[order_a], res.att_t[order_a], res.att_state[order_a]
    if cfg.R in ("R1", "R2", "R3"):
        maxa = cfg.retry.max_retries + 1
    elif cfg.R == "R0":
        maxa = 1
    else:
        maxa = 10**9
    if n_att and np.bincount(ar).max() > maxa:
        bad.append("попыток больше допустимого")
    same = ar[1:] == ar[:-1]
    if np.any(an[1:][same] != an[:-1][same] + 1):
        bad.append("нумерация попыток")
    if n_att and np.any(an[np.concatenate(([True], ~same))] != 0):
        bad.append("первая попытка не №0")
    prev_state = ast[:-1][same]
    if np.any(prev_state == OK):
        bad.append("повтор после успешной попытки")
    # момент неудачи предыдущей попытки
    t_fail = np.where(prev_state == REJECTED, at[:-1][same], at[:-1][same] + cfg.T)
    delay = at[1:][same] - t_fail
    if np.any(delay < -EPS):
        bad.append("повтор раньше момента неудачи")
    nxt_no = an[1:][same]
    if cfg.R == "R1" and delay.size and np.any(np.abs(delay - cfg.retry.fixed_delay) > 1e-7):
        bad.append("R1: задержка != d")
    if cfg.R in ("R2", "R3") and delay.size:
        ub = np.minimum(cfg.retry.cap, cfg.retry.base * cfg.retry.multiplier ** (nxt_no - 1.0))
        if np.any(delay > ub + 1e-7):
            bad.append("R2/R3: задержка выше границы джиттера")
    # финальная неудача: время исхода
    last = np.concatenate((~same, [True])) if n_att else np.array([], bool)
    if n_att:
        lr, lt, ls = ar[last], at[last], ast[last]
        fr = st[lr]
        m = fr == FAIL_TIMEOUT
        if np.any(np.abs(res.req_res_t[lr][m] - (lt[m] + cfg.T)) > 1e-7):
            bad.append("время FAIL_TIMEOUT != t_attempt+T")
        m = fr == FAIL_REJECT
        if np.any(np.abs(res.req_res_t[lr][m] - lt[m]) > 1e-7) or np.any(ls[m] != REJECTED):
            bad.append("FAIL_REJECT")
        # успех: последняя попытка OK, и клиентская задержка = end - первичный приход
        m = fr == SUCCESS
        if np.any(ls[m] != OK):
            bad.append("успешный запрос: последняя попытка не OK")
        if np.any(np.abs(res.req_lat[lr][m] - (end_of[order_a][last][m] - res.req_t[lr][m])) > 1e-7):
            bad.append("задержка успеха != end - первичный приход")
        if np.any(res.req_lat[lr][m] < 0):
            bad.append("отрицательная задержка")
    # 8. R3: скользящий бюджет, пересчёт по моментам решения
    if cfg.R == "R3" and delay.size:
        dec_t = t_fail  # момент решения о выдаче каждого повтора
        prim = res.req_t
        o = np.argsort(dec_t, kind="stable")
        ds = dec_t[o]
        w = cfg.retry.budget_window
        for j in range(len(ds)):
            f = ds[j]
            n_ret = int(np.sum((ds > f - w) & (ds <= f)))  # включая текущий (допущение: включительно)
            n_pr = int(np.sum((prim > f - w) & (prim <= f)))
            if n_ret > cfg.retry.budget_reserve + cfg.retry.budget_ratio * n_pr + 1e-6:
                # равные моменты решения допускают погрешность порядка; фиксируем нарушение
                bad.append(f"R3: бюджет превышен в t={f:.3f}: {n_ret} > {10 + 0.1 * n_pr}")
                break
    return bad


def audit_order(disc: str, seed: int) -> list[str]:
    """Порядок выбора из очереди: перебор (FIFO: старейшая, LIFO: новейшая, ALIFO: по порогу)."""
    cfg = Config(c=2, K=20, T=5.0, R="R1", Q=disc, D="off", S="Exp")
    res = simulate(cfg, [(300.0, 1.3)], 300.0, seed)
    sid = res.starts[:, 0].astype(np.int64)
    st_t = np.full(len(res.att_t), math.inf)
    st_t[sid] = res.starts[:, 1]
    waited = (res.att_state != REJECTED) & (st_t > res.att_t + EPS)
    bad = []
    nq = 0
    maxq = 0
    for x, t in zip(sid, res.starts[:, 1]):
        if not waited[x]:
            continue
        inq = np.flatnonzero(waited & (res.att_t <= t + 1e-12) & (st_t >= t - 1e-12))
        maxq = max(maxq, len(inq))
        if disc == "FIFO":
            want = inq.min()
        elif disc == "LIFO":
            want = inq.max()
        else:
            want = inq.max() if len(inq) > cfg.theta_q * cfg.K else inq.min()
        nq += 1
        if want != x:
            bad.append(f"{disc}: t={t:.3f} выбран {x}, ожидался {want}, |q|={len(inq)}")
            break
    if disc == "ALIFO" and maxq <= cfg.theta_q * cfg.K:
        bad.append("тест ALIFO вырожден: очередь не превысила порог")
    return bad if nq > 50 else bad + ["слишком мало событий"]


def power_selftest() -> dict[str, bool]:
    """Проверка чувствительности аудита: внесённые искажения должны быть обнаружены."""
    import copy

    cfg = Config(R="R3", Q="FIFO", D="off")
    plan = plan_for(cfg, "A", 0.9)
    res = simulate(cfg, plan.schedule, plan.t_stop, 43000)
    r: dict[str, bool] = {"baseline_clean": not audit(cfg, res)}
    x = copy.deepcopy(res)
    x.att_state[np.flatnonzero(x.att_state == OK)[0]] = LATE
    r["detects_state_flip"] = bool(audit(cfg, x))
    x = copy.deepcopy(res)
    x.starts[5, 2] += 3.0
    r["detects_changed_service"] = bool(audit(cfg, x))
    x = copy.deepcopy(res)
    x.att_t[np.flatnonzero(x.att_no == 1)[0]] += 0.5
    r["detects_shifted_retry"] = bool(audit(cfg, x))
    x = copy.deepcopy(res)
    x.samp_q[-1] += 1
    r["detects_queue_sample"] = bool(audit(cfg, x))
    # дисциплина: LIFO-прогон, проверенный по правилу FIFO, должен быть отвергнут
    cfg2 = Config(c=2, K=20, T=5.0, R="R1", Q="LIFO", D="off")
    res2 = simulate(cfg2, [(300.0, 1.3)], 300.0, 41000)
    sid = res2.starts[:, 0].astype(np.int64)
    st_t = np.full(len(res2.att_t), math.inf)
    st_t[sid] = res2.starts[:, 1]
    waited = (res2.att_state != REJECTED) & (st_t > res2.att_t + EPS)
    wrong = 0
    for xx, t in zip(sid, res2.starts[:, 1]):
        if waited[xx]:
            inq = np.flatnonzero(waited & (res2.att_t <= t + 1e-12) & (st_t >= t - 1e-12))
            wrong += int(inq.min() != xx)
    r["order_check_detects_wrong_discipline"] = wrong > 0
    return r


def main() -> None:
    out: dict = {}
    levels = dict(R=["R0", "R1", "R2", "R3"], Q=["FIFO", "LIFO", "ALIFO"], D=["off", "on"], S=["Exp", "LogN", "Pareto"])
    n_runs = 0
    viol: list[str] = []
    seed = 40000
    for r, q, d, s in itertools.product(*levels.values()):
        cfg = Config(R=r, Q=q, D=d, S=s)
        for proto, rho in (("A", 0.6), ("A", 0.95), ("B", 0.6), ("B", 0.95), ("B", 0.3)):
            plan = plan_for(cfg, proto, rho)
            res = simulate(cfg, plan.schedule, plan.t_stop, seed)
            b = audit(cfg, res)
            row = run_protocol(cfg, proto, rho, seed, check=True)  # встроенная проверка тоже
            n_runs += 1
            if b:
                viol.append(f"{r}/{q}/{d}/{s} {proto} rho={rho} seed={seed}: {b[:3]}")
            seed += 1
    out["n_runs_audited"] = n_runs
    out["violations"] = viol
    # порядок дисциплин
    order_bad = {}
    for disc in ("FIFO", "LIFO", "ALIFO"):
        order_bad[disc] = audit_order(disc, 41000)
    out["discipline_order_violations"] = order_bad
    # Пуассон: число первичных в сегментах (pooled z)
    zs = []
    cfg = Config(R="R1")
    cnt, exp_ = 0, 0.0
    for sd_ in range(42000, 42040):
        plan = plan_for(cfg, "A", 0.7)
        res = simulate(cfg, plan.schedule, plan.t_stop, sd_)
        t = res.req_t
        # сегмент триггера [200, 250): ожидание 0.7*4*8*50 = 1120
        cnt += int(np.sum((t >= 200) & (t < 250)))
        exp_ += 0.7 * 4 * 8 * 50
    out["poisson_trigger_segment"] = {"observed": cnt, "expected": exp_, "z": (cnt - exp_) / math.sqrt(exp_)}
    # R3 нагрузочный прогон с отказами бюджета (проверка, что аудит не пустой)
    cfg = Config(R="R3")
    plan = plan_for(cfg, "A", 0.9)
    res = simulate(cfg, plan.schedule, plan.t_stop, 42100)
    out["r3_budget_denied_example"] = res.budget_denied
    out["r3_budget_granted_example"] = res.budget_granted
    out["power_selftest"] = power_selftest()
    Path(ROOT / "reports" / "validation_data").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "validation_data" / "v4.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
