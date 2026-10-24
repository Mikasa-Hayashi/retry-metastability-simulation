"""Метрики по SimResult (MODEL_SPEC §5, D-008) и проверка инварианта V4."""

from __future__ import annotations

import math

import numpy as np

from .model import (
    DROPPED,
    FAIL_REJECT,
    FAIL_TIMEOUT,
    LATE,
    OK,
    PENDING,
    QUEUED,
    REJECTED,
    SERVING,
    SUCCESS,
    InvariantError,
    SimResult,
)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise InvariantError(msg)


def check_v4(res: SimResult) -> dict[str, int]:
    """Баланс (V4): первичные = успехи + неудачи(таймаут) + неудачи(отказ) + в полёте.

    Дополнительно: сохранение попыток, уникальность id, отсутствие отрицательных времён.
    Возвращает счётчики. Нарушение — InvariantError.
    """
    n_req = len(res.req_t)
    st = res.req_status
    n_succ = int(np.sum(st == SUCCESS))
    n_ft = int(np.sum(st == FAIL_TIMEOUT))
    n_fr = int(np.sum(st == FAIL_REJECT))
    n_pend = int(np.sum(st == PENDING))
    _require(n_req == n_succ + n_ft + n_fr + n_pend, "V4: баланс запросов")

    n_att = len(res.att_req)
    _require(len(res.att_no) == n_att == len(res.att_state) == len(res.att_t), "V4: длины попыток")
    _require(int(res.req_att.sum()) == n_att, "V4: сумма попыток по запросам != числу попыток")
    # id попыток уникальны по построению (индекс); id запросов, на которые ссылаются, валидны
    if n_att:
        _require(int(res.att_req.min()) >= 0 and int(res.att_req.max()) < n_req, "V4: id запроса")
        counts = np.bincount(res.att_req, minlength=n_req)
        _require(bool(np.all(counts == res.req_att)), "V4: число попыток запроса")
        # номера попыток запроса 0..k-1 без пропусков: сумма и максимум
        mx = np.zeros(n_req, dtype=np.int64)
        np.maximum.at(mx, res.att_req, res.att_no)
        _require(bool(np.all(mx + 1 == res.req_att)), "V4: нумерация попыток")
    s = res.att_state
    c_ok = int(np.sum(s == OK))
    c_late = int(np.sum(s == LATE))
    c_drop = int(np.sum(s == DROPPED))
    c_rej = int(np.sum(s == REJECTED))
    c_q = int(np.sum(s == QUEUED))
    c_srv = int(np.sum(s == SERVING))
    _require(n_att == c_ok + c_late + c_drop + c_rej + c_q + c_srv, "V4: состояния попыток")
    _require(c_ok == n_succ, "V4: успешные попытки != успешные запросы")
    _require(c_rej == res.n_rejected and c_drop == res.n_dropped, "V4: счётчики отказов")
    _require(c_q == res.q_end and c_srv == res.busy_end, "V4: очередь/воркеры на конец")
    _require(len(res.starts) == c_ok + c_late + c_srv, "V4: число стартов")
    _require(len(res.done) == c_ok + c_late, "V4: число завершений")
    _require(res.orbit_end <= n_pend, "V4: ожидающие повтора должны быть в полёте")

    # времена
    resolved = st != PENDING
    _require(
        bool(np.all(res.req_res_t[resolved] >= res.req_t[resolved])), "V4: время исхода < прихода"
    )
    _require(not bool(np.any(np.isnan(res.req_res_t[resolved]))), "V4: NaN во времени исхода")
    succ = st == SUCCESS
    _require(bool(np.all(res.req_lat[succ] >= 0.0)), "V4: отрицательная задержка")
    if len(res.starts):
        sid = res.starts[:, 0].astype(np.int64)
        _require(bool(np.all(res.starts[:, 1] >= res.att_t[sid])), "V4: старт раньше прихода")
        _require(bool(np.all(res.starts[:, 2] > 0.0)), "V4: неположительное время обслуживания")
        _require(len(np.unique(sid)) == len(sid), "V4: попытка стартовала дважды")
    if len(res.done):
        _require(bool(np.all(np.diff(res.done[:, 1]) >= 0.0)), "V4: завершения не упорядочены")
    _require(bool(np.all(np.diff(res.req_t) >= 0.0)), "V4: приходы не упорядочены")
    _require(bool(np.all(np.diff(res.samp_t) > 0.0)), "V4: сэмплы не упорядочены")
    return {
        "n_primary": n_req,
        "n_success": n_succ,
        "n_fail_timeout": n_ft,
        "n_fail_reject": n_fr,
        "n_inflight": n_pend,
        "n_attempts": n_att,
    }


def window_mask(res: SimResult, lo: float, hi: float) -> np.ndarray:
    return (res.req_t >= lo) & (res.req_t < hi)


def goodput(res: SimResult, lo: float, hi: float, allow_pending: bool = False) -> float:
    """g на [lo, hi): доля первичных запросов, пришедших в окне, получивших успех."""
    m = window_mask(res, lo, hi)
    n = int(m.sum())
    if n == 0:
        return math.nan
    st = res.req_status[m]
    if not allow_pending and bool(np.any(st == PENDING)):
        raise InvariantError("в окне есть неразрешённые запросы: увеличьте tail")
    return float(np.mean(st == SUCCESS))


def goodput_series(res: SimResult, window: float, grid: np.ndarray) -> np.ndarray:
    """g(s) на точках grid: запросы с приходом в [s - window, s); NaN, если их нет."""
    succ = np.concatenate(([0], np.cumsum(res.req_status == SUCCESS)))
    hi = np.searchsorted(res.req_t, grid, side="left")
    lo = np.searchsorted(res.req_t, grid - window, side="left")
    n = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        g = (succ[hi] - succ[lo]) / n
    g = np.where(n > 0, g, np.nan)
    return np.asarray(g, dtype=np.float64)


def recovery_time(
    res: SimResult,
    s0: float,
    t_end: float,
    window: float = 100.0,
    level: float = 0.9,
    hold: float = 50.0,
    step: float = 1.0,
) -> tuple[float, bool]:
    """Время восстановления от s0 (D-008). Возвращает (время, цензурировано)."""
    grid = np.arange(s0, t_end + 1e-9, step)
    g = goodput_series(res, window, grid)
    ok = np.where(np.isnan(g), False, g >= level)
    m = round(hold / step)
    if len(ok) <= m:
        return float(t_end - s0), True
    bad = np.concatenate(([0], np.cumsum(~ok)))
    n_start = len(ok) - m
    window_bad = bad[m + 1 : m + 1 + n_start] - bad[:n_start]
    good_idx = np.flatnonzero(window_bad == 0)
    if len(good_idx) == 0:
        return float(t_end - s0), True
    return float(grid[good_idx[0]] - s0), False


def p99_latency(res: SimResult, lo: float, hi: float) -> float:
    m = window_mask(res, lo, hi) & (res.req_status == SUCCESS)
    if not bool(m.any()):
        return math.nan
    return float(np.percentile(res.req_lat[m], 99))


def amplification(res: SimResult, lo: float, hi: float) -> float:
    """Среднее число попыток на первичный запрос окна."""
    m = window_mask(res, lo, hi)
    if not bool(m.any()):
        return math.nan
    return float(res.req_att[m].mean())


def wasted_fraction(res: SimResult, lo: float, hi: float) -> float:
    """Время обслуживания попыток, завершившихся после срока, / всё время обслуживания (D-004)."""
    d = res.done
    if len(d) == 0:
        return math.nan
    m = (d[:, 1] >= lo) & (d[:, 1] < hi)
    tot = float(d[m, 2].sum())
    if tot <= 0.0:
        return math.nan
    return float(d[m & (d[:, 3] == 1), 2].sum() / tot)


def queue_stats(res: SimResult, lo: float, hi: float) -> tuple[float, float]:
    """Среднее и максимум длины очереди по сэмплам в [lo, hi)."""
    m = (res.samp_t >= lo) & (res.samp_t < hi)
    if not bool(m.any()):
        return math.nan, math.nan
    return float(res.samp_q[m].mean()), float(res.samp_q[m].max())
