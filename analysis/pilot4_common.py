"""Общие функции анализа pilot, часть 4 (CALIBRATION_PLAN_4.md)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runner import ROOT, load_results, load_tasks, stage_dir
from src.thresholds import aggregate, estimate_delta, point_threshold

OUT = ROOT / "results" / "pilot"
CRN_LO, CRN_HI = 100_000, 109_999
INDEP_LO = 110_000
THETA_N = 0.5


def load_crn(columns: list[str] | None = None, tags: list[str] | None = None) -> pd.DataFrame:
    """Результаты части 4 из блока общих случайных чисел."""
    cols = list(columns) if columns else None
    need = ["seed", "tag"]
    if cols is not None:
        cols = list(dict.fromkeys(cols + need))
    res = load_results(stage_dir("pilot"), cols)
    res = res[(res["seed"] >= CRN_LO) & (res["seed"] <= CRN_HI)]
    if tags is not None:
        res = res[res["tag"].isin(tags)]
    return res.reset_index(drop=True)


def load_indep(columns: list[str] | None = None, tags: list[str] | None = None) -> pd.DataFrame:
    cols = list(columns) if columns else None
    if cols is not None:
        cols = list(dict.fromkeys(cols + ["seed", "tag"]))
    res = load_results(stage_dir("pilot"), cols)
    res = res[res["seed"] >= INDEP_LO]
    if tags is not None:
        res = res[res["tag"].isin(tags)]
    return res.reset_index(drop=True)


def cfg_meta() -> pd.DataFrame:
    """Таблица config_hash → факторы и параметры (по cfg_json задач части 4)."""
    tk = load_tasks(stage_dir("pilot"))
    tk = tk[tk["seed"] >= CRN_LO].drop_duplicates("config_hash")
    rows = []
    for h, js in zip(tk["config_hash"], tk["cfg_json"], strict=True):
        c = json.loads(js)
        rows.append(
            {
                "config_hash": h, "R": c["R"], "Q": c["Q"], "D": c["D"], "S": c["S"],
                "T": c["T"], "K": c["K"], "c": c["c"], "m": c["retry"]["max_retries"],
                "theta_q": c["theta_q"], "theta": c["proto"]["theta"],
                "trig": c["proto"]["trigger_mult"], "h_rec": c["proto"]["h_rec"],
                "ol_rho": c["proto"]["overload_rho"], "ol_len": c["proto"]["overload_len"],
            }
        )
    return pd.DataFrame(rows)


def norm_failed(g: np.ndarray, rho: np.ndarray, theta_n: float = THETA_N) -> np.ndarray:
    """Нормированный критерий отказа (эксплораторно): g·max(1, ρ) < θ_n."""
    return (np.asarray(g) * np.maximum(1.0, np.asarray(rho)) < theta_n).astype(int)


def classify(fold_c: str, rel_c: str, d_lo: float) -> str:
    if fold_c == "right" and rel_c == "right":
        return "no_failure"
    if fold_c == "right":
        return "A_never_fails_B_fails"
    if rel_c == "left":
        return "B_always_fails"
    if d_lo > 0:
        return "meta"
    return "no_hysteresis"


def delta_block(
    ra: np.ndarray, fa: np.ndarray, rb: np.ndarray, fb: np.ndarray, hi: float, nb: int, seed: int
) -> dict[str, Any]:
    if len(ra) == 0 or len(rb) == 0:
        return {}
    e = estimate_delta(ra, fa, rb, fb, nb, seed, right_bound=hi)
    return {
        "fold": e.fold.value, "fold_lo": e.fold.lo, "fold_hi": e.fold.hi, "fold_cens": e.fold.censored,
        "rel": e.release.value, "rel_lo": e.release.lo, "rel_hi": e.release.hi, "rel_cens": e.release.censored,
        "delta": e.delta, "d_lo": e.lo, "d_hi": e.hi,
        "cls": classify(e.fold.censored, e.release.censored, e.lo),
        "N1": bool(e.fold.censored != "right" and e.lo > 0),
        "N2": bool(e.fold.censored != "right" and e.delta >= 0.1),
    }


def point_thr(rho: np.ndarray, failed: np.ndarray, hi: float) -> tuple[float, str]:
    x, n, k = aggregate(rho, failed)
    v, c, _ = point_threshold(x, n, k, 0.5, right_bound=hi)
    return v, c


def p_table(rho: np.ndarray, failed: np.ndarray) -> pd.Series:
    return pd.Series(failed, index=np.round(rho, 6)).groupby(level=0).mean()
