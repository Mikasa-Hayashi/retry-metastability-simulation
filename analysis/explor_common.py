"""Общие функции эксплораторного анализа (results/explor, seed 1400000-1999999; DECISIONS D-021).

Всё, что считается здесь, эксплораторное и в подтверждающий анализ не входит. База для сравнения - results/raw
(только чтение, через analysis/main_common.py). Идентификаторы таблиц: results/analysis/explor_<id>.csv.
Пороги и bootstrap - те же методы, что в основном анализе (src.thresholds, main_common.thr_boot).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_common import (  # noqa: E402,F401
    BASE_TAGS, BOOT, DELTA_MIN, FIG, OUT, ROOT, SS, cfg_meta, fmt, load_main, md_table, pct, rng_for, save_fig, thr_boot,
)
from src.runner import load_results, load_tasks, stage_dir  # noqa: E402

EXPLOR_SEED_MIN = 1_400_000
EXPLOR_SEED_MAX = 1_999_999
BASE_REP = 40  # в базе для сравнения берутся реплики с номером < 40 (как в main_sensitivity)
K2_CELLS = [("R1", "FIFO", "off", "Exp"), ("R1", "FIFO", "on", "LogN"), ("R2", "FIFO", "on", "Exp")]
HRECS = [1500, 3000, 6000, 12000, 24000]
RU_CELL = {("R1", "FIFO", "off", "Exp"): "R1/FIFO/выкл/Exp", ("R1", "FIFO", "on", "LogN"): "R1/FIFO/вкл/LogN",
           ("R2", "FIFO", "on", "Exp"): "R2/FIFO/вкл/Exp"}
MARK = "эксплораторно"


def load_explor(columns: list[str] | None = None, tags: list[str] | None = None) -> pd.DataFrame:
    cols = list(columns) if columns else None
    if cols is not None:
        cols = list(dict.fromkeys(cols + ["seed", "tag"]))
    res = load_results(stage_dir("explor"), cols)
    if int(res["seed"].min()) < EXPLOR_SEED_MIN or int(res["seed"].max()) > EXPLOR_SEED_MAX:
        raise RuntimeError("seed вне диапазона explor 1400000-1999999")
    if tags is not None:
        res = res[res["tag"].isin(tags)]
    res = res.copy()
    res["rho_test"] = res["rho_test"].round(6)
    return res.reset_index(drop=True)


def explor_meta() -> pd.DataFrame:
    """config_hash -> факторы и параметры, меняемые в explor."""
    tk = load_tasks(stage_dir("explor")).drop_duplicates("config_hash")
    rows = []
    for h, js in zip(tk["config_hash"], tk["cfg_json"], strict=True):
        c = json.loads(js)
        p, rt = c["proto"], c["retry"]
        rows.append({"config_hash": h, "R": c["R"], "Q": c["Q"], "D": c["D"], "S": c["S"], "h_rec": float(p["h_rec"]),
                     "trig": float(p["trigger_mult"]), "trig_len": float(p["trigger_len"]), "ol_len": float(p["overload_len"]),
                     "rbase": float(rt["base"]), "budget": float(rt["budget_ratio"])})
    return pd.DataFrame(rows)


def base_cfg(r: str, q: str, d: str, s: str) -> str:
    m = cfg_meta()
    sub = m[(m.variant == "base") & (m.R == r) & (m.Q == q) & (m.D == d) & (m.S == s)]
    if len(sub) != 1:
        raise RuntimeError(f"база не найдена: {r}/{q}/{d}/{s}")
    return str(sub.config_hash.iloc[0])


def base_data(cols: list[str], max_rep: int | None = BASE_REP) -> pd.DataFrame:
    res = load_main(cols + ["replicate", "tag"], BASE_TAGS)
    res = res[res["rho_test"] <= 1.0 + 1e-9]
    if max_rep is not None:
        res = res[res["replicate"] < max_rep]
    return res


def thr_of(d: pd.DataFrame, protocol: str, h: str, code: int, rcol: str = "failed") -> dict[str, Any] | None:
    x = d[d["protocol"] == protocol]
    if len(x) == 0:
        return None
    return thr_boot(x["rho_test"].to_numpy(), x[rcol].to_numpy().astype(int), 1.0, h, code)


def delta_row(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Δ = ρ_fold - ρ_release с bootstrap-ДИ и пометками цензурирования."""
    dr = a["draws"] - b["draws"]
    lo, hi = pct(dr)
    delta = a["value"] - b["value"]
    return {"fold": a["value"], "fold_lo": a["lo"], "fold_hi": a["hi"], "fold_cens": a["cens"],
            "rel": b["value"], "rel_lo": b["lo"], "rel_hi": b["hi"], "rel_cens": b["cens"],
            "delta": delta, "d_lo": lo, "d_hi": hi,
            "N1": bool(a["cens"] != "right" and lo > 0), "N2": bool(a["cens"] != "right" and delta >= DELTA_MIN)}


def l_a(d: pd.DataFrame, h: str, code: int, protocol: str = "A") -> dict[str, float]:
    """L_A = ∫ (min(1,1/ρ) - ḡ_A(ρ)) dρ по грубой сетке 0,20…1,00 (трапеции), bootstrap по репликам внутри точки."""
    grid = np.array([round(0.2 + 0.05 * i, 6) for i in range(17)])
    gstar = np.minimum(1.0, 1.0 / grid)
    x = d[d["protocol"] == protocol]
    rng = rng_for(h, code)
    gm = np.empty(len(grid))
    gd = np.empty((BOOT, len(grid)))
    for i, rho in enumerate(grid):
        v = x[np.isclose(x["rho_test"], rho)]["g_final"].to_numpy()
        if len(v) == 0:
            raise RuntimeError(f"нет точки ρ = {rho} для {h}")
        gm[i] = v.mean()
        gd[:, i] = v[rng.integers(0, len(v), (BOOT, len(v)))].mean(axis=1)
    l_hat = float(np.trapezoid(gstar - gm, grid))
    dr = np.trapezoid(gstar[None, :] - gd, grid, axis=1)
    lo, hi = pct(dr)
    return {"L": l_hat, "L_lo": lo, "L_hi": hi, "L_draws": dr}
