"""Общие функции анализа основного прогона (results/raw, seed >= 1000000).

Анализ строго по docs/PREREGISTRATION.md (тег prereg-v1). Данные pilot (seed < 1000000) не читаются:
загрузчик проверяет диапазон seed. Идентификаторы таблиц: results/analysis/<id>_*.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.runner import load_results, load_tasks, stage_dir  # noqa: E402
from src.thresholds import aggregate, bootstrap_thresholds, point_threshold  # noqa: E402

OUT = ROOT / "results" / "analysis"
FIG = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

MAIN_SEED_MIN = 1_000_000
BOOT = 2000
BOOT_SEED = 20261024  # базовая энтропия bootstrap; seed конкретной оценки = [BOOT_SEED, int(config_hash,16), код]
DELTA_MIN = 0.1
THETA = 0.5
RS = ["R0", "R1", "R2", "R3"]
QS = ["FIFO", "LIFO", "ALIFO"]
DS = ["off", "on"]
SS = ["Exp", "LogN", "Pareto"]
GRID = [round(0.2 + 0.05 * i, 6) for i in range(17)]  # грубая сетка 0,20 … 1,00
EXT_GRID = [1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
BASE_TAGS = ["m_scr", "m_ref1", "m_r100"]
EXT_TAGS = ["m_ext"]
SENS_TAGS = ["m_sens", "m_sens_ref"]
ACTIVE12 = [(r, "FIFO", d, s) for r in ("R1", "R2") for d in DS for s in SS]
VARIANTS = ["theta03", "theta07", "T5", "T20", "K64", "K256", "hrec3000", "hrec6000", "trg125", "trg175"]

RU_S = {"Exp": "Exp (cv = 1)", "LogN": "LogN (cv = 2)", "Pareto": "Pareto (α = 2,5)"}


def fmt(x: float, nd: int = 3) -> str:
    """Число с десятичной запятой."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "н/д"
    return f"{x:.{nd}f}".replace(".", ",")


def ci_str(v: float, lo: float, hi: float, nd: int = 3) -> str:
    return f"{fmt(v, nd)} [{fmt(lo, nd)}; {fmt(hi, nd)}]"


def load_main(columns: list[str] | None = None, tags: list[str] | None = None) -> pd.DataFrame:
    """Результаты основного прогона из results/raw. Проверка: только seed >= 1000000."""
    cols = list(columns) if columns else None
    if cols is not None:
        cols = list(dict.fromkeys(cols + ["seed", "tag"]))
    res = load_results(stage_dir("main"), cols)
    if int(res["seed"].min()) < MAIN_SEED_MIN:
        raise RuntimeError("в данных есть seed < 1000000: данные pilot не допускаются в основной анализ")
    if tags is not None:
        res = res[res["tag"].isin(tags)]
    res = res.copy()
    if "rho_test" in res.columns:
        res["rho_test"] = res["rho_test"].round(6)
    return res.reset_index(drop=True)


def cfg_meta() -> pd.DataFrame:
    """config_hash -> факторы, параметры, метка варианта чувствительности ('base' или имя варианта)."""
    tk = load_tasks(stage_dir("main"))
    tk = tk.drop_duplicates("config_hash")
    rows = []
    for h, js in zip(tk["config_hash"], tk["cfg_json"], strict=True):
        c = json.loads(js)
        p = c["proto"]
        rows.append({"config_hash": h, "R": c["R"], "Q": c["Q"], "D": c["D"], "S": c["S"], "T": float(c["T"]), "K": int(c["K"]),
                     "theta": float(p["theta"]), "trig": float(p["trigger_mult"]), "trig_len": float(p["trigger_len"]),
                     "h_rec": float(p["h_rec"]), "ol_len": float(p["overload_len"]), "ol_rho": float(p["overload_rho"]),
                     "theta_q": float(c["theta_q"]), "c": int(c["c"])})
    m = pd.DataFrame(rows)

    def variant(r: Any) -> str:
        dev = []
        if r.theta != 0.5:
            dev.append({0.3: "theta03", 0.7: "theta07"}[round(r.theta, 2)])
        if r["T"] != 10.0:
            dev.append({5.0: "T5", 20.0: "T20"}[r["T"]])
        if r.K != 128:
            dev.append({64: "K64", 256: "K256"}[r.K])
        if r.h_rec != 1500.0:
            dev.append({3000.0: "hrec3000", 6000.0: "hrec6000"}[r.h_rec])
        if r.trig != 1.5:
            dev.append({1.25: "trg125", 1.75: "trg175"}[round(r.trig, 2)])
        if len(dev) > 1:
            raise RuntimeError(f"более одного отклонения от базы: {dev}")
        return dev[0] if dev else "base"

    m["variant"] = [variant(r) for _, r in m.iterrows()]
    return m.sort_values(["variant", "R", "Q", "D", "S"]).reset_index(drop=True)


def base_meta() -> pd.DataFrame:
    m = cfg_meta()
    b = m[m["variant"] == "base"].copy()
    b["R"] = pd.Categorical(b["R"], RS, ordered=True)
    b["Q"] = pd.Categorical(b["Q"], QS, ordered=True)
    b["D"] = pd.Categorical(b["D"], DS, ordered=True)
    b["S"] = pd.Categorical(b["S"], SS, ordered=True)
    b = b.sort_values(["R", "Q", "D", "S"]).reset_index(drop=True)
    for c in ("R", "Q", "D", "S"):
        b[c] = b[c].astype(str)
    if len(b) != 72:
        raise RuntimeError(f"ожидалось 72 базовые конфигурации, найдено {len(b)}")
    return b


def rng_for(h: str, code: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([BOOT_SEED, int(h, 16) % (2**63), code]))


def thr_boot(rho: np.ndarray, failed: np.ndarray, right_bound: float, h: str, code: int, n_boot: int = BOOT) -> dict[str, Any]:
    """Порог P(ρ) = 0,5: точечная оценка (логистика/изотоника, D-011), bootstrap по репликам, цензурирование."""
    x, n, k = aggregate(rho, failed)
    val, cens, method = point_threshold(x, n, k, 0.5, right_bound)
    draws = bootstrap_thresholds(x, n, k, rng_for(h, code), n_boot, 0.5, right_bound)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"value": float(val), "cens": cens, "method": method, "lo": float(lo), "hi": float(hi), "draws": draws,
            "n_rho": int(len(x)), "n_min": int(n.min()), "n_max": int(n.max())}


def norm_failed(g: np.ndarray, rho: np.ndarray, theta: float = THETA) -> np.ndarray:
    """Критерий отказа PREREGISTRATION: g < θ·min(1, 1/ρ)."""
    return (np.asarray(g) < theta * np.minimum(1.0, 1.0 / np.asarray(rho))).astype(int)


def holm(p: np.ndarray) -> np.ndarray:
    """Поправка Холма (скорректированные p-значения)."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def boot_p(draws: np.ndarray, null: float = 0.0) -> float:
    """Двусторонний bootstrap p-value для H0: эффект = null (по перцентильному распределению); нижняя граница 1/(B+1)."""
    d = np.asarray(draws)
    b = len(d)
    p_le = (np.sum(d <= null) + 1) / (b + 1)
    p_ge = (np.sum(d >= null) + 1) / (b + 1)
    return float(min(1.0, 2 * min(p_le, p_ge)))


def pct(d: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def md_table(df: pd.DataFrame, floatfmt: dict[str, int] | None = None, default_nd: int = 3) -> str:
    """Markdown-таблица с десятичной запятой."""
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(fmt(float(v), floatfmt.get(c, default_nd)))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def save_fig(fig: Any, name: str, data: pd.DataFrame) -> None:
    """png + svg + csv с данными рядом (figures/)."""
    fig.savefig(FIG / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    data.to_csv(FIG / f"{name}.csv", index=False, encoding="utf-8")
