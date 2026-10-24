"""Основной анализ, п. 1-2 плана: ρ_fold, ρ_release, Δ по 72 конфигурациям, классификация N1/N2, контроль R0,
вспомогательная область ρ > 1 (Δ_ext).

python analysis/main_thresholds.py
Выход (results/analysis/): t01_thresholds72.csv (идентификатор T01), t01_boot.npz (bootstrap-выборки порогов, B = 2000),
t02_ext72.csv (T02, Δ_ext), t02_boot.npz, t03_curves_base.csv (P_A, P_B, g по ρ; для рисунков), t03b_r0_control.csv.
Правое цензурирование: порог выше сетки = 1,0 (ρ ≤ 1) либо 2,5 (Δ_ext), помечается; значение 1,0 — не оценка, а граница.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_common import (  # noqa: E402
    BASE_TAGS, BOOT, DELTA_MIN, EXT_TAGS, OUT, THETA, base_meta, load_main, norm_failed, thr_boot,
)


def classify(fold_c: str, rel_c: str, d_lo: float) -> str:
    if fold_c == "right" and rel_c == "right":
        return "нет отказов на сетке"
    if fold_c == "right":
        return "A не отказывает, B отказывает"
    if rel_c == "left":
        return "B отказывает на всей сетке"
    if d_lo > 0:
        return "метастабильность (N1)"
    return "нет гистерезиса"


def work(args: tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]) -> dict:
    h, ra, fa, rb, fb, hi, base = args
    a = thr_boot(ra, fa, hi, h, base + 1)
    b = thr_boot(rb, fb, hi, h, base + 2)
    d = a["draws"] - b["draws"]
    lo, up = np.percentile(d, [2.5, 97.5])
    delta = a["value"] - b["value"]
    out = {
        "config_hash": h,
        "n_rho_A": a["n_rho"], "n_rho_B": b["n_rho"], "reps_min": min(a["n_min"], b["n_min"]), "reps_max": max(a["n_max"], b["n_max"]),
        "fold": a["value"], "fold_lo": a["lo"], "fold_hi": a["hi"], "fold_cens": a["cens"], "fold_method": a["method"],
        "rel": b["value"], "rel_lo": b["lo"], "rel_hi": b["hi"], "rel_cens": b["cens"], "rel_method": b["method"],
        "delta": delta, "d_lo": float(lo), "d_hi": float(up),
        "_fold_draws": a["draws"], "_rel_draws": b["draws"],
    }
    out["cls"] = classify(a["cens"], b["cens"], float(lo))
    out["N1"] = bool(a["cens"] != "right" and lo > 0)
    out["N2"] = bool(a["cens"] != "right" and delta >= DELTA_MIN)
    if a["cens"] == "right" and b["cens"] == "right":
        out["delta_status"] = "оба порога цензурированы справа: Δ = 0 по соглашению §6, не оценка"
    elif a["cens"] == "right":
        out["delta_status"] = "ρ_fold цензурирован справа: Δ — нижняя оценка"
    elif b["cens"] == "left":
        out["delta_status"] = "ρ_release цензурирован слева: Δ — нижняя оценка"
    else:
        out["delta_status"] = "оценка"
    return out


def pack(rows: list[dict], meta: pd.DataFrame, prefix: str) -> pd.DataFrame:
    fd = np.stack([r.pop("_fold_draws") for r in rows])
    rd = np.stack([r.pop("_rel_draws") for r in rows])
    df = meta.merge(pd.DataFrame(rows), on="config_hash")
    order = {h: i for i, h in enumerate(meta["config_hash"])}
    idx = np.argsort([order[r["config_hash"]] for r in rows])
    np.savez_compressed(OUT / f"{prefix}_boot.npz", config_hash=np.array([rows[i]["config_hash"] for i in idx]),
                        fold=fd[idx], rel=rd[idx])
    return df


def main() -> int:
    meta = base_meta()
    cols = ["config_hash", "rho_test", "protocol", "failed", "g_final", "replicate", "tag"]
    res = load_main(cols, BASE_TAGS + EXT_TAGS)
    res = res[res["config_hash"].isin(meta["config_hash"])]
    low = res[(res["tag"].isin(BASE_TAGS)) & (res["rho_test"] <= 1.0 + 1e-9)]
    jobs = []
    for i, h in enumerate(meta["config_hash"]):
        d = low[low["config_hash"] == h]
        a, b = d[d["protocol"] == "A"], d[d["protocol"] == "B"]
        jobs.append((h, a["rho_test"].to_numpy(), a["failed"].to_numpy().astype(int), b["rho_test"].to_numpy(),
                     b["failed"].to_numpy().astype(int), 1.0, 10))
    with mp.Pool(7) as pool:
        rows = pool.map(work, jobs, chunksize=2)
    df = pack(rows, meta, "t01")
    df.to_csv(OUT / "t01_thresholds72.csv", index=False, encoding="utf-8")
    print(f"T01: {len(df)} конфигураций, B = {BOOT}; N1 = {int(df.N1.sum())}, N2 = {int(df.N2.sum())}")
    print(df.cls.value_counts().to_string())

    # контроль R0
    r0 = df[df["R"] == "R0"][["R", "Q", "D", "S", "fold", "fold_cens", "rel", "rel_cens", "delta", "d_lo", "d_hi", "cls", "delta_status"]]
    r0.to_csv(OUT / "t03b_r0_control.csv", index=False, encoding="utf-8")
    print("R0: Δ min %.4f max %.4f; d_lo>0: %d; d_hi max %.4f" % (r0.delta.min(), r0.delta.max(), int((r0.d_lo > 0).sum()), r0.d_hi.max()))

    # кривые P_A, P_B, g по ρ (грубая и уточнённая сетки; все реплики)
    full = res[(res["tag"].isin(BASE_TAGS))].copy()
    cur = (full.groupby(["config_hash", "protocol", "rho_test"])
           .agg(n=("failed", "size"), P=("failed", "mean"), g=("g_final", "mean"), g_sd=("g_final", "std")).reset_index())
    cur = meta[["config_hash", "R", "Q", "D", "S"]].merge(cur, on="config_hash")
    cur["P_se"] = np.sqrt(cur["P"] * (1 - cur["P"]) / cur["n"])
    cur.to_csv(OUT / "t03_curves_base.csv", index=False, encoding="utf-8")

    # вспомогательная область ρ > 1: критерий g < θ·min(1,1/ρ), порог в [0,2; 2,5], правая граница 2,5
    allr = res[res["tag"].isin(BASE_TAGS + EXT_TAGS)].copy()
    allr["failed_norm"] = norm_failed(allr["g_final"].to_numpy(), allr["rho_test"].to_numpy(), THETA)
    jobs = []
    for h in meta["config_hash"]:
        d = allr[allr["config_hash"] == h]
        a, b = d[d["protocol"] == "A"], d[d["protocol"] == "B"]
        jobs.append((h, a["rho_test"].to_numpy(), a["failed_norm"].to_numpy(), b["rho_test"].to_numpy(), b["failed_norm"].to_numpy(), 2.5, 20))
    with mp.Pool(7) as pool:
        rows = pool.map(work, jobs, chunksize=2)
    dfe = pack(rows, meta, "t02")
    # N1 для Δ_ext: ρ_fold < 2,5 и нижняя граница ДИ > 0 (как N1)
    dfe.to_csv(OUT / "t02_ext72.csv", index=False, encoding="utf-8")
    print(f"T02 (Δ_ext, ρ ≤ 2,5): N1_ext = {int(dfe.N1.sum())}, N2_ext = {int(dfe.N2.sum())}; ρ_fold цензурирован справа: {int((dfe.fold_cens == 'right').sum())}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
