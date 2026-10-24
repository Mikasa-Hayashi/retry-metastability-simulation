"""Pilot: диагностика слабых триггеров, контроль R0, H_rec, мощность, скорость. Читает results/pilot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runner import load_results, load_tasks, stage_dir
from src.thresholds import estimate_delta, estimate_threshold

CH = "148eb8d011551aa1"  # опорная конфигурация, триггер (1,5; 25)
NB = 2000
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
d = stage_dir("pilot")
res = load_results(d)
tk = load_tasks(d)


def hash_of(tag: str) -> list[str]:
    return list(tk[tk["tag"] == tag]["config_hash"].unique())


def thr(df: pd.DataFrame, seed: int = 1):
    return estimate_threshold(df["rho_test"].to_numpy(), df["failed"].to_numpy(), NB, seed)


def sel(h: str, proto: str, max_rep: int | None = None) -> pd.DataFrame:
    r = res[(res["config_hash"] == h) & (res["protocol"] == proto)]
    return r if max_rep is None else r[r["replicate"] < max_rep]


def fmt(t) -> str:
    return f"{t.value:.3f} [{t.lo:.3f}; {t.hi:.3f}] {t.censored}/{t.method}"


print("== P_A, P_B опорной конфигурации с выбранным триггером (40 реплик)")
pa = sel(CH, "A", 40).groupby("rho_test")["failed"].agg(["mean", "size"])
pb = sel(CH, "B", 40).groupby("rho_test")["failed"].agg(["mean", "size"])
print(pd.concat({"P_A": pa["mean"], "P_B": pb["mean"]}, axis=1).round(3).T.to_string())

print("\n== Диагностика слабее плана (эксплораторно)")
rb = sel(CH, "B", 40)
for h in hash_of("diag_A"):
    c = json.loads(tk[tk["config_hash"] == h]["cfg_json"].iloc[0])["proto"]
    ra = sel(h, "A")
    e = estimate_delta(ra["rho_test"].to_numpy(), ra["failed"].to_numpy(),
                       rb["rho_test"].to_numpy(), rb["failed"].to_numpy(), NB, 3)
    print(f"m_s={c['trigger_mult']} D_s={c['trigger_len']}: fold {fmt(e.fold)}; "
          f"delta {e.delta:.3f} [{e.lo:.3f}; {e.hi:.3f}]")
    print("  P_A:", ra.groupby("rho_test")["failed"].mean().round(2).to_dict())

print("\n== Контроль R0 (триггер 1,5; 25)")
h0 = hash_of("r0_ctrl")[0]
r0 = res[res["config_hash"] == h0]
r0c = r0[r0["replicate"] < 40]
tab = r0c.groupby(["protocol", "rho_test"])["failed"].agg(["sum", "size"]).unstack(0)
print(tab.to_string())
ra0, rb0 = r0c[r0c.protocol == "A"], r0c[r0c.protocol == "B"]
e0 = estimate_delta(ra0["rho_test"].to_numpy(), ra0["failed"].to_numpy(),
                    rb0["rho_test"].to_numpy(), rb0["failed"].to_numpy(), NB, 4)
print(f"R0: fold {fmt(e0.fold)}; release {fmt(e0.release)}; delta {e0.delta:.3f}")
print("failed всего:", int(r0c["failed"].sum()), "из", len(r0c))
print("min g_final:", r0c["g_final"].min(), " recovery_censored:", int(r0c["recovery_censored"].sum()))
print("-- шум R0 при rho = 0,95 (200 реплик):")
for p in ("A", "B"):
    x = r0[(r0.protocol == p) & (r0.rho_test == 0.95)]
    k, n = int(x["failed"].sum()), len(x)
    ph = k / n
    z = 1.96
    w = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    cen = (ph + z * z / (2 * n)) / (1 + z * z / n)
    print(f"  {p}: failed {k}/{n} = {ph:.3f}, Wilson [{cen - w:.3f}; {cen + w:.3f}], "
          f"min g_final {x['g_final'].min():.3f}, theta=0,7 доля {(x['g_final'] < 0.7).mean():.3f}")

print("\n== Чувствительность к H_rec (опорная, триггер 1,5; 25, 40 реплик)")
rows = []
for H, ta, tb in ((1500, None, None), (3000, "hrec3000_A", "hrec3000_B"), (6000, "hrec6000_A", "hrec6000_B")):
    if H == 1500:
        ra, rbb = sel(CH, "A", 40), sel(CH, "B", 40)
    else:
        ra = res[res["config_hash"] == hash_of(ta)[0]]
        ra = ra[ra.protocol == "A"]
        rbb = res[res["config_hash"] == hash_of(tb)[0]]
        rbb = rbb[rbb.protocol == "B"]
    e = estimate_delta(ra["rho_test"].to_numpy(), ra["failed"].to_numpy(),
                       rbb["rho_test"].to_numpy(), rbb["failed"].to_numpy(), NB, 5)
    print(f"H_rec={H}: fold {fmt(e.fold)}; release {fmt(e.release)}; "
          f"delta {e.delta:.3f} [{e.lo:.3f}; {e.hi:.3f}]")
    print("  P_B:", rbb.groupby("rho_test")["failed"].mean().round(2).to_dict())

print("\n== Мощность: полуширина 95% ДИ по подвыборкам реплик (опорная, триггер 1,5; 25)")
for R in (20, 30, 40, 60, 100):
    ra, rbb = sel(CH, "A", R), sel(CH, "B", R)
    e = estimate_delta(ra["rho_test"].to_numpy(), ra["failed"].to_numpy(),
                       rbb["rho_test"].to_numpy(), rbb["failed"].to_numpy(), NB, 6)
    hw = lambda t: (t.hi - t.lo) / 2
    print(f"R={R}: n_A={len(ra)} fold {e.fold.value:.3f} ±{hw(e.fold):.4f}; release "
          f"{e.release.value:.3f} ±{hw(e.release):.4f}; delta {e.delta:.3f} ±{(e.hi - e.lo) / 2:.4f}")

print("\n== Время прогона")
sp = res[res["tag"] == "speed72"]
print("speed72: n", len(sp), "mean runtime", round(sp["runtime"].mean(), 3), "max", round(sp["runtime"].max(), 3))
print(sp.groupby(["S", "D"])["runtime"].mean().round(3).to_string())
print(sp.groupby(["R", "protocol"])["runtime"].mean().round(3).to_string())
print("все прогоны pilot: n", len(res), "mean", round(res["runtime"].mean(), 3))
print(res.groupby(["R", "protocol"])["runtime"].mean().round(3).to_string())
