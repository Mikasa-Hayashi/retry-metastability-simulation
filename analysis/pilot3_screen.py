"""Pilot, часть 3, п. A: скрининг 72 конфигураций (CALIBRATION_PLAN_3.md).

python analysis/pilot3_screen.py
Реплики < R_SCR (8) на точку для всех конфигураций (одинаковый вес данных). Вывод: results/pilot/pilot3_screen.csv|txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config_hash
from src.runner import ROOT, configs_from_spec, load_results, load_tasks, stage_dir
from src.thresholds import estimate_delta

R_SCR = 8
NB = 2000
OUT = ROOT / "results" / "pilot"
LINES: list[str] = []


def emit(s: str = "") -> None:
    print(s)
    LINES.append(s)


def classify(r: pd.Series) -> str:
    if r["fold_cens"] == "right" and r["rel_cens"] == "right":
        return "no_failure"
    if r["fold_cens"] == "right":
        return "A_never_fails_B_fails"
    if r["rel_cens"] == "left":
        return "B_always_fails"
    if r["d_lo"] > 0:
        return "meta"
    return "no_hysteresis"


def on_grid(rho: pd.Series) -> pd.Series:
    """Сетка скрининга: ρ кратно 0,05 в [0,20; 1,00] (отсекает уточнённые/диагностические точки той же конфигурации)."""
    v = rho.to_numpy() * 20.0
    return pd.Series((np.abs(v - np.round(v)) < 1e-6) & (rho.to_numpy() >= 0.2 - 1e-9) & (rho.to_numpy() <= 1.0 + 1e-9), index=rho.index)


def spec_hashes(name: str) -> list[str]:
    spec = yaml.safe_load(open(ROOT / "experiments" / "grids" / f"{name}.yaml", encoding="utf-8"))
    return [config_hash(c) for c in configs_from_spec(spec)]


def screen_rows(res: pd.DataFrame, hashes: list[str]) -> pd.DataFrame:
    rows = []
    for h in hashes:
        r = res[(res["config_hash"] == h) & (res["replicate"] < R_SCR) & on_grid(res["rho_test"])]
        ra, rb = r[r["protocol"] == "A"], r[r["protocol"] == "B"]
        e = estimate_delta(
            ra["rho_test"].to_numpy(), ra["failed"].to_numpy(),
            rb["rho_test"].to_numpy(), rb["failed"].to_numpy(), NB, 11, right_bound=1.0,
        )
        pa = ra.groupby("rho_test")["failed"].mean()
        pb = rb.groupby("rho_test")["failed"].mean()
        rows.append(
            {
                "hash": h, "R": r["R"].iloc[0], "Q": r["Q"].iloc[0], "D": r["D"].iloc[0], "S": r["S"].iloc[0],
                "n_A": len(ra), "n_B": len(rb), "n_rho_A": pa.size, "n_rho_B": pb.size,
                "fold": e.fold.value, "fold_lo": e.fold.lo, "fold_hi": e.fold.hi, "fold_cens": e.fold.censored,
                "rel": e.release.value, "rel_lo": e.release.lo, "rel_hi": e.release.hi, "rel_cens": e.release.censored,
                "delta": e.delta, "d_lo": e.lo, "d_hi": e.hi,
                "maxP_A": pa.max(), "maxP_B": pb.max(),
                "PA_at_1": pa.get(1.0, np.nan), "PB_at_1": pb.get(1.0, np.nan),
            }
        )
    df = pd.DataFrame(rows)
    df["N1"] = (df["fold_cens"] != "right") & (df["d_lo"] > 0)
    df["N2"] = (df["fold_cens"] != "right") & (df["delta"] >= 0.1)
    df["cls"] = df.apply(classify, axis=1)
    return df


def marg(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by).agg(n=("N1", "size"), N1=("N1", "sum"), N2=("N2", "sum"),
                           fold_cens=("fold_cens", lambda s: int((s == "right").sum())),
                           rel_cens=("rel_cens", lambda s: int((s == "right").sum())))
    return g.reset_index()


def main() -> int:
    d = stage_dir("pilot")
    res, tk = load_results(d), load_tasks(d)
    h54 = spec_hashes("pilot3_scr")
    h18 = spec_hashes("pilot3_r0")
    emit(f"# Скрининг (R = {R_SCR} реплик на точку, ρ = 0,20…1,00 шаг 0,05, триггер (1,5; 25), bootstrap {NB})")
    df = screen_rows(res, h54)
    df = df.sort_values(["S", "D", "Q", "R"]).reset_index(drop=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    emit(f"\nКонфигураций не-R0: {len(df)}; точек ρ на протокол: A {df.n_rho_A.min()}-{df.n_rho_A.max()}, "
         f"B {df.n_rho_B.min()}-{df.n_rho_B.max()}; реплик в A/B на конфигурацию {df.n_A.min()}/{df.n_B.min()}")
    cols = ["R", "Q", "D", "S", "fold", "fold_cens", "rel", "rel_cens", "delta", "d_lo", "d_hi", "maxP_A", "maxP_B", "N1", "N2", "cls"]
    emit(df[cols].round(3).to_string())
    n = len(df)
    emit(f"\nN1 (fold<1 и нижняя граница ДИ Δ > 0): {int(df.N1.sum())} из {n}")
    emit(f"N2 (fold<1 и Δ ≥ 0,1): {int(df.N2.sum())} из {n}")
    emit("Классы: " + str(df["cls"].value_counts().to_dict()))
    nc_r = int((df["fold_cens"] == "right").sum() + (df["rel_cens"] == "right").sum())
    nc_l = int((df["fold_cens"] == "left").sum() + (df["rel_cens"] == "left").sum())
    emit(f"Цензурировано порогов: справа {nc_r}, слева {nc_l} из {2 * n} (ρ_fold справа {int((df.fold_cens == 'right').sum())}, ρ_release справа {int((df.rel_cens == 'right').sum())})")
    for by in (["R"], ["Q"], ["D"], ["S"], ["R", "Q"], ["Q", "D"], ["Q", "S"], ["R", "D"], ["R", "S"], ["D", "S"]):
        emit(f"\nМаргинальная таблица по {'×'.join(by)}:")
        emit(marg(df, by).to_string(index=False))
    emit("\nПолная таблица R×Q (строки) × D×S (колонки): N1 (Δ точечная)")
    df["DS"] = df["D"] + "/" + df["S"]
    df["RQ"] = df["R"] + "/" + df["Q"]
    piv = df.assign(v=np.where(df.N1, df.delta.round(2).astype(str), "-")).pivot(index="RQ", columns="DS", values="v")
    emit(piv.to_string())

    # сравнение с частью 2 (40 реплик)
    p2 = pd.read_csv(OUT / "pilot2_configs.csv")
    emit("\nСогласие со скринингом по 10 конфигурациям части 2 (там 40 реплик, здесь 8):")
    cmp_rows = []
    for r in p2.itertuples():
        m = df[(df.R == r.R) & (df.Q == r.Q) & (df.D == r.D) & (df.S == r.S)].iloc[0]
        n1_40 = bool(r.fold_cens != "right" and r.d_lo > 0)
        cmp_rows.append({"cfg": f"{r.R}/{r.Q}/{r.D}/{r.S}", "delta40": r.delta, "delta8": m.delta,
                         "fold40": r.fold, "fold8": m.fold, "rel40": r.rel, "rel8": m.rel,
                         "N1_40": n1_40, "N1_8": bool(m.N1), "N2_40": bool(r.eligible), "N2_8": bool(m.N2)})
    cm = pd.DataFrame(cmp_rows)
    emit(cm.round(3).to_string(index=False))
    emit(f"Совпадение N1: {int((cm.N1_40 == cm.N1_8).sum())} из {len(cm)}; N2: {int((cm.N2_40 == cm.N2_8).sum())} из {len(cm)}; "
         f"max|Δ8−Δ40| = {np.abs(cm.delta40 - cm.delta8).max():.3f}")

    # R0
    emit("\n# Контроль R0 (ρ = 0,80; 0,90; 0,95; 1,00; 8 реплик)")
    rr = []
    for h in h18:
        r = res[(res["config_hash"] == h) & (res["replicate"] < R_SCR) & (res["rho_test"] >= 0.8 - 1e-9)]
        pa = r[r.protocol == "A"].groupby("rho_test")["failed"].agg(["sum", "size"])
        pb = r[r.protocol == "B"].groupby("rho_test")["failed"].agg(["sum", "size"])
        rr.append({"cfg": f"R0/{r.Q.iloc[0]}/{r.D.iloc[0]}/{r.S.iloc[0]}",
                   "A_fail<=0.95": int(pa.loc[pa.index <= 0.95 + 1e-9, "sum"].sum()), "A_n<=0.95": int(pa.loc[pa.index <= 0.95 + 1e-9, "size"].sum()),
                   "B_fail<=0.95": int(pb.loc[pb.index <= 0.95 + 1e-9, "sum"].sum()), "B_n<=0.95": int(pb.loc[pb.index <= 0.95 + 1e-9, "size"].sum()),
                   "A_fail@1.0": int(pa["sum"].get(1.0, -1)), "B_fail@1.0": int(pb["sum"].get(1.0, -1)), "n@1.0": int(pa["size"].get(1.0, 0))})
    rdf = pd.DataFrame(rr).sort_values("cfg")
    emit(rdf.to_string(index=False))
    emit(f"R0: отказов при ρ ≤ 0,95: A {rdf['A_fail<=0.95'].sum()} из {rdf['A_n<=0.95'].sum()}, B {rdf['B_fail<=0.95'].sum()} из {rdf['B_n<=0.95'].sum()}; "
         f"при ρ = 1,0: A {rdf['A_fail@1.0'].sum()} из {rdf['n@1.0'].sum()}, B {rdf['B_fail@1.0'].sum()} из {rdf['n@1.0'].sum()}")

    # H1
    emit("\n# H1: ячейки R×Q×D, Δ при Exp и LogN")
    dl = df.copy()
    dl["cell"] = dl["R"] + "/" + dl["Q"] + "/" + dl["D"]
    pv = dl.pivot(index="cell", columns="S", values="delta")
    pn = dl.pivot(index="cell", columns="S", values="N1")
    pn2 = dl.pivot(index="cell", columns="S", values="N2")
    emit(f"Ячеек без R0: {len(pv)}; N1 при Exp {int(pn['Exp'].sum())}, при LogN {int(pn['LogN'].sum())}, при Pareto {int(pn['Pareto'].sum())}; "
         f"при Exp и LogN одновременно {int((pn['Exp'] & pn['LogN']).sum())}; хотя бы в одном {int((pn['Exp'] | pn['LogN']).sum())}")
    emit(f"N2 (Δ ≥ 0,1) при Exp {int(pn2['Exp'].sum())}, LogN {int(pn2['LogN'].sum())}; одновременно {int((pn2['Exp'] & pn2['LogN']).sum())}")
    rng = np.random.default_rng(np.random.SeedSequence(5))

    def tau(x: np.ndarray, y: np.ndarray) -> float:
        if np.unique(x).size < 2 or np.unique(y).size < 2:
            return float("nan")
        return float(kendalltau(x, y).statistic)

    def tau_ci(x: np.ndarray, y: np.ndarray) -> str:
        t0 = tau(x, y)
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(x), len(x))
            v = tau(x[i], y[i])
            if not np.isnan(v):
                bs.append(v)
        if not bs:
            return f"τ-b = {t0:.3f}, ДИ не определён"
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return f"τ-b = {t0:.3f}, bootstrap по ячейкам [{lo:.3f}; {hi:.3f}] (валидных {len(bs)}/2000)"

    x, y = pv["Exp"].to_numpy(), pv["LogN"].to_numpy()
    emit("(а) 18 ячеек без R0: " + tau_ci(x, y))
    x24, y24 = np.concatenate([x, np.zeros(6)]), np.concatenate([y, np.zeros(6)])
    emit("(б) 24 ячейки, R0 с Δ = 0: " + tau_ci(x24, y24))
    sel = (pn["Exp"] | pn["LogN"]).to_numpy()
    emit(f"(в) ячейки, невырожденные хотя бы в одном S ({int(sel.sum())}): " + (tau_ci(x[sel], y[sel]) if sel.sum() >= 3 else "мало ячеек"))
    both = (pn["Exp"] & pn["LogN"]).to_numpy()
    emit(f"(г) невырожденные в обоих S ({int(both.sum())}): " + (tau_ci(x[both], y[both]) if both.sum() >= 3 else "мало ячеек"))
    emit("Δ (точечная) по ячейкам:")
    emit(pv.round(3).assign(N1_Exp=pn["Exp"], N1_LogN=pn["LogN"]).to_string())

    # H2
    emit("\n# H2: контраст взаимодействия (R3 vs R2) × (ALIFO vs FIFO) по (D, S)")
    ix = df.set_index(["R", "Q", "D", "S"])
    h2 = []
    for dd in ("off", "on"):
        for ss in ("Exp", "LogN", "Pareto"):
            g = lambda R, Q: ix.loc[(R, Q, dd, ss)]
            a, b, c_, e = g("R3", "ALIFO"), g("R3", "FIFO"), g("R2", "ALIFO"), g("R2", "FIFO")
            cc = a.delta - b.delta - c_.delta + e.delta
            h2.append({"D": dd, "S": ss, "d_R3ALIFO": a.delta, "d_R3FIFO": b.delta, "d_R2ALIFO": c_.delta, "d_R2FIFO": e.delta,
                       "contrast": cc, "N1_R3ALIFO": bool(a.N1), "N1_R3FIFO": bool(b.N1), "N1_R2ALIFO": bool(c_.N1), "N1_R2FIFO": bool(e.N1),
                       "all4_N1": bool(a.N1 and b.N1 and c_.N1 and e.N1)})
    hd = pd.DataFrame(h2)
    emit(hd.round(3).to_string(index=False))
    emit(f"Ячеек (D,S) из 6: все 4 Δ невырождены {int(hd.all4_N1.sum())}; R3×ALIFO невырождена {int(hd.N1_R3ALIFO.sum())}; "
         f"R3×FIFO невырождена {int(hd.N1_R3FIFO.sum())}; R2×ALIFO {int(hd.N1_R2ALIFO.sum())}; R2×FIFO {int(hd.N1_R2FIFO.sum())}")
    emit(f"Контраст < 0 в {int((hd.contrast < 0).sum())} из 6; для Exp/LogN при D усреднённом: "
         f"Exp {hd[hd.S == 'Exp'].contrast.mean():.3f}, LogN {hd[hd.S == 'LogN'].contrast.mean():.3f}")

    df.drop(columns=["DS", "RQ"]).to_csv(OUT / "pilot3_screen.csv", index=False)
    hd.to_csv(OUT / "pilot3_screen_h2.csv", index=False)
    (OUT / "pilot3_screen.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
