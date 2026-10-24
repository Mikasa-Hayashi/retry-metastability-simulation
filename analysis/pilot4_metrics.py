"""Pilot, часть 4, п. 3: непрерывные метрики деградации (CALIBRATION_PLAN_4.md).

python analysis/pilot4_metrics.py
Данные: теги p4_scr, p4_ext (общая грубая сетка, реплики 0…39). Для каждой из 72 конфигураций:
  H  = ∫|g_B − g_A| dρ  (площадь петли гистерезиса), на [0,2; 1,0] (le1) и [0,2; 2,5] (ext);
  L_A, L_B = ∫(g*(ρ) − g(ρ)) dρ, g* = min(1, 1/ρ) (потерянный goodput) на тех же интервалах;
  wasted, amplification, p99, recovery_time (ограниченное среднее: цензурированные = горизонт) при ρ ∈ {0,7; 0,9; 1,0}.
ДИ: bootstrap по репликам (одинаковые индексы реплик для всех ρ, обоих протоколов и всех конфигураций: общие
случайные числа), 2000 повторов, перцентильный 95 %. Эффекты факторов — разности средних по ячейкам, те же индексы.
Вывод: results/pilot/pilot4_metrics.csv|txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot4_common import OUT, cfg_meta, load_crn

NB = 2000
NREP = 40
GRID = sorted({round(0.2 + 0.05 * i, 6) for i in range(17)} | {1.05} | {round(1.1 + 0.1 * i, 6) for i in range(15)})
PTS = (0.7, 0.9, 1.0)
VARS = {"g": "g_final", "wasted": "wasted_fraction", "ampl": "amplification", "p99": "p99", "rec": "recovery_time",
        "rec_cens": "recovery_censored"}


def ci(x: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    res = load_crn(["config_hash", "rho_test", "protocol", "replicate", *VARS.values()], ["p4_scr", "p4_ext"])
    res["rho_test"] = res["rho_test"].round(6)
    res = res[res["rho_test"].isin(GRID) & (res["replicate"] < NREP)]
    meta = cfg_meta()
    meta = meta[meta["config_hash"].isin(res["config_hash"].unique())].sort_values(["R", "Q", "D", "S"]).reset_index(drop=True)
    rng = np.random.default_rng(np.random.SeedSequence(2026))
    idx = rng.integers(0, NREP, (NB, NREP))
    grid = np.array(GRID)
    le1 = grid <= 1.0 + 1e-9
    gstar = np.minimum(1.0, 1.0 / grid)
    boots: dict[str, np.ndarray] = {}  # имя метрики → [n_cfg, NB]
    rows = []
    for ci_, h in enumerate(meta["config_hash"]):
        d = res[res["config_hash"] == h]
        mats: dict[tuple[str, str], np.ndarray] = {}
        for pr in ("A", "B"):
            dp = d[d["protocol"] == pr]
            for key, col in VARS.items():
                pv = dp.pivot_table(index="rho_test", columns="replicate", values=col, aggfunc="first").reindex(index=grid, columns=range(NREP))
                mats[(pr, key)] = pv.to_numpy()

        def bcurve(m: np.ndarray) -> np.ndarray:
            return np.nanmean(m[:, idx], axis=2)  # [n_rho, NB]

        bc = {k: bcurve(v) for k, v in mats.items()}
        pc = {k: np.nanmean(v, axis=1) for k, v in mats.items()}
        out: dict[str, float | str] = {"config_hash": h}

        def integ(y: np.ndarray, sel: np.ndarray) -> np.ndarray:
            return np.trapezoid(y[sel], grid[sel], axis=0)

        vals: dict[str, tuple[float, np.ndarray]] = {}
        for nm, sel in (("le1", le1), ("ext", np.ones_like(le1))):
            vals[f"H_{nm}"] = (float(integ(np.abs(pc[("B", "g")] - pc[("A", "g")]), sel)),
                               integ(np.abs(bc[("B", "g")] - bc[("A", "g")]), sel))
            for pr in ("A", "B"):
                vals[f"L{pr}_{nm}"] = (float(integ(gstar - pc[(pr, "g")], sel)),
                                       integ(gstar[:, None] - bc[(pr, "g")], sel))
        for pr in ("A", "B"):
            for key in ("wasted", "ampl", "p99", "rec", "rec_cens", "g"):
                for r0 in PTS:
                    j = int(np.argmin(np.abs(grid - r0)))
                    vals[f"{key}{pr}_{r0}"] = (float(pc[(pr, key)][j]), bc[(pr, key)][j])
        for k, (v, b) in vals.items():
            lo, hi = ci(b)
            out[k], out[k + "_lo"], out[k + "_hi"] = v, lo, hi
            boots.setdefault(k, np.full((len(meta), NB), np.nan))[ci_] = b
        rows.append(out)
    df = meta.merge(pd.DataFrame(rows), on="config_hash")
    df.to_csv(OUT / "pilot4_metrics.csv", index=False)

    lines: list[str] = []
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.max_rows", 500)

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit(f"# pilot4 metrics: {len(df)} конфигураций, общая сетка {len(GRID)} узлов, реплик {NREP}, bootstrap {NB} (общие индексы реплик)")
    fac = ["R", "Q", "D", "S"]
    for blk, cols in (
        ("Площадь петли H и потерянный goodput L (ρ ≤ 1)", ["H_le1", "LA_le1", "LB_le1"]),
        ("Площадь петли H и потерянный goodput L (ρ ≤ 2,5; область перегрузки включена)", ["H_ext", "LA_ext", "LB_ext"]),
    ):
        emit(f"\n## {blk}: оценка [ДИ]")
        t = df[fac].copy()
        for c in cols:
            t[c] = [f"{v:.3f} [{lo:.3f}; {hi:.3f}]" for v, lo, hi in zip(df[c], df[c + "_lo"], df[c + "_hi"], strict=True)]
        emit(t.to_string(index=False))
    for r0 in PTS:
        emit(f"\n## ρ = {r0}: впустую потраченная работа, усиление, p99, время восстановления (ограниченное среднее), доля цензурированных; A | B")
        t = df[fac].copy()
        for key in ("g", "wasted", "ampl", "p99", "rec", "rec_cens"):
            for pr in ("A", "B"):
                c = f"{key}{pr}_{r0}"
                t[f"{key}_{pr}"] = [f"{v:.3f} [{lo:.3f}; {hi:.3f}]" if key in ("g", "wasted", "ampl", "rec_cens") else f"{v:.1f} [{lo:.1f}; {hi:.1f}]"
                                    for v, lo, hi in zip(df[c], df[c + "_lo"], df[c + "_hi"], strict=True)]
        emit(t.to_string(index=False))

    emit("\n## Эффекты факторов (разность средних по ячейкам, bootstrap с общими индексами реплик)")
    lab = df[fac].reset_index(drop=True)
    effects = []
    for met in ("H_le1", "LA_le1", "LB_le1", "H_ext", "LA_ext", "LB_ext", "wastedA_0.9", "wastedB_0.9", "amplA_0.9", "amplB_0.9"):
        b = boots[met]
        for f, pairs in (("Q", [("LIFO", "FIFO"), ("ALIFO", "FIFO"), ("ALIFO", "LIFO")]),
                         ("R", [("R1", "R0"), ("R2", "R0"), ("R3", "R0"), ("R2", "R1"), ("R3", "R2")]),
                         ("D", [("on", "off")]), ("S", [("LogN", "Exp"), ("Pareto", "Exp")])):
            for u, v in pairs:
                mu, mv = (lab[f] == u).to_numpy(), (lab[f] == v).to_numpy()
                dd = np.nanmean(b[mu], axis=0) - np.nanmean(b[mv], axis=0)
                p = float(np.nanmean(df[met].to_numpy()[mu]) - np.nanmean(df[met].to_numpy()[mv]))
                lo, hi = ci(dd)
                effects.append({"metric": met, "factor": f, "contrast": f"{u} − {v}", "effect": p, "lo": lo, "hi": hi})
    ef = pd.DataFrame(effects)
    ef.to_csv(OUT / "pilot4_metrics_effects.csv", index=False)
    emit(ef.round(3).to_string(index=False))

    emit("\n## Контраст взаимодействия на непрерывных метриках (аналог H2; L ≥ 0 тоже ограничена снизу): "
         "C = m(R3,ALIFO) − m(R3,FIFO) − m(R2,ALIFO) + m(R2,FIFO), усреднено по D и указанным S; "
         "а также R3−R2 внутри ALIFO, ALIFO−FIFO внутри R2, R3−R2 внутри FIFO")
    inter = []
    for met in ("H_ext", "LA_le1", "LB_le1", "LA_ext", "LB_ext", "wastedA_0.9", "amplA_0.9"):
        b = boots[met]
        for sname, smask in (("все S", np.ones(len(lab), bool)), ("Exp", (lab.S == "Exp").to_numpy()),
                             ("LogN", (lab.S == "LogN").to_numpy()), ("Pareto", (lab.S == "Pareto").to_numpy())):

            def cell(r: str, q: str, b: np.ndarray = b, smask: np.ndarray = smask) -> np.ndarray:
                m_ = ((lab.R == r) & (lab.Q == q)).to_numpy() & smask
                return np.nanmean(b[m_], axis=0)

            def cellp(r: str, q: str, smask: np.ndarray = smask, met: str = met) -> float:
                m_ = ((lab.R == r) & (lab.Q == q)).to_numpy() & smask
                return float(np.nanmean(df[met].to_numpy()[m_]))

            for nm, fn in (
                ("C(R3xALIFO)", lambda f: f("R3", "ALIFO") - f("R3", "FIFO") - f("R2", "ALIFO") + f("R2", "FIFO")),
                ("R3-R2 | ALIFO", lambda f: f("R3", "ALIFO") - f("R2", "ALIFO")),
                ("ALIFO-FIFO | R2", lambda f: f("R2", "ALIFO") - f("R2", "FIFO")),
                ("R3-R2 | FIFO", lambda f: f("R3", "FIFO") - f("R2", "FIFO")),
            ):
                lo, hi = ci(fn(cell))
                inter.append({"metric": met, "S": sname, "contrast": nm, "est": fn(cellp), "lo": lo, "hi": hi})
    it = pd.DataFrame(inter)
    it.to_csv(OUT / "pilot4_metrics_interaction.csv", index=False)
    emit(it.round(4).to_string(index=False))

    emit("\n## Различимость конфигураций с Δ = 0 (LIFO/ALIFO/R3): ранг по H_ext и LA_ext")
    emit(df.sort_values("LA_ext")[fac + ["LA_le1", "LA_ext", "H_ext", "wastedA_0.9", "amplA_0.9", "recA_0.9", "recB_0.9"]].round(3).to_string(index=False))
    (OUT / "pilot4_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
