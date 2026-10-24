"""Pilot, часть 4, пп. 1, 2, 4: пороги и Δ по конфигурациям (CALIBRATION_PLAN_4.md).

python analysis/pilot4_screen.py --name main --tags p4_scr,p4_ext,p4_ref1,p4_ref2 [--rho-max 2.5]
Для каждой конфигурации: Δ(ρ ≤ 1) [правое цензурирование на 1,0], Δ_ext [ρ ≤ rho_max], Δ_ext по нормированному
критерию g·max(1,ρ) < 0,5 (эксплораторно), точечные Δ_ext для θ_n ∈ {0,3; 0,7; 0,9}. Bootstrap 2000 по репликам.
Вывод: results/pilot/pilot4_screen_<name>.csv|txt, pilot4_curves_<name>.csv.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from pilot4_common import OUT, cfg_meta, delta_block, load_crn, norm_failed, point_thr

NB = 2000
RHO_MAX = 2.5


def work(args: tuple[str, pd.DataFrame, float]) -> dict[str, Any]:
    h, d, rho_max = args
    d = d[d["rho_test"] <= rho_max + 1e-9]
    a, b = d[d["protocol"] == "A"], d[d["protocol"] == "B"]
    ra, rb = a["rho_test"].to_numpy(), b["rho_test"].to_numpy()
    out: dict[str, Any] = {
        "config_hash": h,
        "n_rho_A": a["rho_test"].nunique(),
        "n_rho_B": b["rho_test"].nunique(),
        "n_min": int(min(a.groupby("rho_test").size().min(), b.groupby("rho_test").size().min())),
    }
    m1a, m1b = (ra <= 1.0 + 1e-9), (rb <= 1.0 + 1e-9)
    alla, allb = np.ones(len(ra), bool), np.ones(len(rb), bool)
    for pre, hi, sel_a, sel_b, crit in (
        ("le1", 1.0, m1a, m1b, "plain"),
        ("ext", rho_max, alla, allb, "plain"),
        ("nrm", rho_max, alla, allb, "norm"),
    ):
        fa = a["failed"].to_numpy() if crit == "plain" else norm_failed(a["g_final"].to_numpy(), ra)
        fb = b["failed"].to_numpy() if crit == "plain" else norm_failed(b["g_final"].to_numpy(), rb)
        blk = delta_block(ra[sel_a], fa[sel_a], rb[sel_b], fb[sel_b], hi, NB, 11)
        for k, v in blk.items():
            out[f"{pre}_{k}"] = v
    for tn in (0.3, 0.7, 0.9):
        fa, fb = norm_failed(a["g_final"].to_numpy(), ra, tn), norm_failed(b["g_final"].to_numpy(), rb, tn)
        va, ca = point_thr(ra, fa, rho_max)
        vb, cb = point_thr(rb, fb, rho_max)
        out[f"nrm{tn}_fold"], out[f"nrm{tn}_fold_c"], out[f"nrm{tn}_rel"], out[f"nrm{tn}_rel_c"] = va, ca, vb, cb
        out[f"nrm{tn}_delta"] = va - vb
    pa = a.groupby(a["rho_test"].round(6))["failed"].mean()
    pb = b.groupby(b["rho_test"].round(6))["failed"].mean()
    com = pa.index.intersection(pb.index)
    gap = pb[com] - pa[com]
    out["gap_max_all"] = float(gap.abs().max())
    out["gap_mean_all"] = float(gap.abs().mean())
    hi_idx = com[com > 1.0 + 1e-9]
    out["gap_max_gt1"] = float(gap[hi_idx].abs().max()) if len(hi_idx) else np.nan
    out["gap_mean_gt1"] = float(gap[hi_idx].abs().mean()) if len(hi_idx) else np.nan
    out["maxPA"], out["maxPB"] = float(pa.max()), float(pb.max())
    for nm, p in (("A", pa), ("B", pb)):
        hit = p[p >= 0.5]
        out[f"rho_star_{nm}"] = float(hit.index.min()) if len(hit) else np.nan
    return out


def curves(res: pd.DataFrame) -> pd.DataFrame:
    res = res.copy()
    res["failed_norm"] = norm_failed(res["g_final"].to_numpy(), res["rho_test"].to_numpy())
    res["rho_test"] = res["rho_test"].round(6)
    return (
        res.groupby(["config_hash", "protocol", "rho_test"])
        .agg(
            n=("failed", "size"), P=("failed", "mean"), P_norm=("failed_norm", "mean"), g=("g_final", "mean"),
            g_sd=("g_final", "std"), wasted=("wasted_fraction", "mean"), ampl=("amplification", "mean"),
            p99=("p99", "mean"), qlen=("qlen_mean_final", "mean"), rec=("recovery_time", "mean"),
            rec_cens=("recovery_censored", "mean"), rej=("n_rejected", "mean"), att=("n_attempts", "mean"),
        )
        .reset_index()
    )


def marg(df: pd.DataFrame, by: list[str], pre: str) -> pd.DataFrame:
    g = df.groupby(by).agg(
        n=("config_hash", "size"), N1=(f"{pre}_N1", "sum"), N2=(f"{pre}_N2", "sum"),
        fold_cens=(f"{pre}_fold_cens", lambda s: int((s == "right").sum())),
        rel_cens=(f"{pre}_rel_cens", lambda s: int((s == "right").sum())),
    )
    return g.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--rho-max", type=float, default=RHO_MAX)
    ap.add_argument("--only-tag", default=None, help="брать конфигурации только из задач этого тега")
    ap.add_argument("--reps", type=int, default=1000, help="использовать реплики с номером < reps")
    ap.add_argument("--spec", default=None, help="YAML сетки: брать конфигурации из неё (по config_hash)")
    a = ap.parse_args()
    tags = a.tags.split(",")
    res = load_crn(
        ["config_hash", "rho_test", "protocol", "failed", "g_final", "replicate", "wasted_fraction",
         "amplification", "p99", "qlen_mean_final", "recovery_time", "recovery_censored", "n_rejected", "n_attempts"],
        tags,
    )
    res = res[res["replicate"] < a.reps]
    meta = cfg_meta()
    hashes = set(res["config_hash"])
    if a.only_tag:
        from src.runner import load_tasks, stage_dir

        tk = load_tasks(stage_dir("pilot"))
        hashes = set(tk[tk["tag"] == a.only_tag]["config_hash"]) & hashes
    if a.spec:
        import yaml
        from make_grid_p4 import build_cfgs

        from src.config import config_hash

        with open(a.spec, encoding="utf-8") as fh:
            hashes = {config_hash(c) for c, _p, _n in build_cfgs(yaml.safe_load(fh))} & hashes
    res = res[res["config_hash"].isin(hashes)]
    meta = meta[meta["config_hash"].isin(hashes)]
    cv = curves(res)
    cv = cv[cv["rho_test"] <= a.rho_max + 1e-9]
    cv.to_csv(OUT / f"pilot4_curves_{a.name}.csv", index=False)
    jobs = [
        (h, res[res["config_hash"] == h][["rho_test", "protocol", "failed", "g_final"]], a.rho_max)
        for h in sorted(hashes)
    ]
    with mp.Pool(max(1, (mp.cpu_count() or 2) - 1)) as pool:
        rows = pool.map(work, jobs, chunksize=2)
    df = meta.merge(pd.DataFrame(rows), on="config_hash")
    df.to_csv(OUT / f"pilot4_screen_{a.name}.csv", index=False)
    lines: list[str] = []
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 1000)

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit(f"# pilot4 screen '{a.name}': теги {tags}, конфигураций {len(df)}, ρ_max {a.rho_max}, bootstrap {NB}, "
         f"реплик в точке (min) {df.n_min.min()}")
    fac = [c for c in ["R", "Q", "D", "S", "m", "T", "theta_q"] if df[c].nunique() > 1]
    df = df.sort_values(fac).reset_index(drop=True)
    for pre, title in (
        ("le1", "Δ(ρ ≤ 1) [простой критерий g < 0,5]"),
        ("ext", f"Δ_ext (ρ ≤ {a.rho_max}) [простой критерий]"),
        ("nrm", f"Δ_ext (ρ ≤ {a.rho_max}) [нормированный g·max(1,ρ) < 0,5, эксплораторно]"),
    ):
        emit(f"\n## {title}")
        cols = fac + [f"{pre}_{k}" for k in ("fold", "fold_cens", "rel", "rel_cens", "delta", "d_lo", "d_hi", "N1", "cls")]
        emit(df[cols].round(3).to_string(index=False))
        emit(f"N1 = {int(df[f'{pre}_N1'].sum())} из {len(df)}; N2 = {int(df[f'{pre}_N2'].sum())}; "
             f"классы: {df[f'{pre}_cls'].value_counts().to_dict()}")
        emit(f"цензурировано справа: ρ_fold {int((df[f'{pre}_fold_cens'] == 'right').sum())}, "
             f"ρ_release {int((df[f'{pre}_rel_cens'] == 'right').sum())}; слева: ρ_release {int((df[f'{pre}_rel_cens'] == 'left').sum())}")
        bys = [[c] for c in fac] + ([["R", "Q"]] if {"R", "Q"} <= set(fac) else [])
        for by in bys:
            emit(f"маргинально по {'×'.join(by)}:")
            emit(marg(df, by, pre).to_string(index=False))
    emit("\n## A ≈ B (|P_B − P_A| по общим узлам ρ)")
    emit(df[fac + ["gap_max_all", "gap_mean_all", "gap_max_gt1", "gap_mean_gt1", "rho_star_A", "rho_star_B", "maxPA", "maxPB"]]
         .round(3).to_string(index=False))
    emit("\n## Чувствительность нормированного критерия к θ_n (точечно, без ДИ): ρ_fold / ρ_release / Δ")
    emit(df[fac + [f"nrm{t}_{k}" for t in (0.3, 0.7, 0.9) for k in ("fold", "rel", "delta")]].round(3).to_string(index=False))
    (OUT / f"pilot4_screen_{a.name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
