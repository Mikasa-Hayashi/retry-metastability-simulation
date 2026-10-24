"""Pilot, часть 2.1/2.2: пороги и Δ по конфигурациям покрытия факторов (CALIBRATION_PLAN_2.md).

python analysis/pilot2_configs.py configs   -> результаты по 10 конфигурациям (триггер 1,5; 25)
python analysis/pilot2_configs.py triggers  -> сетка триггеров для отобранных конфигураций
Реплики опорной конфигурации ограничены номерами < 40.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runner import ROOT, load_results, load_tasks, stage_dir
from src.thresholds import estimate_delta

NB = 2000
MAX_REP = 40
TRIG_ORDER = [
    (1.5, 25), (1.5, 50), (2, 25), (1.5, 100), (2, 50), (3, 25), (4, 25),
    (2, 100), (3, 50), (4, 50), (3, 100), (4, 100),
]  # порядок по E из CALIBRATION_PLAN.md
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = stage_dir("pilot")
    return load_results(d), load_tasks(d)


def cfg_rows(res: pd.DataFrame, tk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tags = sorted(t for t in tk["tag"].unique() if str(t).startswith("p2_cfg"))
    for tag in tags:
        h = tk[tk["tag"] == tag]["config_hash"].iloc[0]
        c = json.loads(tk[tk["config_hash"] == h]["cfg_json"].iloc[0])
        r = res[(res["config_hash"] == h) & (res["replicate"] < MAX_REP)]
        ra, rb = r[r["protocol"] == "A"], r[r["protocol"] == "B"]
        e = estimate_delta(
            ra["rho_test"].to_numpy(), ra["failed"].to_numpy(),
            rb["rho_test"].to_numpy(), rb["failed"].to_numpy(), NB, 11,
            right_bound=1.0,
        )
        rows.append(
            {
                "tag": tag, "hash": h, "R": c["R"], "Q": c["Q"], "D": c["D"], "S": c["S"],
                "n_A": len(ra), "n_B": len(rb),
                "fold": e.fold.value, "fold_lo": e.fold.lo, "fold_hi": e.fold.hi, "fold_cens": e.fold.censored,
                "rel": e.release.value, "rel_lo": e.release.lo, "rel_hi": e.release.hi, "rel_cens": e.release.censored,
                "delta": e.delta, "d_lo": e.lo, "d_hi": e.hi,
                "run_mean_s": r["runtime"].mean(), "run_max_s": r["runtime"].max(),
                "v4_viol": 0,
            }
        )
    df = pd.DataFrame(rows)
    df["eligible"] = (df["delta"] >= 0.1) & (df["fold_cens"] != "right")
    df["flag"] = (df["delta"] < 0.1) | (df["fold_cens"] == "right") | (df["rel_cens"] == "left")
    return df


def curves(res: pd.DataFrame, h: str, proto: str) -> str:
    r = res[(res["config_hash"] == h) & (res["protocol"] == proto) & (res["replicate"] < MAX_REP)]
    return str(r.groupby("rho_test")["failed"].mean().round(2).to_dict())


def trig_rows(res: pd.DataFrame, tk: pd.DataFrame, b_hash_by_cfg: dict[str, str]) -> pd.DataFrame:
    rows = []
    tags = sorted(t for t in tk["tag"].unique() if str(t).startswith("p2_trg"))
    for tag in tags:
        for h in tk[tk["tag"] == tag]["config_hash"].unique():
            c = json.loads(tk[tk["config_hash"] == h]["cfg_json"].iloc[0])
            key = f"{c['R']}|{c['Q']}|{c['D']}|{c['S']}"
            rb = res[(res["config_hash"] == b_hash_by_cfg[key]) & (res["protocol"] == "B") & (res["replicate"] < MAX_REP)]
            ra = res[(res["config_hash"] == h) & (res["protocol"] == "A") & (res["replicate"] < MAX_REP)]
            e = estimate_delta(
                ra["rho_test"].to_numpy(), ra["failed"].to_numpy(),
                rb["rho_test"].to_numpy(), rb["failed"].to_numpy(), NB, 13,
            )
            rows.append(
                {
                    "cfg": key, "m_s": c["proto"]["trigger_mult"], "D_s": c["proto"]["trigger_len"],
                    "E": (c["proto"]["trigger_mult"] - 1) * c["proto"]["trigger_len"],
                    "fold": e.fold.value, "fold_cens": e.fold.censored, "rel": e.release.value,
                    "rel_cens": e.release.censored, "delta": e.delta, "d_lo": e.lo, "d_hi": e.hi,
                }
            )
    df = pd.DataFrame(rows).sort_values(["cfg", "E", "m_s"]).reset_index(drop=True)
    df["eligible"] = (df["delta"] >= 0.1) & (df["fold_cens"] != "right")
    return df


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "configs"
    res, tk = _load()
    if mode == "configs":
        df = cfg_rows(res, tk)
        print(df.drop(columns=["hash"]).round(3).to_string())
        n_cens = int((df["fold_cens"] != "none").sum() + (df["rel_cens"] != "none").sum())
        print(f"\nцензурировано порогов: {n_cens} из {2 * len(df)}; недопустимых: {int((~df['eligible']).sum())} из {len(df)}")
        print("справа цензурированных ρ_fold:", int((df["fold_cens"] == "right").sum()))
        for r in df.itertuples():
            print(r.tag, "P_A:", curves(res, r.hash, "A"))
            print(r.tag, "P_B:", curves(res, r.hash, "B"))
        df.to_csv(ROOT / "results" / "pilot" / "pilot2_configs.csv", index=False)
    else:
        df0 = cfg_rows(res, tk)
        bmap = {f"{r.R}|{r.Q}|{r.D}|{r.S}": r.hash for r in df0.itertuples()}
        df = trig_rows(res, tk, bmap)
        print(df.round(3).to_string())
        df.to_csv(ROOT / "results" / "pilot" / "pilot2_triggers.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
