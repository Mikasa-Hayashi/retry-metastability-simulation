"""Pilot: пороги и Δ по триггерам (results/pilot), выбор триггера по правилу CALIBRATION_PLAN.

python analysis/pilot_calibration.py [--n-boot 2000] -> печатает таблицу и пишет
results/pilot/calibration_table.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runner import ROOT, load_results, load_tasks, stage_dir
from src.thresholds import estimate_delta


def trigger_table(n_boot: int, a_tag: str = "calib_A", b_tag: str = "calib_B") -> pd.DataFrame:
    d = stage_dir("pilot")
    res = load_results(d)
    tk = load_tasks(d)
    cfg = {r.config_hash: json.loads(r.cfg_json) for r in tk.drop_duplicates("config_hash").itertuples()}
    b_hash = tk[tk["tag"] == b_tag]["config_hash"].iloc[0]
    rb = res[(res["config_hash"] == b_hash) & (res["protocol"] == "B")]
    rows = []
    for h in tk[tk["tag"] == a_tag]["config_hash"].unique():
        ra = res[(res["config_hash"] == h) & (res["protocol"] == "A")]
        est = estimate_delta(
            ra["rho_test"].to_numpy(),
            ra["failed"].to_numpy(),
            rb["rho_test"].to_numpy(),
            rb["failed"].to_numpy(),
            n_boot=n_boot,
            seed=7,
        )
        m, ln = cfg[h]["proto"]["trigger_mult"], cfg[h]["proto"]["trigger_len"]
        rows.append(
            {
                "hash": h,
                "m_s": m,
                "D_s": ln,
                "E": (m - 1) * ln,
                "n_A": len(ra),
                "fold": est.fold.value,
                "fold_lo": est.fold.lo,
                "fold_hi": est.fold.hi,
                "fold_cens": est.fold.censored,
                "release": est.release.value,
                "release_lo": est.release.lo,
                "release_hi": est.release.hi,
                "rel_cens": est.release.censored,
                "delta": est.delta,
                "delta_lo": est.lo,
                "delta_hi": est.hi,
            }
        )
    df = pd.DataFrame(rows).sort_values(["E", "m_s"]).reset_index(drop=True)
    df["eligible"] = (df["delta"] >= 0.1) & (df["fold_cens"] != "right")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    df = trigger_table(a.n_boot)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(df.drop(columns=["hash"]).round(3).to_string())
    elig = df[df["eligible"]]
    if len(elig):
        print("\nвыбран (самый слабый допустимый):", elig.iloc[0][["hash", "m_s", "D_s"]].to_dict())
    else:
        print("\nдопустимых триггеров нет")
    df.to_csv(ROOT / "results" / "pilot" / "calibration_table.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
