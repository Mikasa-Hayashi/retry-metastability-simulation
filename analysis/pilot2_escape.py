"""Pilot, часть 2.3: единичные «восстановления» в протоколе B на опорной конфигурации (CALIBRATION_PLAN_2.md п. 3).

python analysis/pilot2_escape.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runner import load_results, load_tasks, stage_dir

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - w, c + w


def main() -> int:
    d = stage_dir("pilot")
    res = load_results(d)
    tk = load_tasks(d)
    ref = "148eb8d011551aa1"
    tags = {t: tk[tk["tag"] == t]["config_hash"].iloc[0] for t in ("p3_c4", "p3_c16", "p3_h6000")}
    conds = [
        ("c=8,K=128,H=1500,rho=0.4", ref, 0.4, 1500),
        ("c=8,K=128,H=1500,rho=0.5", ref, 0.5, 1500),
        ("c=8,K=128,H=1500,rho=0.6", ref, 0.6, 1500),
        ("c=4,K=64,H=1500,rho=0.5", tags["p3_c4"], 0.5, 1500),
        ("c=16,K=256,H=1500,rho=0.5", tags["p3_c16"], 0.5, 1500),
        ("c=8,K=128,H=6000,rho=0.5", tags["p3_h6000"], 0.5, 6000),
    ]
    rows = []
    for name, h, rho, H in conds:
        x = res[(res["config_hash"] == h) & (res["protocol"] == "B") & (res["rho_test"] == rho)]
        n = len(x)
        rec = x[x["failed"] == 0]
        k = len(rec)
        lo, hi = wilson(k, n)
        ret = x[(x["failed"] == 1) & (x["recovery_censored"] == 0)]
        lam = -math.log(1 - k / n) / H if 0 < k < n else float("nan")
        rt = rec["recovery_time"]
        rows.append(
            {
                "условие": name, "n": n, "восст.": k, "доля": round(k / n, 4),
                "Wilson_lo": round(lo, 4), "Wilson_hi": round(hi, 4),
                "возврат(failed=1, был здоров)": len(ret),
                "lambda=-ln(1-p)/H": lam,
                "g_final восст. min/med": (round(rec["g_final"].min(), 3), round(rec["g_final"].median(), 3)) if k else None,
                "rec_time восст. quartiles": tuple(np.round(rt.quantile([0.25, 0.5, 0.75]).to_numpy(), 0)) if k else None,
                "rec_time max, H": (round(rt.max(), 0), H) if k else None,
                "g_final failed med": round(x[x["failed"] == 1]["g_final"].median(), 3),
                "qlen_mean failed med": round(x[x["failed"] == 1]["qlen_mean_final"].median(), 1),
                "qlen_mean восст. med": round(rec["qlen_mean_final"].median(), 1) if k else None,
                "n_cens_rec(восст.)": int(rec["recovery_censored"].sum()) if k else 0,
            }
        )
    out = pd.DataFrame(rows)
    print(out.T.to_string())
    out.to_csv(stage_dir("pilot") / "pilot2_escape.csv", index=False)
    # достаточно ли горизонта 6000 для равномерного распределения моментов выхода
    h = tags["p3_h6000"]
    x = res[(res["config_hash"] == h) & (res["protocol"] == "B") & (res["rho_test"] == 0.5) & (res["failed"] == 0)]
    print("\nH=6000: моменты первого достижения g>=0.9 у восстановившихся:", sorted(np.round(x["recovery_time"].to_numpy(), 0).tolist()))
    x = res[(res["config_hash"] == ref) & (res["protocol"] == "B") & (res["rho_test"] == 0.5) & (res["failed"] == 0)]
    print("H=1500:", sorted(np.round(x["recovery_time"].to_numpy(), 0).tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
