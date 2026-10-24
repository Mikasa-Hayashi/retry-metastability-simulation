"""Уточнение сетки ρ для части 4 (общие случайные числа). Смотрит только на failed (P_A или P_B), ρ, протокол.

python experiments/refine_grid_p4.py --stage pilot --tags p4_scr,p4_ext --tag-new p4_ref1 [--crit failed|norm]

Для каждой конфигурации и протокола по данным CRN-блока (seed >= 100000, реплики < R) по всей сетке считается порог
(src.thresholds.point_threshold, правое ограничение hi). Нецензурированный порог <= split: шаг 0,01, окно ±0,05;
порог > split: шаг step_hi (0,02), окно ±0,10. Цензурированные пороги не уточняются. Критерий: failed (θ = конфиг)
или norm (g_final·max(1, ρ) < theta_n, эксплораторно).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_grid_p4 import CRN_BASE, add_tasks

from src.runner import _cfg_from_json, load_results, load_tasks, refine_rhos, stage_dir
from src.thresholds import aggregate, point_threshold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="pilot")
    ap.add_argument("--tags", required=True, help="теги исходных задач через запятую")
    ap.add_argument("--tag-new", required=True)
    ap.add_argument("--crit", default="failed", choices=["failed", "norm"])
    ap.add_argument("--theta-n", type=float, default=0.5)
    ap.add_argument("--lo", type=float, default=0.2)
    ap.add_argument("--hi", type=float, default=2.5)
    ap.add_argument("--split", type=float, default=1.0)
    ap.add_argument("--step-lo", type=float, default=0.01)
    ap.add_argument("--hw-lo", type=float, default=0.05)
    ap.add_argument("--step-hi", type=float, default=0.02)
    ap.add_argument("--hw-hi", type=float, default=0.10)
    ap.add_argument("--replicates", type=int, default=40)
    ap.add_argument("--hash-file", default=None, help="файл со списком config_hash (по одному в строке); иначе все")
    ap.add_argument("--max-rho", type=float, default=None, help="данные выше этого ρ игнорируются (в т.ч. при поиске порога)")
    ap.add_argument("--refine-censored", action="store_true",
                    help="для порогов, цензурированных справа, добавить окно [1,0 − hw; 1,0] шагом step-lo")
    a = ap.parse_args()
    d = stage_dir(a.stage)
    tags = a.tags.split(",")
    tasks = load_tasks(d)
    tasks = tasks[tasks["tag"].isin(tags)]
    res = load_results(d, ["config_hash", "rho_test", "protocol", "failed", "g_final", "seed", "replicate"])
    res = res[(res["seed"] >= CRN_BASE) & (res["replicate"] < a.replicates)]
    hashes = set(tasks["config_hash"].unique())
    if a.hash_file:
        hashes &= set(Path(a.hash_file).read_text().split())
    total = 0
    n_cfg = 0
    for h in sorted(hashes):
        r = res[res["config_hash"] == h]
        if a.max_rho is not None:
            r = r[r["rho_test"] <= a.max_rho + 1e-9]
        if r.empty:
            continue
        cfgs = [(_cfg_from_json(str(tasks[tasks["config_hash"] == h]["cfg_json"].iloc[0])), [], "")]
        for proto in ("A", "B"):
            rp = r[r["protocol"] == proto]
            if rp.empty:
                continue
            rho = rp["rho_test"].to_numpy()
            if a.crit == "norm":
                failed = (rp["g_final"].to_numpy() * np.maximum(1.0, rho) < a.theta_n).astype(int)
            else:
                failed = rp["failed"].to_numpy().astype(int)
            x, n, k = aggregate(rho, failed)
            thr, cens, _ = point_threshold(x, n, k, 0.5, right_bound=a.hi)
            if cens == "right" and a.refine_censored:
                new = refine_rhos(rho, failed, a.step_lo, a.hw_lo, a.lo, 1.0)
                new = [v for v in new if v <= 1.0 + 1e-9]
            elif cens != "none":
                continue
            elif thr <= a.split:
                new = refine_rhos(rho, failed, a.step_lo, a.hw_lo, a.lo, a.hi)
            else:
                new = refine_rhos(rho, failed, a.step_hi, a.hw_hi, a.lo, a.hi)
            if not new:
                continue
            cfgs[0] = (cfgs[0][0], [proto], "")
            t = add_tasks(a.stage, [cfgs[0]], lambda p, new=new: new, a.replicates, a.tag_new, "crn")
            total += len(t)
            n_cfg += 1
    print(f"уточнение: пар (конфигурация, протокол) {n_cfg}, добавлено задач {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
