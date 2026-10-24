"""Уточнение сетки ρ в stage explor: правило основного прогона (один раунд, шаг 0,01, окно ±0,05 вокруг каждого
нецензурированного порога), отдельно по протоколам A и B, по данным ρ ≤ 1,0 (только failed и ρ).

python experiments/refine_grid_explor.py --src-tag x_k2 --tag-new x_k2_ref [--replicates 40] [--dry]

Цензурированные пороги не уточняются; новые точки — кратные 0,01 в [thr − 0,05; thr + 0,05] ∩ [0,2; 1,0], которых
ещё нет. Правило то же, что в refine_grid_main.py (src.runner.refine_rhos); читается только results/explor.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runner import _cfg_from_json, load_results, load_tasks, make_tasks, refine_rhos, stage_dir
from src.thresholds import aggregate, point_threshold

LO, HI, STEP, HW = 0.2, 1.0, 0.01, 0.05


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-tag", required=True, help="теги исходных задач через запятую")
    ap.add_argument("--tag-new", required=True)
    ap.add_argument("--replicates", type=int, default=40)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    d = stage_dir("explor")
    tasks = load_tasks(d)
    tasks = tasks[tasks["tag"].isin(a.src_tag.split(","))]
    res = load_results(d, ["config_hash", "rho_test", "protocol", "failed", "replicate"])
    res = res[(res["replicate"] < a.replicates) & (res["rho_test"] <= HI + 1e-9)]
    total = n_pairs = 0
    for h in sorted(tasks["config_hash"].unique()):
        r = res[res["config_hash"] == h]
        for proto in sorted(tasks[tasks["config_hash"] == h]["protocol"].unique()):
            rp = r[r["protocol"] == proto]
            if rp.empty:
                continue
            rho = rp["rho_test"].to_numpy()
            failed = rp["failed"].to_numpy().astype(int)
            x, n, k = aggregate(rho, failed)
            _thr, cens, _m = point_threshold(x, n, k, 0.5, right_bound=HI)
            if cens != "none":
                continue
            new = refine_rhos(rho, failed, STEP, HW, LO, HI)
            if not new:
                continue
            n_pairs += 1
            if a.dry:
                total += len(new) * a.replicates
                continue
            js = str(tasks[tasks["config_hash"] == h]["cfg_json"].iloc[0])
            total += len(make_tasks("explor", [_cfg_from_json(js)], new, [proto], a.replicates, a.tag_new))
    print(f"уточнение: пар (конфигурация, протокол) {n_pairs}, {'задач по плану' if a.dry else 'добавлено задач'} {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
