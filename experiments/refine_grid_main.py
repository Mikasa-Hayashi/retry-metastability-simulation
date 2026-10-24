"""Уточнение сетки ρ основного прогона: один раунд, шаг 0,01, окно ±0,05 вокруг каждого нецензурированного порога.

python experiments/refine_grid_main.py --src-tag m_scr --tag-new m_ref1 [--replicates 40] [--hash-file F] [--dry]

Правило (PREREGISTRATION, «Сетка ρ»; src.runner.refine_rhos): по протоколам A и B отдельно, по данным ρ ≤ 1,0
(только доли отказов P_A, P_B и ρ; правое ограничение 1,0), порог оценивается src.thresholds.point_threshold;
цензурированные пороги не уточняются; новые точки — кратные 0,01 в [thr − 0,05; thr + 0,05] ∩ [0,2; 1,0], которых
ещё нет. Используются реплики с номером < replicates. Seed новых задач уникальны (make_tasks).
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
    ap.add_argument("--hash-file", default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    d = stage_dir("main")
    tags = a.src_tag.split(",")
    tasks = load_tasks(d)
    tasks = tasks[tasks["tag"].isin(tags)]
    res = load_results(d, ["config_hash", "rho_test", "protocol", "failed", "replicate"])
    res = res[(res["replicate"] < a.replicates) & (res["rho_test"] <= HI + 1e-9)]
    hashes = sorted(set(tasks["config_hash"].unique()))
    if a.hash_file:
        allowed = {line.split()[0] for line in Path(a.hash_file).read_text().splitlines() if line.strip()}
        hashes = [h for h in hashes if h in allowed]
    total = n_pairs = 0
    pending: list[tuple[str, str, list[float]]] = []
    for h in hashes:
        r = res[res["config_hash"] == h]
        if r.empty:
            continue
        for proto in ("A", "B"):
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
            if new:
                pending.append((h, proto, new))
    for h, proto, new in pending:
        n_pairs += 1
        if a.dry:
            total += len(new) * a.replicates
            continue
        js = str(tasks[tasks["config_hash"] == h]["cfg_json"].iloc[0])
        t = make_tasks("main", [_cfg_from_json(js)], new, [proto], a.replicates, a.tag_new)
        total += len(t)
    print(f"уточнение: пар (конфигурация, протокол) {n_pairs}, {'задач по плану' if a.dry else 'добавлено задач'} {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
