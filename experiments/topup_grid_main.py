"""Доведение числа реплик до R на уже существующих точках ρ ≤ 1 (грубых и уточнённых) выбранных конфигураций.

python experiments/topup_grid_main.py --spec experiments/grids/main_active12.yaml --tag m_r100 --src-tags m_scr,m_ref1
python experiments/topup_grid_main.py --escalate --exclude-spec experiments/grids/main_active12.yaml \
    --tag m_esc --replicates 100 --src-tags m_scr,m_ref1 --hash-out results/raw/work/escalation_hashes.txt

--escalate: правило эскалации PREREGISTRATION: ячейка вне 12 активных с точечным Δ ≥ 0,1 и ρ_fold < 1 (не цензурирован)
на ГРУБОЙ сетке (первый тег из --src-tags, реплики < 40, правое ограничение 1,0) доводится до R = 100. Смотрит только на
P_A, P_B. Одно применение (после прогона эскалированных ячеек правило не пересматривается).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_grid_main import build_cfgs_main

from src.config import config_hash
from src.runner import _cfg_from_json, load_results, load_tasks, make_tasks, rkey, stage_dir
from src.thresholds import aggregate, point_threshold

DELTA_MIN = 0.1


def escalation_hashes(coarse_tag: str, exclude: set[str]) -> list[tuple[str, float, float]]:
    d = stage_dir("main")
    tasks = load_tasks(d)
    hs = sorted(set(tasks[tasks["tag"] == coarse_tag]["config_hash"]) - exclude)
    res = load_results(d, ["config_hash", "rho_test", "protocol", "failed", "replicate"])
    res = res[res["replicate"] < 40]
    coarse = set(tasks[tasks["tag"] == coarse_tag]["rho_test"].map(rkey))
    out: list[tuple[str, float, float]] = []
    for h in hs:
        r = res[(res["config_hash"] == h) & (res["rho_test"].map(rkey).isin(coarse)) & (res["rho_test"] <= 1.0 + 1e-9)]
        thr = {}
        for proto in ("A", "B"):
            rp = r[r["protocol"] == proto]
            x, n, k = aggregate(rp["rho_test"].to_numpy(), rp["failed"].to_numpy().astype(int))
            thr[proto] = point_threshold(x, n, k, 0.5, right_bound=1.0)
        fold, fc, _ = thr["A"]
        rel, _rc, _ = thr["B"]
        if fc == "none" and fold < 1.0 and (fold - rel) >= DELTA_MIN:
            out.append((h, fold, rel))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=None)
    ap.add_argument("--escalate", action="store_true")
    ap.add_argument("--exclude-spec", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--replicates", type=int, default=100)
    ap.add_argument("--src-tags", default="m_scr,m_ref1")
    ap.add_argument("--hash-out", default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    d = stage_dir("main")
    src_tags = a.src_tags.split(",")
    if a.escalate:
        excl: set[str] = set()
        if a.exclude_spec:
            with open(a.exclude_spec, encoding="utf-8") as fh:
                excl = {config_hash(c) for c, _p in build_cfgs_main(yaml.safe_load(fh))}
        esc = escalation_hashes(src_tags[0], excl)
        hashes = [h for h, _f, _r in esc]
        print(f"эскалация: ячеек {len(hashes)}")
        for h, f, r in esc:
            print(f"  {h} fold={f:.4f} release={r:.4f}")
        if a.hash_out:
            Path(a.hash_out).write_text("\n".join(f"{h} {f:.6f} {r:.6f}" for h, f, r in esc) + "\n")
    else:
        assert a.spec
        with open(a.spec, encoding="utf-8") as fh:
            hashes = [config_hash(c) for c, _p in build_cfgs_main(yaml.safe_load(fh))]
    tasks = load_tasks(d)
    tasks = tasks[tasks["tag"].isin(src_tags)]
    total = 0
    for h in hashes:
        sel = tasks[tasks["config_hash"] == h]
        if sel.empty:
            raise SystemExit(f"нет задач для {h}")
        cfg = _cfg_from_json(str(sel["cfg_json"].iloc[0]))
        for proto in ("A", "B"):
            rhos = sorted({rkey(v) for v in sel[sel["protocol"] == proto]["rho_test"] if v <= 1.0 + 1e-9})
            if rhos and not a.dry:
                total += len(make_tasks("main", [cfg], rhos, [proto], a.replicates, a.tag))
            elif rhos:
                total += len(rhos) * (a.replicates - 40)
    print(f"ячеек {len(hashes)}, {'задач по плану' if a.dry else 'добавлено задач'} {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
