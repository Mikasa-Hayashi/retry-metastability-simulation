"""Список задач основного прогона (stage main) по YAML-спецификации; пишет results/main -> results/raw.

python experiments/make_grid_main.py experiments/grids/main_scr.yaml [--dry]

Seed задачи = 1000000 + порядковый номер задачи в списке этапа (src.runner.make_tasks): уникален для каждой задачи,
общие случайные числа не используются (D-018). Идемпотентно: повторный вызов ничего не добавляет.
Спецификация: base_config, set, sweep (декартово произведение), необязательные variants (поправки к каждой
ячейке sweep, у варианта свой список протоколов), rho_parts либо rho_list, protocols, replicates, tag.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.runner import ROOT, _set_dotted, make_tasks, rkey


def build_cfgs_main(spec: dict[str, Any]) -> list[tuple[Config, list[str]]]:
    """[(конфигурация, протоколы)] в детерминированном порядке."""
    base_path = Path(spec["base_config"])
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    with open(base_path, encoding="utf-8") as fh:
        base = yaml.safe_load(fh) or {}
    for k, v in (spec.get("set") or {}).items():
        _set_dotted(base, k, v)
    sweep: dict[str, list[Any]] = spec.get("sweep") or {}
    keys = list(sweep)
    variants = spec.get("variants") or [{"name": "", "set": {}, "protocols": spec["protocols"]}]
    out: list[tuple[Config, list[str]]] = []
    for var in variants:
        for combo in itertools.product(*(sweep[k] for k in keys)):
            data = json.loads(json.dumps(base))
            for k, v in zip(keys, combo, strict=True):
                _set_dotted(data, k, v)
            for k, v in (var.get("set") or {}).items():
                _set_dotted(data, k, v)
            out.append((Config.model_validate(data), list(var.get("protocols", spec.get("protocols")))))
    return out


def rho_points_main(spec: dict[str, Any]) -> list[float]:
    if "rho_list" in spec:
        return sorted({rkey(v) for v in spec["rho_list"]})
    pts: set[float] = set()
    for part in spec.get("rho_parts", []):
        n = round((part["stop"] - part["start"]) / part["step"]) + 1
        pts |= {rkey(part["start"] + i * part["step"]) for i in range(n)}
    return sorted(pts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    with open(a.spec, encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    if spec["stage"] != "main":
        raise SystemExit("эта утилита только для stage: main")
    cfgs = build_cfgs_main(spec)
    rhos = rho_points_main(spec)
    if not rhos:
        raise SystemExit("в спецификации нет сетки ρ (rho_parts/rho_list): используйте topup_grid_main.py")
    reps = int(spec["replicates"])
    n_plan = sum(len(p) for _, p in cfgs) * len(rhos) * reps
    if a.dry:
        print(f"конфигураций {len(cfgs)}, точек ρ {len(rhos)}, задач (до удаления дубликатов) {n_plan}")
        return 0
    groups: dict[tuple[str, ...], list[Config]] = {}
    for cfg, protos in cfgs:
        groups.setdefault(tuple(protos), []).append(cfg)
    total = 0
    lo = hi = None
    for protos, cs in groups.items():
        new = make_tasks("main", cs, rhos, list(protos), reps, spec["tag"])
        total += len(new)
        if len(new):
            lo = int(new.seed.min()) if lo is None else min(lo, int(new.seed.min()))
            hi = int(new.seed.max()) if hi is None else max(hi, int(new.seed.max()))
    print(f"конфигураций {len(cfgs)}, добавлено задач {total} (план {n_plan})"
          + (f", seeds {lo}-{hi}" if total else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
