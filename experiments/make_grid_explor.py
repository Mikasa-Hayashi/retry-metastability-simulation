"""Список задач эксплораторных дозапусков (stage explor, results/explor/); основной прогон не затрагивается.

python experiments/make_grid_explor.py experiments/grids/explor_k2.yaml [--dry]

Seed задачи = 1400000 + порядковый номер задачи в списке этапа (src.runner.make_tasks): уникален для каждой задачи.
Спецификация: base_config, set, cells (список ячеек: ключи через точку) и/или sweep (декартово произведение),
variants (поправки к каждой ячейке; у варианта свой список протоколов), rho_parts, protocols, replicates, tag.
Идемпотентно: повторный вызов ничего не добавляет.
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


def build_cfgs(spec: dict[str, Any]) -> list[tuple[Config, list[str], str]]:
    """[(конфигурация, протоколы, имя варианта)] в детерминированном порядке."""
    base_path = Path(spec["base_config"])
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    with open(base_path, encoding="utf-8") as fh:
        base = yaml.safe_load(fh) or {}
    for k, v in (spec.get("set") or {}).items():
        _set_dotted(base, k, v)
    cells: list[dict[str, Any]] = list(spec.get("cells") or [])
    sweep: dict[str, list[Any]] = spec.get("sweep") or {}
    if sweep:
        keys = list(sweep)
        cells += [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*(sweep[k] for k in keys))]
    if not cells:
        cells = [{}]
    variants = spec.get("variants") or [{"name": "", "set": {}, "protocols": spec["protocols"]}]
    out: list[tuple[Config, list[str], str]] = []
    for var in variants:
        for cell in cells:
            data = json.loads(json.dumps(base))
            for k, v in cell.items():
                _set_dotted(data, k, v)
            for k, v in (var.get("set") or {}).items():
                _set_dotted(data, k, v)
            out.append(
                (Config.model_validate(data), list(var.get("protocols", spec.get("protocols"))), str(var.get("name", "")))
            )
    return out


def rho_points(spec: dict[str, Any]) -> list[float]:
    pts: set[float] = set()
    for part in spec["rho_parts"]:
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
    if spec["stage"] != "explor":
        raise SystemExit("эта утилита только для stage: explor")
    cfgs = build_cfgs(spec)
    rhos = rho_points(spec)
    reps = int(spec["replicates"])
    n_plan = sum(len(p) for _, p, _ in cfgs) * len(rhos) * reps
    if a.dry:
        print(f"конфигураций {len(cfgs)}, точек ρ {len(rhos)}, задач {n_plan}")
        return 0
    groups: dict[tuple[str, ...], list[Config]] = {}
    for cfg, protos, _name in cfgs:
        groups.setdefault(tuple(protos), []).append(cfg)
    total = 0
    lo = hi = None
    for protos, cs in groups.items():
        new = make_tasks("explor", cs, rhos, list(protos), reps, spec["tag"])
        total += len(new)
        if len(new):
            lo = int(new.seed.min()) if lo is None else min(lo, int(new.seed.min()))
            hi = int(new.seed.max()) if hi is None else max(hi, int(new.seed.max()))
    print(f"конфигураций {len(cfgs)}, добавлено задач {total} (план {n_plan})" + (f", seeds {lo}-{hi}" if total else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
