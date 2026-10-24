"""Список задач для pilot, часть 4 (CALIBRATION_PLAN_4.md): общие случайные числа и независимые блоки.

python experiments/make_grid_p4.py experiments/grids/pilot4_scr.yaml

seed_mode: crn   seed = CRN_BASE + номер реплики (одни и те же числа для всех конфигураций, ρ и протоколов;
                 блок 100000-109999; реплика r одной конфигурации в точке ρ соответствует реплике r другой)
seed_mode: indep seed = следующий свободный номер, начиная с INDEP_BASE (независимые блоки: эталоны, покрытие)

Идемпотентно: задачи, уже имеющиеся в списке этапа (тот же config_hash, ρ, протокол и seed либо тег и реплика),
не дублируются. Код src/ не меняется: используются функции src.runner.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, config_hash
from src.runner import (
    ROOT,
    _next_part,
    _set_dotted,
    _write_new,
    cfg_to_json,
    load_tasks,
    rkey,
    stage_dir,
    stage_spec,
)

CRN_BASE = 100_000
CRN_END = 109_999
INDEP_BASE = 110_000


def build_cfgs(spec: dict[str, Any]) -> list[tuple[Config, list[str], str]]:
    """Возвращает [(конфигурация, протоколы, имя варианта)]."""
    base_path = Path(spec["base_config"])
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    with open(base_path, encoding="utf-8") as fh:
        base = yaml.safe_load(fh) or {}
    for k, v in (spec.get("set") or {}).items():
        _set_dotted(base, k, v)
    out: list[tuple[Config, list[str], str]] = []
    if "variants" in spec:
        for var in spec["variants"]:
            data = json.loads(json.dumps(base))
            for k, v in (var.get("set") or {}).items():
                _set_dotted(data, k, v)
            out.append((Config.model_validate(data), list(var["protocols"]), str(var["name"])))
        return out
    sweep: dict[str, list[Any]] = spec.get("sweep") or {}
    keys = list(sweep)
    prune = spec.get("prune") or {}
    for combo in itertools.product(*(sweep[k] for k in keys)):
        d = dict(zip(keys, combo, strict=True))
        if prune and d.get("Q") != prune["theta_q_only_for"] and d.get("theta_q") != prune["theta_q_default"]:
            continue
        data = json.loads(json.dumps(base))
        for k, v in d.items():
            _set_dotted(data, k, v)
        out.append((Config.model_validate(data), list(spec["protocols"]), ""))
    return out


def rho_points(spec: dict[str, Any], protocol: str) -> list[float]:
    if "rho_by_protocol" in spec:
        return sorted({rkey(v) for v in spec["rho_by_protocol"].get(protocol, [])})
    pts: set[float] = set()
    for part in spec["rho_parts"]:
        n = round((part["stop"] - part["start"]) / part["step"]) + 1
        pts |= {rkey(part["start"] + i * part["step"]) for i in range(n)}
    return sorted(pts)


def add_tasks(
    stage: str,
    cfgs: list[tuple[Config, list[str], str]],
    rho_for: Any,
    replicates: int,
    tag: str,
    mode: str,
    rep_start: int = 0,
    root: Path | None = None,
) -> pd.DataFrame:
    spec = stage_spec(stage)
    d = stage_dir(stage, root)
    existing = load_tasks(d)
    ex_keys = set(
        zip(
            existing["config_hash"].astype(str),
            [rkey(r) for r in existing["rho_test"]],
            existing["protocol"].astype(str),
            existing["seed"].astype(int),
            strict=True,
        )
    )
    ex_tag = set(
        zip(
            existing["tag"].astype(str),
            existing["config_hash"].astype(str),
            [rkey(r) for r in existing["rho_test"]],
            existing["protocol"].astype(str),
            existing["replicate"].astype(int),
            strict=True,
        )
    )
    seeds_ex = existing["seed"].astype(int).to_numpy()
    next_indep = int(max(INDEP_BASE - 1, seeds_ex[seeds_ex >= INDEP_BASE].max() if (seeds_ex >= INDEP_BASE).any() else 0)) + 1
    rows: list[dict[str, Any]] = []
    for cfg, protos, _name in cfgs:
        h, js = config_hash(cfg), cfg_to_json(cfg)
        for proto in protos:
            for rho in rho_for(proto):
                for rep in range(rep_start, rep_start + replicates):
                    if mode == "crn":
                        seed = CRN_BASE + rep
                        if seed > CRN_END:
                            raise ValueError("реплика вне общего блока")
                        if (h, rho, proto, seed) in ex_keys:
                            continue
                    else:
                        if (tag, h, rho, proto, rep) in ex_tag:
                            continue
                        seed = next_indep
                        next_indep += 1
                    if not (spec.seed_lo <= seed <= spec.seed_hi):
                        raise ValueError(f"seed {seed} вне диапазона этапа")
                    rows.append(
                        {
                            "config_hash": h,
                            "cfg_json": js,
                            "rho_test": rho,
                            "protocol": proto,
                            "replicate": rep,
                            "seed": seed,
                            "tag": tag,
                        }
                    )
    new = pd.DataFrame(rows, columns=list(existing.columns))
    if len(new):
        _write_new(new, _next_part(d, "tasks"))
    return new


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    with open(a.spec, encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    cfgs = build_cfgs(spec)
    rho_for = lambda p: rho_points(spec, p)
    if a.dry:
        n = sum(len(rho_for(p)) for _, ps, _ in cfgs for p in ps) * int(spec["replicates"])
        print(f"конфигураций {len(cfgs)}, задач (до удаления дубликатов) {n}")
        return 0
    new = add_tasks(
        spec["stage"], cfgs, rho_for, int(spec["replicates"]), spec["tag"],
        spec.get("seed_mode", "crn"), int(spec.get("rep_start", 0)),
    )
    print(f"конфигураций {len(cfgs)}, добавлено задач {len(new)}"
          + (f", seeds {new.seed.min()}-{new.seed.max()}" if len(new) else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
