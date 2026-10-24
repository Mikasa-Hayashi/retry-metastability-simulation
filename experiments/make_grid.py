"""Строит список задач этапа по YAML-спецификации сетки и дописывает его в results/<stage>/tasks-XXXX.parquet.

python experiments/make_grid.py experiments/grids/pilot_calib_A.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runner import configs_from_spec, make_tasks, rho_list


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    args = ap.parse_args()
    with open(args.spec, encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    cfgs = configs_from_spec(spec)
    new = make_tasks(
        spec["stage"], cfgs, rho_list(spec), spec["protocols"], int(spec["replicates"]), spec["tag"]
    )
    print(f"конфигураций {len(cfgs)}, добавлено задач {len(new)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
