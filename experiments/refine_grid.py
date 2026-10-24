"""Уточнение сетки ρ вокруг порога по P_A или P_B (правило: docs/PREREGISTRATION.md / CALIBRATION_PLAN.md).

python experiments/refine_grid.py --stage pilot --tag-src calib_A --protocol A --tag-new refine_A \
    --step 0.025 --halfwidth 0.075 --lo 0.3 --hi 0.95 --replicates 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runner import refine_tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--tag-src", required=True)
    ap.add_argument("--tag-new", required=True)
    ap.add_argument("--protocol", required=True, choices=["A", "B"])
    ap.add_argument("--step", type=float, required=True)
    ap.add_argument("--halfwidth", type=float, required=True)
    ap.add_argument("--lo", type=float, required=True)
    ap.add_argument("--hi", type=float, required=True)
    ap.add_argument("--replicates", type=int, required=True)
    a = ap.parse_args()
    new = refine_tasks(
        a.stage, a.tag_src, a.protocol, a.step, a.halfwidth, a.lo, a.hi, a.replicates, a.tag_new
    )
    print(f"добавлено задач {len(new)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
