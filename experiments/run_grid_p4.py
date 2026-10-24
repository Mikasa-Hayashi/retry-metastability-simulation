"""Выполняет невыполненные задачи этапа с ограничением по номеру реплики (CALIBRATION_PLAN_4.md, дополнение).

python experiments/run_grid_p4.py --stage pilot --max-rep 20 [--tag T] [--workers N] [--batch 2000]
Отличие от run_grid.py только в том, что из невыполненных задач берутся реплики с номером < max-rep
(общие случайные числа: реплики 0…max-rep−1 образуют префикс полного набора, остальные выполняются позже).
src/runner.py не меняется: подменяется функция отбора невыполненных задач в этом процессе.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import runner


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--max-rep", type=int, required=True)
    ap.add_argument("--min-rep", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    orig = runner.pending_tasks

    def filtered(d: Path):  # type: ignore[no-untyped-def]
        t = orig(d)
        return t[(t["replicate"] < a.max_rep) & (t["replicate"] >= a.min_rep)].reset_index(drop=True)

    runner.pending_tasks = filtered
    info = runner.run_stage(a.stage, a.batch, a.workers, None, a.tag)
    print(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
