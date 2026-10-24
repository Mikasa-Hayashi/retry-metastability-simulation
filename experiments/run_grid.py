"""Выполняет невыполненные задачи этапа (возобновление по ключу), пишет part-XXXX.parquet.

python experiments/run_grid.py --stage pilot [--workers N] [--batch 500] [--limit N] [--tag T]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runner import run_stage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--workers", type=int, default=None, help="по умолчанию число ядер − 1")
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    info = run_stage(a.stage, a.batch, a.workers, a.limit, a.tag)
    print(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
