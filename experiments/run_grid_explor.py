"""Запуск невыполненных задач stage explor (results/explor/); сессия пишется в results/explor/sessions.jsonl.

python experiments/run_grid_explor.py --tag x_k2 [--workers 7] [--batch 500]
python experiments/run_grid_explor.py --tag x_k2 --sample 200 --sample-seed 1   # замер скорости на 200 случайных задачах

Возобновление по ключу (config_hash, ρ, протокол, seed). --sample N: N случайных невыполненных задач тега
(детерминированно по sample-seed); результаты сохраняются как обычные.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import runner


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--sample-seed", type=int, default=1)
    a = ap.parse_args()
    if a.sample is not None:
        orig = runner.pending_tasks

        def sampled(d: Path):  # type: ignore[no-untyped-def]
            t = orig(d)
            t = t[t["tag"] == a.tag].reset_index(drop=True)
            n = min(a.sample, len(t))
            idx = np.sort(np.random.default_rng(a.sample_seed).choice(len(t), size=n, replace=False))
            return t.iloc[idx].reset_index(drop=True)

        runner.pending_tasks = sampled
    info = runner.run_stage("explor", a.batch, a.workers, None, a.tag)
    print(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
