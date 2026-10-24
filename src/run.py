"""CLI: python -m src.run --config configs/x.yaml --seed N --rho 0.9 --protocol A [--out f.parquet]."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from .config import config_hash, load_config
from .protocols import Protocol, run_protocol


def write_parquet(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Дописывает строки в parquet (читает существующий файл и пересохраняет)."""
    p = Path(path)
    df = pd.DataFrame(rows)
    if p.exists():
        df = pd.concat([pd.read_parquet(p), df], ignore_index=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Прогон протокола A/B (MODEL_SPEC §6)")
    ap.add_argument("--config", required=True, help="YAML-конфиг")
    ap.add_argument("--seed", type=int, required=True, help="seed первого прогона")
    ap.add_argument("--n-seeds", type=int, default=1, help="число подряд идущих seed")
    ap.add_argument(
        "--rho", type=float, nargs="+", default=[0.9], help="ρ_test (одно или несколько)"
    )
    ap.add_argument("--protocol", choices=["A", "B", "both"], default="A")
    ap.add_argument("--out", default=None, help="parquet для дозаписи результатов")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    protocols: list[str] = ["A", "B"] if args.protocol == "both" else [args.protocol]
    rows: list[dict[str, Any]] = []
    for rho in args.rho:
        for proto in protocols:
            for seed in range(args.seed, args.seed + args.n_seeds):
                rows.append(run_protocol(cfg, cast(Protocol, proto), rho, seed))
    if args.out:
        write_parquet(rows, args.out)
    print(f"config_hash={config_hash(cfg)}")
    for r in rows:
        print(json.dumps(r, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
