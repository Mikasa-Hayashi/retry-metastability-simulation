"""Раннер серий прогонов: список задач, параллельный запуск, возобновление, журнал ошибок.

Хранение (results/<stage>/):
  tasks-XXXX.parquet   список задач (config_hash, cfg_json, rho_test, protocol, replicate, seed, tag);
                       файлы только добавляются
  part-XXXX.parquet    результаты, один файл на пачку задач, пишется сразу после пачки
  errors-XXXX.parquet  ошибки задач (задача + трейсбек), ничего не удаляется
  sessions.jsonl       журнал запусков раннера (время, число задач, процессы)

Ключ возобновления: (config_hash, ρ_test, protocol, seed). Seed задачи = stage_base + порядковый
номер задачи в списке этапа (детерминированно). Pilot: 0–999999, main: от 1000000.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import Config, config_hash
from .protocols import Protocol, git_commit, package_versions, run_protocol
from .thresholds import aggregate, point_threshold

ROOT = Path(__file__).resolve().parent.parent
KEY_COLS = ["config_hash", "rho_test", "protocol", "seed"]
RHO_DECIMALS = 6


@dataclass(frozen=True)
class StageSpec:
    name: str
    seed_lo: int
    seed_hi: int  # включительно


STAGES: dict[str, StageSpec] = {
    "pilot": StageSpec("pilot", 0, 999_999),
    "main": StageSpec("main", 1_000_000, 2**31 - 1),
    # эксплораторные дозапуски (results/explor/), вне основного анализа
    "explor": StageSpec("explor", 1_400_000, 1_999_999),
}


def stage_spec(stage: str) -> StageSpec:
    if stage not in STAGES:
        raise ValueError(f"неизвестный этап {stage!r}; допустимы {sorted(STAGES)}")
    return STAGES[stage]


def stage_dir(stage: str, root: Path | None = None) -> Path:
    stage_spec(stage)
    if root is None and stage == "main":
        return ROOT / "results" / "raw"  # основной прогон хранится в results/raw/
    return (root or ROOT / "results") / stage


def src_hash() -> str:
    """SHA-256 конкатенации src/*.py в алфавитном порядке (как в reports/validation.md)."""
    h = hashlib.sha256()
    for f in sorted((ROOT / "src").glob("*.py")):
        if f.name == "runner.py":
            continue  # раннер не влияет на семантику модели
        h.update(f.read_bytes())
    return h.hexdigest()


def cfg_to_json(cfg: Config) -> str:
    return json.dumps(cfg.model_dump(), sort_keys=True, separators=(",", ":"))


def rkey(rho: float) -> float:
    return round(float(rho), RHO_DECIMALS)


# ---------------------------------------------------------------- файлы

def _parts(d: Path, prefix: str) -> list[Path]:
    return sorted(d.glob(f"{prefix}-*.parquet"))


def _next_part(d: Path, prefix: str) -> Path:
    n = len(_parts(d, prefix))
    return d / f"{prefix}-{n:04d}.parquet"


def _write_new(df: pd.DataFrame, path: Path) -> None:
    """Пишет во временный файл и переименовывает: готовый файл не бывает недописанным."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def load_tasks(d: Path) -> pd.DataFrame:
    ps = _parts(d, "tasks")
    if not ps:
        return pd.DataFrame(
            columns=["config_hash", "cfg_json", "rho_test", "protocol", "replicate", "seed", "tag"]
        )
    return pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True)


def load_results(d: Path, columns: list[str] | None = None) -> pd.DataFrame:
    ps = _parts(d, "part")
    if not ps:
        return pd.DataFrame(columns=columns or KEY_COLS)
    return pd.concat([pd.read_parquet(p, columns=columns) for p in ps], ignore_index=True)


def load_errors(d: Path) -> pd.DataFrame:
    ps = _parts(d, "errors")
    if not ps:
        return pd.DataFrame(columns=[*KEY_COLS, "error", "traceback", "attempt_time"])
    return pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True)


def done_keys(d: Path) -> set[tuple[str, float, str, int]]:
    df = load_results(d, KEY_COLS)
    return {
        (str(h), rkey(r), str(p), int(s))
        for h, r, p, s in zip(
            df["config_hash"], df["rho_test"], df["protocol"], df["seed"], strict=True
        )
    }


def _task_key(h: Any, r: Any, p: Any, s: Any) -> tuple[str, float, str, int]:
    return (str(h), rkey(r), str(p), int(s))


# ---------------------------------------------------------------- задачи

def make_tasks(
    stage: str,
    cfgs: Sequence[Config],
    rhos: Iterable[float],
    protocols: Sequence[str],
    replicates: int,
    tag: str,
    root: Path | None = None,
) -> pd.DataFrame:
    """Добавляет в список задач этапа недостающие реплики (до `replicates` на точку).

    Идемпотентно: повторный вызов с теми же аргументами ничего не добавляет. Seed =
    stage_base + порядковый номер задачи в списке этапа.
    """
    spec = stage_spec(stage)
    d = stage_dir(stage, root)
    existing = load_tasks(d)
    counts: dict[tuple[str, float, str], int] = {}
    for h, r, p in zip(
        existing["config_hash"], existing["rho_test"], existing["protocol"], strict=True
    ):
        k = (str(h), rkey(r), str(p))
        counts[k] = counts.get(k, 0) + 1
    next_idx = len(existing)
    rows: list[dict[str, Any]] = []
    rho_list = sorted({rkey(r) for r in rhos})
    for cfg in cfgs:
        h, js = config_hash(cfg), cfg_to_json(cfg)
        for rho, proto in itertools.product(rho_list, protocols):
            have = counts.get((h, rho, proto), 0)
            for rep in range(have, replicates):
                seed = spec.seed_lo + next_idx + len(rows)
                if seed > spec.seed_hi:
                    raise ValueError(f"seed {seed} вне диапазона этапа {stage}")
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


def pending_tasks(d: Path) -> pd.DataFrame:
    tasks = load_tasks(d)
    done = done_keys(d)
    mask = [
        _task_key(h, r, p, s) not in done
        for h, r, p, s in zip(
            tasks["config_hash"],
            tasks["rho_test"],
            tasks["protocol"],
            tasks["seed"],
            strict=True,
        )
    ]
    return tasks[np.asarray(mask, dtype=bool)].reset_index(drop=True)


# ---------------------------------------------------------------- исполнение

@lru_cache(maxsize=64)
def _cfg_from_json(js: str) -> Config:
    return Config.model_validate(json.loads(js))


def _work(rec: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Выполняет одну задачу. Исключение превращается в запись об ошибке."""
    try:
        cfg = _cfg_from_json(rec["cfg_json"])
        row = run_protocol(cfg, cast(Protocol, rec["protocol"]), rec["rho_test"], rec["seed"])
        row["replicate"] = int(rec["replicate"])
        row["tag"] = rec["tag"]
        row["src_hash"] = src_hash()
        return True, row
    except Exception as exc:  # noqa: BLE001 - любая ошибка задачи логируется, прогон продолжается
        return False, {
            "config_hash": rec["config_hash"],
            "rho_test": rkey(rec["rho_test"]),
            "protocol": rec["protocol"],
            "seed": int(rec["seed"]),
            "tag": rec["tag"],
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "attempt_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def _iter_results(
    recs: list[dict[str, Any]], pool: Any | None
) -> Iterator[tuple[bool, dict[str, Any]]]:
    if pool is None:
        yield from (_work(r) for r in recs)
    else:
        yield from pool.imap_unordered(_work, recs, chunksize=2)


def run_stage(
    stage: str,
    batch_size: int = 500,
    workers: int | None = None,
    limit: int | None = None,
    tag: str | None = None,
    root: Path | None = None,
    log: Any = None,
) -> dict[str, Any]:
    """Выполняет невыполненные задачи этапа партиями; каждая партия сразу пишется в part-XXXX."""
    d = stage_dir(stage, root)
    log = log or sys.stderr
    pend = pending_tasks(d)
    if tag is not None:
        pend = pend[pend["tag"] == tag].reset_index(drop=True)
    if limit is not None:
        pend = pend.iloc[:limit]
    total = len(pend)
    workers = workers or default_workers()
    t0 = time.time()
    n_ok = n_err = 0
    pool = mp.get_context("spawn").Pool(workers) if workers > 1 else None
    try:
        for start in range(0, total, batch_size):
            recs = cast(list[dict[str, Any]], pend.iloc[start : start + batch_size].to_dict("records"))
            oks: list[dict[str, Any]] = []
            errs: list[dict[str, Any]] = []
            for ok, row in _iter_results(recs, pool):
                (oks if ok else errs).append(row)
            if oks:
                _write_new(pd.DataFrame(oks), _next_part(d, "part"))
            if errs:
                _write_new(pd.DataFrame(errs), _next_part(d, "errors"))
            n_ok += len(oks)
            n_err += len(errs)
            el = time.time() - t0
            n_done = n_ok + n_err
            eta = el / n_done * (total - n_done) if n_done else float("nan")
            print(
                f"[{stage}] {n_done}/{total} ok={n_ok} err={n_err} "
                f"elapsed={el:.0f}s eta={eta:.0f}s",
                file=log,
                flush=True,
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    info = {
        "start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
        "end": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": time.time() - t0,
        "workers": workers,
        "n_tasks": total,
        "n_ok": n_ok,
        "n_err": n_err,
        "tag_filter": tag,
        "git_commit": git_commit(),
        "src_hash": src_hash(),
        "versions": package_versions(),
    }
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "sessions.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(info, ensure_ascii=False) + "\n")
    return info


# ---------------------------------------------------------------- уточнение сетки

def refine_rhos(
    rho: np.ndarray,
    failed: np.ndarray,
    step: float,
    halfwidth: float,
    lo: float,
    hi: float,
    level: float = 0.5,
) -> list[float]:
    """Новые точки ρ с шагом `step` вокруг порога (по долям отказов P(ρ) одного протокола).

    Окно [thr − halfwidth, thr + halfwidth]; при цензурировании справа — [hi − halfwidth, hi],
    слева — [lo, lo + halfwidth]. Кратные step точки, которых ещё нет в сетке.
    """
    x, n, k = aggregate(rho, failed)
    thr, cens, _ = point_threshold(x, n, k, level, right_bound=hi)
    if cens == "right":
        a, b = hi - halfwidth, hi
    elif cens == "left":
        a, b = lo, lo + halfwidth
    else:
        a, b = thr - halfwidth, thr + halfwidth
    a, b = max(a, lo), min(b, hi)
    i0, i1 = int(np.ceil(a / step - 1e-9)), int(np.floor(b / step + 1e-9))
    have = {rkey(v) for v in x}
    return [rkey(i * step) for i in range(i0, i1 + 1) if rkey(i * step) not in have]


def refine_tasks(
    stage: str,
    tag_src: str,
    protocol: str,
    step: float,
    halfwidth: float,
    lo: float,
    hi: float,
    replicates: int,
    tag_new: str,
    root: Path | None = None,
) -> pd.DataFrame:
    """Добавляет задачи уточнения. Читает только failed/ρ/протокол (P_A или P_B), не другие метрики."""
    d = stage_dir(stage, root)
    tasks = load_tasks(d)
    sel = tasks[(tasks["tag"] == tag_src) & (tasks["protocol"] == protocol)]
    res = load_results(d, ["config_hash", "rho_test", "protocol", "failed"])
    res = res[res["protocol"] == protocol]
    out: list[pd.DataFrame] = []
    for h in sel["config_hash"].unique():
        r = res[res["config_hash"] == h]
        if r.empty:
            continue
        new_rho = refine_rhos(
            r["rho_test"].to_numpy(), r["failed"].to_numpy(), step, halfwidth, lo, hi
        )
        if not new_rho:
            continue
        js = str(sel[sel["config_hash"] == h]["cfg_json"].iloc[0])
        out.append(
            make_tasks(stage, [_cfg_from_json(js)], new_rho, [protocol], replicates, tag_new, root)
        )
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------------------------------------------------------- проверка полноты и манифест

def check_complete(stage: str, root: Path | None = None) -> dict[str, Any]:
    d = stage_dir(stage, root)
    tasks = load_tasks(d)
    res = load_results(d, KEY_COLS)
    tk = [
        _task_key(h, r, p, s)
        for h, r, p, s in zip(
            tasks["config_hash"],
            tasks["rho_test"],
            tasks["protocol"],
            tasks["seed"],
            strict=True,
        )
    ]
    rk = [
        _task_key(h, r, p, s)
        for h, r, p, s in zip(
            res["config_hash"], res["rho_test"], res["protocol"], res["seed"], strict=True
        )
    ]
    ts, rs = set(tk), set(rk)
    err = load_errors(d)
    err_keys = {
        _task_key(h, r, p, s)
        for h, r, p, s in zip(
            err["config_hash"], err["rho_test"], err["protocol"], err["seed"], strict=True
        )
    }
    spec = stage_spec(stage)
    seeds = tasks["seed"].to_numpy() if len(tasks) else np.array([], dtype=int)
    return {
        "n_tasks": len(tk),
        "n_tasks_unique": len(ts),
        "n_results": len(rk),
        "n_missing": len(ts - rs),
        "n_duplicates": len(rk) - len(rs),
        "n_extra": len(rs - ts),
        "n_error_records": len(err),
        "n_error_tasks": len(err_keys),
        "n_error_unresolved": len(err_keys - rs),
        "seed_in_range": bool(((seeds >= spec.seed_lo) & (seeds <= spec.seed_hi)).all()),
        "complete": len(ts - rs) == 0 and len(rk) == len(rs) and not (rs - ts),
    }


def write_manifest(stage: str, root: Path | None = None) -> Path:
    """results/<stage>/MANIFEST.md: файлы, число строк, SHA-256."""
    d = stage_dir(stage, root)
    lines = [
        f"# MANIFEST results/{stage}",
        "",
        f"Сформирован: {time.strftime('%Y-%m-%d %H:%M:%S')}. Файлы только добавляются, не удаляются.",
        "",
        "| файл | строк | SHA-256 |",
        "|---|---|---|",
    ]
    for f in sorted(d.glob("*.parquet")):
        n = pq.ParquetFile(f).metadata.num_rows
        lines.append(f"| {f.name} | {n} | {hashlib.sha256(f.read_bytes()).hexdigest()} |")
    for f in sorted(d.glob("*.jsonl")):
        n = len(f.read_text(encoding="utf-8").splitlines())
        lines.append(f"| {f.name} | {n} | {hashlib.sha256(f.read_bytes()).hexdigest()} |")
    lines += [
        "",
        "Описание: tasks-* — список задач; part-* — результаты (MODEL_SPEC §10 + служебные поля);",
        "errors-* — ошибки задач с трейсбеком; sessions.jsonl — журнал запусков раннера.",
    ]
    out = d / "MANIFEST.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------- спецификация сетки (YAML)

def _set_dotted(d: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def configs_from_spec(spec: dict[str, Any]) -> list[Config]:
    """Базовый YAML + декартово произведение `sweep` (ключи через точку, напр. proto.trigger_mult)."""
    import yaml

    base_path = Path(spec["base_config"])
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    with open(base_path, encoding="utf-8") as fh:
        base = yaml.safe_load(fh) or {}
    for k, v in (spec.get("set") or {}).items():
        _set_dotted(base, k, v)
    sweep: dict[str, list[Any]] = spec.get("sweep") or {}
    keys = list(sweep)
    out: list[Config] = []
    for combo in itertools.product(*(sweep[k] for k in keys)):
        data = json.loads(json.dumps(base))
        for k, v in zip(keys, combo, strict=True):
            _set_dotted(data, k, v)
        out.append(Config.model_validate(data))
    return out


def rho_list(spec: dict[str, Any]) -> list[float]:
    r = spec["rho"]
    if isinstance(r, list):
        return [rkey(v) for v in r]
    n = round((r["stop"] - r["start"]) / r["step"]) + 1
    return [rkey(r["start"] + i * r["step"]) for i in range(n)]
