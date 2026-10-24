"""Сверка tasks с результатами; пишет reports/run_log_<stage>.md и results/<stage>/MANIFEST.md."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.protocols import git_commit, package_versions
from src.runner import ROOT, check_complete, stage_dir, write_manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    a = ap.parse_args()
    rep = check_complete(a.stage)
    d = stage_dir(a.stage)
    sessions = []
    sf = d / "sessions.jsonl"
    if sf.exists():
        sessions = [json.loads(line) for line in sf.read_text(encoding="utf-8").splitlines() if line]
    wall = sum(s["elapsed_s"] for s in sessions)
    lines = [
        f"# Журнал прогона: {a.stage}",
        "",
        f"Сформирован: {time.strftime('%Y-%m-%d %H:%M:%S')}; commit: {git_commit()}",
        f"Версии: {package_versions()}",
        "",
        "| показатель | значение |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in rep.items()]
    lines += [f"| время выполнения (сумма сессий), с | {wall:.0f} |", "", "## Сессии раннера", ""]
    lines += [
        f"- {s['start']} — {s['end']}: задач {s['n_tasks']}, ok {s['n_ok']}, ошибок {s['n_err']}, "
        f"процессов {s['workers']}, {s['elapsed_s']:.0f} с, tag={s['tag_filter']}, "
        f"commit {s['git_commit']}"
        for s in sessions
    ]
    out = ROOT / "reports" / f"run_log_{a.stage}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_manifest(a.stage)
    print(json.dumps(rep, ensure_ascii=False))
    return 0 if rep["complete"] and rep["n_error_unresolved"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
