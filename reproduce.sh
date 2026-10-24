#!/usr/bin/env bash
# Воспроизведение таблиц (results/analysis/) и графиков (figures/) из сырых данных results/raw/ (seed >= 1000000).
# Использование: bash reproduce.sh   (интерпретатор: .venv, либо переменная PY). Время около 4 минут на 8 ядрах.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
if [ -z "${PY:-}" ]; then
  if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
  elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
  else PY=python; fi
fi
for s in main_thresholds main_metrics main_sensitivity main_hypotheses main_factorial main_addendum main_figures main_addendum_figures main_report_tables; do
  echo "== analysis/$s.py"
  "$PY" "analysis/$s.py"
done
echo "Готово: results/analysis/*.csv, figures/F*.png|svg|csv"
