# Воспроизведение таблиц (results/analysis/) и графиков (figures/) из сырых данных results/raw/ (seed >= 1000000).
# Использование: powershell -File reproduce.ps1   (интерпретатор: .venv\Scripts\python.exe, либо $env:PY)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$py = $env:PY
if (-not $py) { if (Test-Path ".venv\Scripts\python.exe") { $py = ".venv\Scripts\python.exe" } else { $py = "python" } }
foreach ($s in "main_thresholds","main_metrics","main_sensitivity","main_hypotheses","main_factorial","main_addendum","main_figures","main_addendum_figures","main_report_tables") {
  Write-Host "== analysis/$s.py"
  & $py "analysis/$s.py"
  if ($LASTEXITCODE -ne 0) { throw "сбой: $s" }
}
Write-Host "Готово: results/analysis/*.csv, figures/F*.png|svg|csv"
