set -e
export PYTHONUTF8=1
P=.venv/Scripts/python.exe
L=results/raw/logs
G=experiments/grids
$P experiments/refine_grid_main.py --src-tag m_scr --tag-new m_ref1 > $L/m_ref1_make.log 2>&1
$P experiments/run_grid_main.py --tag m_ref1 > $L/m_ref1.log 2>&1
$P experiments/topup_grid_main.py --spec $G/main_active12.yaml --tag m_r100 --src-tags m_scr,m_ref1 > $L/m_r100_make.log 2>&1
$P experiments/run_grid_main.py --tag m_r100 > $L/m_r100.log 2>&1
$P experiments/topup_grid_main.py --escalate --exclude-spec $G/main_active12.yaml --tag m_esc --replicates 100 --src-tags m_scr,m_ref1 --hash-out results/raw/work/escalation_hashes.txt > $L/m_esc_make.log 2>&1
$P experiments/run_grid_main.py --tag m_esc > $L/m_esc.log 2>&1
$P experiments/make_grid_main.py $G/main_ext.yaml > $L/m_ext_make.log 2>&1
$P experiments/run_grid_main.py --tag m_ext > $L/m_ext.log 2>&1
$P experiments/make_grid_main.py $G/main_sens.yaml > $L/m_sens_make.log 2>&1
$P experiments/run_grid_main.py --tag m_sens > $L/m_sens.log 2>&1
$P experiments/refine_grid_main.py --src-tag m_sens --tag-new m_sens_ref > $L/m_sens_ref_make.log 2>&1
$P experiments/run_grid_main.py --tag m_sens_ref > $L/m_sens_ref.log 2>&1
echo CHAIN_DONE > $L/chain_done.txt
