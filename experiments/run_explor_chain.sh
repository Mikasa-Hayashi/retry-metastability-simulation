set -e
export PYTHONUTF8=1
P=.venv/Scripts/python.exe
L=results/explor/logs
# порядок: от самых коротких к самому длинному (x_k2)
for t in x_trg x_sens_r2 x_sens_r3 x_ol x_k2; do
  $P experiments/run_grid_explor.py --tag $t > $L/$t.log 2>&1
  $P experiments/refine_grid_explor.py --src-tag $t --tag-new ${t}_ref > $L/${t}_ref_make.log 2>&1
  $P experiments/run_grid_explor.py --tag ${t}_ref > $L/${t}_ref.log 2>&1
done
echo CHAIN_DONE > $L/chain_done.txt
