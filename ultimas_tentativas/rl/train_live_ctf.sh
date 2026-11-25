#!/usr/bin/env bash
# Training with the REAL tournament opponents (live recordings, see docs: live_ctf.py / train_ctf_bc.py):
#   1. data/ctf_ao_vivo -> data/live_ctf.npz            (observation + inferred action of every robot)
#   2. behavior cloning of each real team               -> runs/bc_<team>_<att|def>/model.pt
#   3. PPO of the attacker, starting from the deployed one (ctf_att_4), with the imitated real defenders
#      in the defender pool next to the old simulated ones (and the imitated real attackers as "other").
#
#   cd ~/ros2_ws/rl; ./train_live_ctf.sh [nome=ctf_att_real_1] [iters=400] [init=runs/ctf_att_4/model.pt]
set -euo pipefail
cd "$(dirname "$0")"
PY=~/ros2_ws/.venv/bin/python
NAME=${1:-ctf_att_real_1}; IT=${2:-400}; INIT=${3:-runs/ctf_att_4/model.pt}
[ -f data/live_ctf.npz ] || $PY live_ctf.py
[ -f runs/bc_todos_def/model.pt ] || $PY train_ctf_bc.py --all
BC_DEF=($(ls runs/bc_*_def/model.pt | grep -v bc_todos_)); BC_ATT=($(ls runs/bc_*_att/model.pt | grep -v bc_todos_))
DEF=(runs/ctf_def_1/model.pt runs/ctf_def_2/model.pt runs/ctf_def_3/model.pt runs/ctf_def_4/best.pt "${BC_DEF[@]}" runs/bc_todos_def/model.pt)
ATT=(runs/ctf_att_1/model.pt runs/ctf_att_2/model.pt runs/ctf_att_3/model.pt runs/ctf_att_4/model.pt "${BC_ATT[@]}" runs/bc_todos_att/model.pt)
echo "defensores no pool: ${#DEF[@]} (${#BC_DEF[@]} imitados do torneio + todos juntos como 'latest')"
# main (red defender): 50 % pool (4 simulated + 7 real), 15 % all real teams together, 25 % script, 10 % stopped
$PY train_ctf.py --role att --iters "$IT" --name "$NAME" --init "$INIT" --init-std 0.3 \
    --att-pool "${ATT[@]}" --def-pool "${DEF[@]}" \
    --main-mix '{"pool":0.5,"latest":0.15,"script":0.25,"stopped":0.1}' \
    --other-mix '{"latest":0.3,"pool":0.2,"script":0.3,"stopped":0.2}' \
    --easy 0.3 --ent 0.004 2>&1 | grep --line-buffered -v -i warn
echo "avaliação contra os defensores reais imitados:"
./eval_live_ctf.sh "runs/$NAME/model.pt"
