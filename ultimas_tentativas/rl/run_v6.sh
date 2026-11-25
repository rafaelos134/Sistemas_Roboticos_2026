#!/usr/bin/env bash
# After run_v5.sh: (5) team training with PFSP, (6) an exploiter of that team, team again with the exploiter
# in the pools, (7) the same team with a GRU memory. Then every candidate is ranked in psr_sim (select_ctf.py).
cd "$(dirname "$0")"
PY=~/ros2_ws/.venv/bin/python
F="grep --line-buffered -v -i warn"
A="runs/ctf_att_1/model.pt runs/ctf_att_2/model.pt runs/ctf_att_3/model.pt runs/ctf_att_4/model.pt runs/ctf_att_5/model.pt"
D="runs/ctf_def_1/model.pt runs/ctf_def_2/model.pt runs/ctf_def_3/model.pt runs/ctf_def_4/best.pt runs/ctf_def_5/model.pt"
echo "== team_1 (5 + PFSP)"
$PY train_ctf_team.py --name ctf_team_1 --iters 800 --init-att runs/ctf_att_5/model.pt --init-def runs/ctf_def_5/model.pt \
    --att-pool $A --def-pool $D --pfsp 2>&1 | $F
echo "== exploiter of team_1 (6)"
$PY train_ctf_team.py --name ctf_expl_1 --iters 400 --init-att runs/ctf_att_5/model.pt --init-def runs/ctf_def_5/model.pt \
    --att-pool runs/ctf_team_1/att.pt --def-pool runs/ctf_team_1/def.pt --main-mix '{"latest": 1.0}' 2>&1 | $F
echo "== team_2 (pools + team_1 + exploiter, PFSP)"
$PY train_ctf_team.py --name ctf_team_2 --iters 600 --init-att runs/ctf_team_1/att.pt --init-def runs/ctf_team_1/def.pt \
    --att-pool $A runs/ctf_expl_1/att.pt runs/ctf_team_1/att.pt --def-pool $D runs/ctf_expl_1/def.pt runs/ctf_team_1/def.pt \
    --pfsp 2>&1 | $F
echo "== team_rnn (7: GRU memory, from team_2)"
$PY train_ctf_team.py --name ctf_team_rnn --iters 500 --rnn --episode-s 90 --init-att runs/ctf_team_2/att.pt \
    --init-def runs/ctf_team_2/def.pt --att-pool $A runs/ctf_expl_1/att.pt runs/ctf_team_2/att.pt \
    --def-pool $D runs/ctf_expl_1/def.pt runs/ctf_team_2/def.pt --pfsp 2>&1 | $F
echo "== selection in psr_sim"
$PY select_ctf.py --role team --minutes 3 --cks runs/ctf_att_4/model.pt:script runs/ctf_att_5/model.pt:script \
    runs/ctf_att_5/model.pt:runs/ctf_def_5/model.pt runs/ctf_team_1/att.pt:runs/ctf_team_1/def.pt \
    runs/ctf_team_2/att.pt:runs/ctf_team_2/def.pt runs/ctf_team_2/att.pt:script \
    runs/ctf_team_rnn/att.pt:runs/ctf_team_rnn/def.pt runs/ctf_team_rnn/att.pt:script 2>&1 | $F
echo V6DONE
