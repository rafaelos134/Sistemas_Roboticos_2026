#!/usr/bin/env bash
# Futebol RL: atacantes A1 -> A2 -> A2w, que geraram o projeto_final/rl/soccer.py (soccer_att_a2w + goleiro por script).
# Requer GPU (Genesis + PyTorch) e o kit em ../kit (ou PSR_KIT=/caminho/do/kit).
set -e; cd "$(dirname "$0")"; PY=${PY:-python3}
$PY train_soccer.py --stage A1 --iters 600  --name soccer_att_a1                                       # time parado
$PY train_soccer.py --stage A2 --iters 1000 --name soccer_att_a2  --init runs/soccer_att_a1/model.pt   # scripts + cópias
$PY train_soccer.py --stage A2 --iters 1000 --name soccer_att_a2w --init runs/soccer_att_a2/model.pt --wall 2.0  # bola longe das paredes
mkdir -p exports
$PY export_soccer.py --att runs/soccer_att_a2w/model.pt --no-unjam --out exports/soccer.py
