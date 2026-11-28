#!/usr/bin/env bash
# Rouba-bandeira RL: atacante PPO + defensor por script (ctf_script.py do kit), que geraram o projeto_final/rl/ctf.py
# (ctf_zs_robusto). Currículo: (1) do zero, parado -> mistura; (2) fuga: largadas perto da bandeira inimiga;
# (3) robusto: adversários variados + latência. Adversários "er" = máquinas de estados estimadas dos 8 times do
# torneio (eng_reversa/estimados/ctf). Requer GPU e o kit em ../kit.
set -e; cd "$(dirname "$0")"; PY=${PY:-python3}
$PY train_ctf_zero.py --name ctf_zs_base --net separate --def-script --iters 1500
$PY train_ctf_zero.py --name ctf_zs_fuga --net separate --def-script --init runs/ctf_zs_base --stage 1 --iters 600 \
    --init-std 0.3 --ent 0.003 --easy 0.8 --score 20 --att-styles "[1,0,0,0]" --def-styles "[1,0,0,0,0]" \
    --strong 1.0 --dr-delay 0.3 --mix '{"script": 0.8, "er": 0.2}'
$PY train_ctf_zero.py --name ctf_zs_robusto --net separate --def-script --init runs/ctf_zs_fuga --stage 1 --iters 800 \
    --init-std 0.15 --ent 0.002 --easy 0.3 --score 20 --att-styles "[0.4,0.2,0.2,0.2]" --def-styles "[0.7,0.1,0.1,0.05,0.05]" \
    --strong 0.8 --dr-delay 0.6 --mix '{"stopped": 0.05, "script": 0.55, "er": 0.4}'
mkdir -p exports
$PY export_ctf.py --att runs/ctf_zs_robusto/att.pt --def-script --out exports/ctf.py
