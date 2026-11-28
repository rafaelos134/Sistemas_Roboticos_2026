#!/usr/bin/env bash
# Partidas e vídeos que faltavam para a entrega (versões finais). Rodar no container.
cd ~/ros2_ws/rl
PY=~/ros2_ws/.venv/bin/python; R=results/entrega_final; V=$R/videos
N=exports/ctf_zs/ctf_zs_robusto.py; H=exports/ctf/ctf_hibrido_v4.py; G=../classico/entrega/eval_ctf_classico.py
S6=exports/sumo/sumo_rl_v6_fusao.py; SC=../classico/entrega/sumo_classico.py
export OMP_NUM_THREADS=2 DISPLAY=${DISPLAY:-:2}
ev() { out=$1; shift; $PY eval_ctf.py "$@" --minutes 5 --matches 3 --csv $R/$out.csv > $R/$out.log 2>&1; }
ev ctf_final_vs_parado --blue $N &
ev ctf_parado_vs_final --red $N &
ev ctf_final_vs_hibrido_v4 --blue $N --red $H &
ev ctf_hibrido_v4_vs_final --blue $H --red $N &
wait
q() { "$@" > /dev/null 2>&1; }
q $PY record_video.py $S6 --rounds 3 --title "Sumo: RL fase 6 (azul) x parado" --out $V/sumo_rl_vs_parado.mp4 &
q $PY record_video.py $S6 --opponent $SC --rounds 3 --title "Sumo: RL fase 6 (azul) x classico (laranja)" --out $V/sumo_rl_vs_classico.mp4 &
q $PY record_ctf.py --blue $N --seconds 60 --title "Rouba-bandeira: RL final (azul) x parado" --out $V/ctf_rl_vs_parado.mp4 &
q $PY record_ctf.py --blue $N --red $G --seconds 120 --title "Rouba-bandeira: RL final (azul) x classico (vermelho)" --out $V/ctf_rl_vs_classico.mp4 &
wait
echo FIM > $R/DONE
