#!/usr/bin/env bash
# Avaliações (supõe esta pasta montada em ~/ros2_ws/exercicios e o kit de avaliação em ~/ros2_ws/rl)
# que faltavam: futebol RL (A2w) e clássico, e CTF RL v4 x híbrido v4 (híbrido como vermelho).
# Rodar no container: cd ~/ros2_ws/exercicios/avaliacao && ./rodar_avaliacoes_2.sh
cd ~/ros2_ws/rl
PY=~/ros2_ws/.venv/bin/python; E=../exercicios; A=$E/avaliacao; R=$E/registros/simulador; F=exports/soccer; C=exports/ctf
SC=$A/eval_soccer_classico.py; P=$A/soccer_parado.py
run() { local out=$1; shift; { echo "### $*"; "$@" 2>&1 | grep -v -i warn | grep -E '^(match|RESULT|Traceback|  File|[A-Za-z]*Error)'; } > "$R/log_$out.txt"; }
soc() { local out=$1 b=$2 r=$3; run "$out" $PY eval_soccer.py --blue "$b" --red "$r" --minutes 5 --matches 3 --csv "$R/$out.csv"; }
# futebol RL (A2w, entregue) x parado e x A2 (versão anterior), dos dois lados
soc soccer_rl_vs_parado $F/soccer_rl.py $P &
soc soccer_parado_vs_rl $P $F/soccer_rl.py &
soc soccer_rl_vs_a2 $F/soccer_rl.py $F/soccer_rl_a2.py &
soc soccer_a2_vs_rl $F/soccer_rl_a2.py $F/soccer_rl.py &
wait
# futebol clássico x parado, x script simples (versão anterior/base), x RL A2w, dos dois lados
soc soccer_classico_vs_parado $SC $P &
soc soccer_parado_vs_classico $P $SC &
soc soccer_classico_vs_script $SC $F/soccer_script.py &
soc soccer_script_vs_classico $F/soccer_script.py $SC &
wait
soc soccer_classico_vs_rl $SC $F/soccer_rl.py &
soc soccer_rl_vs_classico $F/soccer_rl.py $SC &
run ctf_rl_v4_vs_hibrido_v4 $PY eval_ctf.py --blue $C/ctf_rl_v4.py --red $C/ctf_hibrido_v4.py --minutes 5 --matches 3 --csv $R/ctf_rl_v4_vs_hibrido_v4.csv &
wait
echo FIM
