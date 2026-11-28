#!/usr/bin/env bash
# Avaliações (supõe esta pasta montada em ~/ros2_ws/exercicios e o kit de avaliação em ~/ros2_ws/rl)
# dos clássicos no psr_sim (rodar no container: cd ~/ros2_ws/exercicios/avaliacao && ./rodar_avaliacoes.sh)
cd ~/ros2_ws/rl
PY=~/ros2_ws/.venv/bin/python; E=../exercicios; A=$E/avaliacao; R=$E/registros/simulador; S=exports/sumo; C=exports/ctf
run() { echo "### $*"; "$@" 2>&1 | grep -v -i warn | grep -E '^(round|match|RESULT)'; }
# sumô: 10 rodadas de cada lado (20 no total)
for s in 0 1; do
  run $PY eval_sumo.py $E/sumo/sumo_classico.py --rounds 10 --side $s --csv $R/sumo_classico_vs_parado_side$s.csv
  run $PY eval_sumo.py $E/sumo/sumo_classico.py --opponent $E/sumo/anteriores/sumo_classico_simples.py --rounds 10 --side $s --csv $R/sumo_classico_vs_simples_side$s.csv
  run $PY eval_sumo.py $E/sumo/sumo_classico.py --opponent $S/sumo_rl_v5.py --rounds 10 --side $s --csv $R/sumo_classico_vs_rl_v5_side$s.csv
done
# rouba-bandeira: 3 partidas de 5 min, dos dois lados
run $PY eval_ctf.py --blue $A/eval_ctf_classico.py --minutes 5 --matches 3 --csv $R/ctf_classico_vs_parado.csv
run $PY eval_ctf.py --red  $A/eval_ctf_classico.py --minutes 5 --matches 3 --csv $R/ctf_parado_vs_classico.csv
run $PY eval_ctf.py --blue $A/eval_ctf_classico.py --red $A/eval_ctf_classico_anterior.py --minutes 5 --matches 3 --csv $R/ctf_classico_vs_anterior.csv
run $PY eval_ctf.py --blue $A/eval_ctf_classico_anterior.py --red $A/eval_ctf_classico.py --minutes 5 --matches 3 --csv $R/ctf_anterior_vs_classico.csv
run $PY eval_ctf.py --blue $A/eval_ctf_classico.py --red $C/ctf_hibrido_v4.py --minutes 5 --matches 3 --csv $R/ctf_classico_vs_hibrido_v4.csv
run $PY eval_ctf.py --blue $C/ctf_hibrido_v4.py --red $A/eval_ctf_classico.py --minutes 5 --matches 3 --csv $R/ctf_hibrido_v4_vs_classico.csv
echo FIM
