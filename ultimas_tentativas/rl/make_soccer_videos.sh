#!/usr/bin/env bash
# Record the soccer versions against the opponents: docs/videos/futebol_<azul>_vs_<vermelho>.mp4 (90 s each)
cd "$(dirname "$0")"
PY=~/ros2_ws/.venv/bin/python; S=exports/soccer; V=docs/videos/futebol
rec() {  # blue red(or "") title out [seed]
  $PY record_soccer.py --blue "$S/$1" ${2:+--red "$S/$2"} --seconds 90 --seed "${5:-3}" --title "$3" --out "$V/$4.mp4" 2>&1 | grep "^saved"
}
rec soccer_rl_a1.py "" "A1 (azul) x parado" futebol_a1_vs_parado &
rec soccer_rl_a2.py "" "A2 (azul) x parado" futebol_a2_vs_parado &
rec soccer_rl.py "" "A2w final (azul) x parado" futebol_final_vs_parado &
wait
rec soccer_rl_a1.py soccer_script.py "A1 (azul) x script (vermelho)" futebol_a1_vs_script &
rec soccer_rl_a2.py soccer_script.py "A2 (azul) x script (vermelho)" futebol_a2_vs_script &
rec soccer_rl.py soccer_script.py "A2w final (azul) x script (vermelho)" futebol_final_vs_script &
wait
rec soccer_rl.py soccer_script.py "A2w final x script - partida com travamento (seed 1)" futebol_final_vs_script_travamento 1 &
rec soccer_rl_a3.py soccer_script.py "A3 self-play (azul) x script (vermelho)" futebol_a3_vs_script &
rec soccer_rl_hibrido.py soccer_script.py "A2w + supervisor (azul) x script (vermelho)" futebol_hibrido_vs_script &
wait
rec soccer_rl.py soccer_rl_a2.py "A2w final (azul) x A2 (vermelho)" futebol_final_vs_a2 &
rec soccer_rl_a3.py soccer_rl.py "A3 (azul) x A2w final (vermelho)" futebol_a3_vs_final &
rec soccer_script.py soccer_rl.py "script (azul) x A2w final (vermelho)" futebol_script_vs_final &
wait
rec soccer_script.py soccer_script.py "script x script (referência)" futebol_script_vs_script
echo VIDEOSDONE
