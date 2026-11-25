#!/usr/bin/env bash
# ./make_ctf_videos.sh [v4]  (v4: only the v4 videos)
# Record the capture-the-flag nodes against each other: docs/videos/ctf/<azul>_vs_<vermelho>.mp4 (180 s = 2 rounds)
cd "$(dirname "$0")"
PY=~/ros2_ws/.venv/bin/python; S=exports/ctf; V=docs/videos/ctf
rec() {  # blue("" = stopped) red("" = stopped) title out [seed]
  $PY record_ctf.py ${1:+--blue "$S/$1"} ${2:+--red "$S/$2"} --seconds 180 --seed "${5:-1}" --title "$3" \
      --out "$V/$4.mp4" 2>&1 | grep "^saved\|Error"
}
if [ "$1" != "v4" ]; then
rec ctf_rl.py "" "RL v3 (att3+def3, azul) x parado" rl_vs_parado &
rec ctf_rl_def4.py "" "RL att3+def4 (azul) x parado" rldef4_vs_parado &
rec ctf_hibrido.py "" "Hibrido att3+def script (azul) x parado" hibrido_vs_parado &
wait
rec ctf_rl.py ctf_script.py "RL v3 (azul) x script (vermelho)" rl_vs_script &
rec ctf_script.py ctf_rl.py "script (azul) x RL v3 (vermelho)" script_vs_rl &
rec ctf_script.py ctf_rl_def4.py "script (azul) x RL att3+def4 (vermelho)" script_vs_rldef4 &
wait
rec ctf_script.py ctf_hibrido.py "script (azul) x Hibrido (vermelho)" script_vs_hibrido &
rec ctf_hibrido.py ctf_rl.py "Hibrido (azul) x RL v3 (vermelho)" hibrido_vs_rl &
rec ctf_rl_def4.py ctf_rl.py "RL att3+def4 (azul) x RL v3 (vermelho)" rldef4_vs_rl &
wait
rec ctf_script.py ctf_script.py "script x script (referencia)" script_vs_script &
rec ctf_script.py "" "script (azul) x parado (referencia)" script_vs_parado &
wait
fi
# v4: attacker fine-tuned with stopped robots (att4), defender def4
rec ctf_rl_v4.py "" "RL v4 (att4+def4, azul) x parado" v4_vs_parado &
rec ctf_hibrido_v4.py "" "Hibrido v4 (att4 + def script, azul) x parado" hibv4_vs_parado &
rec ctf_rl_v4.py ctf_script.py "RL v4 (azul) x script (vermelho)" v4_vs_script &
wait
rec ctf_script.py ctf_rl_v4.py "script (azul) x RL v4 (vermelho)" script_vs_v4 &
rec ctf_rl_v4.py ctf_rl.py "RL v4 (azul) x RL v3 (vermelho)" v4_vs_rl &
rec ctf_hibrido_v4.py ctf_rl_v4.py "Hibrido v4 (azul) x RL v4 (vermelho)" hibv4_vs_v4 &
wait
echo VIDEOSDONE
