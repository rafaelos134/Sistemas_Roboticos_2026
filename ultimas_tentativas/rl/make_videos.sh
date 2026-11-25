#!/usr/bin/env bash
# Record every phase against every opponent: docs/videos/<fase>_vs_<adversario>.mp4
cd "$(dirname "$0")"
PY=~/ros2_ws/.venv/bin/python; S=exports/sumo; V=docs/videos/sumo
export DISPLAY=${DISPLAY:-:2}
rec() {  # node opponent(or "") title out
  $PY record_video.py "$S/$1" ${2:+--opponent "$S/$2"} --rounds 3 --title "$3" --out "$V/$4.mp4" 2>&1 | grep "^saved"
}
declare -A PH=([fase1]=sumo_rl_v1.py [fase2]=sumo_rl_v2.py [fase3]=sumo_rl_v3.py)
declare -A OP=([parado]="" [chaser]=sumo_chaser.py [flanker]=sumo_flanker.py)
for f in fase1 fase2 fase3; do
  for o in parado chaser flanker; do
    rec "${PH[$f]}" "${OP[$o]}" "${f/fase/Fase } (azul) x $o (laranja)" "${f}_vs_$o" &
  done
  wait
done
rec sumo_rl_v2.py sumo_rl_v1.py "Fase 2 (azul) x Fase 1 (laranja)" fase2_vs_fase1 &
rec sumo_rl_v3.py sumo_rl_v1.py "Fase 3 (azul) x Fase 1 (laranja)" fase3_vs_fase1 &
rec sumo_rl_v3.py sumo_rl_v2.py "Fase 3 (azul) x Fase 2 (laranja)" fase3_vs_fase2 &
wait
rec sumo_rl_v3.py sumo_rl_v3.py "Fase 3 (azul) x Fase 3 (laranja)" fase3_vs_fase3
echo VIDEOSDONE
