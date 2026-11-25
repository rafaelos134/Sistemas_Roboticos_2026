#!/usr/bin/env bash
# Alternating training of capture the flag: the attack trains against the defense, the defense against the attack.
#   round 1: ctf_att_1 vs scripted defenders          -> ctf_def_1 vs ctf_att_1 (+ scripts)
#   round k: ctf_att_k vs ctf_def_{k-1} (+ older ones) -> ctf_def_k vs ctf_att_k (+ older ones)
# Each version starts from the previous one of its role; the pools keep every earlier version, so the
# networks do not forget how to play against older opponents (fictitious self-play).
#
#   cd ~/ros2_ws/rl; ./ctf_league.sh [rounds=3] [iters_att=500] [iters_def=400]  2>&1 | tee runs/ctf_league.log
set -euo pipefail
cd "$(dirname "$0")"
PY=~/ros2_ws/.venv/bin/python
ROUNDS=${1:-3}; IA=${2:-500}; ID=${3:-400}
ATT=(); DEF=()
for k in $(seq 1 "$ROUNDS"); do
  init=(); [ "$k" -gt 1 ] && init=(--init "runs/ctf_att_$((k-1))/model.pt")
  echo "=== round $k: attack (vs ${DEF[*]:-scripts})"
  $PY train_ctf.py --role att --iters "$IA" --name "ctf_att_$k" "${init[@]}" \
      ${ATT[@]:+--att-pool "${ATT[@]}"} ${DEF[@]:+--def-pool "${DEF[@]}"} 2>&1 | grep --line-buffered -v -i warn
  ATT+=("runs/ctf_att_$k/model.pt")

  init=(); [ "$k" -gt 1 ] && init=(--init "runs/ctf_def_$((k-1))/model.pt")
  echo "=== round $k: defense (vs ${ATT[*]})"
  $PY train_ctf.py --role def --iters "$ID" --name "ctf_def_$k" "${init[@]}" \
      --att-pool "${ATT[@]}" ${DEF[@]:+--def-pool "${DEF[@]}"} 2>&1 | grep --line-buffered -v -i warn
  DEF+=("runs/ctf_def_$k/model.pt")
done
echo LEAGUEDONE
