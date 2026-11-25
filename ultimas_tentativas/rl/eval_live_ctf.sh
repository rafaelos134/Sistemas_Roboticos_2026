#!/usr/bin/env bash
# Evaluates an attacker network against the defenders imitated from the real tournament (runs/bc_*_def)
# and the simulator's script. Usage (container): ./eval_live_ctf.sh runs/ctf_att_4/model.pt [saida.jsonl]
MODEL=${1:-runs/ctf_att_4/model.pt}; OUT=${2:-runs/eval_live_$(basename $(dirname $MODEL)).jsonl}
PY=${PY:-$HOME/ros2_ws/.venv/bin/python}
: > "$OUT"
for d in runs/bc_*_def/model.pt script; do
  if [ "$d" = script ]; then args=(--main script); nome=script; else args=(--def "$d" --main latest); nome=$(basename $(dirname $d)); fi
  r=$($PY eval_ctf_vec.py --role att --model "$MODEL" "${args[@]}" --other script --rounds 2 2>/dev/null | tail -1)
  echo "{\"modelo\": \"$MODEL\", \"defensor\": \"$nome\", \"r\": $r}" | tee -a "$OUT"
done
