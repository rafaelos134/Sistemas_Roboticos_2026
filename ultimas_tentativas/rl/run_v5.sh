#!/usr/bin/env bash
# v5: fine-tune of both roles against the new script styles + reference gains, new attacker reward terms
# (avoid / immune pickup) and domain randomization; checkpoints every 100 iterations (chosen in psr_sim).
cd "$(dirname "$0")"
PY=~/ros2_ws/.venv/bin/python
COMMON=(--att-styles "[0.4,0.2,0.2,0.2]" --def-styles "[0.3,0.2,0.2,0.15,0.15]" --strong 0.5
        --dr-speed 0.1 --dr-delay 0.3 --obs-noise 0.004 --yaw-noise 0.03 --save-every 100)
A="runs/ctf_att_1/model.pt runs/ctf_att_2/model.pt runs/ctf_att_3/model.pt runs/ctf_att_4/model.pt"
D="runs/ctf_def_1/model.pt runs/ctf_def_2/model.pt runs/ctf_def_3/model.pt runs/ctf_def_4/best.pt"
$PY train_ctf.py --role att --iters 1000 --name ctf_att_5 --init runs/ctf_att_4/model.pt --att-pool $A --def-pool $D \
    --main-mix '{"script":0.5,"latest":0.2,"pool":0.15,"stopped":0.15}' \
    --other-mix '{"latest":0.35,"script":0.35,"stopped":0.3}' "${COMMON[@]}" 2>&1 | grep --line-buffered -v -i warn
$PY train_ctf.py --role def --iters 800 --name ctf_def_5 --init runs/ctf_def_4/best.pt --att-pool $A runs/ctf_att_5/model.pt \
    --def-pool $D --main-mix '{"script":0.5,"latest":0.25,"pool":0.15,"stopped":0.1}' \
    --other-mix '{"latest":0.35,"script":0.35,"stopped":0.3}' "${COMMON[@]}" 2>&1 | grep --line-buffered -v -i warn
echo V5DONE
