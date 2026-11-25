#!/usr/bin/env bash
# Compare the critics (symmetric x asymmetric x asymmetric + Dijkstra), same actor training (stage A2, --wall 2):
#  1) vectorized: 512 kickoffs of 60 s against the scripted team (eval_soccer_vec.py)
#  2) real psr_sim: exported node, 6 matches of 5 min against soccer_script.py (seeds 1..6)
# Results in results/critic_*.txt / .csv
cd "$(dirname "$0")"
PY=/home/ubuntu/ros2_ws/.venv/bin/python; S=exports/soccer
mkdir -p results
declare -A RUN=([a2w]=soccer_att_a2w [symfresh]=soccer_att_a2w_symfresh [asym]=soccer_att_a2w_asym [dij]=soccer_att_a2w_dij)
for k in "${!RUN[@]}"; do
  $PY export_soccer.py --att runs/${RUN[$k]}/model.pt --no-unjam --out $S/soccer_rl_critic_$k.py > /dev/null
done
for k in "${!RUN[@]}"; do
  $PY eval_soccer_vec.py runs/${RUN[$k]}/model.pt 2>&1 | grep -v -i warn > results/critic_vec_$k.txt
done
for k in "${!RUN[@]}"; do
  $PY eval_soccer.py --blue $S/soccer_rl_critic_$k.py --red $S/soccer_script.py --minutes 5 --matches 6 --seed 1 \
      --csv results/critic_sim_$k.csv 2>&1 | grep -E "^(match|RESULT)" > results/critic_sim_$k.txt &
done
wait
for k in a2w symfresh asym dij; do echo "== $k"; cat results/critic_vec_$k.txt results/critic_sim_$k.txt; done
