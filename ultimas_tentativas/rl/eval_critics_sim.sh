#!/usr/bin/env bash
# psr_sim part of eval_critics.sh: one process per match (Genesis can be initialized only once per process)
cd "$(dirname "$0")"
PY=/home/ubuntu/ros2_ws/.venv/bin/python; S=exports/soccer
one() { $PY eval_soccer.py --blue $S/soccer_rl_critic_$1.py --red $S/soccer_script.py --minutes 5 --seed $2 2>&1 | grep "^match" | sed "s/^match 1/seed $2/" >> results/critic_sim_$1.txt; }
for seeds in "2 3" "4 5" "6"; do
  for k in a2w symfresh asym dij; do for s in $seeds; do one $k $s & done; done
  wait
done
