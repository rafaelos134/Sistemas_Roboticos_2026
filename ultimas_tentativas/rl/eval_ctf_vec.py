"""Evaluate a CTF network in the vectorized env (deterministic policy), one opponent setting at a time.

    ~/ros2_ws/.venv/bin/python eval_ctf_vec.py --role att --model runs/ctf_att_3/model.pt \
        --att runs/ctf_att_3/model.pt --def runs/ctf_def_3/model.pt --main stopped --other stopped
Kinds for --main / --other: stopped, script, latest (the networks given with --att / --def).
Prints points scored / conceded per 90 s round (kit start, no easy starts).
"""
import argparse
import json

import torch

from ctf_env import CtfEnv
from train_ctf import load_pool

p = argparse.ArgumentParser()
p.add_argument("--role", default="att", choices=("att", "def"))
p.add_argument("--model", required=True)
p.add_argument("--att", nargs="*", default=[])
p.add_argument("--def", dest="dfn", nargs="*", default=[])
p.add_argument("--main", default="stopped")
p.add_argument("--other", default="latest")
p.add_argument("--envs", type=int, default=1024)
p.add_argument("--rounds", type=float, default=1.0, help="simulated time in 90 s rounds")
a = p.parse_args()
env = CtfEnv(n_envs=a.envs, mode="attack" if a.role == "att" else "defend", episode_s=90.0, random_start=0.0,
             easy_start=0.0, obs_noise=0.0, main_mix={a.main: 1.0}, other_mix={a.other: 1.0},
             att_pool=load_pool(a.att), def_pool=load_pool(a.dfn))
me = load_pool([a.model])
obs = env.reset_all()
env.steps[:] = 0
env.ep_steps[:] = 0
tot = {k: 0 for k in ("scored", "conceded", "pickup", "tagged", "tagger")}
for _ in range(int(a.rounds * env.max_steps)):
    obs, rew, term, trunc, info = env.step(me.act(0, obs))
    for k in tot:
        tot[k] += int(info[k].sum())
print(json.dumps({"role": a.role, "main": a.main, "other": a.other,
                  **{k + "_per_round": round(v / a.envs / a.rounds, 3) for k, v in tot.items()}}))
