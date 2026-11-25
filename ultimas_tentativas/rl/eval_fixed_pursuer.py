"""Vectorized eval against pursuer_bi with the fixed gains of exports/sumo/sumo_pursuer.py
(kw 5, lead 0.3 s, full speed, no edge guard).  Usage: eval_fixed_pursuer.py ckpt [ckpt ...]"""
import sys

import torch

from eval_vec import load
from sumo_env import RewardConfig, SumoEnv

env = SumoEnv(n_envs=512, reward=RewardConfig(draw=-3.0), random_start=0.0, obs_noise=0.0)
import itertools
path = sys.argv[1]
model = load(path)
for KW, LEAD, SPD, GUARD, KIND in [(5, .3, 1, 0, "pursuer_bi"), (3, .3, 1, 0, "pursuer_bi"), (8, .3, 1, 0, "pursuer_bi"),
                                    (5, 0, 1, 0, "pursuer_bi"), (5, .5, 1, 0, "pursuer_bi"), (5, .3, .85, 0, "pursuer_bi"),
                                    (5, .3, 1, 1, "pursuer_bi"), (5, .3, 1, 0, "pursuer"), (2, .1, .7, 0, "pursuer")]:
    env.set_opponent_mix({KIND: 1.0})
    obs = env.reset_all()

    def fix():
        env.opp_params[:, 0] = KW; env.opp_params[:, 1] = LEAD + 0.3; env.opp_params[:, 2] = SPD; env.opp_params[:, 5] = 0.2 if GUARD else 0.0
    fix()
    alive = torch.ones(env.n, dtype=torch.bool, device="cuda")
    res = {k: torch.zeros(env.n, dtype=torch.bool, device="cuda") for k in ("win", "lose", "draw")}
    with torch.no_grad():
        for _ in range(env.max_steps + 2):
            obs, _, term, trunc, info = env.step(model.actor(model.norm(obs)).clamp(-1, 1))
            fix()
            new = (term | trunc) & alive
            for k in res:
                res[k] |= info[k] & new
            alive &= ~new
            if not alive.any():
                break
    print("%-11s kw %d lead %.1f spd %.2f guard %d  win %5.1f lose %5.1f draw %5.1f" % (KIND, KW, LEAD, SPD, GUARD, *[100.0 * res[k].float().mean().item() for k in ("win", "lose", "draw")]))
