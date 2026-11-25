"""Smoke test of ctf_env.py: the learning robot also uses the script; prints outcomes, events and speed.

    ~/ros2_ws/.venv/bin/python check_ctf_env.py [--mode attack|defend] [--envs 512] [--seconds 120]
"""
import argparse
import time

import torch

from ctf_env import CtfEnv
from ctf_scripts import attacker_actions, defender_actions

p = argparse.ArgumentParser()
p.add_argument("--mode", default="attack")
p.add_argument("--envs", type=int, default=512)
p.add_argument("--seconds", type=float, default=120.0)
p.add_argument("--stopped", action="store_true", help="main opponent stopped")
a = p.parse_args()
env = CtfEnv(n_envs=a.envs, mode=a.mode, random_start=0.0, easy_start=0.0,
             main_mix={"stopped": 1.0} if a.stopped else {"script": 1.0})
obs = env.reset_all()
assert obs.shape == (a.envs, 38) and torch.isfinite(obs).all(), obs.shape
tot = {k: 0 for k in ("scored", "conceded", "timeout", "pickup", "tagged", "tagger")}
rew_sum, steps = 0.0, int(a.seconds / env.dt)
t0 = time.time()
for _ in range(steps):
    slot = env.agent
    v = env.view(0, env.state)
    mine, theirs, own, enemy, carry, taken, hold, imm = v
    g = env.gains[slot]
    act = (attacker_actions(mine, theirs, own, enemy, carry, hold, g) if slot == 0
           else defender_actions(mine, theirs, own, taken, hold, imm, g))
    obs, rew, term, trunc, info = env.step(act)
    assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    rew_sum += float(rew.mean())
    for k in tot:
        tot[k] += int(info[k].sum())
dt = time.time() - t0
print("mode %s: %d envs x %.0f s simulated in %.1f s (%.0f env-steps/s, %.0fx real time per env)"
      % (a.mode, a.envs, a.seconds, dt, a.envs * steps / dt, a.envs * a.seconds / dt))
print("per env per 90 s:", {k: round(v / a.envs * 90 / a.seconds, 3) for k, v in tot.items()},
      "mean reward/step %.4f" % (rew_sum / steps))
