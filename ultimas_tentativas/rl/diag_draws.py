"""Diagnose sumo draws: in the drawn rounds, how far apart are the robots, where are they, how fast?

    ~/ros2_ws/.venv/bin/python diag_draws.py runs/sumo_ppo_v3/model.pt pursuer_bi
"""
import sys

import torch

from eval_vec import load
from sumo_env import RewardConfig, SumoEnv

model = load(sys.argv[1])
kind = sys.argv[2] if len(sys.argv) > 2 else "pursuer_bi"
env = SumoEnv(n_envs=512, reward=RewardConfig(draw=-3.0), random_start=0.0, obs_noise=0.0)
env.set_opponent_mix({kind: 1.0})
obs = env.reset_all()
T = env.max_steps
rec = {k: torch.zeros(T, env.n, device="cuda") for k in ("d", "r_me", "r_op", "v_me", "v_op", "w_me", "face", "oface")}
alive = torch.ones(env.n, dtype=torch.bool, device="cuda")
res = torch.zeros(env.n, dtype=torch.long, device="cuda")  # 0 running, 1 win, 2 lose, 3 draw
with torch.no_grad():
    for t in range(T):
        me, op = env.state
        dx, dy = op[:, 0] - me[:, 0], op[:, 1] - me[:, 1]
        rec["d"][t] = torch.hypot(dx, dy)
        rec["r_me"][t] = torch.hypot(me[:, 0], me[:, 1])
        rec["r_op"][t] = torch.hypot(op[:, 0], op[:, 1])
        rec["v_me"][t], rec["v_op"][t], rec["w_me"][t] = me[:, 3], op[:, 3], me[:, 4]
        b = torch.atan2(dy, dx)
        rec["face"][t] = torch.cos(b - me[:, 2])           # 1 = I face the opponent
        rec["oface"][t] = torch.cos(b + 3.14159 - op[:, 2])  # 1 = it faces me
        act = model.actor(model.norm(obs)).clamp(-1, 1)
        obs, _, term, trunc, info = env.step(act)
        new = (term | trunc) & alive
        res[new & info["win"]] = 1; res[new & info["lose"]] = 2; res[new & info["draw"]] = 3
        alive &= ~new
        if not alive.any():
            break
drawn = res == 3
print(kind, "win %d lose %d draw %d of %d" % ((res == 1).sum(), (res == 2).sum(), drawn.sum(), env.n))
if drawn.any():
    for k, v in rec.items():
        x = v[:, drawn]
        print("%-6s mean %.3f  |abs| %.3f  (last 10 s mean %.3f)" % (k, x.mean(), x.abs().mean(), x[-200:].mean()))
    # contact: distance < 0.08 m (2 radii + margin)
    print("contact fraction %.2f" % (rec["d"][:, drawn] < 0.08).float().mean())
    i = drawn.nonzero()[0, 0]
    for t in range(0, T, 40):
        print("t %4.1f d %.3f r_me %.2f r_op %.2f v_me %+.2f v_op %+.2f w_me %+.2f face %+.2f oface %+.2f" % (
            t * env.control_dt, *[float(rec[k][t, i]) for k in ("d", "r_me", "r_op", "v_me", "v_op", "w_me", "face", "oface")]))
