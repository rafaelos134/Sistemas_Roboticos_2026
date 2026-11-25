"""Sanity check: a scripted chaser in the vectorized env must push the stopped
opponent out in about the same time as in psr_sim (3.76 s from the standard start)."""
import math
import torch
from sumo_env import SumoEnv

env = SumoEnv(n_envs=4096, random_start=0.0, obs_noise=0.0)
done_t = torch.full((env.n,), float("nan"), device="cuda")
won = torch.zeros(env.n, dtype=torch.bool, device="cuda")
alive = torch.ones(env.n, dtype=torch.bool, device="cuda")
for k in range(1, 200):
    me, opp = env.state
    a = torch.atan2(opp[:, 1] - me[:, 1], opp[:, 0] - me[:, 0])
    e = torch.remainder(a - me[:, 2] + math.pi, 2 * math.pi) - math.pi
    act = torch.stack([(e.abs() < 0.5).float(), (4.0 * e / 4.0).clamp(-1, 1)], 1)
    _, _, term, trunc, info = env.step(act)
    new = (term | trunc) & alive
    done_t[new] = k * env.control_dt
    won[new] = info["win"][new]
    alive &= ~new
    if not alive.any():
        break
print(f"envs {env.n}: win {won.float().mean():.1%}, push-out time mean {done_t.nanmean():.2f} s "
      f"(min {done_t[~done_t.isnan()].min():.2f}, max {done_t[~done_t.isnan()].max():.2f}); psr_sim: 3.76 s")
