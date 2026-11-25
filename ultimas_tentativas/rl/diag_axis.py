"""Does the per-axis force limit (0.2 N on x and on y) favour a push at an angle to the world axes?
Clinch start (touching, face to face, random line angle phi). The opponent is a pursuer (lead 0,
full speed). The agent drives at full speed with its heading kept at (line angle + delta).
After 3 s: how far the pair moved towards the opponent (positive = the agent wins ground)."""
import math

import torch

from opponents import SCRIPTED
from sumo_env import RewardConfig, SumoEnv

env = SumoEnv(n_envs=2048, reward=RewardConfig(), random_start=0.0, clinch_start=1.0, obs_noise=0.0)
env.set_opponent_mix({"pursuer": 1.0})
for delta_deg in (0, 15, 30, 45):
    env.reset_all()
    env.opp_params[:, 0] = 5.0; env.opp_params[:, 1] = 0.3; env.opp_params[:, 2] = 1.0; env.opp_params[:, 5] = 0.0
    me0, op0 = [s.clone() for s in env.state]
    phi = torch.atan2(op0[:, 1] - me0[:, 1], op0[:, 0] - me0[:, 0])
    u = torch.stack([torch.cos(phi), torch.sin(phi)], 1)
    c0 = (me0[:, :2] + op0[:, :2]) / 2
    sign = torch.where(torch.rand(env.n, device="cuda") < 0.5, 1.0, -1.0)
    with torch.no_grad():
        for t in range(60):
            me, op = env.state
            line = torch.atan2(op[:, 1] - me[:, 1], op[:, 0] - me[:, 0])
            e = torch.remainder(line + sign * math.radians(delta_deg) - me[:, 2] + math.pi, 2 * math.pi) - math.pi
            act = torch.stack([torch.ones(env.n, device="cuda"), (5.0 * e / 4.0).clamp(-1, 1)], 1)
            env.step(act)
    me, op = env.state
    c1 = (me[:, :2] + op[:, :2]) / 2
    gain = ((c1 - c0) * u).sum(1)  # displacement towards the opponent
    a = torch.remainder(phi, math.pi / 2)
    a = torch.minimum(a, math.pi / 2 - a)  # 0 = line along an axis, pi/4 = diagonal
    out = env.steps < 60  # some rounds ended (reset): ignore them
    ok = ~out
    print("delta %2d deg:" % delta_deg, " ".join(
        "phi%2d-%2d %+.3f" % (lo, lo + 15, float(gain[ok & (a >= math.radians(lo)) & (a < math.radians(lo + 15))].mean()))
        for lo in (0, 15, 30)), " ended %d" % int(out.sum()))
