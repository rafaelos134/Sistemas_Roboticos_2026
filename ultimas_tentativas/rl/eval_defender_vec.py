"""Defender vs scripted keeper, same situations: blue keeper (id 0) against red attackers =
frozen copies of our attacker network (and the blue attackers = the same network).
Episodes of 60 s from the kickoff and from 'easy' shots at our goal.

    ~/ros2_ws/.venv/bin/python eval_defender_vec.py runs/soccer_def/model.pt --attacker runs/soccer_att_a3/model.pt
"""
import argparse

import torch

from ppo import actor_state

from frozen_pool import FrozenPool
from soccer_env import SoccerEnv
from soccer_scripts import keeper_actions
from train_soccer import make_model


@torch.no_grad()
def run(env, policy):
    obs = env.reset_all()
    env.script_p[:] = torch.tensor([4.0, 0.5, 1.0, 0.08], device="cuda")
    res = torch.zeros(3, device="cuda")
    alive = torch.ones(env.n, dtype=torch.bool, device="cuda")
    for _ in range(env.max_steps + 1):
        if policy is None:
            rob, ball = env.state
            a = keeper_actions(env._team_views(rob, ball)[0], env.script_p[0])
        else:
            a = policy.actor(policy.norm(obs)).clamp(-1, 1)
        obs, r, term, trunc, info = env.step(a)
        new = (info["scored"] | info["conceded"] | info["timeout"]) & alive
        res += torch.stack([(info["conceded"] & new).sum(), (info["scored"] & new).sum(), (info["timeout"] & new).sum()]).float()
        alive &= ~new
        if not alive.any():
            break
    return (100 * res / env.n).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--attacker", required=True)
    p.add_argument("--envs", type=int, default=512)
    a = p.parse_args()
    pol = make_model().cuda().eval()
    pol.load_state_dict(actor_state(torch.load(a.ckpt, map_location="cuda", weights_only=False)["model"]), strict=False)
    pool = FrozenPool(make_model)
    pool.add(torch.load(a.attacker, map_location="cuda", weights_only=False)["model"], "att")
    for name, (rs, es) in (("kickoff", (0.0, 0.0)), ("shots at our goal", (0.0, 1.0))):
        env = SoccerEnv(n_envs=a.envs, mode="defend", random_start=rs, easy_start=es, obs_noise=0.0,
                        red_mix={"latest": 1.0}, att_pool=pool)
        for label, policy in (("scripted keeper", None), ("defender (RL)", pol)):
            c, s, t = run(env, policy)
            print("%-18s %-16s conceded %5.1f%%  scored %5.1f%%  no goal in 60 s %5.1f%%" % (name, label, c, s, t), flush=True)


if __name__ == "__main__":
    main()
