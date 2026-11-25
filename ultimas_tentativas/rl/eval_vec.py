"""Quick evaluation in the vectorized environment: deterministic policy (mean action)
against each opponent, one round per environment, standard start (as in psr_sim).

    ~/ros2_ws/.venv/bin/python eval_vec.py runs/sumo_ppo_v2/model.pt
    ~/ros2_ws/.venv/bin/python eval_vec.py runs/sumo_ppo_v3/model.pt --vs runs/sumo_ppo_v2/model.pt runs/sumo_ppo_v1/model.pt

Prints win / lose / draw (%) and the mean round time per opponent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from opponents import SCRIPTED, PolicyPool
from ppo import ActorCritic
from sumo_env import RewardConfig, SumoEnv
from sumo_obs import OBS_DIM


def load(path):
    m = ActorCritic(OBS_DIM, 2).cuda().eval()
    m.load_state_dict(torch.load(path, map_location="cuda", weights_only=False)["model"])
    return m


@torch.no_grad()
def play(env, model, kind_name, slot=None):
    env.set_opponent_mix({kind_name: 1.0})
    obs = env.reset_all()
    if slot is not None:  # force one pool slot
        env.opp_kind[:] = len(SCRIPTED) + slot
    alive = torch.ones(env.n, dtype=torch.bool, device="cuda")
    out = {k: torch.zeros(env.n, dtype=torch.bool, device="cuda") for k in ("win", "lose", "draw")}
    t = torch.zeros(env.n, device="cuda")
    for k in range(1, env.max_steps + 2):
        act = model.actor(model.norm(obs)).clamp(-1, 1)
        obs, _, term, trunc, info = env.step(act)
        new = (term | trunc) & alive
        for key in out:
            out[key] |= info[key] & new
        t[new] = k * env.control_dt
        alive &= ~new
        if not alive.any():
            break
    n = env.n
    w = out["win"]
    return (100.0 * w.sum().item() / n, 100.0 * out["lose"].sum().item() / n, 100.0 * out["draw"].sum().item() / n,
            t[w].mean().item() if w.any() else float("nan"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--vs", nargs="*", default=[], help="checkpoints of other policies to play against")
    p.add_argument("--envs", type=int, default=1024)
    args = p.parse_args()

    model = load(args.ckpt)
    pool = PolicyPool(lambda: ActorCritic(OBS_DIM, 2))
    for path in args.vs:
        pool.add(torch.load(path, map_location="cuda", weights_only=False)["model"], Path(path).parent.name)
    env = SumoEnv(n_envs=args.envs, reward=RewardConfig(draw=-3.0), random_start=0.0, obs_noise=0.0, pool=pool)
    print("%s  (%d rounds per opponent, standard start)" % (args.ckpt, args.envs))
    print("%-22s %6s %6s %6s %9s" % ("opponent", "win%", "lose%", "draw%", "t_win(s)"))
    for name in SCRIPTED:
        print("%-22s %6.1f %6.1f %6.1f %9.2f" % ((name,) + play(env, model, name)))
    for k, name in enumerate(pool.names):
        print("%-22s %6.1f %6.1f %6.1f %9.2f" % (("policy " + name,) + play(env, model, "pool", slot=k)))


if __name__ == "__main__":
    main()
