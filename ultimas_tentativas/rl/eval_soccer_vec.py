"""Quick soccer evaluation in the vectorized env: episodes of 30 s from the kickoff,
deterministic (mean) vs stochastic attacker policy, against a given red team.

    ~/ros2_ws/.venv/bin/python eval_soccer_vec.py runs/soccer_att_a2/model.pt --red script
"""
import argparse

import torch

from ppo import actor_state

from soccer_env import SoccerEnv
from train_soccer import make_model


@torch.no_grad()
def run(env, m, stochastic, fixed_gains=False):
    obs = env.reset_all()
    if fixed_gains:  # the gains of exercicios/soccer/soccer_script.py
        env.script_p[:] = torch.tensor([4.0, 0.5, 1.0, 0.08], device="cuda")
    res = torch.zeros(3, device="cuda")
    alive = torch.ones(env.n, dtype=torch.bool, device="cuda")
    for _ in range(env.max_steps + 1):
        on = m.norm(obs)
        a = m.dist(on).sample() if stochastic else m.actor(on)
        obs, r, term, trunc, info = env.step(a)
        new = (info["scored"] | info["conceded"] | info["timeout"]) & alive
        res += torch.stack([(info["scored"] & new).sum(), (info["conceded"] & new).sum(), (info["timeout"] & new).sum()]).float()
        alive &= ~new
        if not alive.any():
            break
    return (100 * res / env.n).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--red", default="script", choices=("stopped", "script"))
    p.add_argument("--keeper", default="script")
    p.add_argument("--envs", type=int, default=512)
    a = p.parse_args()
    m = make_model().cuda().eval()
    m.load_state_dict(actor_state(torch.load(a.ckpt, map_location="cuda", weights_only=False)["model"]), strict=False)
    env = SoccerEnv(n_envs=a.envs, random_start=0.0, easy_start=0.0, obs_noise=0.0, red_mix={a.red: 1.0}, keeper=a.keeper)
    for fixed in (False, True):
        for stoch in (False, True):
            s, c, t = run(env, m, stoch, fixed)
            print("kickoff, %-22s %-13s scored %5.1f%%  conceded %5.1f%%  no goal in %.0f s %5.1f%%"
                  % ("fixed script gains" if fixed else "random script gains", "stochastic" if stoch else "deterministic",
                     s, c, env.max_steps * env.control_dt, t), flush=True)


if __name__ == "__main__":
    main()
