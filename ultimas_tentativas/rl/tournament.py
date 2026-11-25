"""Pick the strongest checkpoint: each candidate plays 1024 rounds against a fixed panel
(the scripted opponents + earlier policies + some phase 3 snapshots), deterministic, standard start.
Score = mean over the panel of (win% - lose%).

    ~/ros2_ws/.venv/bin/python tournament.py
"""
import sys
from pathlib import Path

import torch

from eval_vec import load, play
from opponents import SCRIPTED, PolicyPool
from ppo import ActorCritic
from sumo_env import RewardConfig, SumoEnv
from sumo_obs import OBS_DIM

R = Path(__file__).resolve().parent / "runs"
PANEL = [R / "sumo_ppo_v1/model.pt", R / "sumo_ppo_v2/model.pt"] + [R / "sumo_ppo_v3/pool" / f"it{k:04d}.pt" for k in (400, 700, 1000, 1300)]
CANDIDATES = [R / "sumo_ppo_v2/model.pt"] + [R / "sumo_ppo_v3/pool" / f"it{k:04d}.pt" for k in range(500, 1501, 100)]

pool = PolicyPool(lambda: ActorCritic(OBS_DIM, 2))
for p in PANEL:
    pool.add(torch.load(p, map_location="cuda", weights_only=False)["model"], p.stem if "pool" in str(p) else p.parent.name)
env = SumoEnv(n_envs=1024, reward=RewardConfig(draw=-3.0), random_start=0.0, obs_noise=0.0, pool=pool)
names = list(SCRIPTED) + pool.names
print("%-22s" % "candidate" + "".join("%9s" % n[:9] for n in names) + "   score")
rows = []
for c in CANDIDATES:
    model = load(c)
    res = [play(env, model, s) for s in SCRIPTED] + [play(env, model, "pool", slot=k) for k in range(len(pool))]
    score = sum(w - l for w, l, _, _ in res) / len(res)
    tag = c.stem if "pool" in str(c) else c.parent.name
    rows.append((score, tag))
    print("%-22s" % tag + "".join("%5.0f/%-3.0f" % (w, l) for w, l, _, _ in res) + "  %6.1f" % score, flush=True)
print("cells: win%/lose%. best:", max(rows))
