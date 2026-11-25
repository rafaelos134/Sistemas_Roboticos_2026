"""Pick the phase 5 checkpoint: 512 rounds per opponent against the 7 scripts (pursuers with the
stronger phase 4/5 gains) and a panel of policies (v3, v4, v5 snapshots). Score = mean(win% - lose%)."""
from pathlib import Path

import torch

from eval_vec import load, play
from opponents import SCRIPTED, PolicyPool
from ppo import ActorCritic
from sumo_env import RewardConfig, SumoEnv
from sumo_obs import OBS_DIM

R = Path(__file__).resolve().parent / "runs"
PANEL = [R / "sumo_ppo_v3/model.pt", R / "sumo_ppo_v4/stop_it830.pt", R / "sumo_ppo_v5/pool/it0400.pt",
         R / "sumo_ppo_v5/pool/it0800.pt"]
CANDIDATES = [R / "sumo_ppo_v4/stop_it830.pt"] + sorted((R / "sumo_ppo_v5/pool").glob("it*.pt"))[3:] + [R / "sumo_ppo_v5/model.pt"]

pool = PolicyPool(lambda: ActorCritic(OBS_DIM, 2))
for p in PANEL:
    pool.add(torch.load(p, map_location="cuda", weights_only=False)["model"], p.parent.parent.name[-2:] + p.stem if "pool" in str(p) else p.parent.name)
env = SumoEnv(n_envs=512, reward=RewardConfig(draw=-3.0), random_start=0.0, obs_noise=0.0, pool=pool)
names = list(SCRIPTED) + pool.names
print("%-18s" % "candidate" + "".join("%10s" % n[-9:] for n in names) + "   score")
rows = []
for c in CANDIDATES:
    model = load(c)
    res = [play(env, model, s) for s in SCRIPTED] + [play(env, model, "pool", slot=k) for k in range(len(pool))]
    score = sum(w - l for w, l, _, _ in res) / len(res)
    tag = c.parent.parent.name[-2:] + "/" + c.stem if "pool" in str(c) else c.parent.name + "/" + c.stem
    rows.append((score, tag))
    print("%-18s" % tag[-18:] + "".join("%5.0f/%-4.0f" % (w, l) for w, l, _, _ in res) + "  %6.1f" % score, flush=True)
print("cells: win%/lose%. best:", max(rows))
