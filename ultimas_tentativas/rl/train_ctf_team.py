"""Train the blue TEAM (attacker + defender networks together, shared result) with PPO, see ctf_team_env.py.

    $PY train_ctf_team.py --name ctf_team_1 --init-att runs/ctf_att_5/model.pt --init-def runs/ctf_def_5/model.pt \
        --att-pool runs/ctf_att_*/model.pt --def-pool runs/ctf_def_*/model.pt --pfsp

Opponents (both red robots) from --main-mix: stopped, script (styles), latest (last network of each pool),
pool (older ones). --pfsp: the older pool versions are drawn with prioritized fictitious self-play weights,
w ~ (1 - win rate against it)^2 + 0.05, so the opponents we lose to come more often (AlphaStar).
An exploiter is the same script with --main-mix '{"latest": 1}' and the current main team as the last pool
entries: it trains only to beat that team; added to the pools afterwards it covers its weaknesses.

Writes runs/<name>/: att.pt, def.pt (same format as train_ctf.py: export_ctf.py reads them), ck_att_/ck_def_<it>.pt,
log.csv, config.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from ctf_team_env import CtfTeamEnv, TeamReward
from ctf_obs import OBS_DIM
from ppo import PPO, PPOConfig
from ppo_rnn import PPORnn, RecurrentActorCritic
from train_ctf import HIDDEN, load_pool, make_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--envs", type=int, default=2048)
    p.add_argument("--iters", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-att", required=True)
    p.add_argument("--init-def", required=True)
    p.add_argument("--init-std", type=float, default=0.35)
    p.add_argument("--att-pool", nargs="*", default=[])
    p.add_argument("--def-pool", nargs="*", default=[])
    p.add_argument("--main-mix", default='{"script": 0.4, "latest": 0.25, "pool": 0.25, "stopped": 0.1}')
    p.add_argument("--pfsp", action="store_true")
    p.add_argument("--easy", type=float, default=0.3)
    p.add_argument("--episode-s", type=float, default=30.0)
    p.add_argument("--ent", type=float, default=0.005)
    p.add_argument("--att-styles", default="[0.4,0.2,0.2,0.2]")
    p.add_argument("--def-styles", default="[0.3,0.2,0.2,0.15,0.15]")
    p.add_argument("--strong", type=float, default=0.5)
    p.add_argument("--dr-speed", type=float, default=0.1)
    p.add_argument("--dr-delay", type=float, default=0.3)
    p.add_argument("--obs-noise", type=float, default=0.004)
    p.add_argument("--yaw-noise", type=float, default=0.03)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--dual", action="store_true", help="robot 1 is a second attacker (ctf_team_env.py)")
    p.add_argument("--gate", type=float, default=None, help="supervisor of robot 1 (scripted defender on threat)")
    p.add_argument("--time-cost", type=float, default=0.0, help="team reward per second without a point (cost)")
    p.add_argument("--draw", type=float, default=0.0, help="team reward at the 90 s draw (e.g. -5)")
    p.add_argument("--score", type=float, default=10.0)
    p.add_argument("--conceded", type=float, default=-10.0)
    p.add_argument("--rnn", action="store_true", help="MLP + GRU memory (ppo_rnn.py); can start from MLP checkpoints")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    att_pool, def_pool = load_pool(args.att_pool), load_pool(args.def_pool)
    main_mix = json.loads(args.main_mix)
    cfg = PPOConfig(gamma=0.995, ent_coef=args.ent)
    models = []
    for path in (args.init_att, args.init_def):
        m = (RecurrentActorCritic(OBS_DIM, 2, hidden=HIDDEN) if args.rnn else make_model()).cuda()
        missing, _ = m.load_state_dict(torch.load(path, map_location="cuda", weights_only=False)["model"], strict=False)
        if missing:
            print("%s: new parameters %s (start at zero / random)" % (path, sorted({k.split(".")[0] for k in missing})))
        with torch.no_grad():
            m.log_std.fill_(float(torch.log(torch.tensor(args.init_std))))
        models.append(m)
    ppos = [(PPORnn if args.rnn else PPO)(m, cfg) for m in models]
    trew = TeamReward(time_cost=args.time_cost, draw=args.draw, score=args.score, conceded=args.conceded)
    env = CtfTeamEnv(team_reward=trew, dual=args.dual, gate=args.gate, n_envs=args.envs, gamma=cfg.gamma, seed=args.seed, easy_start=args.easy, episode_s=args.episode_s,
                     obs_noise=args.obs_noise, yaw_noise=args.yaw_noise, att_styles=json.loads(args.att_styles),
                     def_styles=json.loads(args.def_styles), strong=args.strong, dr_speed=args.dr_speed,
                     dr_delay=args.dr_delay, main_mix=main_mix, other_mix={"script": 1.0},
                     att_pool=att_pool, def_pool=def_pool)
    out = Path(__file__).resolve().parent / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    cfgd = {"args": vars(args), "ppo": asdict(cfg), "team_reward": asdict(env.trew), "att_reward": asdict(env.arew),
            "att_pool": att_pool.names, "def_pool": def_pool.names}
    (out / "config.json").write_text(json.dumps(cfgd, indent=2))
    log = open(out / "log.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["iter", "steps", "fps", "episodes", "scored", "conceded", "nopoint", "pickups_per_ep", "def_pickups_per_ep",
                 "def_enemy_half", "ev_att", "ev_def", "kl_att", "kl_def", "std_att", "std_def"]
                + ["win_vs_" + k for k in ("stopped", "script", "networks")])

    # PFSP statistics per pool index of each role (EMA of win rate when that version played)
    win = [torch.full((max(len(att_pool), 1),), 0.5, device="cuda"), torch.full((max(len(def_pool), 1),), 0.5, device="cuda")]

    def pfsp_weights():
        for role, pool in ((0, att_pool), (1, def_pool)):
            n_old = len(pool) - 1
            if n_old >= 1:
                w = (1.0 - win[role][:n_old]).pow(2) + 0.05
                env.pfsp[role] = w / w.sum()

    env.pfsp = [None, None]
    if args.pfsp:
        pfsp_weights()

    T = cfg.rollout
    obs = env.reset_all()
    hs = [m.init_hidden(env.n) for m in models] if args.rnn else None
    t_start, total = time.time(), 0
    for it in range(1, args.iters + 1):
        bufs = [{k: [] for k in ("obs", "act", "logp", "val", "rew", "term", "done", "next_val")} for _ in range(2)]
        res = torch.zeros(3, 3, device="cuda")  # (stopped, script, networks) x (scored, conceded, no point)
        ev = torch.zeros(3, device="cuda")
        n_ep = 0
        t0 = time.time()
        h0 = [h.clone() for h in hs] if args.rnn else None
        for _ in range(T):
            acts = []
            with torch.no_grad():
                for r, m in enumerate(models):
                    m.norm.update(obs[r])
                    if args.rnn:
                        d, hs[r] = m.dist_h(m.norm(obs[r]), hs[r])
                    else:
                        d = m.dist(m.norm(obs[r]))
                    a = d.sample()
                    bufs[r]["obs"].append(obs[r]); bufs[r]["act"].append(a)
                    bufs[r]["logp"].append(d.log_prob(a).sum(-1)); bufs[r]["val"].append(m.value(m.norm(obs[r])))
                    acts.append(a)
            kinds = env.kind[2:4].clone()
            nobs, rew, term, trunc, info = env.step(torch.stack(acts))
            done = term | trunc
            with torch.no_grad():
                for r, m in enumerate(models):
                    bufs[r]["rew"].append(rew[r]); bufs[r]["term"].append(term.float()); bufs[r]["done"].append(done.float())
                    bufs[r]["next_val"].append(m.value(m.norm(info["final_obs"][r])))
                    if r == 1 and args.gate is not None:
                        bufs[r].setdefault("mask", []).append((~info["gated"]).float())
            if args.rnn:
                hs = [h * (~done).float().unsqueeze(1) for h in hs]
            ev += torch.stack([info["pickup"].sum(), info["pickup_def"].sum(), info["enemy_half_def"].sum() / env.n]).float()
            if done.any():
                n_ep += int(done.sum())
                g = info["main_kind"].clamp(max=2)
                nop = info["timeout"] | info["truncated"]
                for j, m_ in enumerate((info["scored"], info["conceded"], nop)):
                    res[:, j] += torch.bincount(g[m_ & done], minlength=3)[:3].float()
                # PFSP: win rate per pool version (older ones), for each red role
                for role in (0, 1):
                    k = kinds[role][done] - 2
                    s = info["scored"][done].float()
                    for j in range(len(win[role])):
                        sel = k == j
                        if sel.any():
                            win[role][j] = 0.98 * win[role][j] + 0.02 * s[sel].mean()
            obs = nobs
        stats = []
        for r in range(2):
            b = {k: torch.stack(v) for k, v in bufs[r].items()}
            if args.rnn:
                b["h0"] = h0[r]
            stats.append(ppos[r].update(b))
        if args.pfsp:
            pfsp_weights()
        total += T * env.n
        fps = T * env.n / (time.time() - t0)
        tot = res.sum(0)
        sc, co, no = (tot / tot.sum().clamp_min(1)).tolist()
        wins = (res[:, 0] / res.sum(1).clamp_min(1)).tolist()
        e = (ev / max(n_ep, 1)).tolist()
        e[2] = float(ev[2]) / T  # fraction of time the defender is in the enemy half
        stds = [float(m.log_std.detach().exp().mean()) for m in models]
        wr.writerow([it, total, round(fps), n_ep, round(sc, 4), round(co, 4), round(no, 4), round(e[0], 3), round(e[1], 3),
                     round(e[2], 3), round(stats[0]["explained_var"], 3), round(stats[1]["explained_var"], 3),
                     round(stats[0]["kl"], 5), round(stats[1]["kl"], 5), round(stds[0], 3), round(stds[1], 3)]
                    + [round(x, 4) for x in wins])
        log.flush()
        for r, nm in ((0, "att"), (1, "def")):
            ck = {"model": models[r].state_dict(), "iter": it, "role": nm, "config": cfgd, "scored": sc, "conceded": co}
            torch.save(ck, out / ("%s.pt" % nm))
            if args.save_every and it % args.save_every == 0:
                torch.save(ck, out / ("ck_%s_%04d.pt" % (nm, it)))
        if it % 10 == 0 or it == 1:
            print(f"it {it:4d} {total/1e6:6.1f}M {fps/1e3:4.0f}k/s ep {n_ep:5d} ponto {sc:5.1%} sofrido {co:5.1%} "
                  f"sem-ponto {no:5.1%} | pega {e[0]:.2f} (def {e[1]:.2f}) def-no-ataque {e[2]:4.0%} | "
                  f"EV {stats[0]['explained_var']:.2f}/{stats[1]['explained_var']:.2f} | vence parado {wins[0]:4.0%} "
                  f"script {wins[1]:4.0%} redes {wins[2]:4.0%}"
                  + (" | pfsp " + " ".join("%.2f" % x for x in win[1][:max(len(def_pool) - 1, 0)].tolist()) if args.pfsp else ""),
                  flush=True)
    print(f"done in {(time.time()-t_start)/60:.1f} min; saved in {out}")


if __name__ == "__main__":
    main()
