"""Capture the flag: PPO FROM SCRATCH for the blue team (attacker robot 0 + defender robot 1), ctf_zero_env.py.

    $PY train_ctf_zero.py --name ctf_zero_shared --net shared     # one network for both robots (+ role one-hot)
    $PY train_ctf_zero.py --name ctf_zero_sep    --net separate   # one network per robot

Curriculum (automatic, the red team is one controller per episode):
  stage 0  red stopped                                   -> next when win rate vs stopped >= --promote
  stage 1  stopped 0.1 / script 0.45 / estimated 0.45    (scripts: ctf_scripts.py styles; estimated: the 8 teams
                                                           of eng_reversa, state machines fitted to the tournament)
Focus: the attacker scoring. The team reward of ctf_team_env.py, with a larger point (--score) and the
attacker shaping (flag / home potentials, avoid the defenders, pickup bonus).

Writes runs/<name>/: model.pt (shared) or att.pt + def.pt (separate), ck_*.pt, log.csv, config.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from ctf_obs import OBS_DIM
from ctf_team_env import TeamReward
from ctf_zero_env import ER_KIND, CtfZeroEnv, load_er_teams
from ppo import PPO, ActorCritic, PPOConfig

HIDDEN = (128, 128)
STAGES = [{"stopped": 1.0}, {"stopped": 0.1, "script": 0.45, "er": 0.45}]


def role_obs(obs):
    """(2, n, OBS) -> (2, n, OBS + 2): role one-hot appended (shared network)."""
    n = obs.shape[1]
    oh = torch.zeros(2, n, 2, device=obs.device)
    oh[0, :, 0] = 1.0
    oh[1, :, 1] = 1.0
    return torch.cat([obs, oh], 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--net", choices=["shared", "separate"], default="shared")
    p.add_argument("--envs", type=int, default=2048)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-std", type=float, default=0.5)
    p.add_argument("--ent", type=float, default=0.005)
    p.add_argument("--easy", type=float, default=0.4, help="fraction of episodes starting near the enemy flag")
    p.add_argument("--episode-s", type=float, default=30.0)
    p.add_argument("--score", type=float, default=15.0)
    p.add_argument("--conceded", type=float, default=-10.0)
    p.add_argument("--promote", type=float, default=0.7)
    p.add_argument("--min-stage-iters", type=int, default=50)
    p.add_argument("--stage", type=int, default=0, help="start stage")
    p.add_argument("--att-styles", default="[0.4,0.2,0.2,0.2]")
    p.add_argument("--def-styles", default="[0.3,0.2,0.2,0.15,0.15]")
    p.add_argument("--strong", type=float, default=0.5)
    p.add_argument("--dr-speed", type=float, default=0.1)
    p.add_argument("--dr-delay", type=float, default=0.3)
    p.add_argument("--obs-noise", type=float, default=0.004)
    p.add_argument("--yaw-noise", type=float, default=0.03)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--easy-mate", type=float, default=0.0, help="episodes with robot 1 starting near the enemy flag")
    p.add_argument("--d-guard", type=float, default=0.3, help="defender potential towards the guard point")
    p.add_argument("--mix", default=None, help="JSON: replaces the stage 1 opponent mix")
    p.add_argument("--def-script", action="store_true",
                   help="robot 1 = standard scripted defender (reference gains); only the attacker network trains")
    p.add_argument("--init", default=None, help="start from this run (its model.pt or att.pt/def.pt)")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    if args.mix:
        STAGES[1] = json.loads(args.mix)
    if args.def_script and args.net != "separate":
        raise SystemExit("--def-script needs --net separate")

    er = load_er_teams()
    er_names = [e[0] for e in er]
    shared = args.net == "shared"
    cfg = PPOConfig(gamma=0.995, ent_coef=args.ent)
    if shared:
        models = [ActorCritic(OBS_DIM + 2, 2, hidden=HIDDEN, init_std=args.init_std).cuda()]
    else:
        models = [ActorCritic(OBS_DIM, 2, hidden=HIDDEN, init_std=args.init_std).cuda() for _ in range(2)]
    if args.init:
        src = Path(args.init)
        for m, nm in zip(models, ["model"] if shared else ["att", "def"]):
            m.load_state_dict(torch.load(src / f"{nm}.pt", map_location="cuda", weights_only=False)["model"])
            with torch.no_grad():
                m.log_std.fill_(float(torch.log(torch.tensor(args.init_std))))
    ppos = [PPO(m, cfg) for m in models]
    stage = args.stage
    trew = TeamReward(score=args.score, conceded=args.conceded, d_guard=args.d_guard)
    env = CtfZeroEnv(er_teams=er, team_reward=trew, n_envs=args.envs, gamma=cfg.gamma, seed=args.seed,
                     easy_start=args.easy, easy_mate=args.easy_mate, episode_s=args.episode_s, obs_noise=args.obs_noise, yaw_noise=args.yaw_noise,
                     att_styles=json.loads(args.att_styles), def_styles=json.loads(args.def_styles), strong=args.strong,
                     dr_speed=args.dr_speed, dr_delay=args.dr_delay, main_mix=STAGES[stage], other_mix={"script": 1.0},
                     def_script=args.def_script)
    out = Path(__file__).resolve().parent / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    cfgd = {"args": vars(args), "ppo": asdict(cfg), "team_reward": asdict(env.trew), "att_reward": asdict(env.arew),
            "er_teams": er_names, "stages": STAGES, "obs": "ctf_obs.py" + (" + role one-hot (att, def)" if shared else "")}
    (out / "config.json").write_text(json.dumps(cfgd, indent=2))
    log = open(out / "log.csv", "w", newline="")
    wr = csv.writer(log)
    groups = ["stopped", "script", "er"]
    wr.writerow(["iter", "stage", "steps", "fps", "episodes", "scored", "conceded", "nopoint", "pickups_per_ep",
                 "ev0", "ev1", "kl0", "std0"] + ["win_" + g for g in groups] + ["lose_" + g for g in groups]
                + ["win_er_" + n for n in er_names])

    win_ema = torch.zeros(3)  # EMA of the score rate per group
    er_ema = torch.zeros(len(er))
    stage_it = 0
    T = cfg.rollout
    obs = env.reset_all()
    t_start, total = time.time(), 0
    for it in range(1, args.iters + 1):
        nb = len(models)
        keys = ("obs", "act", "logp", "val", "rew", "term", "done", "next_val")
        bufs = [{k: [] for k in keys} for _ in range(nb)]
        res = torch.zeros(3, 3, device="cuda")  # group x (scored, conceded, no point)
        er_res = torch.zeros(len(er), 2, device="cuda")  # (scored, episodes)
        pick = 0.0
        n_ep = 0
        t0 = time.time()
        for _ in range(T):
            with torch.no_grad():
                if shared:
                    m = models[0]
                    x = role_obs(obs).reshape(2 * env.n, -1)
                    m.norm.update(x)
                    xn = m.norm(x)
                    d = m.dist(xn)
                    a = d.sample()
                    b = bufs[0]
                    b["obs"].append(x); b["act"].append(a); b["logp"].append(d.log_prob(a).sum(-1))
                    b["val"].append(m.value(xn))
                    acts = a.reshape(2, env.n, 2)
                else:
                    al = []
                    for r, m in enumerate(models):
                        m.norm.update(obs[r])
                        xn = m.norm(obs[r])
                        d = m.dist(xn)
                        a = d.sample()
                        b = bufs[r]
                        b["obs"].append(obs[r]); b["act"].append(a); b["logp"].append(d.log_prob(a).sum(-1))
                        b["val"].append(m.value(xn))
                        al.append(a)
                    acts = torch.stack(al)
            nobs, rew, term, trunc, info = env.step(acts)
            done = term | trunc
            with torch.no_grad():
                if shared:
                    m = models[0]
                    fo = role_obs(info["final_obs"]).reshape(2 * env.n, -1)
                    b = bufs[0]
                    b["rew"].append(rew.reshape(-1)); b["term"].append(term.float().repeat(2))
                    b["done"].append(done.float().repeat(2)); b["next_val"].append(m.value(m.norm(fo)))
                else:
                    for r, m in enumerate(models):
                        b = bufs[r]
                        b["rew"].append(rew[r]); b["term"].append(term.float()); b["done"].append(done.float())
                        b["next_val"].append(m.value(m.norm(info["final_obs"][r])))
            pick += float(info["pickup"].sum())
            if done.any():
                n_ep += int(done.sum())
                kind = info["main_kind"]
                g = torch.where(kind >= ER_KIND, torch.full_like(kind, 2), kind.clamp(max=1))
                nop = info["timeout"] | info["truncated"]
                for j, m_ in enumerate((info["scored"], info["conceded"], nop)):
                    res[:, j] += torch.bincount(g[m_ & done], minlength=3)[:3].float()
                er_d = done & (kind >= ER_KIND)
                if er_d.any():
                    tj = kind[er_d] - ER_KIND
                    er_res[:, 0] += torch.bincount(tj[info["scored"][er_d]], minlength=len(er)).float()
                    er_res[:, 1] += torch.bincount(tj, minlength=len(er)).float()
            obs = nobs
        train = list(zip(ppos, bufs))[:1] if args.def_script else list(zip(ppos, bufs))
        stats = [ppo.update({k: torch.stack(v) for k, v in b.items()}) for ppo, b in train]
        total += T * env.n
        fps = T * env.n / (time.time() - t0)
        tot = res.sum(0)
        sc, co, no = (tot / tot.sum().clamp_min(1)).tolist()
        cnt = res.sum(1)
        wins = (res[:, 0] / cnt.clamp_min(1)).cpu()
        loses = (res[:, 1] / cnt.clamp_min(1)).cpu()
        for gi in range(3):
            if cnt[gi] > 0:
                win_ema[gi] = 0.9 * win_ema[gi] + 0.1 * wins[gi]
        er_w = (er_res[:, 0] / er_res[:, 1].clamp_min(1)).cpu()
        for j in range(len(er)):
            if er_res[j, 1] > 0:
                er_ema[j] = 0.9 * er_ema[j] + 0.1 * er_w[j]
        stds = float(models[0].log_std.detach().clamp(-3, 0).exp().mean())
        wr.writerow([it, stage, total, round(fps), n_ep, round(sc, 4), round(co, 4), round(no, 4), round(pick / max(n_ep, 1), 3),
                     round(stats[0]["explained_var"], 3), round(stats[-1]["explained_var"], 3), round(stats[0]["kl"], 5),
                     round(stds, 3)] + [round(float(x), 4) for x in wins] + [round(float(x), 4) for x in loses]
                    + [round(float(x), 4) for x in er_w])
        log.flush()
        names = ["model"] if shared else ["att", "def"]
        for m, nm in zip(models, names):
            ck = {"model": m.state_dict(), "iter": it, "stage": stage, "role": nm, "config": cfgd, "shared": shared,
                  "win_ema": win_ema.tolist(), "er_ema": dict(zip(er_names, er_ema.tolist()))}
            torch.save(ck, out / f"{nm}.pt")
            if args.save_every and it % args.save_every == 0:
                torch.save(ck, out / f"ck_{nm}_{it:04d}.pt")
        stage_it += 1
        if it % 10 == 0 or it == 1:
            print(f"it {it:4d} st {stage} {total/1e6:6.1f}M {fps/1e3:3.0f}k/s ep {n_ep:5d} ponto {sc:5.1%} sofrido {co:5.1%} "
                  f"| pega/ep {pick/max(n_ep,1):.2f} | EV {stats[0]['explained_var']:.2f} std {stds:.2f} | "
                  f"vence parado {win_ema[0]:4.0%} script {win_ema[1]:4.0%} er {win_ema[2]:4.0%} | "
                  + " ".join(f"{n[:4]} {x:.0%}" for n, x in zip(er_names, er_ema.tolist())), flush=True)
        if stage + 1 < len(STAGES) and stage_it >= args.min_stage_iters and win_ema[stage] >= args.promote:
            stage += 1
            stage_it = 0
            env.set_mix(STAGES[stage], {"script": 1.0})
            torch.save({"model": [m.state_dict() for m in models], "iter": it}, out / f"stage{stage - 1}_end.pt")
            print(f"*** estágio {stage}: {STAGES[stage]} (it {it})", flush=True)
    print(f"done in {(time.time()-t_start)/60:.1f} min; saved in {out}")


if __name__ == "__main__":
    main()
