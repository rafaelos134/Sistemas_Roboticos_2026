"""Train one capture-the-flag role with PPO, against frozen networks of the other role (alternating training).

    cd ~/ros2_ws/rl; PY=~/ros2_ws/.venv/bin/python
    $PY train_ctf.py --role att --iters 400 --name ctf_att_1                                   # vs scripts
    $PY train_ctf.py --role def --iters 400 --name ctf_def_1 --att-pool runs/ctf_att_1/model.pt
    $PY train_ctf.py --role att --iters 400 --name ctf_att_2 --init runs/ctf_att_1/model.pt \
        --att-pool runs/ctf_att_1/model.pt --def-pool runs/ctf_def_1/model.pt
    (ctf_league.sh runs the whole alternation)

The pools hold frozen networks of each role, the LAST file is the "latest". The main opponent (red defender
when attacking, red attacker when defending) is drawn per episode from `--main-mix`; the teammate and the
other opponent from `--other-mix`. Kinds: stopped, script (random gains), latest, pool (older versions).

Writes runs/<name>/: model.pt, best.pt, log.csv, config.json, tb/ (TensorBoard).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from ctf_env import CtfEnv
from ctf_obs import OBS_DIM
from frozen_pool import FrozenPool
from ppo import PPO, ActorCritic, PPOConfig

HIDDEN = (128, 128)
GROUPS = ["stopped", "script", "latest", "older"]


def make_model():
    return ActorCritic(OBS_DIM, 2, hidden=HIDDEN)


def load_pool(paths):
    pool = FrozenPool(make_model)
    for p in paths:
        pool.add(torch.load(p, map_location="cuda", weights_only=False)["model"], Path(p).parent.name)
    return pool


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True, choices=("att", "def"))
    p.add_argument("--envs", type=int, default=2048)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", default=None, help="checkpoint to start from (previous version of this role)")
    p.add_argument("--init-std", type=float, default=0.4)
    p.add_argument("--att-pool", nargs="*", default=[], help="frozen attacker networks (last = latest)")
    p.add_argument("--def-pool", nargs="*", default=[], help="frozen defender networks (last = latest)")
    p.add_argument("--main-mix", default=None, help='JSON, e.g. {"latest": 0.5, "pool": 0.25, "script": 0.25}')
    p.add_argument("--other-mix", default=None)
    p.add_argument("--easy", type=float, default=0.4, help="fraction of easy starts (near the flag)")
    p.add_argument("--episode-s", type=float, default=30.0, help="training episode cut (s), bootstrapped")
    p.add_argument("--ent", type=float, default=0.005)
    p.add_argument("--att-styles", default=None, help="JSON list, weights of the attacker script styles (ctf_scripts.py)")
    p.add_argument("--def-styles", default=None, help="JSON list, weights of the defender script styles")
    p.add_argument("--strong", type=float, default=0.0, help="fraction of scripts with the reference gains")
    p.add_argument("--dr-speed", type=float, default=0.0, help="command scale per robot in [1-x, 1+x]")
    p.add_argument("--dr-delay", type=float, default=0.0, help="fraction of episodes with one step of latency")
    p.add_argument("--obs-noise", type=float, default=0.002, help="position noise (m)")
    p.add_argument("--yaw-noise", type=float, default=0.0, help="yaw noise (rad)")
    p.add_argument("--save-every", type=int, default=0, help="keep a checkpoint ck_<iter>.pt every N iterations")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    mode = "attack" if args.role == "att" else "defend"
    att_pool, def_pool = load_pool(args.att_pool), load_pool(args.def_pool)
    main_pool = def_pool if mode == "attack" else att_pool
    if args.main_mix:
        main_mix = json.loads(args.main_mix)
    elif len(main_pool):
        main_mix = {"latest": 0.5, "pool": 0.2, "script": 0.25, "stopped": 0.05} if len(main_pool) > 1 else \
                   {"latest": 0.7, "script": 0.25, "stopped": 0.05}
    else:  # first version, no network of the other role yet: half of the opponents stopped
        main_mix = {"script": 0.6, "stopped": 0.4}
    other_mix = json.loads(args.other_mix) if args.other_mix else {"latest": 0.5, "script": 0.5}

    cfg = PPOConfig(gamma=0.995, ent_coef=args.ent)
    model = make_model().cuda()
    if args.init:
        ck = torch.load(args.init, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"])
        with torch.no_grad():
            model.log_std.fill_(float(torch.log(torch.tensor(args.init_std))))
        print("start from %s (iter %s), std reset to %.2f" % (args.init, ck.get("iter"), args.init_std))

    out = Path(__file__).resolve().parent / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    env = CtfEnv(n_envs=args.envs, mode=mode, gamma=cfg.gamma, seed=args.seed, easy_start=args.easy,
                 episode_s=args.episode_s, obs_noise=args.obs_noise, yaw_noise=args.yaw_noise,
                 att_styles=json.loads(args.att_styles) if args.att_styles else None,
                 def_styles=json.loads(args.def_styles) if args.def_styles else None,
                 strong=args.strong, dr_speed=args.dr_speed, dr_delay=args.dr_delay,
                 main_mix=main_mix, other_mix=other_mix, att_pool=att_pool, def_pool=def_pool)
    ppo = PPO(model, cfg)
    cfgd = {"args": vars(args), "mode": mode, "main_mix": main_mix, "other_mix": other_mix, "ppo": asdict(cfg),
            "hidden": HIDDEN, "reward": asdict(env.arew if mode == "attack" else env.drew), "control_dt": env.dt,
            "obs_dim": OBS_DIM, "att_pool": att_pool.names, "def_pool": def_pool.names}
    (out / "config.json").write_text(json.dumps(cfgd, indent=2))
    tb = SummaryWriter(str(out / "tb"))
    log = open(out / "log.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["iter", "steps", "fps", "episodes", "scored", "conceded", "timeout", "pickup_per_ep", "tagged_per_ep",
                 "tagger_per_ep", "ep_len_s", "reward_step", "pi_loss", "v_loss", "entropy", "kl", "lr", "std_lin",
                 "std_ang", "explained_var"] + ["scored_" + g for g in GROUPS] + ["conceded_" + g for g in GROUPS]
                + ["n_" + g for g in GROUPS])

    npool = len(main_pool)

    def group(kind):  # 0 stopped, 1 script, 2 latest, 3 older pool versions
        g = kind.clamp(max=2).clone()
        if npool:
            g[kind == 2 + npool - 1] = 2
            g[(kind >= 2) & (kind < 2 + npool - 1)] = 3
        return g

    T = cfg.rollout
    obs = env.reset_all()
    ep_len = torch.zeros(env.n, device="cuda")
    best, total, t_start = -1e9, 0, time.time()
    for it in range(1, args.iters + 1):
        buf = {k: [] for k in ("obs", "act", "logp", "val", "rew", "term", "done", "next_val")}
        res = torch.zeros(4, 3, device="cuda")  # group x (scored, conceded, no point: 90 s draw or cut episode)
        ev = torch.zeros(3, device="cuda")
        lens = []
        t0 = time.time()
        for _ in range(T):
            with torch.no_grad():
                model.norm.update(obs)
                d = model.dist(model.norm(obs))
                act = d.sample()
                logp = d.log_prob(act).sum(-1)
                val = model.value(model.norm(obs))
            nobs, rew, term, trunc, info = env.step(act)
            done = term | trunc
            with torch.no_grad():
                nv = model.value(model.norm(info["final_obs"]))
            for k, v in (("obs", obs), ("act", act), ("logp", logp), ("val", val), ("rew", rew),
                         ("term", term.float()), ("done", done.float()), ("next_val", nv)):
                buf[k].append(v)
            ev += torch.stack([info["pickup"].sum(), info["tagged"].sum(), info["tagger"].sum()]).float()
            ep_len += 1
            if done.any():
                g = group(info["main_kind"])
                nopoint = info["timeout"] | info["truncated"]
                for j, m in enumerate((info["scored"], info["conceded"], nopoint)):
                    res[:, j] += torch.bincount(g[m & done], minlength=4)[:4].float()
                lens.append(ep_len[done].clone())
                ep_len[done] = 0
            obs = nobs
        buf = {k: torch.stack(v) for k, v in buf.items()}
        stats = ppo.update(buf)
        total += T * env.n
        fps = T * env.n / (time.time() - t0)
        tot = res.sum(0)
        n_ep = int(tot.sum())
        sc, co, to = (tot / max(n_ep, 1)).tolist()
        ep_ev = (ev / max(n_ep, 1)).tolist()
        per_s = (res[:, 0] / res.sum(1).clamp_min(1)).tolist()
        per_c = (res[:, 1] / res.sum(1).clamp_min(1)).tolist()
        mean_len = float(torch.cat(lens).mean()) * env.dt if lens else 0.0
        std = model.log_std.exp().tolist()
        wr.writerow([it, total, round(fps), n_ep, round(sc, 4), round(co, 4), round(to, 4)]
                    + [round(x, 3) for x in ep_ev]
                    + [round(mean_len, 2), round(float(buf["rew"].mean()), 5), round(stats["pi_loss"], 5),
                       round(stats["v_loss"], 5), round(stats["entropy"], 4), round(stats["kl"], 5), stats["lr"],
                       round(std[0], 4), round(std[1], 4), round(stats["explained_var"], 4)]
                    + [round(x, 4) for x in per_s] + [round(x, 4) for x in per_c] + [int(x) for x in res.sum(1).tolist()])
        log.flush()
        for k, v in (("resultado/ponto_feito", sc), ("resultado/ponto_sofrido", co), ("resultado/sem_ponto", to),
                     ("eventos/pegadas_por_rodada", ep_ev[0]), ("eventos/segurado_por_rodada", ep_ev[1]),
                     ("eventos/segurou_por_rodada", ep_ev[2]), ("episodio/duracao_s", mean_len),
                     ("episodio/recompensa_por_passo", float(buf["rew"].mean())), ("ppo/kl", stats["kl"]),
                     ("ppo/entropia", stats["entropy"]), ("ppo/variancia_explicada_critic", stats["explained_var"]),
                     ("ppo/std_linear", std[0]), ("ppo/std_angular", std[1]), ("treino/passos_por_s", fps)):
            tb.add_scalar(k, v, total)
        for g, a, b, n in zip(GROUPS, per_s, per_c, res.sum(1).tolist()):
            if n > 0:
                tb.add_scalar("ponto_feito_por_adversario/" + g, a, total)
                tb.add_scalar("ponto_sofrido_por_adversario/" + g, b, total)
        ck = {"model": model.state_dict(), "iter": it, "role": args.role, "config": cfgd, "scored": sc, "conceded": co}
        torch.save(ck, out / "model.pt")
        if args.save_every and it % args.save_every == 0:
            torch.save(ck, out / ("ck_%04d.pt" % it))
        # best: against the non-stopped opponents
        w = res[1:].sum(1)
        s_ns = float(res[1:, 0].sum() / w.sum().clamp_min(1))
        c_ns = float(res[1:, 1].sum() / w.sum().clamp_min(1))
        score = (s_ns - c_ns) if mode == "attack" else (-c_ns - 0.02 * ep_ev[0] + 0.1 * s_ns)
        if n_ep >= 100 and it > args.iters // 4 and score >= best:
            best = score
            torch.save(ck, out / "best.pt")
        if it % 10 == 0 or it == 1:
            print(f"it {it:4d} {total/1e6:6.1f}M {fps/1e3:4.0f}k/s ep {n_ep:5d} ponto {sc:5.1%} sofrido {co:5.1%} "
                  f"sem-ponto {to:5.1%} | pega {ep_ev[0]:.2f} segurado {ep_ev[1]:.2f} segurou {ep_ev[2]:.2f} | "
                  f"EV {stats['explained_var']:.2f} len {mean_len:4.1f}s std {std[0]:.2f},{std[1]:.2f} | "
                  + " ".join("%s %3.0f/%3.0f%%" % (g, 100 * a, 100 * b) for g, a, b, n
                             in zip(GROUPS, per_s, per_c, res.sum(1).tolist()) if n > 0), flush=True)
    tb.close()
    print(f"done in {(time.time()-t_start)/60:.1f} min; saved in {out}")


if __name__ == "__main__":
    main()
