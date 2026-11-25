"""Train the soccer policies with PPO (curriculum, like the sumo).

    cd ~/ros2_ws/rl; PY=~/ros2_ws/.venv/bin/python
    # attackers (blue 1 and 2 share one network)
    $PY train_soccer.py --stage A1 --iters 600  --name soccer_att_a1                    # red team stopped
    $PY train_soccer.py --stage A2 --iters 1000 --name soccer_att_a2 --init runs/soccer_att_a1/model.pt   # scripted red team
    $PY train_soccer.py --stage A3 --iters 1500 --name soccer_att_a3 --init runs/soccer_att_a2/model.pt   # self-play
    # defender (blue 0) against our own attackers
    $PY train_soccer.py --stage D  --iters 800  --name soccer_def --attacker runs/soccer_att_a3/model.pt

    # asymmetric critic (soccer_critic.py; the actor and the exported node do not change)
    $PY train_soccer.py --stage A2 --iters 1000 --wall 2 --critic asym     --name soccer_att_a2_asym --init runs/soccer_att_a2/model.pt
    $PY train_soccer.py --stage A2 --iters 1000 --wall 2 --critic asym_dij --name soccer_att_a2_dij  --init runs/soccer_att_a2/model.pt

Writes runs/<name>/: model.pt, best.pt, log.csv, config.json, tb/ (TensorBoard), pool/ (A3).
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

from frozen_pool import FrozenPool
from ppo import PPO, ActorCritic, PPOConfig, actor_state
from soccer_critic import critic_dim
from soccer_env import DefenderReward, SoccerEnv, SoccerReward
from soccer_obs import OBS_DIM

HIDDEN = (128, 128)
STAGES = {
    "A1": dict(mode="attack", red={"stopped": 1.0}, keeper="stopped"),
    "A2": dict(mode="attack", red={"script": 0.85, "stopped": 0.15}, keeper="script"),
    "A3": dict(mode="attack", red={"pool": 0.45, "latest": 0.20, "script": 0.30, "stopped": 0.05}, keeper="script"),
    "D": dict(mode="defend", red={"latest": 0.85, "script": 0.15}, keeper="stopped"),
}
GROUPS = ["stopped", "script", "self"]


def make_model(critic="sym"):
    return ActorCritic(OBS_DIM, 2, hidden=HIDDEN, critic_dim=critic_dim(critic))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--envs", type=int, default=1024)
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", default=None, help="checkpoint to start from (previous stage)")
    p.add_argument("--init-std", type=float, default=0.35)
    p.add_argument("--attacker", default=None, help="stage D: the frozen attacker network")
    p.add_argument("--snapshot-every", type=int, default=100)
    p.add_argument("--wall", type=float, default=0.0, help="weight of the 'ball away from the long walls' potential")
    p.add_argument("--critic", default="sym", choices=("sym", "asym", "asym_dij"),
                   help="critic input: actor obs (sym), + field view and ball still time (asym), + Dijkstra (asym_dij)")
    p.add_argument("--fresh-critic", action="store_true", help="do not load the critic of --init (always for asym)")
    p.add_argument("--ent", type=float, default=None, help="entropy coefficient (default: 0.01 in A1, 0.005 after)")
    args = p.parse_args()
    st = STAGES[args.stage]
    torch.manual_seed(args.seed)

    # more exploration than the sumo at the start (a goal is a much rarer event), less in the later stages
    cfg = PPOConfig(ent_coef=args.ent if args.ent is not None else (0.01 if args.stage == "A1" else 0.005))
    model = make_model(args.critic).cuda()
    if args.init:
        ck = torch.load(args.init, map_location="cuda", weights_only=False)
        sd = ck["model"]
        same = (args.critic == "sym") == ("cnorm.mean" not in sd) and not args.fresh_critic
        if same and args.critic != "sym" and sd["cnorm.mean"].shape != model.cnorm.mean.shape:
            same = False
        model.load_state_dict(sd if same else actor_state(sd), strict=same)
        print("critic:", args.critic, "(loaded from --init)" if same else "(new)")
        with torch.no_grad():
            model.log_std.fill_(float(torch.log(torch.tensor(args.init_std))))
        print("start from %s (iter %s), std reset to %.2f" % (args.init, ck.get("iter"), args.init_std))

    out = Path(__file__).resolve().parent / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    att_pool = def_pool = None
    if args.stage == "A3":
        att_pool = FrozenPool(make_model)
        att_pool.add(model.state_dict(), "start")
        att_pool.add(model.state_dict(), "latest")
        (out / "pool").mkdir(exist_ok=True)
    if args.stage == "D":
        if not args.attacker:
            raise SystemExit("--attacker is required for stage D")
        att_pool = FrozenPool(make_model)
        att_pool.add(torch.load(args.attacker, map_location="cuda", weights_only=False)["model"], "attacker")

    env = SoccerEnv(n_envs=args.envs, mode=st["mode"], gamma=cfg.gamma, seed=args.seed, red_mix=st["red"],
                    keeper=st["keeper"], att_pool=att_pool, def_pool=def_pool, critic=args.critic)
    env.rew.shape_wall = args.wall
    ppo = PPO(model, cfg)
    cfgd = {"args": vars(args), "stage": {k: v for k, v in st.items()}, "ppo": asdict(cfg), "hidden": HIDDEN,
            "reward": asdict(env.rew) if st["mode"] == "attack" else asdict(env.drew),
            "control_dt": env.control_dt, "max_steps": env.max_steps, "obs_dim": OBS_DIM,
            "critic": args.critic, "critic_dim": critic_dim(args.critic)}
    (out / "config.json").write_text(json.dumps(cfgd, indent=2))
    tb = SummaryWriter(str(out / "tb"))
    log = open(out / "log.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["iter", "steps", "fps", "episodes", "scored", "conceded", "timeout", "ep_len_s", "reward_step",
                 "pi_loss", "v_loss", "entropy", "kl", "lr", "std_lin", "std_ang", "explained_var", "stalled_frac"]
                + ["scored_" + g for g in GROUPS] + ["conceded_" + g for g in GROUPS] + ["n_" + g for g in GROUPS])

    T = cfg.rollout
    obs = env.reset_all()
    cobs = env.critic_obs()
    asym = cobs is not None
    NA = obs.shape[0]
    ep_len = torch.zeros(env.n, device="cuda")
    best, total, t_start = -1e9, 0, time.time()
    for it in range(1, args.iters + 1):
        buf = {k: [] for k in ("obs", "act", "logp", "val", "rew", "term", "done", "next_val") + (("cobs",) if asym else ())}
        res = torch.zeros(3, 3, device="cuda")  # group x (scored, conceded, timeout)
        lens = []
        stall_frac = 0.0
        t0 = time.time()
        for _ in range(T):
            with torch.no_grad():
                model.norm.update(obs)
                if asym:
                    model.cnorm.update(cobs)
                on = model.norm(obs)
                d = model.dist(on)
                act = d.sample()
                logp = d.log_prob(act).sum(-1)
                val = model.value(model.critic_in(obs, cobs))
            nobs, rew, term, trunc, info = env.step(act)
            with torch.no_grad():
                nv = model.value(model.critic_in(info["final_obs"], info["final_cobs"]))
            done = term | trunc
            for k, v in (("obs", obs), ("act", act), ("logp", logp), ("val", val), ("rew", rew),
                         ("term", term.float()), ("done", done.float()), ("next_val", nv)):
                buf[k].append(v)
            if asym:
                buf["cobs"].append(cobs)
                cobs = env.critic_obs()
            ep_len += 1
            stall_frac += float(info["stalled"].float().mean())
            de = done[: env.n]
            if de.any():
                g = info["red_kind"].clamp(max=2)
                for j, key in enumerate(("scored", "conceded", "timeout")):
                    res[:, j] += torch.bincount(g[info[key] & de], minlength=3)[:3].float()
                lens.append(ep_len[de].clone())
                ep_len[de] = 0
            obs = nobs
        buf = {k: torch.stack(v) for k, v in buf.items()}
        stats = ppo.update(buf)
        if att_pool is not None and args.stage == "A3":
            att_pool.set(len(att_pool) - 1, model.state_dict(), "latest")
            if it % args.snapshot_every == 0:
                att_pool.snapshot(model.state_dict(), "it%04d" % it)
                torch.save({"model": model.state_dict(), "iter": it}, out / "pool" / ("it%04d.pt" % it))
        total += T * NA
        fps = T * NA / (time.time() - t0)
        tot = res.sum(0)
        n_ep = int(tot.sum())
        sc, co, to = (tot / max(n_ep, 1)).tolist()
        per_s = (res[:, 0] / res.sum(1).clamp_min(1)).tolist()
        per_c = (res[:, 1] / res.sum(1).clamp_min(1)).tolist()
        mean_len = float(torch.cat(lens).mean()) * env.control_dt if lens else 0.0
        std = model.log_std.exp().tolist()
        wr.writerow([it, total, round(fps), n_ep, round(sc, 4), round(co, 4), round(to, 4), round(mean_len, 2),
                     round(float(buf["rew"].mean()), 5), round(stats["pi_loss"], 5), round(stats["v_loss"], 5),
                     round(stats["entropy"], 4), round(stats["kl"], 5), stats["lr"], round(std[0], 4), round(std[1], 4),
                     round(stats["explained_var"], 4), round(stall_frac / T, 4)]
                    + [round(x, 4) for x in per_s] + [round(x, 4) for x in per_c] + [int(x) for x in res.sum(1).tolist()])
        log.flush()
        tb.add_scalar("resultado/fracao_de_passos_com_bola_travada", stall_frac / T, total)
        for k, v in (("resultado/gol_feito", sc), ("resultado/gol_sofrido", co), ("resultado/sem_gol_30s", to),
                     ("episodio/duracao_s", mean_len), ("episodio/recompensa_por_passo", float(buf["rew"].mean())),
                     ("ppo/perda_actor", stats["pi_loss"]), ("ppo/perda_critic", stats["v_loss"]),
                     ("ppo/entropia", stats["entropy"]), ("ppo/kl", stats["kl"]), ("ppo/lr", stats["lr"]),
                     ("ppo/variancia_explicada_critic", stats["explained_var"]),
                     ("ppo/std_linear", std[0]), ("ppo/std_angular", std[1]), ("treino/passos_por_s", fps)):
            tb.add_scalar(k, v, total)
        for g, a, b, n in zip(GROUPS, per_s, per_c, res.sum(1).tolist()):
            if n > 0:
                tb.add_scalar("gol_feito_por_adversario/" + g, a, total)
                tb.add_scalar("gol_sofrido_por_adversario/" + g, b, total)
        ck = {"model": model.state_dict(), "iter": it, "stage": args.stage, "config": cfgd,
              "scored": sc, "conceded": co}
        torch.save(ck, out / "model.pt")
        score = (sc - co) if st["mode"] == "attack" else (-co - 0.2 * sc)
        if n_ep >= 50 and score >= best:
            best = score
            torch.save(ck, out / "best.pt")
        if it % 10 == 0 or it == 1:
            print(f"it {it:4d} {total/1e6:6.1f}M {fps/1e3:5.0f}k/s ep {n_ep:5d} gol {sc:5.1%} sofrido {co:5.1%} "
                  f"sem-gol {to:5.1%} travada {stall_frac / T:5.1%} EV {stats['explained_var']:.2f} len {mean_len:5.2f}s std {std[0]:.2f},{std[1]:.2f} | "
                  + " ".join("%s %3.0f/%3.0f%%" % (g, 100 * a, 100 * b) for g, a, b, n
                             in zip(GROUPS, per_s, per_c, res.sum(1).tolist()) if n > 0), flush=True)
    tb.close()
    print(f"done in {(time.time()-t_start)/60:.1f} min; saved in {out}")


if __name__ == "__main__":
    main()
