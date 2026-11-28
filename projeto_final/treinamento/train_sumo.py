"""Train the sumo policy with PPO in the vectorized Genesis environment (curriculum in phases).

    cd ~/ros2_ws/rl; PY=~/ros2_ws/.venv/bin/python
    # phase 1: stopped opponent, from scratch
    $PY train_sumo.py --phase 1 --iters 500 --name sumo_ppo_v1
    # phase 2: scripted opponents with random gains, starting from phase 1
    $PY train_sumo.py --phase 2 --iters 800 --name sumo_ppo_v2 --init runs/sumo_ppo_v1/model.pt
    # phase 3: self-play against frozen copies (+ the scripts), starting from phase 2
    $PY train_sumo.py --phase 3 --iters 1500 --name sumo_ppo_v3 --init runs/sumo_ppo_v2/model.pt
    # v6: v5 + noisy sensors and the SumoEKF (sumo_fusion.py); the critic sees the noise-free observation
    $PY train_sumo.py --phase 5 --fusion --iters 1500 --name sumo_ppo_v6_fus --init runs/sumo_ppo_v5/model.pt \
        --pool-from runs/sumo_ppo_v5/pool

Writes runs/<name>/: model.pt (last), best.pt, log.csv (win/lose/draw per opponent), config.json,
and in phase 3 pool/<k>.pt (the frozen copies).
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

from opponents import SCRIPTED, PolicyPool
from ppo import PPO, ActorCritic, PPOConfig, actor_state
from sumo_env import RewardConfig, SumoEnv
from sumo_obs import OBS_DIM

PHASES = {
    1: {"stopped": 1.0},
    2: {"stopped": 0.10, "chaser": 0.30, "careful": 0.25, "evader": 0.20, "random": 0.15},
    # phase 3: ~50% older copies, ~20% the newest copy, ~30% the scripts (so they are not forgotten)
    3: {"pool": 0.50, "latest": 0.20, "stopped": 0.03, "chaser": 0.10, "careful": 0.08, "evader": 0.05, "random": 0.04},
    # phase 4: the relentless pursuers (draws by orbiting), keep the pool so self-play is not forgotten
    5: {"pool": 0.22, "latest": 0.10, "pursuer": 0.22, "pursuer_bi": 0.22, "chaser": 0.10, "careful": 0.06,
        "evader": 0.05, "random": 0.02, "stopped": 0.01},
    4: {"pool": 0.28, "latest": 0.12, "pursuer": 0.18, "pursuer_bi": 0.16, "chaser": 0.12, "careful": 0.07,
        "evader": 0.04, "random": 0.02, "stopped": 0.01},
}


def make_model(asym=False):
    """asym: the critic reads the noise-free observation (same 12 inputs, own normalization)."""
    return ActorCritic(OBS_DIM, 2, critic_dim=OBS_DIM if asym else None)


def policy_sd(sd):
    """State dict loadable by the symmetric model of the pool (drops the critic normalization cnorm)."""
    return {k: v for k, v in sd.items() if not k.startswith("cnorm.")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=int, default=1, choices=(1, 2, 3, 4, 5))
    p.add_argument("--envs", type=int, default=2048)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--name", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", default=None, help="checkpoint to start from (previous phase)")
    p.add_argument("--init-std", type=float, default=0.35,
                   help="reset the policy std when starting from a checkpoint (exploration for the new phase)")
    p.add_argument("--snapshot-every", type=int, default=100, help="phase 3: add the policy to the pool every N iters")
    p.add_argument("--no-shaping", action="store_true", help="base reward only (win / still / self_out / draw)")
    p.add_argument("--time-cost", type=float, default=None, help="reward per step (default: -0.01 in phase 4, 0 before)")
    p.add_argument("--clinch", type=float, default=None, help="fraction of clinch starts (default 0.3 in phase 5)")
    p.add_argument("--ent", type=float, default=None, help="entropy coefficient (default 0.003; 0.01 in phase 5)")
    p.add_argument("--pool-from", default=None, help="phase 4: directory with frozen copies (*.pt) to seed the pool")
    p.add_argument("--fusion", action="store_true",
                   help="noisy sensors + SumoEKF (sumo_fusion.py); the critic sees the noise-free observation")
    p.add_argument("--noise-max", type=float, default=2.0, help="noise level drawn per episode in U(0, max)")
    args = p.parse_args()
    name = args.name or "sumo_ppo_phase%d" % args.phase

    torch.manual_seed(args.seed)
    cfg = PPOConfig()
    if args.ent is not None or args.phase == 5:
        cfg.ent_coef = args.ent if args.ent is not None else 0.01
    rcfg = RewardConfig()
    if args.phase >= 2:
        rcfg.draw = -3.0  # a stalemate is not a safe outcome any more
        rcfg.shape_center = 1.0  # stay closer to the centre than the opponent
    rcfg.time_cost = args.time_cost if args.time_cost is not None else (-0.01 if args.phase >= 4 else 0.0)
    if args.no_shaping:
        rcfg.shape_approach = rcfg.shape_push = rcfg.shape_center = 0.0

    model = make_model(args.fusion).cuda()
    if args.init:
        ck = torch.load(args.init, map_location="cuda", weights_only=False)
        sd = dict(ck["model"])
        if args.fusion and "cnorm.mean" not in sd:
            for k in ("mean", "var", "count"):  # same input: reuse the critic normalization
                sd["cnorm." + k] = sd["norm." + k].clone()
        if not args.fusion:
            sd = policy_sd(sd)
        model.load_state_dict(sd)
        with torch.no_grad():
            model.log_std.fill_(float(torch.log(torch.tensor(args.init_std))))
        print("start from %s (iter %s); policy std reset to %.2f" % (args.init, ck.get("iter"), args.init_std))

    pool = None
    out = Path(__file__).resolve().parent / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    if args.phase >= 3:
        pool = PolicyPool(make_model)
        (out / "pool").mkdir(exist_ok=True)
        pool.add(policy_sd(model.state_dict()), "start")  # slot 0: the policy this phase started from, always kept
        if args.pool_from:
            for f in sorted(Path(args.pool_from).glob("it*.pt"))[-12:]:
                pool.add(policy_sd(torch.load(f, map_location="cuda", weights_only=False)["model"]), f.stem)
            print("pool seeded with", pool.names)
        pool.add(policy_sd(model.state_dict()), "latest")  # last slot: refreshed every iteration
        torch.save({"model": model.state_dict()}, out / "pool" / "000_start.pt")
    env = SumoEnv(n_envs=args.envs, reward=rcfg, gamma=cfg.gamma, seed=args.seed,
                  opponent_mix=PHASES[args.phase], pool=pool,
                  clinch_start=args.clinch if args.clinch is not None else (0.3 if args.phase == 5 else 0.0),
                  sensing="fusion" if args.fusion else "truth", noise_max=args.noise_max)
    ppo = PPO(model, cfg)

    (out / "config.json").write_text(json.dumps({
        "args": vars(args), "ppo": asdict(cfg), "reward": asdict(rcfg), "opponent_mix": PHASES[args.phase],
        "control_dt": env.control_dt, "max_steps": env.max_steps, "obs_dim": OBS_DIM,
    }, indent=2))
    groups = list(SCRIPTED) + (["self"] if args.phase >= 3 else [])
    log = open(out / "log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["iter", "steps", "fps", "episodes", "win", "lose", "draw", "ep_len_s", "reward_step",
                     "pi_loss", "v_loss", "entropy", "kl", "lr", "std_lin", "std_ang", "pool_size"]
                    + ["win_" + g for g in groups] + ["n_" + g for g in groups])

    tb = SummaryWriter(str(out / "tb"))  # live curves: tensorboard --logdir runs
    T, N = cfg.rollout, args.envs
    obs = env.reset_all()
    cobs = env.critic_obs()
    asym = cobs is not None
    best, total = -1e9, 0
    ep_len = torch.zeros(N, device="cuda")
    t_start = time.time()
    for it in range(1, args.iters + 1):
        buf = {k: [] for k in ("obs", "act", "logp", "val", "rew", "term", "done", "next_val") + (("cobs",) if asym else ())}
        res = torch.zeros(len(groups), 3, device="cuda")  # win, lose, draw per opponent group
        lens = []
        t0 = time.time()
        for _ in range(T):
            with torch.no_grad():
                model.norm.update(obs)
                if asym:
                    model.cnorm.update(cobs)
                obs_n = model.norm(obs)
                d = model.dist(obs_n)
                act = d.sample()
                logp = d.log_prob(act).sum(-1)
                val = model.value(model.critic_in(obs, cobs))
            nobs, rew, term, trunc, info = env.step(act)
            with torch.no_grad():
                next_val = model.value(model.critic_in(info["final_obs"], info["final_cobs"]))
            done = term | trunc
            for k, v in (("obs", obs), ("act", act), ("logp", logp), ("val", val), ("rew", rew),
                         ("term", term.float()), ("done", done.float()), ("next_val", next_val)):
                buf[k].append(v)
            if asym:
                buf["cobs"].append(cobs)
                cobs = env.critic_obs()
            ep_len += 1
            if done.any():
                g = info["opp_kind"].clamp(max=len(groups) - 1)  # every pool slot -> "self"
                for j, key in enumerate(("win", "lose", "draw")):
                    res[:, j] += torch.bincount(g[info[key] & done], minlength=len(groups))[:len(groups)].float()
                lens.append(ep_len[done].clone())
                ep_len[done] = 0
            obs = nobs
        buf = {k: torch.stack(v) for k, v in buf.items()}
        stats = ppo.update(buf)
        if pool is not None:
            pool.set(len(pool) - 1, policy_sd(model.state_dict()), "latest")
            if it % args.snapshot_every == 0:
                pool.snapshot(policy_sd(model.state_dict()), "it%04d" % it)
                torch.save({"model": model.state_dict(), "iter": it}, out / "pool" / ("it%04d.pt" % it))
        total += T * N
        fps = T * N / (time.time() - t0)
        tot = res.sum(0)
        n_ep = int(tot.sum())
        win, lose, draw = (tot / max(n_ep, 1)).tolist()
        per = (res[:, 0] / res.sum(1).clamp_min(1)).tolist()
        mean_len = float(torch.cat(lens).mean()) * env.control_dt if lens else 0.0
        std = model.log_std.exp().tolist()
        writer.writerow([it, total, round(fps), n_ep, round(win, 4), round(lose, 4), round(draw, 4),
                         round(mean_len, 2), round(float(buf["rew"].mean()), 5), round(stats["pi_loss"], 5),
                         round(stats["v_loss"], 5), round(stats["entropy"], 4), round(stats["kl"], 5), stats["lr"],
                         round(std[0], 4), round(std[1], 4), len(pool) if pool else 0]
                        + [round(x, 4) for x in per] + [int(x) for x in res.sum(1).tolist()])
        log.flush()
        for k, v in (("resultado/vitoria", win), ("resultado/derrota", lose), ("resultado/empate", draw),
                     ("episodio/duracao_s", mean_len), ("episodio/recompensa_por_passo", float(buf["rew"].mean())),
                     ("ppo/perda_actor", stats["pi_loss"]), ("ppo/perda_critic", stats["v_loss"]),
                     ("ppo/entropia", stats["entropy"]), ("ppo/kl", stats["kl"]), ("ppo/lr", stats["lr"]),
                     ("ppo/std_linear", std[0]), ("ppo/std_angular", std[1]), ("treino/passos_por_s", fps)):
            tb.add_scalar(k, v, total)
        for g, x, cnt in zip(groups, per, res.sum(1).tolist()):
            if cnt > 0:
                tb.add_scalar("vitoria_por_adversario/" + g, x, total)
        if pool is not None:
            tb.add_scalar("treino/tamanho_do_grupo", len(pool), total)
        ckpt = {"model": model.state_dict(), "iter": it, "win": win, "phase": args.phase,
                "config": json.loads((out / "config.json").read_text())}
        torch.save(ckpt, out / "model.pt")
        # best: on the scripted opponents (phase 3 pool opponents get stronger, so their win rate is not comparable)
        scripted = res[:len(SCRIPTED)].sum(0)
        score = float(scripted[0] / scripted.sum().clamp_min(1)) if scripted.sum() >= 50 else -1e9
        if score >= best:
            best = score
            torch.save(ckpt, out / "best.pt")
        if it % 10 == 0 or it == 1:
            groups_txt = " ".join("%s %3.0f%%" % (g[:5], 100 * x) for g, x in zip(groups, per))
            print(f"it {it:4d} {total/1e6:6.1f}M {fps/1e3:5.0f}k/s ep {n_ep:5d} win {win:5.1%} lose {lose:5.1%} "
                  f"draw {draw:5.1%} len {mean_len:5.2f}s std {std[0]:.2f},{std[1]:.2f} | {groups_txt}", flush=True)
    tb.close()
    print(f"done in {(time.time()-t_start)/60:.1f} min; saved in {out}")


if __name__ == "__main__":
    main()
