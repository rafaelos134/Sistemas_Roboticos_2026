"""Capture the flag with Double DQN (van Hasselt et al., 2016) + dueling network (Wang et al., 2016) + n-step
returns, on the team environment (ctf_team_env.py), with the SAME reward, opponents and randomization as PPO.

Discrete actions: 5 linear x 5 angular commands (25), in [-1, 1] x [-1, 1].
One Q-network is shared by both blue robots (the observation is relative to the robot, "me" / "teammate"),
so each env step gives 2 transitions. With --dual both robots are attackers (as train_ctf_team.py --dual).

    $PY train_ctf_dqn.py --name ctf_dqn_1 --dual --minutes 40 --att-pool ... --def-pool ...

Q(s, a) = V(s) + A(s, a) - mean_a A(s, a). Target: r_t + ... + g^n * Q_target(s_t+n, argmax_a Q(s_t+n, a)).
The checkpoint stores the dueling heads merged into one linear layer (Q is linear in the last hidden layer),
in the same "actor.*" format as PPO + an "actions" table: export_ctf.py exports it as a node (argmax).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from ctf_obs import OBS_DIM
from ctf_team_env import CtfTeamEnv, TeamReward
from ppo import RunningNorm, mlp
from train_ctf import load_pool

LIN = (-1.0, -0.5, 0.0, 0.5, 1.0)
ANG = (-1.0, -0.5, 0.0, 0.5, 1.0)
ACTIONS = torch.tensor([(v, w) for v in LIN for w in ANG])  # (25, 2)


class DuelingQ(nn.Module):
    def __init__(self, obs_dim, n_act, hidden=(256, 256)):
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.body = mlp(obs_dim, hidden[-1], hidden[:-1])
        self.v = nn.Linear(hidden[-1], 1)
        self.a = nn.Linear(hidden[-1], n_act)

    def forward(self, obs):
        h = torch.tanh(self.body(self.norm(obs)))
        a = self.a(h)
        return self.v(h) + a - a.mean(-1, keepdim=True)

    def merged_state(self):
        """PPO-actor-like state dict: actor.0 .. actor.2k (Linear layers), last = merged dueling heads."""
        lins = [m for m in self.body if isinstance(m, nn.Linear)]
        sd = {"norm.mean": self.norm.mean.cpu(), "norm.var": self.norm.var.cpu()}
        for j, lin in enumerate(lins):
            sd["actor.%d.weight" % (2 * j)] = lin.weight.detach().cpu()
            sd["actor.%d.bias" % (2 * j)] = lin.bias.detach().cpu()
        wa, ba = self.a.weight.detach(), self.a.bias.detach()
        w = self.v.weight.detach() + wa - wa.mean(0, keepdim=True)
        b = self.v.bias.detach() + ba - ba.mean()
        k = 2 * len(lins)
        sd["actor.%d.weight" % k], sd["actor.%d.bias" % k] = w.cpu(), b.cpu()
        return sd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--envs", type=int, default=1024)
    p.add_argument("--minutes", type=float, default=40.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dual", action="store_true")
    p.add_argument("--att-pool", nargs="*", default=[])
    p.add_argument("--def-pool", nargs="*", default=[])
    p.add_argument("--main-mix", default='{"script": 0.5, "latest": 0.2, "pool": 0.2, "stopped": 0.1}')
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--nstep", type=int, default=3)
    p.add_argument("--buffer", type=int, default=2_000_000)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--updates", type=int, default=2, help="gradient steps per env step")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-frac", type=float, default=0.3, help="fraction of the run for the epsilon decay")
    p.add_argument("--time-cost", type=float, default=0.3)
    p.add_argument("--draw", type=float, default=-5.0)
    p.add_argument("--easy", type=float, default=0.4)
    p.add_argument("--strong", type=float, default=0.7)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda"
    env = CtfTeamEnv(team_reward=TeamReward(time_cost=args.time_cost, draw=args.draw), dual=args.dual,
                     n_envs=args.envs, gamma=args.gamma, seed=args.seed, easy_start=args.easy, episode_s=30.0,
                     obs_noise=0.004, yaw_noise=0.03, att_styles=[0.4, 0.2, 0.2, 0.2],
                     def_styles=[0.3, 0.2, 0.2, 0.15, 0.15], strong=args.strong, dr_speed=0.1, dr_delay=0.3,
                     main_mix=json.loads(args.main_mix), other_mix={"script": 1.0},
                     att_pool=load_pool(args.att_pool), def_pool=load_pool(args.def_pool))
    n_act = len(ACTIONS)
    acts_tab = ACTIONS.to(dev)
    q, qt = DuelingQ(OBS_DIM, n_act).to(dev), DuelingQ(OBS_DIM, n_act).to(dev)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)
    out = Path(__file__).resolve().parent / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"args": vars(args), "actions": ACTIONS.tolist()}, indent=2))
    log = open(out / "log.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["minute", "env_steps", "transitions", "eps", "episodes", "scored", "conceded", "nopoint", "q_mean", "loss"])

    # replay (GPU): n-step transitions of both robots
    N, B = args.buffer, 2 * args.envs
    R_obs = torch.zeros(N, OBS_DIM, device=dev)
    R_nobs = torch.zeros(N, OBS_DIM, device=dev)
    R_act = torch.zeros(N, dtype=torch.long, device=dev)
    R_rew = torch.zeros(N, device=dev)
    R_disc = torch.zeros(N, device=dev)  # gamma^k of the bootstrap (0 when terminal)
    ptr, full = 0, False
    # n-step queues per robot/env
    qo, qa, qr, qd, fins = [], [], [], [], []  # n-step queues of (B, ...) tensors

    def push(o, a, r, no, disc):
        nonlocal ptr, full
        k = o.shape[0]
        idx = (torch.arange(k, device=dev) + ptr) % N
        R_obs[idx], R_act[idx], R_rew[idx], R_nobs[idx], R_disc[idx] = o, a, r, no, disc
        ptr = (ptr + k) % N
        full = full or ptr < k

    obs = env.reset_all()
    t0 = time.time()
    total_sec = 60.0 * args.minutes
    steps, n_ep, res = 0, 0, torch.zeros(3, device=dev)
    q_mean = loss_v = 0.0
    next_log = 60.0
    best = -1e9
    while time.time() - t0 < total_sec:
        frac = (time.time() - t0) / total_sec
        eps = max(args.eps_end, args.eps_start + (args.eps_end - args.eps_start) * min(1.0, frac / args.eps_frac))
        o = obs.reshape(B, OBS_DIM)
        with torch.no_grad():
            q.norm.update(o)
            greedy = q(o).argmax(-1)
        rnd = torch.randint(0, n_act, (B,), device=dev)
        a = torch.where(torch.rand(B, device=dev) < eps, rnd, greedy)
        nobs, rew, term, trunc, info = env.step(acts_tab[a].reshape(2, env.n, 2))
        done = term | trunc
        steps += 1
        # n-step bookkeeping: rewards accumulate until n steps or the end of the episode
        fin = info["final_obs"].reshape(B, OBS_DIM)
        d2 = done.repeat(2)
        t2 = term.repeat(2)
        qo.append(o); qa.append(a); qr.append(rew.reshape(B)); qd.append(torch.stack([d2, t2])); fins.append(fin)
        if len(qo) >= args.nstep:
            ret, disc_b, boot = nstep_target(qr, qd, fins, qo, args.gamma, args.nstep)
            push(qo[0], qa[0], ret, boot, disc_b)
            for z in (qo, qa, qr, qd, fins):
                z.pop(0)
        obs = nobs
        if done.any():
            n_ep += int(done.sum())
            res += torch.stack([info["scored"][done].sum(), info["conceded"][done].sum(),
                                (info["timeout"] | info["truncated"])[done].sum()]).float()
        # learning
        size = N if full else ptr
        if size > 50_000:
            for _ in range(args.updates):
                i = torch.randint(0, size, (args.batch,), device=dev)
                with torch.no_grad():
                    na = q(R_nobs[i]).argmax(-1, keepdim=True)
                    target = R_rew[i] + R_disc[i] * qt(R_nobs[i]).gather(1, na).squeeze(1)
                qv = q(R_obs[i]).gather(1, R_act[i].unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(qv, target)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()
                with torch.no_grad():
                    for pt, pq in zip(qt.parameters(), q.parameters()):
                        pt.lerp_(pq, args.tau)
                    qt.norm.load_state_dict(q.norm.state_dict())
            q_mean, loss_v = float(qv.mean()), float(loss)
        el = time.time() - t0
        if el >= next_log:
            next_log += 60.0
            sc, co, no = (res / res.sum().clamp_min(1)).tolist()
            wr.writerow([round(el / 60, 1), steps, steps * B, round(eps, 3), n_ep, round(sc, 4), round(co, 4), round(no, 4),
                         round(q_mean, 3), round(loss_v, 4)])
            log.flush()
            print(f"{el/60:5.1f} min  {steps*B/1e6:6.1f}M trans  eps {eps:.2f}  ep {n_ep:6d}  ponto {sc:5.1%}  "
                  f"sofrido {co:5.1%}  sem-ponto {no:5.1%}  Q {q_mean:6.2f}  loss {loss_v:.4f}", flush=True)
            ck = {"model": q.merged_state(), "actions": ACTIONS, "dqn": q.state_dict(), "minute": el / 60,
                  "scored": sc, "conceded": co}
            torch.save(ck, out / "model.pt")
            if int(el / 60) % 10 == 0:
                torch.save(ck, out / ("ck_%03dmin.pt" % int(el / 60)))
            if frac > 0.4 and sc - co >= best:
                best = sc - co
                torch.save(ck, out / "best.pt")
            res.zero_()
            n_ep = 0
    print(f"done in {(time.time()-t0)/60:.1f} min; saved in {out}")


def nstep_target(qr, qd, fins, qo, gamma, n):
    """n-step return of the oldest transition in the queues.
    qr[k]: reward after step k; qd[k] = (done_k, terminal_k); fins[k]: final observation of step k
    (= next observation, or the last one before a reset). Returns (return, bootstrap discount, bootstrap obs)."""
    B = qr[0].shape[0]
    dev = qr[0].device
    ret = torch.zeros(B, device=dev)
    disc = torch.ones(B, device=dev)
    alive = torch.ones(B, dtype=torch.bool, device=dev)
    boot_obs = fins[n - 1].clone()
    boot_disc = torch.full((B,), gamma ** n, device=dev)
    for k in range(n):
        ret = ret + torch.where(alive, disc * qr[k], torch.zeros_like(ret))
        ended = alive & qd[k][0]
        terminal = alive & qd[k][1]
        boot_obs = torch.where(ended.unsqueeze(1), fins[k], boot_obs)
        boot_disc = torch.where(ended, torch.where(terminal, torch.zeros_like(disc), disc * gamma), boot_disc)
        alive = alive & ~qd[k][0]
        disc = disc * gamma
    return ret, boot_disc, boot_obs


if __name__ == "__main__":
    main()
