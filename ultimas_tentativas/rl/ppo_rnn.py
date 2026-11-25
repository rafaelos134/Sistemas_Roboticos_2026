"""Recurrent policy for PPO: the MLP actor of ppo.ActorCritic plus a GRU memory added as a residual.

    mean(s_t, h_t) = actor(norm(s_t)) + head(h_t),   h_t = GRUCell(norm(s_t), h_{t-1})   (h = 0 at episode start)

`head` starts at zero, so a recurrent model initialized from a trained MLP checkpoint plays exactly like it
and only learns to use the memory (e.g. to recognize the style of the opponent during the round).
The critic stays feed-forward (V(s)); log_std and the observation normalization are the ones of the MLP.

PPO with recurrence: rollouts of T steps keep the hidden state of every env at the start (h0); the update
replays whole sequences (minibatches of envs, not of steps), resetting h where an episode ended.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ppo import PPO, ActorCritic


class RecurrentActorCritic(ActorCritic):
    def __init__(self, obs_dim, act_dim, hidden=(128, 128), mem=64, **kw):
        super().__init__(obs_dim, act_dim, hidden=hidden, **kw)
        self.mem = mem
        self.gru = nn.GRUCell(obs_dim, mem)
        self.head = nn.Linear(mem, act_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def init_hidden(self, n, device="cuda"):
        return torch.zeros(n, self.mem, device=device)

    def dist_h(self, obs_n, h):
        h = self.gru(obs_n, h)
        mean = self.actor(obs_n) + self.head(h)
        return torch.distributions.Normal(mean, self.log_std.clamp(-3.0, 0.0).exp()), h

    def mean_h(self, obs_n, h):
        h = self.gru(obs_n, h)
        return (self.actor(obs_n) + self.head(h)).clamp(-1, 1), h


class PPORnn(PPO):
    def update(self, buf):
        """buf: obs, act, logp, val, rew, term, done, next_val (T, N, ...) and h0 (N, mem)."""
        c, m = self.cfg, self.m
        T, N = buf["rew"].shape
        adv = torch.zeros_like(buf["rew"])
        last = torch.zeros(N, device=adv.device)
        for t in reversed(range(T)):
            delta = buf["rew"][t] + c.gamma * buf["next_val"][t] * (1.0 - buf["term"][t]) - buf["val"][t]
            last = delta + c.gamma * c.lam * (1.0 - buf["done"][t]) * last
            adv[t] = last
        ret = adv + buf["val"]
        ev = float(1.0 - (ret - buf["val"]).var() / ret.var().clamp_min(1e-8))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        obs_n = m.norm(buf["obs"].reshape(T * N, -1)).reshape(T, N, -1)
        keep = 1.0 - buf["done"]
        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0}
        n_up = 0
        mb = N // c.minibatches
        for _ in range(c.epochs):
            perm = torch.randperm(N, device=adv.device)
            for k in range(c.minibatches):
                i = perm[k * mb:(k + 1) * mb]
                h = buf["h0"][i]
                logps, ents = [], []
                for t in range(T):
                    d, h = m.dist_h(obs_n[t, i], h)
                    logps.append(d.log_prob(buf["act"][t, i]).sum(-1))
                    ents.append(d.entropy().sum(-1))
                    h = h * keep[t, i].unsqueeze(1)
                logp, ent = torch.stack(logps), torch.stack(ents).mean()
                ratio = torch.exp(logp - buf["logp"][:, i])
                a_ = adv[:, i]
                pi_loss = -torch.min(ratio * a_, ratio.clamp(1 - c.clip, 1 + c.clip) * a_).mean()
                v_loss = 0.5 * (m.value(obs_n[:, i].reshape(-1, obs_n.shape[-1])) - ret[:, i].reshape(-1)).pow(2).mean()
                loss = pi_loss + c.vf_coef * v_loss - c.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), c.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    kl = (buf["logp"][:, i] - logp).mean().item()
                stats["pi_loss"] += pi_loss.item(); stats["v_loss"] += v_loss.item()
                stats["entropy"] += ent.item(); stats["kl"] += kl
                n_up += 1
        for k in stats:
            stats[k] /= n_up
        if stats["kl"] > 2.0 * c.target_kl:
            self.lr = max(1e-5, self.lr / 1.5)
        elif stats["kl"] < 0.5 * c.target_kl:
            self.lr = min(1e-3, self.lr * 1.5)
        for g in self.opt.param_groups:
            g["lr"] = self.lr
        stats["lr"] = self.lr
        stats["explained_var"] = ev
        return stats
