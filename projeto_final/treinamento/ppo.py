"""PPO (Schulman et al., 2017) with a Gaussian actor and a separate critic, in plain PyTorch.

Actor  (policy)  pi(a|s):  obs(12) -> Linear 64 -> tanh -> Linear 64 -> tanh -> Linear 2 = mean
                           log_std: 2 free parameters (state independent)
                           a ~ Normal(mean, exp(log_std)); the environment clips a to [-1, 1]
Critic (value)   V(s):     obs(12) -> Linear 64 -> tanh -> Linear 64 -> tanh -> Linear 1

The two networks do not share weights. Observations are normalized with running
mean / std (the statistics are frozen and exported together with the actor).
Advantages: GAE(lambda). Truncated episodes (40 s timeout) bootstrap with V(s_final).

Asymmetric critic (critic_dim given): the critic reads its own, larger input (privileged information
available only in training, see soccer_critic.py) with its own running normalization `cnorm`.
The actor, its normalization and log_std are unchanged, so the exported node is the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def mlp(n_in, n_out, hidden=(64, 64), out_gain=1.0):
    layers, last = [], n_in
    for h in hidden:
        lin = nn.Linear(last, h)
        nn.init.orthogonal_(lin.weight, gain=2 ** 0.5)
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        last = h
    out = nn.Linear(last, n_out)
    nn.init.orthogonal_(out.weight, gain=out_gain)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, clip=5.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.eps, self.clip = eps, clip

    @torch.no_grad()
    def update(self, x):
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        self.var = (self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot) / tot
        self.count = tot

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + self.eps)).clamp(-self.clip, self.clip)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(64, 64), init_std=0.6, critic_dim=None):
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.actor = mlp(obs_dim, act_dim, hidden, out_gain=0.01)
        self.asym = critic_dim is not None
        if self.asym:
            self.cnorm = RunningNorm(critic_dim)
        self.critic = mlp(critic_dim or obs_dim, 1, hidden, out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(torch.log(torch.tensor(init_std)))))

    def dist(self, obs_n):
        # std limited to [0.05, 1]: the action is clipped to [-1, 1], so a larger std would only make the
        # policy bang-bang (and the entropy bonus would keep pushing it up for free)
        return torch.distributions.Normal(self.actor(obs_n), self.log_std.clamp(-3.0, 0.0).exp())

    def value(self, obs_n):
        return self.critic(obs_n).squeeze(-1)

    def critic_in(self, obs, cobs=None):
        """Normalized critic input: cnorm(cobs) for the asymmetric critic, norm(obs) otherwise."""
        return self.cnorm(cobs) if self.asym else self.norm(obs)


def actor_state(sd):
    """Only the policy part of a state dict (actor, norm, log_std): to load it into a model whose
    critic has another shape. Use with load_state_dict(..., strict=False)."""
    return {k: v for k, v in sd.items() if not k.startswith(("critic.", "cnorm."))}


@dataclass
class PPOConfig:
    rollout: int = 32  # steps per env per iteration
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    ent_coef: float = 0.003
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    target_kl: float = 0.02  # lr is lowered when KL is above, raised when far below


class PPO:
    def __init__(self, model: ActorCritic, cfg: PPOConfig):
        self.m, self.cfg = model, cfg
        self.opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        self.lr = cfg.lr

    def update(self, buf):
        """buf: dict of tensors (T, N, ...): obs, act, logp, val, rew, done, next_val (+ cobs, asymmetric critic)."""
        c = self.cfg
        T, N = buf["rew"].shape
        adv = torch.zeros_like(buf["rew"])
        last = torch.zeros(N, device=adv.device)
        for t in reversed(range(T)):
            nonterm = 1.0 - buf["term"][t]
            delta = buf["rew"][t] + c.gamma * buf["next_val"][t] * nonterm - buf["val"][t]
            last = delta + c.gamma * c.lam * (1.0 - buf["done"][t]) * last
            adv[t] = last
        ret = adv + buf["val"]
        # explained variance of the critic on this batch (1 = perfect, 0 = no better than the mean)
        ev = float(1.0 - (ret - buf["val"]).var() / ret.var().clamp_min(1e-8))

        flat = lambda x: x.reshape(T * N, *x.shape[2:])  # noqa: E731
        obs, act, logp_old, adv, ret = map(flat, (buf["obs"], buf["act"], buf["logp"], adv, ret))
        # optional mask: steps whose action did not come from the policy (e.g. a supervisor took over) are
        # left out of the policy loss (they still count for the critic and for the advantages)
        pmask = flat(buf["mask"]) if "mask" in buf else None
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        obs_n = self.m.norm(obs)
        cin = self.m.critic_in(obs, flat(buf["cobs"]) if "cobs" in buf else None)

        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0}
        n_up = 0
        mb = T * N // c.minibatches
        for _ in range(c.epochs):
            perm = torch.randperm(T * N, device=obs.device)
            for k in range(c.minibatches):
                i = perm[k * mb:(k + 1) * mb]
                d = self.m.dist(obs_n[i])
                logp = d.log_prob(act[i]).sum(-1)
                ratio = torch.exp(logp - logp_old[i])
                surr = torch.min(ratio * adv[i], ratio.clamp(1 - c.clip, 1 + c.clip) * adv[i])
                if pmask is None:
                    pi_loss = -surr.mean()
                else:
                    pi_loss = -(surr * pmask[i]).sum() / pmask[i].sum().clamp_min(1.0)
                v_loss = 0.5 * (self.m.value(cin[i]) - ret[i]).pow(2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pi_loss + c.vf_coef * v_loss - c.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.m.parameters(), c.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    kl = (logp_old[i] - logp).mean().item()
                stats["pi_loss"] += pi_loss.item(); stats["v_loss"] += v_loss.item()
                stats["entropy"] += ent.item(); stats["kl"] += kl
                n_up += 1
        for k in stats:
            stats[k] /= n_up
        # adaptive learning rate on the KL divergence
        if stats["kl"] > 2.0 * c.target_kl:
            self.lr = max(1e-5, self.lr / 1.5)
        elif stats["kl"] < 0.5 * c.target_kl:
            self.lr = min(1e-3, self.lr * 1.5)
        for g in self.opt.param_groups:
            g["lr"] = self.lr
        stats["lr"] = self.lr
        stats["explained_var"] = ev
        return stats
