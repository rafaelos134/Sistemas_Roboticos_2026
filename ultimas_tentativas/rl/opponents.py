"""Opponents of the sumo training, batched in torch (one opponent per environment).

Scripted (phase 2), each with gains drawn at random per episode:
  STOPPED   holds its position (like robot1:=none)
  CHASER    turns to the agent and drives straight at it (what most teams will do)
  CAREFUL   chaser that goes back to the centre when it gets near the edge
  EVADER    circles around the centre and runs sideways when the agent comes close
  RANDOM    random (v, w) held for ~1 s, going back to the centre near the edge
  PURSUER   relentless chaser: never stops, v = speed * cos(heading error), aims ahead of the
            agent (lead pursuit); it ends up orbiting an agent that sidesteps (phase 4)
  PURSUER_BI the same, bidirectional: drives backwards when the agent is behind it

Policies (phase 3): frozen copies of the agent's own network (checkpoints), see PolicyPool.

The held-out opponent used only for evaluation (sumo_flanker.py, pushes towards the
nearest edge from the side) is NOT here on purpose.

Every function gets the opponent state `me` and the agent state `tgt`, both (n, 5):
x, y, yaw, v, w, and returns actions (n, 2) in [-1, 1] (fraction of V_MAX, W_MAX).
"""

from __future__ import annotations

import math

import torch

from sumo_obs import V_MAX, W_MAX, observation_torch

STOPPED, CHASER, CAREFUL, EVADER, RANDOM, PURSUER, PURSUER_BI = range(7)
SCRIPTED = ("stopped", "chaser", "careful", "evader", "random", "pursuer", "pursuer_bi")
N_PARAMS = 6


def wrap(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def sample_params(kind: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Per-episode gains, (n, N_PARAMS):
    0 kw (rad/s per rad of heading error), 1 alignment threshold (rad), 2 speed fraction,
    3 edge guard radius (m), 4 circle radius (evader), 5 flee distance (evader) / circle direction sign."""
    n = kind.shape[0]
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, device=kind.device, generator=gen)  # noqa: E731
    p = torch.stack([u(2.0, 6.0), u(0.3, 0.8), u(0.6, 1.0), u(0.24, 0.32), u(0.12, 0.22), u(0.12, 0.20)], 1)
    # pursuers: stronger and faster (kw in [4, 8], speed in [0.85, 1]): the hard cases that draw
    pur = (kind == PURSUER) | (kind == PURSUER_BI)
    p[:, 0] = torch.where(pur, p[:, 0] + 2.0, p[:, 0])
    p[:, 2] = torch.where(pur, 0.85 + 0.15 * (p[:, 2] - 0.6) / 0.4, p[:, 2])
    return p


def go_to(me, tx, ty, p, reverse_ok=False):
    """Heading controller towards (tx, ty): turn proportionally, drive when aligned."""
    e = wrap(torch.atan2(ty - me[:, 1], tx - me[:, 0]) - me[:, 2])
    w = (p[:, 0] * e / W_MAX).clamp(-1, 1)
    v = torch.where(e.abs() < p[:, 1], p[:, 2], torch.zeros_like(e))
    return torch.stack([v, w], 1)


def scripted_actions(kind, p, me, tgt, hold):
    """Actions of all scripted kinds (rows with a policy kind are ignored by the caller)."""
    n = me.shape[0]
    act = torch.zeros(n, 2, device=me.device)
    r_me = torch.hypot(me[:, 0], me[:, 1])
    near_edge = r_me > p[:, 3]
    zeros = torch.zeros(n, device=me.device)

    chase = go_to(me, tgt[:, 0], tgt[:, 1], p)
    home = go_to(me, zeros, zeros, p)
    act = torch.where((kind == CHASER).unsqueeze(1), chase, act)
    act = torch.where(((kind == CAREFUL) & ~near_edge).unsqueeze(1), chase, act)
    act = torch.where(((kind == CAREFUL) & near_edge).unsqueeze(1), home, act)

    # evader: target a point ahead on a circle; when the agent is close, run sideways
    sign = torch.where(p[:, 5] > 0.16, 1.0, -1.0)
    th = torch.atan2(me[:, 1], me[:, 0]) + sign * 0.6
    cx, cy = p[:, 4] * torch.cos(th), p[:, 4] * torch.sin(th)
    dx, dy = me[:, 0] - tgt[:, 0], me[:, 1] - tgt[:, 1]
    d = torch.hypot(dx, dy).clamp_min(1e-6)
    fx = me[:, 0] + 0.15 * (dx * 0.5 - sign * dy * 0.866) / d  # away from the agent, rotated 60 deg
    fy = me[:, 1] + 0.15 * (dy * 0.5 + sign * dx * 0.866) / d
    fr = torch.hypot(fx, fy).clamp_min(1e-6)
    scale = torch.clamp(0.25 / fr, max=1.0)  # keep the flee point inside r = 0.25
    flee = d < p[:, 5]
    ev = go_to(me, torch.where(flee, fx * scale, cx), torch.where(flee, fy * scale, cy), p)
    ev[:, 0] = torch.where(flee, torch.ones_like(ev[:, 0]) * (ev[:, 0] > 0), ev[:, 0])  # run at full speed
    act = torch.where(((kind == EVADER) & ~near_edge).unsqueeze(1), ev, act)
    act = torch.where(((kind == EVADER) & near_edge).unsqueeze(1), home, act)

    act = torch.where(((kind == RANDOM) & ~near_edge).unsqueeze(1), hold, act)
    act = torch.where(((kind == RANDOM) & near_edge).unsqueeze(1), home, act)

    # pursuers (phase 4): aim at where the agent will be (lead time p[1] - 0.3 in [0, 0.5] s),
    # turn with gain kw and never stop driving: the orbit an agent that only sidesteps gets stuck in
    lead = (p[:, 1] - 0.3).clamp_min(0.0)
    px = tgt[:, 0] + lead * tgt[:, 3] * torch.cos(tgt[:, 2])
    py = tgt[:, 1] + lead * tgt[:, 3] * torch.sin(tgt[:, 2])
    e = wrap(torch.atan2(py - me[:, 1], px - me[:, 0]) - me[:, 2])
    pu = torch.stack([p[:, 2] * torch.cos(e).clamp_min(0.0), (p[:, 0] * e / W_MAX).clamp(-1, 1)], 1)
    back = e.abs() > math.pi / 2  # bidirectional: the rear is as good as the front on the HeRo
    eb = torch.where(back, wrap(e - math.pi), e)
    pb = torch.stack([p[:, 2] * torch.cos(e), (p[:, 0] * eb / W_MAX).clamp(-1, 1)], 1)
    # half of the pursuers keep an edge guard (p[5] > 0.16), the others chase all the way
    guard = near_edge & (p[:, 5] > 0.16)
    act = torch.where(((kind == PURSUER) & ~guard).unsqueeze(1), pu, act)
    act = torch.where(((kind == PURSUER_BI) & ~guard).unsqueeze(1), pb, act)
    act = torch.where((((kind == PURSUER) | (kind == PURSUER_BI)) & guard).unsqueeze(1), home, act)
    return act


class PolicyPool:
    """Frozen copies of the agent (phase 3). Slot k is opponent kind len(SCRIPTED) + k."""

    def __init__(self, make_model, max_size: int = 16):
        self.make_model, self.max_size = make_model, max_size
        self.models: list = []
        self.names: list[str] = []

    def __len__(self):
        return len(self.models)

    def add(self, state_dict, name: str):
        m = self.make_model().cuda().eval()
        m.load_state_dict(state_dict)
        for q in m.parameters():
            q.requires_grad_(False)
        if len(self.models) >= self.max_size:  # forget the oldest (slot 0 is kept: the start of phase 3)
            del self.models[1], self.names[1]
        self.models.append(m)
        self.names.append(name)

    def snapshot(self, state_dict, name: str):
        """Insert a frozen copy before the last ("latest") slot; when full, drop the oldest
        copy but keep slot 0 (the policy phase 3 started from)."""
        m = self.make_model().cuda().eval()
        m.load_state_dict(state_dict)
        for q in m.parameters():
            q.requires_grad_(False)
        if len(self.models) >= self.max_size:
            del self.models[1], self.names[1]
        self.models.insert(len(self.models) - 1, m)
        self.names.insert(len(self.names) - 1, name)

    def set(self, k: int, state_dict, name: str):
        self.models[k].load_state_dict(state_dict)
        self.names[k] = name

    @torch.no_grad()
    def actions(self, kind, me, tgt, prev):
        """Deterministic actions (mean of the Gaussian) of every pool member on its rows."""
        act = torch.zeros(me.shape[0], 2, device=me.device)
        base = len(SCRIPTED)
        for k, m in enumerate(self.models):
            rows = (kind == base + k).nonzero(as_tuple=False).flatten()
            if rows.numel():
                obs = observation_torch(me[rows], tgt[rows], prev[rows])
                act[rows] = m.actor(m.norm(obs)).clamp(-1, 1)
        return act
