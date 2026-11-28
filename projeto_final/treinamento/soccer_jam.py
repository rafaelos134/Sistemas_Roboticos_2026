"""Stuck-ball rule of the tournament server ("Bola presa: reposta no ponto neutro"), batched in torch.

The kit (psr_sim/scenarios/soccer.py) does not have this rule; it was inferred from the 646 events
recorded live in the tournament (torneio_logs/dados/ao_vivo/soccer, analysis in the docs):

  * trigger: the ball stays ~10 s near the same point (almost always pinned against a long wall
    by a robot). Modelled as: anchor = ball position; if the ball leaves a circle of radius
    `radius` around the anchor, the anchor moves there and the timer restarts; timer >= `hold_s`
    -> teleport. hold_s / radius are drawn per episode around (10 s, 5 cm), because the exact
    criterion of the server could not be recovered from the ~3 Hz samples.
  * destination: the 5 neutral points (+-0.45, +-0.30) and (0, 0); the one nearest to the ball
    among those with no robot closer than `free_r` (fallback: the nearest one).
  * after the teleport the ball is HELD at the neutral point for `pause_s` (2 s) while the robots
    keep playing (they are not frozen or moved); then the game goes on.
"""

from __future__ import annotations

import torch

NEUTRAL = ((-0.45, -0.30), (-0.45, 0.30), (0.45, -0.30), (0.45, 0.30), (0.0, 0.0))


class JamReferee:
    def __init__(self, n, control_dt, gen, hold_s=(9.0, 11.0), radius=(0.04, 0.07), pause_s=2.0, free_r=0.15):
        dev = "cuda"
        self.n, self.dt, self.gen = n, control_dt, gen
        self.hold_rng, self.radius_rng = hold_s, radius
        self.pause_steps = int(round(pause_s / control_dt))
        self.free_r = free_r
        self.neutral = torch.tensor(NEUTRAL, device=dev)  # (5, 2)
        self.anchor = torch.zeros(n, 2, device=dev)
        self.still = torch.zeros(n, dtype=torch.long, device=dev)  # steps near the anchor
        self.held = torch.zeros(n, dtype=torch.long, device=dev)  # steps left with the ball held
        self.hold_pos = torch.zeros(n, 2, device=dev)
        self.limit = torch.zeros(n, dtype=torch.long, device=dev)
        self.radius = torch.zeros(n, device=dev)
        self.count = torch.zeros(n, dtype=torch.long, device=dev)  # teleports in this episode

    def reset(self, idx, ball_xy):
        k = idx.numel()
        u = lambda lo, hi: lo + (hi - lo) * torch.rand(k, device="cuda", generator=self.gen)  # noqa: E731
        self.anchor[idx] = ball_xy
        self.still[idx] = 0
        self.held[idx] = 0
        self.count[idx] = 0
        self.limit[idx] = (u(*self.hold_rng) / self.dt).round().long()
        self.radius[idx] = u(*self.radius_rng)

    def update(self, ball_xy, robots_xy):
        """After a control step. ball_xy (n, 2), robots_xy (6, n, 2). Returns (teleport (n,) bool, target (n, 2)):
        the envs whose ball must be moved now (the caller holds it there while `self.held > 0`)."""
        self.held = (self.held - 1).clamp_min(0)
        free = self.held == 0
        moved = torch.hypot(ball_xy[:, 0] - self.anchor[:, 0], ball_xy[:, 1] - self.anchor[:, 1]) > self.radius
        reset = moved | ~free
        self.anchor = torch.where(reset.unsqueeze(1), ball_xy, self.anchor)
        self.still = torch.where(reset, torch.zeros_like(self.still), self.still + 1)
        fire = free & (self.still >= self.limit)
        target = self.hold_pos.clone()
        if fire.any():
            d_ball = torch.cdist(ball_xy.unsqueeze(1), self.neutral.unsqueeze(0).expand(self.n, -1, -1)).squeeze(1)  # (n, 5)
            d_rob = (robots_xy.permute(1, 0, 2).unsqueeze(2) - self.neutral.view(1, 1, 5, 2)).norm(dim=3).amin(1)  # (n, 5)
            busy = d_rob < self.free_r
            cost = d_ball + 100.0 * busy.float()  # nearest free point, else the nearest one
            pick = cost.argmin(1)
            target = torch.where(fire.unsqueeze(1), self.neutral[pick], target)
            self.hold_pos = target
            self.held = torch.where(fire, torch.full_like(self.held, self.pause_steps), self.held)
            self.anchor = torch.where(fire.unsqueeze(1), target, self.anchor)
            self.still[fire] = 0
            self.count += fire.long()
        return fire, target


class JamRuleNp:
    """The same rule for the real psr_sim (eval_soccer.py --jam), fixed at hold_s / radius."""

    def __init__(self, hold_s=10.0, radius=0.05, pause_s=2.0, free_r=0.15):
        self.hold_s, self.radius, self.pause_s, self.free_r = hold_s, radius, pause_s, free_r
        self.anchor, self.t_anchor, self.held_until, self.pos, self.count = None, 0.0, -1.0, None, 0

    def reset(self):
        self.anchor, self.held_until = None, -1.0

    def update(self, t, ball, robots):
        """ball (x, y), robots [(x, y)], t in s. Returns the point where the ball must be now (held or
        just teleported), or None."""
        import math

        if t < self.held_until:
            return self.pos
        if self.anchor is None or math.hypot(ball[0] - self.anchor[0], ball[1] - self.anchor[1]) > self.radius:
            self.anchor, self.t_anchor = ball, t
            return None
        if t - self.t_anchor < self.hold_s:
            return None
        cost = []
        for q in NEUTRAL:
            busy = min(math.hypot(q[0] - r[0], q[1] - r[1]) for r in robots) < self.free_r
            cost.append(math.hypot(q[0] - ball[0], q[1] - ball[1]) + (100.0 if busy else 0.0))
        self.pos = NEUTRAL[cost.index(min(cost))]
        self.held_until = t + self.pause_s
        self.anchor, self.t_anchor = self.pos, self.held_until
        self.count += 1
        return self.pos
