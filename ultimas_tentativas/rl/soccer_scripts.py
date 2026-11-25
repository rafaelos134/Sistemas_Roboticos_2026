"""Deterministic soccer controllers, batched in torch, in the TEAM frame (the team attacks +x).

Used as the red team of phase A2 (gains drawn per episode) and as the blue keeper when
the attackers train with keeper="script".

  keeper:   stays on its goal line, y = ball y clipped to the goal mouth; faces the ball there.
  striker:  goes to the point behind the ball on the line ball -> their goal (going around the
            ball if it is on the wrong side), then pushes the ball towards the goal.
  support:  the attacker farther from the ball waits behind it, at (ball_x - 0.30, ball_y / 2).

Gains per robot p (..., 4): kw, alignment threshold, speed fraction, distance behind the ball.
"""

from __future__ import annotations

import math

import torch

HALF_L, HALF_W, GOAL_HALF = 0.9, 0.6, 0.2
W_MAX = 4.0


def _wrap(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def go_to(me, tx, ty, p, face=None):
    """Heading controller towards (tx, ty). Within 3 cm it only turns to `face` (an angle) if given."""
    dx, dy = tx - me[:, 0], ty - me[:, 1]
    e = _wrap(torch.atan2(dy, dx) - me[:, 2])
    w = (p[:, 0] * e / W_MAX).clamp(-1, 1)
    v = torch.where(e.abs() < p[:, 1], p[:, 2], torch.zeros_like(e))
    if face is not None:
        near = torch.hypot(dx, dy) < 0.03
        ef = _wrap(face - me[:, 2])
        w = torch.where(near, (p[:, 0] * ef / W_MAX).clamp(-1, 1), w)
        v = torch.where(near, torch.zeros_like(v), v)
    return torch.stack([v, w], 1)


def keeper_actions(view, p):
    mine, _, ball = view
    me = mine[0]
    tx = torch.full_like(ball[:, 0], -HALF_L + 0.07)
    ty = ball[:, 1].clamp(-GOAL_HALF + 0.03, GOAL_HALF - 0.03)
    face = torch.atan2(ball[:, 1] - me[:, 1], ball[:, 0] - me[:, 0])
    return go_to(me, tx, ty, p, face=face)


def striker(me, ball, p):
    gx, gy = HALF_L, 0.0
    ux, uy = gx - ball[:, 0], gy - ball[:, 1]
    d = torch.hypot(ux, uy).clamp_min(1e-6)
    ux, uy = ux / d, uy / d
    bx, by = ball[:, 0] - p[:, 3] * ux, ball[:, 1] - p[:, 3] * uy  # behind the ball
    rx, ry = ball[:, 0] - me[:, 0], ball[:, 1] - me[:, 1]
    along = rx * ux + ry * uy  # > 0: I am behind the ball
    lateral = (-rx * uy + ry * ux).abs()
    aligned = (along > 0.02) & (lateral < 0.035)
    # wrong side: aim at a point to the side of the ball first
    side = torch.where((-rx * uy + ry * ux) > 0, -1.0, 1.0)
    sx, sy = bx + side * 0.09 * -uy, by + side * 0.09 * ux
    tx = torch.where(aligned, ball[:, 0] + 0.10 * ux, torch.where(along < 0, sx, bx))
    ty = torch.where(aligned, ball[:, 1] + 0.10 * uy, torch.where(along < 0, sy, by))
    tx = tx.clamp(-HALF_L + 0.04, HALF_L - 0.04)
    ty = ty.clamp(-HALF_W + 0.04, HALF_W - 0.04)
    return go_to(me, tx, ty, p)


def team_script_actions(view, p):
    """Actions (3, n, 2) of a scripted team (keeper + striker + support). p: (3, n, 4)."""
    mine, _, ball = view
    out = torch.zeros(3, ball.shape[0], 2, device=ball.device)
    out[0] = keeper_actions(view, p[0])
    d1 = torch.hypot(mine[1][:, 0] - ball[:, 0], mine[1][:, 1] - ball[:, 1])
    d2 = torch.hypot(mine[2][:, 0] - ball[:, 0], mine[2][:, 1] - ball[:, 1])
    first = (d1 <= d2).unsqueeze(1)
    s1, s2 = striker(mine[1], ball, p[1]), striker(mine[2], ball, p[2])
    sx = (ball[:, 0] - 0.30).clamp(-HALF_L + 0.15, HALF_L - 0.15)
    sy = ball[:, 1] / 2
    h1, h2 = go_to(mine[1], sx, sy, p[1]), go_to(mine[2], sx, sy, p[2])
    out[1] = torch.where(first, s1, h1)
    out[2] = torch.where(first, h2, s2)
    return out
