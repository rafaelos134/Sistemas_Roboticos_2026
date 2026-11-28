"""Deterministic capture-the-flag controllers, batched in torch, in the TEAM frame (base at -x, attack +x).

Used as opponents / teammates while the networks train, with gains drawn per episode. The same logic,
with fixed gains (GAINS_REF), is the kit node kit/exercicios/ctf/ctf_script.py (the reference opponent).

  attacker: carrying -> straight back to x = 0 (reverse allowed), then home.
            not carrying -> to the enemy flag; in the enemy half it is pushed away from an active defender
            closer than `avoid`; if the defender guards the flag (closer than `wait` to it and closer to it
            than me) it waits at the border on the side opposite to the defender.
  defender: an opponent carries my flag -> intercept it a little ahead of it (towards x = 0);
            an active intruder in my half closer than `chase` to my flag -> go at it (tag);
            else guard: on the segment own flag -> their attacker, `guard` m from the flag (x <= -0.04).

Gains p (n, 7): kw, alignment threshold, speed fraction, avoid, wait, guard, chase.

Styles (drawn per episode, `style` (n,) long), to train against the kinds of teams other people write:
  attacker  0 standard (above)          1 rush: straight to the flag, never avoids / waits
            2 flank: along the side wall (y = +-0.6) to x ~ 0.45, then to the flag
            3 return-safe: like 0, carrying it goes home away from the defenders (wide arc)
  defender  0 standard (above)          1 on the flag: stands over its flag, chases only the carrier
            2 hunter: chases any active intruder anywhere in its half
            3 line keeper: on x = -0.30, follows the y of the attacker
            4 second attacker: plays the attacker logic (both robots attack)
""" 

from __future__ import annotations

import math

import torch

BASE_X, HALF = 0.62, 0.8
W_MAX = 4.0
GAIN_LO = (3.0, 0.4, 0.7, 0.12, 0.10, 0.05, 0.25)
GAIN_HI = (6.0, 0.9, 1.0, 0.25, 0.25, 0.18, 0.60)
GAINS_REF = (4.0, 0.6, 1.0, 0.18, 0.18, 0.10, 0.45)


def _wrap(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def drive(me, tx, ty, p, stop_r=0.02):
    """Heading controller towards (tx, ty) that drives backwards when the target is behind. (n, 2) in [-1, 1]."""
    dx, dy = tx - me[:, 0], ty - me[:, 1]
    e = _wrap(torch.atan2(dy, dx) - me[:, 2])
    back = e.abs() > math.pi / 2
    e2 = torch.where(back, _wrap(e + math.pi), e)
    w = (p[:, 0] * e2 / W_MAX).clamp(-1, 1)
    v = torch.where(e2.abs() < p[:, 1], p[:, 2], torch.zeros_like(e2))
    v = torch.where(back, -v, v)
    near = torch.hypot(dx, dy) < stop_r
    v = torch.where(near, torch.zeros_like(v), v)
    w = torch.where(near, torch.zeros_like(w), w)
    return torch.stack([v, w], 1)


def _dist(a, b):
    return torch.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1])


ATT_STYLES, DEF_STYLES = 4, 5


def attacker_actions(mine, theirs, own_flag, enemy_flag, carry, hold, p, style=None, role=0):
    me, dfn = mine[role], theirs[1]
    # carrying: to x = 0 first (straight line), then home
    home_x = torch.where(me[:, 0] > -0.03, torch.full_like(me[:, 0], -0.15), torch.full_like(me[:, 0], -BASE_X))
    home_y = torch.where(me[:, 0] > -0.03, me[:, 1], torch.zeros_like(me[:, 1]))
    # not carrying: to the flag, pushed away from the active defender in the enemy half
    active = hold[3] <= 0
    d_def = _dist(me, dfn).clamp_min(1e-3)
    to_f = enemy_flag - me[:, :2]
    to_f = to_f / to_f.norm(dim=1, keepdim=True).clamp_min(1e-3)
    away = (me[:, :2] - dfn[:, :2]) / d_def.unsqueeze(1)
    push = (active & (me[:, 0] > -0.05) & (d_def < p[:, 3])).float() * (p[:, 3] / d_def).clamp(max=3.0)
    tgt = me[:, :2] + 0.2 * to_f + 0.2 * push.unsqueeze(1) * away
    guarded = active & (_dist(dfn, enemy_flag) < p[:, 4]) & (_dist(me, enemy_flag) > _dist(dfn, enemy_flag) + 0.05)
    wait_y = torch.where(dfn[:, 1] > 0, torch.full_like(dfn[:, 1], -0.3), torch.full_like(dfn[:, 1], 0.3))
    tx = torch.where(guarded, torch.full_like(tgt[:, 0], -0.06), tgt[:, 0])
    ty = torch.where(guarded, wait_y, tgt[:, 1])
    if style is not None:
        # 1 rush: straight at the flag
        rush = style == 1
        tx = torch.where(rush, enemy_flag[:, 0], tx)
        ty = torch.where(rush, enemy_flag[:, 1], ty)
        # 2 flank: along the side wall until x ~ 0.45, then to the flag
        side = torch.where(me[:, 1] >= 0, torch.full_like(me[:, 1], 0.6), torch.full_like(me[:, 1], -0.6))
        fl = (style == 2) & (me[:, 0] < 0.45)
        tx = torch.where(fl, torch.full_like(tx, 0.5), tx)
        ty = torch.where(fl, side, ty)
        fl2 = (style == 2) & (me[:, 0] >= 0.45)
        tx = torch.where(fl2, enemy_flag[:, 0], tx)
        ty = torch.where(fl2, enemy_flag[:, 1], ty)
    c = carry[role]
    if style is not None:
        # 3 return-safe: carrying, through the side away from both opponents, then home
        oy = 0.5 * (theirs[0][:, 1] + theirs[1][:, 1])
        sy = torch.where(oy > 0, torch.full_like(oy, -0.55), torch.full_like(oy, 0.55))
        safe = (style == 3) & (me[:, 0] > -0.03)
        home_x = torch.where(safe, torch.full_like(home_x, -0.2), home_x)
        home_y = torch.where(safe, sy, home_y)
    tx = torch.where(c, home_x, tx).clamp(-HALF + 0.05, HALF - 0.05)
    ty = torch.where(c, home_y, ty).clamp(-HALF + 0.05, HALF - 0.05)
    return drive(me, tx, ty, p)


def defender_actions(mine, theirs, own_flag, flag_taken, hold, immune, p, style=None, enemy_flag=None, carry=None):
    me, att = mine[1], theirs[0]
    home = torch.tensor([-BASE_X, 0.0], device=me.device).expand_as(own_flag)
    # guard point
    u = att[:, :2] - home
    u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-3)
    g = home + p[:, 5:6] * u
    gx, gy = g[:, 0].clamp(max=-0.04), g[:, 1]
    # intruder
    free = (hold[2] <= 0) & (immune[2] <= 0)
    chase = free & (att[:, 0] < 0) & (_dist(att, home) < p[:, 6])
    tx = torch.where(chase, att[:, 0], gx)
    ty = torch.where(chase, att[:, 1], gy)
    # my flag taken: intercept it a little ahead (the carrier runs towards x = 0)
    inter = flag_taken & (own_flag[:, 0] < 0)
    tx = torch.where(inter, own_flag[:, 0] + 0.05, tx)
    ty = torch.where(inter, own_flag[:, 1], ty)
    if style is None:
        tx, ty = tx.clamp(-HALF + 0.05, -0.02), ty.clamp(-HALF + 0.05, HALF - 0.05)
        return drive(me, tx, ty, p, stop_r=0.015)
    # 1 on the flag
    on = (style == 1) & ~inter
    tx = torch.where(on, home[:, 0], tx)
    ty = torch.where(on, home[:, 1], ty)
    # 2 hunter: any active intruder of either kind in my half (the closest one)
    att2 = theirs[1]
    free2 = (hold[3] <= 0) & (immune[3] <= 0)
    in1, in2 = free & (att[:, 0] < 0), free2 & (att2[:, 0] < 0)
    d1, d2 = _dist(me, att), _dist(me, att2)
    pick2 = in2 & (~in1 | (d2 < d1))
    hx = torch.where(pick2, att2[:, 0], att[:, 0])
    hy = torch.where(pick2, att2[:, 1], att[:, 1])
    hunt = (style == 2) & ~inter & (in1 | in2)
    tx = torch.where(hunt, hx, tx)
    ty = torch.where(hunt, hy, ty)
    # 3 line keeper
    line = (style == 3) & ~inter
    tx = torch.where(line, torch.full_like(tx, -0.30), tx)
    ty = torch.where(line, (att[:, 1] * 0.8).clamp(-0.5, 0.5), ty)
    tx, ty = tx.clamp(-HALF + 0.05, -0.02), ty.clamp(-HALF + 0.05, HALF - 0.05)
    a = drive(me, tx, ty, p, stop_r=torch.where(on, 0.005, 0.015))
    # 4 second attacker
    sec = style == 4
    if sec.any() and enemy_flag is not None:
        b = attacker_actions(mine, theirs, own_flag, enemy_flag, carry, hold, p, style=None, role=1)
        a = torch.where(sec.unsqueeze(1), b, a)
    return a


def draw_gains(k, gen, device="cuda", strong=0.0):
    """Gains drawn uniformly in [GAIN_LO, GAIN_HI]; a fraction `strong` gets the reference gains (GAINS_REF)."""
    lo = torch.tensor(GAIN_LO, device=device)
    hi = torch.tensor(GAIN_HI, device=device)
    g = lo + (hi - lo) * torch.rand(k, len(GAIN_LO), generator=gen, device=device)
    if strong > 0:
        ref = torch.rand(k, generator=gen, device=device) < strong
        g[ref] = torch.tensor(GAINS_REF, device=device)
    return g


def draw_styles(k, weights, gen, device="cuda"):
    w = torch.tensor(weights, device=device, dtype=torch.float32)
    return torch.multinomial(w / w.sum(), k, replacement=True, generator=gen)
