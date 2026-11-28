"""Observation of the capture-the-flag policies (attacker and defender), in torch (training) and
numpy (the ROS node). Everything is in the TEAM frame: the team's base is at (-0.62, 0) and it
attacks +x (for red the arena is rotated by 180 deg). Relative vectors are in the robot's own frame.

Robots are ordered by role: mine = (my attacker, my defender), theirs = (their attacker, their defender).
Each robot: (x, y, yaw, v, w).

  0-5   own x / 0.8, y / 0.8, cos yaw, sin yaw, v / V_MAX, w / W_MAX
  6-7   enemy flag (where it is now) relative to me                 / 1 m
  8-9   own flag (where it is now) relative to me                   / 1 m
  10-11 own base (-0.62, 0) relative to me                          / 1 m
  12-13 enemy base (+0.62, 0) relative to me                        / 1 m
  14-15 teammate relative to me                                     / 1 m
  16-17 their attacker relative to me                               / 1 m
  18-19 their defender relative to me                               / 1 m
  20-23 heading of their attacker, their defender relative to mine: cos, sin
  24    I carry the enemy flag;  25 my teammate carries it;  26 an opponent carries my flag
  27-30 remaining hold time / 4 s         (me, teammate, their attacker, their defender)
  31-34 remaining immunity / 1.5 s        (same order)
  35    round time / 90 s
  36-37 previous action (lin, ang) in [-1, 1]
"""

import math

OBS_DIM = 38
HALF = 0.8
BASE_X = 0.62
V_MAX, W_MAX = 0.20, 4.0
HOLD_TIME, IMMUNE_TIME, ROUND_TIME = 4.0, 1.5, 90.0
ROLES = ("attacker", "defender")


def to_team_frame(x):
    """Rotate robot states (..., 5) by 180 deg (red team)."""
    import torch

    y = x.clone()
    y[..., 0] = -x[..., 0]
    y[..., 1] = -x[..., 1]
    y[..., 2] = torch.remainder(x[..., 2] + 2 * math.pi, 2 * math.pi) - math.pi
    return y


def observation_torch(role, mine, theirs, own_flag, enemy_flag, carry, flag_taken, hold, immune, round_t, prev):
    """role: 0 attacker / 1 defender. mine, theirs: (2, n, 5); flags (n, 2); carry: (2, n) my robots carrying
    the enemy flag; flag_taken: (n,); hold, immune: (4, n) remaining seconds (me-team att, def, their att, def);
    round_t: (n,) seconds; prev: (n, 2). Returns (n, 38)."""
    import torch

    me = mine[role]
    mate = mine[1 - role]
    c, s = torch.cos(me[:, 2]), torch.sin(me[:, 2])

    def rel(px, py):
        dx, dy = px - me[:, 0], py - me[:, 1]
        return c * dx + s * dy, -s * dx + c * dy

    def head(o):
        d = o[:, 2] - me[:, 2]
        return torch.cos(d), torch.sin(d)

    ex, ey = rel(enemy_flag[:, 0], enemy_flag[:, 1])
    fx, fy = rel(own_flag[:, 0], own_flag[:, 1])
    hx, hy = rel(torch.full_like(c, -BASE_X), torch.zeros_like(c))
    bx, by = rel(torch.full_like(c, BASE_X), torch.zeros_like(c))
    mx, my = rel(mate[:, 0], mate[:, 1])
    ax, ay = rel(theirs[0][:, 0], theirs[0][:, 1])
    dx, dy = rel(theirs[1][:, 0], theirs[1][:, 1])
    ac, as_ = head(theirs[0])
    dc, ds = head(theirs[1])
    order = (role, 1 - role, 2, 3)
    cols = [me[:, 0] / HALF, me[:, 1] / HALF, c, s, me[:, 3] / V_MAX, me[:, 4] / W_MAX,
            ex, ey, fx, fy, hx, hy, bx, by, mx, my, ax, ay, dx, dy, ac, as_, dc, ds,
            carry[role].float(), carry[1 - role].float(), flag_taken.float()]
    cols += [hold[k] / HOLD_TIME for k in order]
    cols += [immune[k] / IMMUNE_TIME for k in order]
    cols += [round_t / ROUND_TIME]
    return torch.cat([torch.stack(cols, 1), prev], 1)


# >>> numpy-begin (this block is copied verbatim into the ROS node by export_ctf.py)
def team_frame_np(p, flip):
    """(x, y, yaw, v, w) of a robot in the team frame (flip=True for the red team)."""
    if not flip:
        return p
    yaw = (p[2] + 2.0 * math.pi) % (2.0 * math.pi) - math.pi
    return (-p[0], -p[1], yaw, p[3], p[4])


def observation_np(role, mine, theirs, own_flag, enemy_flag, carry, flag_taken, hold, immune, round_t, prev):
    """Same as observation_torch for one robot. mine, theirs: 2 tuples (x, y, yaw, v, w) in the team frame,
    in role order; flags (x, y) in the team frame; carry: 2 bools (my attacker, my defender carry the enemy
    flag); hold, immune: 4 remaining seconds (my att, my def, their att, their def). Returns 38 floats."""
    me, mate = mine[role], mine[1 - role]
    c, s = math.cos(me[2]), math.sin(me[2])

    def rel(px, py):
        dx, dy = px - me[0], py - me[1]
        return c * dx + s * dy, -s * dx + c * dy

    out = [me[0] / HALF, me[1] / HALF, c, s, me[3] / V_MAX, me[4] / W_MAX]
    for p in (enemy_flag, own_flag, (-BASE_X, 0.0), (BASE_X, 0.0), mate, theirs[0], theirs[1]):
        out += list(rel(p[0], p[1]))
    for o in theirs:
        out += [math.cos(o[2] - me[2]), math.sin(o[2] - me[2])]
    out += [float(bool(carry[role])), float(bool(carry[1 - role])), float(bool(flag_taken))]
    order = (role, 1 - role, 2, 3)
    out += [hold[k] / HOLD_TIME for k in order]
    out += [immune[k] / IMMUNE_TIME for k in order]
    out += [round_t / ROUND_TIME, float(prev[0]), float(prev[1])]
    return out
# <<< numpy-end
