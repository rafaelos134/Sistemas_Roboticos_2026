"""Observation of the soccer policies (attacker and defender), in torch (training) and numpy
(the ROS node). Everything is in the TEAM frame (the team attacks +x; for red the field
is rotated by 180 deg) and relative vectors are in the robot's own frame.

Input: mine (3 robots of my team, ordered by role: 0 keeper, 1, 2 attackers), theirs (3),
each robot (x, y, yaw, v, w); ball (x, y, vx, vy); role of the robot; its previous action.

  0-5   own x / 0.9, y / 0.6, cos yaw, sin yaw, v / V_MAX, w / W_MAX
  6-7   ball relative to me, own frame                    / 1 m
  8-9   ball x / 0.9, y / 0.6 (team frame)
  10-11 ball velocity in the own frame                    / 0.5 m/s
  12-13 their goal centre (0.9, 0) relative to me, own frame / 1.8 m
  14-15 our goal centre (-0.9, 0) relative to me, own frame  / 1.8 m
  16-19 the two teammates relative to me, own frame (in role order)  / 1 m
  20-25 the three opponents relative to me, own frame, nearest first  / 1 m
  26-27 previous action (lin, ang) in [-1, 1]
"""

import math

OBS_DIM = 28
HALF_L, HALF_W = 0.9, 0.6
V_MAX, W_MAX = 0.20, 4.0


def to_team_frame(x, is_ball=False):
    """Rotate states by 180 deg (red team). Robots (..., 5) or ball (n, 4)."""
    import torch

    y = x.clone()
    if is_ball:
        return -y
    y[..., 0] = -x[..., 0]
    y[..., 1] = -x[..., 1]
    y[..., 2] = torch.remainder(x[..., 2] + 2 * math.pi, 2 * math.pi) - math.pi
    return y


def observation_torch(mine, theirs, ball, role, prev):
    """mine, theirs: (3, n, 5); ball: (n, 4); role: 0/1/2; prev: (n, 2). Returns (n, 28)."""
    import torch

    me = mine[role]
    c, s = torch.cos(me[:, 2]), torch.sin(me[:, 2])

    def rel(px, py):
        dx, dy = px - me[:, 0], py - me[:, 1]
        return c * dx + s * dy, -s * dx + c * dy

    def rot(vx, vy):
        return c * vx + s * vy, -s * vx + c * vy

    bx, by = rel(ball[:, 0], ball[:, 1])
    bvx, bvy = rot(ball[:, 2], ball[:, 3])
    gx, gy = rel(torch.full_like(c, HALF_L), torch.zeros_like(c))
    ox, oy = rel(torch.full_like(c, -HALF_L), torch.zeros_like(c))
    mates = [k for k in range(3) if k != role]
    m = [rel(mine[k][:, 0], mine[k][:, 1]) for k in mates]
    op = torch.stack([torch.stack(rel(theirs[k][:, 0], theirs[k][:, 1]), 1) for k in range(3)], 1)  # (n, 3, 2)
    order = op.norm(dim=2).argsort(dim=1)
    op = torch.gather(op, 1, order.unsqueeze(2).expand(-1, -1, 2)).reshape(-1, 6)
    cols = [me[:, 0] / HALF_L, me[:, 1] / HALF_W, c, s, me[:, 3] / V_MAX, me[:, 4] / W_MAX,
            bx, by, ball[:, 0] / HALF_L, ball[:, 1] / HALF_W, bvx / 0.5, bvy / 0.5,
            gx / 1.8, gy / 1.8, ox / 1.8, oy / 1.8, m[0][0], m[0][1], m[1][0], m[1][1]]
    return torch.cat([torch.stack(cols, 1), op, prev], 1)


# >>> numpy-begin (this block is copied verbatim into the ROS node by export_soccer.py)
def team_frame_np(p, flip):
    """(x, y, yaw, v, w) of a robot in the team frame (flip=True for the red team)."""
    if not flip:
        return p
    yaw = (p[2] + 2.0 * math.pi) % (2.0 * math.pi) - math.pi
    return (-p[0], -p[1], yaw, p[3], p[4])


def observation_np(mine, theirs, ball, role, prev):
    """mine, theirs: 3 tuples (x, y, yaw, v, w) in the team frame (mine in role order);
    ball: (x, y, vx, vy) in the team frame; prev: (lin, ang). Returns a list of 28 floats."""
    me = mine[role]
    c, s = math.cos(me[2]), math.sin(me[2])

    def rel(px, py):
        dx, dy = px - me[0], py - me[1]
        return c * dx + s * dy, -s * dx + c * dy

    bx, by = rel(ball[0], ball[1])
    bvx, bvy = c * ball[2] + s * ball[3], -s * ball[2] + c * ball[3]
    gx, gy = rel(HALF_L, 0.0)
    ox, oy = rel(-HALF_L, 0.0)
    m = [rel(mine[k][0], mine[k][1]) for k in range(3) if k != role]
    op = sorted((rel(t[0], t[1]) for t in theirs), key=lambda q: q[0] * q[0] + q[1] * q[1])
    out = [me[0] / HALF_L, me[1] / HALF_W, c, s, me[3] / V_MAX, me[4] / W_MAX,
           bx, by, ball[0] / HALF_L, ball[1] / HALF_W, bvx / 0.5, bvy / 0.5,
           gx / 1.8, gy / 1.8, ox / 1.8, oy / 1.8, m[0][0], m[0][1], m[1][0], m[1][1]]
    for q in op:
        out += [q[0], q[1]]
    return out + [float(prev[0]), float(prev[1])]
# <<< numpy-end
