"""Privileged input of the ASYMMETRIC critic (soccer). Used only in training: the critic is not
exported to the ROS node, so it may see things the actor cannot (or that would be expensive
to compute at 20 Hz in the node). The actor keeps the 28 inputs of soccer_obs.py.

Critic input of one agent (team frame, without the observation noise):

  0-27   the actor observation of that agent (so the critic knows WHICH robot it evaluates)
  28-67  field view: the 6 robots (x/0.9, y/0.6, cos yaw, sin yaw, v/V, w/W), mine in role order then
         theirs in id order, + ball (x/0.9, y/0.6, vx/0.5, vy/0.5)
  68-69  ball still time: min(t_still / stall_s, 3) and the flag "jammed" (t_still > stall_s)
  70-74  (critic "asym_dij" only) geodesic distances computed with Dijkstra on a grid:
           me   -> approach point behind the ball (/1.8 m)
           mate -> approach point                 (/1.8 m)
           detour of me = geodesic - straight line (/0.5 m)
           ball -> their goal, around the opponents (/1.8 m)
           detour of the ball                       (/0.5 m)

Dijkstra on a grid, batched on the GPU: 4 cm cells (45 x 30), 16-connected, edge cost = length x
mean cell cost of its two ends (1 free, OBST_COST inside an obstacle: robots can be pushed, so an
obstacle is expensive but not forbidden). The search runs from the target to every cell at once
(one distance field per environment). Instead of a priority queue, every step relaxes all cells in
parallel (Bellman-Ford on the grid); after enough sweeps the result is the same shortest-path
distance Dijkstra returns, and it vectorizes over the 1024 environments.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from soccer_obs import HALF_L, HALF_W, OBS_DIM, V_MAX, W_MAX

FIELD_DIM, STILL_DIM, DIJ_DIM = 40, 2, 5
ROBOT_R, BALL_R = 0.037, 0.0214
BEHIND = 0.08  # approach point: this far behind the ball, on the line goal -> ball
OBST_COST = 15.0


def critic_dim(kind):
    return {"sym": None, "truth": OBS_DIM, "asym": OBS_DIM + FIELD_DIM + STILL_DIM,
            "asym_dij": OBS_DIM + FIELD_DIM + STILL_DIM + DIJ_DIM}[kind]


def field_view(mine, theirs, ball):
    cols = []
    for r in list(mine) + list(theirs):
        cols += [r[:, 0] / HALF_L, r[:, 1] / HALF_W, torch.cos(r[:, 2]), torch.sin(r[:, 2]), r[:, 3] / V_MAX,
                 r[:, 4] / W_MAX]
    cols += [ball[:, 0] / HALF_L, ball[:, 1] / HALF_W, ball[:, 2] / 0.5, ball[:, 3] / 0.5]
    return torch.stack(cols, 1)


def still_features(ball_still_steps, control_dt, stall_s):
    t = ball_still_steps.float() * control_dt / stall_s
    return torch.stack([t.clamp(max=3.0), (t > 1.0).float()], 1)


class GridDijkstra:
    """Shortest-path distance fields on the field, for n environments at once."""

    # 16-connected grid (8 neighbours + the 8 "knight" moves): the length of a straight segment is then
    # overestimated by at most ~2.7 % (8-connected: ~8 %)
    OFFS = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)
            if (dx or dy) and max(abs(dx), abs(dy)) <= 2 and math.gcd(abs(dx), abs(dy)) == 1]

    def __init__(self, n, res=0.04, sweeps=40, chunk=10, device="cuda"):
        self.res, self.n = res, n
        self.nx, self.ny = int(round(2 * HALF_L / res)), int(round(2 * HALF_W / res))
        self.sweeps, self.chunk = sweeps, chunk
        xs = -HALF_L + res * (torch.arange(self.nx, device=device) + 0.5)
        ys = -HALF_W + res * (torch.arange(self.ny, device=device) + 0.5)
        self.cy, self.cx = torch.meshgrid(ys, xs, indexing="ij")  # (ny, nx)
        self.w = [math.hypot(dx, dy) * res for dy, dx in self.OFFS]
        self._relax = torch.compile(self._relax_eager, dynamic=False)

    def _shift(self, x, dy, dx, fill):
        p = F.pad(x, (2, 2, 2, 2), value=fill)
        return p[:, 2 + dy:2 + dy + self.ny, 2 + dx:2 + dx + self.nx]

    def _relax_eager(self, d, cost):
        # edge cost from neighbour (dy, dx) to the cell = length x mean cost of the two cells (computed on
        # the fly: storing the 16 edge maps made every sweep memory bound)
        for _ in range(self.chunk):
            best = d
            for (dy, dx), w in zip(self.OFFS, self.w):
                best = torch.minimum(best, self._shift(d, dy, dx, 1e3) + (0.5 * w) * (cost + self._shift(cost, dy, dx, 1e3)))
            d = best
        return d

    def cell(self, p):
        """Index (iy, ix) of the cell that contains the points p (n, 2), clamped to the grid."""
        ix = ((p[:, 0] + HALF_L) / self.res).long().clamp(0, self.nx - 1)
        iy = ((p[:, 1] + HALF_W) / self.res).long().clamp(0, self.ny - 1)
        return iy, ix

    def cost_map(self, obstacles, radius):
        """obstacles: (k, n, 2). Cost (n, ny, nx): 1 outside, OBST_COST inside any disk."""
        n = obstacles.shape[1]
        occ = torch.zeros(n, self.ny, self.nx, dtype=torch.bool, device=obstacles.device)
        for o in obstacles:
            occ |= (self.cx - o[:, 0, None, None]) ** 2 + (self.cy - o[:, 1, None, None]) ** 2 < radius ** 2
        return 1.0 + (OBST_COST - 1.0) * occ.float()

    def distance_field(self, target, cost):
        """Distance from target (n, 2) to every cell centre (n, ny, nx)."""
        # start: straight-line distance in the 5 x 5 block of cells around the target, "infinite" elsewhere
        iy, ix = self.cell(target)
        near = ((torch.arange(self.ny, device=target.device)[None, :, None] - iy[:, None, None]).abs() <= 2) & \
               ((torch.arange(self.nx, device=target.device)[None, None, :] - ix[:, None, None]).abs() <= 2)
        eu = torch.hypot(self.cx - target[:, 0, None, None], self.cy - target[:, 1, None, None]) * cost
        d = torch.where(near, eu, torch.full_like(eu, 1e3))
        for _ in range(self.sweeps // self.chunk):
            d = self._relax(d.contiguous(), cost.contiguous())
        return d

    def lookup(self, field, p):
        """Bilinear interpolation of the field at the points p (n, 2)."""
        fx = ((p[:, 0] + HALF_L) / self.res - 0.5).clamp(0, self.nx - 1 - 1e-4)
        fy = ((p[:, 1] + HALF_W) / self.res - 0.5).clamp(0, self.ny - 1 - 1e-4)
        x0, y0 = fx.long(), fy.long()
        x1, y1 = (x0 + 1).clamp(max=self.nx - 1), (y0 + 1).clamp(max=self.ny - 1)
        ax, ay = fx - x0, fy - y0
        r = torch.arange(p.shape[0], device=p.device)
        return ((1 - ay) * ((1 - ax) * field[r, y0, x0] + ax * field[r, y0, x1])
                + ay * ((1 - ax) * field[r, y1, x0] + ax * field[r, y1, x1]))


def approach_point(ball):
    """Point BEHIND metres behind the ball on the line (their goal centre -> ball), inside the field."""
    ux, uy = HALF_L - ball[:, 0], -ball[:, 1]
    un = torch.hypot(ux, uy).clamp_min(1e-6)
    px = (ball[:, 0] - BEHIND * ux / un).clamp(-HALF_L + 0.01, HALF_L - 0.01)
    py = (ball[:, 1] - BEHIND * uy / un).clamp(-HALF_W + 0.01, HALF_W - 0.01)
    return torch.stack([px, py], 1)


def dijkstra_features(grid, mine, theirs, ball, roles):
    """mine, theirs: (3, n, 5) team frame; ball (n, 4). Returns {role: (n, 5)} for the attacker roles."""
    b = ball[:, :2]
    # 1) robots -> approach point: obstacles = ball (it must be circled, not pushed the wrong way), keeper and
    #    the 3 opponents. The two attackers are not obstacles (each one reads the field at its own cell).
    obst = torch.stack([b, mine[0][:, :2]] + [t[:, :2] for t in theirs])
    cost = grid.cost_map(obst[1:], 2 * ROBOT_R + 0.005)
    ball_disk = (grid.cx - b[:, 0, None, None]) ** 2 + (grid.cy - b[:, 1, None, None]) ** 2 < (ROBOT_R + BALL_R) ** 2
    cost = torch.where(ball_disk, torch.full_like(cost, OBST_COST), cost)
    target = approach_point(b)
    f_app = grid.distance_field(target, cost)
    # 2) ball -> their goal: obstacles = the 3 opponents
    goal = torch.stack([torch.full_like(b[:, 0], HALF_L - 0.01), torch.zeros_like(b[:, 0])], 1)
    f_goal = grid.distance_field(goal, grid.cost_map(torch.stack([t[:, :2] for t in theirs]), ROBOT_R + BALL_R + 0.005))
    g_ball = grid.lookup(f_goal, b)
    det_ball = g_ball - torch.hypot(goal[:, 0] - b[:, 0], goal[:, 1] - b[:, 1])
    geo = {r: grid.lookup(f_app, mine[r][:, :2]) for r in roles}
    eu = {r: torch.hypot(target[:, 0] - mine[r][:, 0], target[:, 1] - mine[r][:, 1]) for r in roles}
    out = {}
    for r in roles:
        mate = [q for q in roles if q != r]
        gm = geo[mate[0]] if mate else torch.zeros_like(geo[r])
        out[r] = torch.stack([geo[r] / 1.8, gm / 1.8, (geo[r] - eu[r]) / 0.5, g_ball / 1.8, det_ball / 0.5], 1)
    return out
