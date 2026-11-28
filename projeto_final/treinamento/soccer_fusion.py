"""Sensor noise model + state estimation (odometry x lidar fusion) for the soccer policies.

The next version of the kit will add noise to the sensors, with levels not yet published. So:

* SensorNoise (training only, torch): per episode a noise LEVEL s is drawn (s = 0 in 20 % of the
  episodes, else U(0, s_max)) and every sensor gets s x the nominal values below:
    - odometry pose  = true pose + drift (random walk, per robot) + white noise
    - odometry twist = true (v, w) + white noise
    - lidar          = 8 beams (same model as psr_sim: walls, the other robots, the ball) + white
                       noise, with some beams dropped (reported as the max range)
    - ball           = true position / velocity + white noise
  Domain randomization over the level is the hedge against not knowing it.

* Fusion (numpy reference = what runs in the ROS node; torch batched copy for training):
    - our 3 robots: EKF per robot, state (x, y, yaw, bx, by, byaw) = pose + odometry bias.
        predict: odometry twist (v, w);  update 1: odometry pose (z = pose + bias);
        update 2: the 8 lidar beams against the KNOWN field walls (ray cast in the map; beams that
        hit a robot or the ball are shorter than the wall -> rejected by a chi gate; beams with no
        wall within range are not used). Near a wall the lidar makes the bias observable, so the
        drift of the odometry is corrected; in the middle of the field the EKF follows the odometry.
        A jump > 0.25 m of the odometry (kickoff after a goal) re-initialises the pose.
    - ball: constant-velocity Kalman filter (position + velocity measurements, both axes share
      the covariance); a jump > 0.10 m (stuck-ball teleport, kickoff) re-initialises it.
    - opponents: exponential smoothing of the position (we only have their odometry).
"""

from __future__ import annotations

import math

import numpy as np

# >>> numpy-begin (this block is copied verbatim into the ROS node by export_soccer.py)
FUS_RMIN, FUS_RMAX = 0.037, 0.50
FUS_BEAMS = 8
FUS_ANGLES = [-math.pi + k * 2.0 * math.pi / FUS_BEAMS for k in range(FUS_BEAMS)]
# nominal noise the filter is tuned for (noise level s = 1)
FUS = {"sig_p": 0.01, "sig_yaw": 0.03, "drift_p": 0.01, "drift_yaw": 0.02, "sig_v": 0.01, "sig_w": 0.10,
       "sig_r": 0.01, "sig_ball": 0.005, "sig_ball_v": 0.03,
       "slip_v": 0.02, "slip_w": 0.1}  # unmodelled motion (pushes, collisions) in the prediction
FUS_JUMP, FUS_BALL_JUMP, FUS_OPP_JUMP = 0.25, 0.10, 0.20


def field_segments():
    """Wall segments (x0, y0, x1, y1) of the soccer field, same boxes as psr_sim/scenarios/soccer.py."""
    L, W, G, D, t = 1.8, 1.2, 0.40, 0.10, 0.02
    side = (W - G) / 2
    boxes = [(0.0, W / 2 + t / 2, L + 2 * t, t), (0.0, -W / 2 - t / 2, L + 2 * t, t)]
    for sx in (-1, 1):
        xw = sx * (L / 2 + t / 2)
        for sy in (-1, 1):
            boxes.append((xw, sy * (G / 2 + side / 2), t, side))
        boxes.append((sx * (L / 2 + D + t / 2), 0.0, t, G + 2 * t))
        for sy in (-1, 1):
            boxes.append((sx * (L / 2 + D / 2), sy * (G / 2 + t / 2), D, t))
    segs = []
    for x, y, w, h in boxes:
        x0, x1, y0, y1 = x - w / 2, x + w / 2, y - h / 2, y + h / 2
        segs += [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]
    return np.array(segs, dtype=np.float64)


FUS_SEGS = field_segments()


def wall_rays(px, py, angles):
    """Range to the nearest wall along each world angle, and the unit normal of that wall.
    Returns r (B,) (inf: no wall), nx (B,), ny (B,)."""
    a = FUS_SEGS[:, :2]
    e = FUS_SEGS[:, 2:] - a
    dx, dy = np.cos(angles), np.sin(angles)
    wx, wy = a[:, 0] - px, a[:, 1] - py
    den = dx[:, None] * e[None, :, 1] - dy[:, None] * e[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (wx[None, :] * e[None, :, 1] - wy[None, :] * e[None, :, 0]) / den
        u = (wx[None, :] * dy[:, None] - wy[None, :] * dx[:, None]) / den
    ok = (np.abs(den) > 1e-12) & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
    t = np.where(ok, t, np.inf)
    k = t.argmin(axis=1)
    r = t[np.arange(len(angles)), k]
    el = np.hypot(e[k, 0], e[k, 1])
    return r, -e[k, 1] / el, e[k, 0] / el


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


FUS_LV_MIN, FUS_LV_MAX, FUS_LV_FLOOR = 0.3, 3.0, 0.0  # assumed noise level: limits; residual variance at level 0


def _scaled(p, a):
    """Noise parameters for the noise level a (the nominal values are level 1)."""
    q = dict(p)
    for k in ("sig_p", "sig_yaw", "drift_p", "drift_yaw", "sig_v", "sig_w", "sig_ball", "sig_ball_v"):
        q[k] = p[k] * a
    return q


class LevelEstimator:
    """Online estimate of the noise level of the odometry (one per team, shared by the filters).

    The yaw residual e = z_yaw(k) - z_yaw(k-1) - w_odom * dt is not affected by pushes (the twist is
    measured), so its variance is ~ s^2 * (2 sig_yaw^2 + (sig_w dt)^2 + drift_yaw^2 dt): an
    exponential average of e^2 gives the level s. The filters then use max(s, FUS_LV_MIN)."""

    BETA = 0.02  # ~1 s time constant at 20 Hz

    def __init__(self, n=3):
        self.prev = [None] * n
        self.v = None

    def update(self, zs, ws, dt):
        c = FUS["drift_yaw"] ** 2 * dt + 2 * FUS["sig_yaw"] ** 2 + (FUS["sig_w"] * dt) ** 2
        if self.v is None:
            self.v = c  # start assuming the nominal level
        es = []
        for i, (z, w) in enumerate(zip(zs, ws)):
            pv = self.prev[i]
            self.prev[i] = (z[0], z[1], z[2])
            if pv is None or math.hypot(z[0] - pv[0], z[1] - pv[1]) > FUS_JUMP:
                continue
            e = _wrap(z[2] - pv[2] - w * dt)
            es.append(min(e * e, 0.04))
        if es:
            self.v += self.BETA * (sum(es) / len(es) - self.v)
        return min(max(math.sqrt(max(self.v - FUS_LV_FLOOR, 0.0) / c), FUS_LV_MIN), FUS_LV_MAX)


class PoseEKF:
    """EKF of one robot: state (x, y, yaw, bias_x, bias_y, bias_yaw), see the module docstring."""

    def __init__(self, prm=None):
        self.p = dict(FUS if prm is None else prm)
        self.m, self.P = None, None

    def reset(self, z, a=1.0):
        p = _scaled(self.p, a)
        self.m = np.array([z[0], z[1], z[2], 0.0, 0.0, 0.0])
        self.P = np.diag([p["sig_p"] ** 2 + 1e-4, p["sig_p"] ** 2 + 1e-4, p["sig_yaw"] ** 2 + 1e-3,
                          (2 * p["drift_p"]) ** 2 + 1e-6, (2 * p["drift_p"]) ** 2 + 1e-6, (2 * p["drift_yaw"]) ** 2 + 1e-6])

    def step(self, z, v, w, ranges, dt, a=1.0):
        """z = odometry pose (x, y, yaw), (v, w) = odometry twist, ranges = 8 lidar ranges (beam k at
        FUS_ANGLES[k] from the heading), a = assumed noise level. Returns the estimated (x, y, yaw)."""
        p = _scaled(self.p, a)
        if self.m is None:
            self.reset(z, a)
        else:
            m, P = self.m, self.P
            c, s = math.cos(m[2]), math.sin(m[2])
            m[0] += v * c * dt
            m[1] += v * s * dt
            m[2] = _wrap(m[2] + w * dt)
            F = np.eye(6)
            F[0, 2], F[1, 2] = -v * s * dt, v * c * dt
            q = [(p["sig_v"] * dt) ** 2 + (p["slip_v"] * dt) ** 2] * 2 + [(p["sig_w"] * dt) ** 2 + (p["slip_w"] * dt) ** 2] + \
                [p["drift_p"] ** 2 * dt + 1e-9] * 2 + [p["drift_yaw"] ** 2 * dt + 1e-9]
            P = F.dot(P).dot(F.T) + np.diag(q)
            y = np.array([z[0] - m[0] - m[3], z[1] - m[1] - m[4], _wrap(z[2] - m[2] - m[5])])
            if math.hypot(y[0], y[1]) > FUS_JUMP:  # teleported (kickoff): keep the bias, restart the pose
                m[0], m[1], m[2] = z[0] - m[3], z[1] - m[4], _wrap(z[2] - m[5])
                P[:3, :] = 0.0
                P[:, :3] = 0.0
                P[0, 0] = P[1, 1] = p["sig_p"] ** 2 + 1e-4
                P[2, 2] = p["sig_yaw"] ** 2 + 1e-3
            else:
                pht = P[:, :3] + P[:, 3:]  # P H^T, H = [I I]
                S = pht[:3] + pht[3:] + np.diag([p["sig_p"] ** 2 + 1e-6, p["sig_p"] ** 2 + 1e-6, p["sig_yaw"] ** 2 + 1e-6])
                K = np.linalg.solve(S, pht.T).T
                m += K.dot(y)
                P = P - K.dot(pht.T)
            m[2] = _wrap(m[2])
            self.m, self.P = m, 0.5 * (P + P.T)
        self._lidar(ranges)
        return float(self.m[0]), float(self.m[1]), float(self.m[2])

    def _lidar(self, ranges):
        m, P = self.m, self.P
        ang = m[2] + np.array(FUS_ANGLES)
        rh, nx, ny = wall_rays(m[0], m[1], ang)
        dx, dy = np.cos(ang), np.sin(ang)
        nu = nx * dx + ny * dy
        nup = -nx * dy + ny * dx
        rz = np.asarray(ranges, dtype=np.float64)
        sr2 = self.p["sig_r"] ** 2 + 0.005 ** 2
        okg = (rh < FUS_RMAX - 0.01) & (rz < FUS_RMAX - 0.005) & (np.abs(nu) > 0.25)
        nu_s = np.where(okg, nu, 1.0)
        rh_s = np.where(okg, rh, 0.0)
        H = np.zeros((FUS_BEAMS, 6))
        H[:, 0] = -nx / nu_s
        H[:, 1] = -ny / nu_s
        H[:, 2] = -rh_s * nup / nu_s
        y = rz - rh_s
        sk = np.einsum("ki,ij,kj->k", H, P, H) + sr2
        ok = okg & (np.abs(y) < 3.0 * np.sqrt(sk) + 0.01)
        H[~ok] = 0.0
        y = np.where(ok, y, 0.0)
        pht = P.dot(H.T)
        S = H.dot(pht) + np.diag(np.where(ok, sr2, 1.0))
        K = np.linalg.solve(S, pht.T).T
        m = m + K.dot(y)
        m[2] = _wrap(m[2])
        P = P - K.dot(pht.T)
        self.m, self.P = m, 0.5 * (P + P.T)


class BallKF:
    """Constant-velocity Kalman filter of the ball (both axes share the 2 x 2 covariance)."""

    SIG_A = 2.0  # m/s^2, process noise (kicks)

    def __init__(self, prm=None):
        self.p = dict(FUS if prm is None else prm)
        self.x, self.P = None, None  # x: (2 axes, [pos, vel])

    def step(self, z, zv, dt, a=1.0):
        p = _scaled(self.p, a)
        if self.x is not None:
            x = self.x.copy()
            x[:, 0] += x[:, 1] * dt
            if math.hypot(z[0] - x[0, 0], z[1] - x[1, 0]) > FUS_BALL_JUMP:
                self.x = None
        if self.x is None:
            self.x = np.array([[z[0], zv[0]], [z[1], zv[1]]], dtype=np.float64)
            self.P = np.diag([p["sig_ball"] ** 2 + 1e-5, p["sig_ball_v"] ** 2 + 1e-3])
            return float(z[0]), float(z[1]), float(zv[0]), float(zv[1])
        F = np.array([[1.0, dt], [0.0, 1.0]])
        a2 = self.SIG_A ** 2
        Q = a2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        P = F.dot(self.P).dot(F.T) + Q
        R = np.diag([p["sig_ball"] ** 2 + 1e-6, p["sig_ball_v"] ** 2 + 1e-4])
        K = np.linalg.solve(P + R, P).T  # P (P + R)^-1, P symmetric
        zz = np.array([[z[0], zv[0]], [z[1], zv[1]]])
        x = x + (zz - x).dot(K.T)
        self.P = P - K.dot(P)
        self.P = 0.5 * (self.P + self.P.T)
        self.x = x
        return float(x[0, 0]), float(x[1, 0]), float(x[0, 1]), float(x[1, 1])


def opp_alpha(a):
    """Smoothing gain of the opponent positions for the noise level a."""
    return 1.0 / (1.0 + 0.7 * a)


class Smooth2D:
    """Exponential smoothing of an opponent position, restarted on jumps (kickoff)."""

    def __init__(self):
        self.q = None

    def step(self, z, a=1.0):
        if self.q is None or math.hypot(z[0] - self.q[0], z[1] - self.q[1]) > FUS_OPP_JUMP:
            self.q = (z[0], z[1])
        else:
            g = opp_alpha(a)
            self.q = (self.q[0] + g * (z[0] - self.q[0]), self.q[1] + g * (z[1] - self.q[1]))
        return self.q
# <<< numpy-end


# ======================================================================================================
# torch: noise model and the batched copy of the filters (training)
# ======================================================================================================
import torch  # noqa: E402

SEGS_T = None


def _segs():
    global SEGS_T
    if SEGS_T is None:
        SEGS_T = torch.tensor(FUS_SEGS, dtype=torch.float32, device="cuda")
    return SEGS_T


def wrap_t(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def ray_cast_t(px, py, ang, circles=None, with_normal=False):
    """px, py: (B,), ang: (B, K) world angles. circles: (B, C, 3) or None.
    Returns r (B, K) (inf when nothing is hit) [and nx, ny of the wall hit]."""
    S = _segs()
    a, e = S[:, :2], S[:, 2:] - S[:, :2]
    dx, dy = torch.cos(ang), torch.sin(ang)  # (B, K)
    wx = a[:, 0].view(1, 1, -1) - px.view(-1, 1, 1)  # (B, 1, S)
    wy = a[:, 1].view(1, 1, -1) - py.view(-1, 1, 1)
    ex, ey = e[:, 0].view(1, 1, -1), e[:, 1].view(1, 1, -1)
    den = dx.unsqueeze(2) * ey - dy.unsqueeze(2) * ex  # (B, K, S)
    sden = torch.where(den.abs() > 1e-12, den, torch.ones_like(den))
    t = (wx * ey - wy * ex) / sden
    u = (wx * dy.unsqueeze(2) - wy * dx.unsqueeze(2)) / sden
    ok = (den.abs() > 1e-12) & (t >= 0) & (u >= 0) & (u <= 1)
    t = torch.where(ok, t, torch.full_like(t, float("inf")))
    r, k = t.min(2)
    out = r
    if circles is not None:
        fx = px.view(-1, 1, 1) - circles[:, :, 0].unsqueeze(1)  # (B, 1, C)
        fy = py.view(-1, 1, 1) - circles[:, :, 1].unsqueeze(1)
        b = dx.unsqueeze(2) * fx + dy.unsqueeze(2) * fy  # (B, K, C)
        c = fx ** 2 + fy ** 2 - circles[:, :, 2].unsqueeze(1) ** 2
        disc = b * b - c
        tc = -b - torch.sqrt(disc.clamp_min(0))
        okc = (disc >= 0) & (tc >= 0)
        tc = torch.where(okc, tc, torch.full_like(tc, float("inf")))
        out = torch.minimum(r, tc.amin(2))
    if not with_normal:
        return out
    el = torch.hypot(e[:, 0], e[:, 1])
    nx, ny = -e[k, 1] / el[k], e[k, 0] / el[k]
    return out, nx, ny


ANG_T = None


def _angles():
    global ANG_T
    if ANG_T is None:
        ANG_T = torch.tensor(FUS_ANGLES, dtype=torch.float32, device="cuda")
    return ANG_T


def _sc(p, k, a):
    """p[k] scaled by the level a (b,) -> (b,) tensor."""
    return p[k] * a


class LevelEstimatorTorch:
    """Batched copy of LevelEstimator: one per env (the 3 robots of the blue team)."""

    def __init__(self, n):
        self.n = n
        self.prev = torch.zeros(3, n, 3, device="cuda")
        self.has = torch.zeros(3, n, dtype=torch.bool, device="cuda")
        self.v = torch.zeros(n, device="cuda")
        self.init = torch.zeros(n, dtype=torch.bool, device="cuda")

    def forget(self, idx):
        self.has[:, idx] = False
        self.init[idx] = False

    def update(self, z, w, dt, idx):
        """z (3, b, 3) odometry poses, w (3, b) of the envs idx -> level (b,)."""
        c = FUS["drift_yaw"] ** 2 * dt + 2 * FUS["sig_yaw"] ** 2 + (FUS["sig_w"] * dt) ** 2
        v = torch.where(self.init[idx], self.v[idx], torch.full_like(self.v[idx], c))
        pv, has = self.prev[:, idx], self.has[:, idx]
        ok = has & (torch.hypot(z[..., 0] - pv[..., 0], z[..., 1] - pv[..., 1]) <= FUS_JUMP)
        e = wrap_t(z[..., 2] - pv[..., 2] - w * dt)
        e2 = torch.where(ok, (e * e).clamp(max=0.04), torch.zeros_like(e))
        cnt = ok.float().sum(0)
        v = torch.where(cnt > 0, v + LevelEstimator.BETA * (e2.sum(0) / cnt.clamp_min(1) - v), v)
        self.v[idx], self.init[idx] = v, True
        self.prev[:, idx], self.has[:, idx] = z, True
        return torch.sqrt((v - FUS_LV_FLOOR).clamp_min(0) / c).clamp(FUS_LV_MIN, FUS_LV_MAX)


class PoseEKFTorch:
    """Batched copy of PoseEKF (B robots). step(..., rows) updates only those rows."""

    def __init__(self, B, prm=None):
        self.p = dict(FUS if prm is None else prm)
        self.B = B
        self.m = torch.zeros(B, 6, device="cuda")
        self.P = torch.zeros(B, 6, 6, device="cuda")
        self.init = torch.zeros(B, dtype=torch.bool, device="cuda")  # False: initialise on the next step

    def forget(self, rows):
        self.init[rows] = False

    def _p0(self, a):
        p = self.p
        sp, sy, dp, dy = (_sc(p, k, a) for k in ("sig_p", "sig_yaw", "drift_p", "drift_yaw"))
        return torch.diag_embed(torch.stack([sp ** 2 + 1e-4, sp ** 2 + 1e-4, sy ** 2 + 1e-3, (2 * dp) ** 2 + 1e-6,
                                             (2 * dp) ** 2 + 1e-6, (2 * dy) ** 2 + 1e-6], 1))

    def step(self, z, v, w, ranges, dt, rows=None, a=None):
        """z (b, 3), v, w (b,), ranges (b, 8), a (b,) noise level, for `rows` (all if None).
        Returns the estimated pose (b, 3)."""
        p = self.p
        if rows is None:
            rows = torch.arange(self.B, device="cuda")
        b = rows.numel()
        if a is None:
            a = torch.ones(b, device="cuda")
        sp, sy, dp, dy, sv, sw = (_sc(p, k, a) for k in ("sig_p", "sig_yaw", "drift_p", "drift_yaw", "sig_v", "sig_w"))
        m, P, init = self.m[rows].clone(), self.P[rows].clone(), self.init[rows]
        c, s = torch.cos(m[:, 2]), torch.sin(m[:, 2])
        m[:, 0] += v * c * dt
        m[:, 1] += v * s * dt
        m[:, 2] = wrap_t(m[:, 2] + w * dt)
        F = torch.eye(6, device="cuda").expand(b, 6, 6).clone()
        F[:, 0, 2], F[:, 1, 2] = -v * s * dt, v * c * dt
        qxy = (sv * dt) ** 2 + (p["slip_v"] * dt) ** 2
        q = torch.stack([qxy, qxy, (sw * dt) ** 2 + (p["slip_w"] * dt) ** 2, dp ** 2 * dt + 1e-9, dp ** 2 * dt + 1e-9,
                         dy ** 2 * dt + 1e-9], 1)
        P = F @ P @ F.transpose(1, 2) + torch.diag_embed(q)
        y = torch.stack([z[:, 0] - m[:, 0] - m[:, 3], z[:, 1] - m[:, 1] - m[:, 4], wrap_t(z[:, 2] - m[:, 2] - m[:, 5])], 1)
        jump = torch.hypot(y[:, 0], y[:, 1]) > FUS_JUMP
        # normal odometry update (H = [I I])
        pht = P[:, :, :3] + P[:, :, 3:]  # (b, 6, 3)
        R = torch.diag_embed(torch.stack([sp ** 2 + 1e-6, sp ** 2 + 1e-6, sy ** 2 + 1e-6], 1))
        S = pht[:, :3] + pht[:, 3:] + R
        K = torch.linalg.solve(S, pht.transpose(1, 2)).transpose(1, 2)  # (b, 6, 3)
        mu = m + (K @ y.unsqueeze(2)).squeeze(2)
        Pu = P - K @ pht.transpose(1, 2)
        # jump: restart the pose, keep the bias
        mj = m.clone()
        mj[:, 0], mj[:, 1], mj[:, 2] = z[:, 0] - m[:, 3], z[:, 1] - m[:, 4], wrap_t(z[:, 2] - m[:, 5])
        Pj = P.clone()
        Pj[:, :3, :] = 0
        Pj[:, :, :3] = 0
        Pj[:, 0, 0] = sp ** 2 + 1e-4
        Pj[:, 1, 1] = sp ** 2 + 1e-4
        Pj[:, 2, 2] = sy ** 2 + 1e-3
        j = jump.view(-1, 1)
        m = torch.where(j, mj, mu)
        P = torch.where(j.unsqueeze(2), Pj, Pu)
        m[:, 2] = wrap_t(m[:, 2])
        P = 0.5 * (P + P.transpose(1, 2))
        # first step after a reset: initialise from the odometry
        new = ~init
        m0 = torch.cat([z, torch.zeros_like(z)], 1)
        m = torch.where(new.view(-1, 1), m0, m)
        P = torch.where(new.view(-1, 1, 1), self._p0(a), P)
        m, P = self._lidar(m, P, ranges)
        self.m[rows], self.P[rows], self.init[rows] = m, P, True
        return m[:, :3]

    def _lidar(self, m, P, ranges):
        b = m.shape[0]
        ang = m[:, 2:3] + _angles().view(1, -1)  # (b, 8)
        rh, nx, ny = ray_cast_t(m[:, 0], m[:, 1], ang, with_normal=True)
        dx, dy = torch.cos(ang), torch.sin(ang)
        nu = nx * dx + ny * dy
        nup = -nx * dy + ny * dx
        sr2 = self.p["sig_r"] ** 2 + 0.005 ** 2
        okg = (rh < FUS_RMAX - 0.01) & (ranges < FUS_RMAX - 0.005) & (nu.abs() > 0.25)
        nu_s = torch.where(okg, nu, torch.ones_like(nu))
        rh_s = torch.where(okg, rh, torch.zeros_like(rh))
        H = torch.zeros(b, FUS_BEAMS, 6, device="cuda")
        H[:, :, 0] = -nx / nu_s
        H[:, :, 1] = -ny / nu_s
        H[:, :, 2] = -rh_s * nup / nu_s
        y = ranges - rh_s
        sk = torch.einsum("bki,bij,bkj->bk", H, P, H) + sr2
        ok = okg & (y.abs() < 3.0 * torch.sqrt(sk) + 0.01)
        H = H * ok.unsqueeze(2)
        y = torch.where(ok, y, torch.zeros_like(y))
        pht = P @ H.transpose(1, 2)  # (b, 6, 8)
        Rd = torch.where(ok, torch.full_like(y, sr2), torch.ones_like(y))
        S = H @ pht + torch.diag_embed(Rd)
        K = torch.linalg.solve(S, pht.transpose(1, 2)).transpose(1, 2)  # (b, 6, 8)
        m = m + (K @ y.unsqueeze(2)).squeeze(2)
        m[:, 2] = wrap_t(m[:, 2])
        P = P - K @ pht.transpose(1, 2)
        return m, 0.5 * (P + P.transpose(1, 2))


class BallKFTorch:
    def __init__(self, n, prm=None):
        self.p = dict(FUS if prm is None else prm)
        self.n = n
        self.x = torch.zeros(n, 2, 2, device="cuda")  # (n, axis, [pos, vel])
        self.P = torch.zeros(n, 2, 2, device="cuda")
        self.init = torch.zeros(n, dtype=torch.bool, device="cuda")

    def forget(self, rows):
        self.init[rows] = False

    def step(self, z, zv, dt, rows=None, a=None):
        """z, zv (b, 2), a (b,). Returns (b, 4): x, y, vx, vy."""
        p = self.p
        if rows is None:
            rows = torch.arange(self.n, device="cuda")
        b = rows.numel()
        if a is None:
            a = torch.ones(b, device="cuda")
        sb, sbv = _sc(p, "sig_ball", a), _sc(p, "sig_ball_v", a)
        x = self.x[rows].clone()
        x[:, :, 0] += x[:, :, 1] * dt
        jump = torch.hypot(z[:, 0] - x[:, 0, 0], z[:, 1] - x[:, 1, 0]) > FUS_BALL_JUMP
        new = ~self.init[rows] | jump
        F = torch.tensor([[1.0, dt], [0.0, 1.0]], device="cuda")
        a2 = BallKF.SIG_A ** 2
        Q = a2 * torch.tensor([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]], device="cuda")
        P = F @ self.P[rows] @ F.T + Q
        R = torch.diag_embed(torch.stack([sb ** 2 + 1e-6, sbv ** 2 + 1e-4], 1))
        K = torch.linalg.solve(P + R, P).transpose(1, 2)
        zz = torch.stack([z, zv], 2)  # (b, axis, 2)
        xu = x + (zz - x) @ K.transpose(1, 2)
        Pu = P - K @ P
        Pu = 0.5 * (Pu + Pu.transpose(1, 2))
        P0 = torch.diag_embed(torch.stack([sb ** 2 + 1e-5, sbv ** 2 + 1e-3], 1))
        xn = torch.where(new.view(-1, 1, 1), zz, xu)
        self.x[rows] = xn
        self.P[rows] = torch.where(new.view(-1, 1, 1), P0, Pu)
        self.init[rows] = True
        return torch.stack([xn[:, 0, 0], xn[:, 1, 0], xn[:, 0, 1], xn[:, 1, 1]], 1)


class SmoothTorch:
    def __init__(self, B):
        self.q = torch.zeros(B, 2, device="cuda")
        self.init = torch.zeros(B, dtype=torch.bool, device="cuda")

    def forget(self, rows):
        self.init[rows] = False

    def step(self, z, rows=None, a=None):
        if rows is None:
            rows = torch.arange(self.q.shape[0], device="cuda")
        if a is None:
            a = torch.ones(rows.numel(), device="cuda")
        q = self.q[rows]
        jump = torch.hypot(z[:, 0] - q[:, 0], z[:, 1] - q[:, 1]) > FUS_OPP_JUMP
        new = ~self.init[rows] | jump
        g = (1.0 / (1.0 + 0.7 * a)).view(-1, 1)
        q = torch.where(new.view(-1, 1), z, q + g * (z - q))
        self.q[rows] = q
        self.init[rows] = True
        return q


class SensorNoise:
    """Per-episode noise level and the noisy sensors of the 6 robots and the ball (training only)."""

    def __init__(self, n, gen, s_max=2.0, p_clean=0.2, drop=0.02, fixed_level=None):
        self.n, self.gen = n, gen
        self.s_max, self.p_clean, self.drop, self.fixed = s_max, p_clean, drop, fixed_level
        self.level = torch.zeros(n, device="cuda")
        self.bias = torch.zeros(6, n, 3, device="cuda")

    def _randn(self, *shape):
        return torch.randn(*shape, device="cuda", generator=self.gen)

    def reset(self, idx):
        k = idx.numel()
        if self.fixed is not None:
            lv = torch.full((k,), float(self.fixed), device="cuda")
        else:
            lv = self.s_max * torch.rand(k, device="cuda", generator=self.gen)
            lv = torch.where(torch.rand(k, device="cuda", generator=self.gen) < self.p_clean, torch.zeros_like(lv), lv)
        self.level[idx] = lv
        self.bias[:, idx] = 0.0

    def measure(self, rob, ball, dt, ball_r, idx, robot_r=0.037):
        """rob (6, b, 5) true (x, y, yaw, v, w) and ball (b, 4) of the envs `idx`. Advances their odometry drift.
        Returns odom pose (6, b, 3), twist (6, b, 2), lidar of the blue robots (3, b, 8), ball (b, 4)."""
        b = idx.numel()
        lvl = self.level[idx]
        s = lvl.view(1, -1)
        N = FUS
        bias = self.bias[:, idx]
        bias[..., :2] += N["drift_p"] * s.unsqueeze(2) * math.sqrt(dt) * self._randn(6, b, 2)
        bias[..., 2] += N["drift_yaw"] * s * math.sqrt(dt) * self._randn(6, b)
        self.bias[:, idx] = bias
        pose = rob[..., :3] + bias
        pose[..., :2] += N["sig_p"] * s.unsqueeze(2) * self._randn(6, b, 2)
        pose[..., 2] = wrap_t(pose[..., 2] + N["sig_yaw"] * s * self._randn(6, b))
        tw = rob[..., 3:5].clone()
        tw[..., 0] += N["sig_v"] * s * self._randn(6, b)
        tw[..., 1] += N["sig_w"] * s * self._randn(6, b)
        # lidar of our 3 robots: walls + the other 5 robots + the ball (true geometry, like psr_sim)
        scans = []
        for i in range(3):
            others = [j for j in range(6) if j != i]
            circ = torch.stack([torch.stack([rob[j, :, 0], rob[j, :, 1], torch.full_like(rob[j, :, 0], robot_r)], 1)
                                for j in others] + [torch.stack([ball[:, 0], ball[:, 1], torch.full_like(ball[:, 0], ball_r)], 1)], 1)
            ang = rob[i, :, 2:3] + _angles().view(1, -1)
            r = ray_cast_t(rob[i, :, 0], rob[i, :, 1], ang, circ).clamp(FUS_RMIN, FUS_RMAX)
            r = r + N["sig_r"] * lvl.view(-1, 1) * self._randn(b, FUS_BEAMS)
            dropped = torch.rand(b, FUS_BEAMS, device="cuda", generator=self.gen) < self.drop * (lvl.view(-1, 1) > 0).float()
            r = torch.where(dropped, torch.full_like(r, FUS_RMAX), r).clamp(FUS_RMIN, FUS_RMAX)
            scans.append(r)
        bm = ball.clone()
        bm[:, :2] += N["sig_ball"] * lvl.view(-1, 1) * self._randn(b, 2)
        bm[:, 2:] += N["sig_ball_v"] * lvl.view(-1, 1) * self._randn(b, 2)
        return pose, tw, torch.stack(scans), bm


class FusionTorch:
    """Estimated state of the field from the noisy sensors, as the blue team's node computes it."""

    def __init__(self, n, dt):
        self.n, self.dt = n, dt
        self.level = LevelEstimatorTorch(n)
        self.ekf = PoseEKFTorch(3 * n)
        self.ball = BallKFTorch(n)
        self.opp = SmoothTorch(3 * n)
        self.a = torch.ones(n, device="cuda")  # last estimated level (for logging)

    def _rows3(self, idx):
        return torch.cat([idx + k * self.n for k in range(3)])

    def forget(self, idx):
        r3 = self._rows3(idx)
        self.level.forget(idx)
        self.ekf.forget(r3)
        self.opp.forget(r3)
        self.ball.forget(idx)

    def step(self, pose, tw, scans, ball, idx):
        """Measurements of the envs `idx` (from SensorNoise.measure) -> estimated rob (6, b, 5), ball (b, 4)."""
        b = idx.numel()
        r3 = self._rows3(idx)  # robot k of env idx[j] -> row k * n + idx[j]
        a = self.level.update(pose[:3], tw[:3, :, 1], self.dt, idx)  # (b,)
        self.a[idx] = a
        a3 = a.repeat(3)
        z = pose[:3].reshape(3 * b, 3)
        v, w = tw[:3, :, 0].reshape(-1), tw[:3, :, 1].reshape(-1)
        est = self.ekf.step(z, v, w, scans.reshape(3 * b, FUS_BEAMS), self.dt, r3, a3).view(3, b, 3)
        opp = self.opp.step(pose[3:, :, :2].reshape(3 * b, 2), r3, a3).view(3, b, 2)
        rob = torch.cat([torch.cat([est, tw[:3]], 2), torch.cat([opp, pose[3:, :, 2:3], tw[3:]], 2)], 0)
        bl = self.ball.step(ball[:, :2], ball[:, 2:], self.dt, idx, a)
        return rob, bl


class SensorNoiseNp:
    """numpy copy of SensorNoise for the real psr_sim (eval_soccer.py --noise): corrupts the state dict
    that the nodes receive (odometry of the 6 robots, lidar, ball), at a fixed level."""

    def __init__(self, level, seed=0, drop=0.02):
        self.s, self.drop = float(level), drop
        self.rng = np.random.default_rng(seed)
        self.bias = np.zeros((6, 3))

    def apply(self, robots, puck, dt):
        s, N, g = self.s, FUS, self.rng
        out = []
        for r in robots:
            i = r["id"]
            self.bias[i, :2] += N["drift_p"] * s * math.sqrt(dt) * g.standard_normal(2)
            self.bias[i, 2] += N["drift_yaw"] * s * math.sqrt(dt) * g.standard_normal()
            q = dict(r)
            q["x"] = r["x"] + self.bias[i, 0] + N["sig_p"] * s * g.standard_normal()
            q["y"] = r["y"] + self.bias[i, 1] + N["sig_p"] * s * g.standard_normal()
            q["yaw"] = _wrap(r["yaw"] + self.bias[i, 2] + N["sig_yaw"] * s * g.standard_normal())
            q["v"] = r["v"] + N["sig_v"] * s * g.standard_normal()
            q["w"] = r["w"] + N["sig_w"] * s * g.standard_normal()
            rg = np.asarray(r["ranges"], dtype=float) + N["sig_r"] * s * g.standard_normal(len(r["ranges"]))
            if s > 0:
                rg = np.where(g.random(len(rg)) < self.drop, FUS_RMAX, rg)
            q["ranges"] = np.clip(rg, FUS_RMIN, FUS_RMAX).tolist()
            out.append(q)
        b = dict(puck)
        b["x"] += N["sig_ball"] * s * g.standard_normal()
        b["y"] += N["sig_ball"] * s * g.standard_normal()
        b["vx"] += N["sig_ball_v"] * s * g.standard_normal()
        b["vy"] += N["sig_ball_v"] * s * g.standard_normal()
        return out, b
