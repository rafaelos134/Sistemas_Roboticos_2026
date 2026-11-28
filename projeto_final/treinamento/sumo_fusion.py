"""Sensor noise model + state estimation for the sumo policy (same approach as soccer_fusion.py).

The sumo ring has no walls: the 8 lidar beams only ever hit the opponent. So instead of matching the lidar
against a map (soccer), the filter uses it as a drift-free measurement of WHERE THE OPPONENT IS relative to us.

* SumoSensorNoise (training, torch): per episode a noise LEVEL s (s = 0 in 20 % of the episodes, else
  U(0, s_max)); every sensor gets s x the nominal values of SF (same nominal values as the soccer):
    - odometry pose  = true pose + drift (random walk per robot) + white noise. The drift does not start
      at zero: a round is one of many in a match, so the initial drift is the walk of U(0, 120) s;
    - odometry twist = true (v, w) + white noise;
    - lidar          = 8 beams against the opponent (psr_sim model) + white noise on the beams that hit
                       (a beam that hits nothing reads the max range), 2 % of the beams dropped.
  The filter only takes a beam as a hit below RMAX - 3 sig_r (robust to noise on the max range too).

* SumoEKF (numpy, what runs in the ROS node; SumoFusionTorch is the batched copy for training):
  one EKF with both robots, state (x, y, yaw, bx, by, byaw) x 2 = pose + odometry drift of each robot.
    predict:   each robot with its odometry twist (v, w);
    update 1:  the odometry poses of both robots (z = pose + drift);
    update 2:  every lidar beam that hit something hit the opponent: its centre is ~ p + (r + R_robot) d,
               precise along the beam, +-R_robot/sqrt(3) across it (the beam may hit off-centre). This ties
               the opponent to us without the drift of either odometry. Our yaw is not corrected by the lidar
               (a version that did, with the exact circle constraint, diverged in yaw for large drifts);
    update 3:  at the start of a round both robots are at the known spawn (x = -/+0.18, y ~ 0, facing each
               other), which makes the drift of the absolute pose observable (the distance to the ring edge).
               The round start is detected by a jump of the odometry (teleport) or announced by /game/state.
  The noise level is estimated online from the yaw residual of the odometry (as in the soccer).
"""

from __future__ import annotations

import math

import numpy as np

# >>> numpy-begin (this block is copied verbatim into the ROS node by export_sumo.py --fusion)
SF_RMIN, SF_RMAX, SF_BEAMS = 0.037, 0.50, 8
SF_ANGLES = [-math.pi + k * 2.0 * math.pi / SF_BEAMS for k in range(SF_BEAMS)]
SF_ROBOT_R = 0.037
# nominal noise the filter is tuned for (noise level s = 1)
SF = {"sig_p": 0.01, "sig_yaw": 0.03, "drift_p": 0.01, "drift_yaw": 0.02, "sig_v": 0.01, "sig_w": 0.10,
      "sig_r": 0.01, "slip_v": 0.05, "slip_w": 0.3}  # slip: unmodelled motion (pushed sideways) in the prediction
SF_SPAWN_X = 0.18
SF_SPAWN_SIG = (0.003, 0.035, 0.17)  # spawn: x exact, y ~ U(-0.06, 0.06), yaw within ~0.3 rad
SF_BIAS_T0 = 60.0  # s of drift assumed for the prior of the drift state
SF_JUMP, SF_SPAWN_GATE, SF_SPAWN_WAIT, SF_LIDAR_GATE = 0.15, 0.15, 20, 0.30
SF_LV_MIN, SF_LV_MAX = 0.3, 3.0
SF_POSE = [0, 1, 2, 6, 7, 8]
SF_BIAS = [3, 4, 5, 9, 10, 11]


def _sf_wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _sf_scaled(a):
    q = dict(SF)
    for k in ("sig_p", "sig_yaw", "drift_p", "drift_yaw", "sig_v", "sig_w", "sig_r"):
        q[k] = SF[k] * a
    return q


def _sf_p0(p):
    pose = [p["sig_p"] ** 2 + 1e-4, p["sig_p"] ** 2 + 1e-4, p["sig_yaw"] ** 2 + 1e-3]
    bias = [p["drift_p"] ** 2 * SF_BIAS_T0 + 1e-6] * 2 + [p["drift_yaw"] ** 2 * SF_BIAS_T0 + 1e-6]
    return np.diag(pose + bias + pose + bias)


class SumoLevel:
    """Online estimate of the noise level from the yaw residual of the odometry of both robots
    (e = z_yaw(k) - z_yaw(k-1) - w dt is not affected by pushes: w is measured)."""

    BETA = 0.02

    def __init__(self, n=2):
        self.prev = [None] * n
        self.v = None

    def update(self, zs, ws, dt):
        c = SF["drift_yaw"] ** 2 * dt + 2 * SF["sig_yaw"] ** 2 + (SF["sig_w"] * dt) ** 2
        if self.v is None:
            self.v = c
        es = []
        for i, (z, w) in enumerate(zip(zs, ws)):
            pv = self.prev[i]
            self.prev[i] = (z[0], z[1], z[2])
            if pv is None or math.hypot(z[0] - pv[0], z[1] - pv[1]) > SF_JUMP:
                continue
            e = _sf_wrap(z[2] - pv[2] - w * dt)
            es.append(min(e * e, 0.04))
        if es:
            self.v += self.BETA * (sum(es) / len(es) - self.v)
        return min(max(math.sqrt(max(self.v, 0.0) / c), SF_LV_MIN), SF_LV_MAX)


class SumoEKF:
    """EKF of both robots, see the module docstring. robot: our id (0 starts at -0.18, 1 at +0.18)."""

    def __init__(self, robot=0, spawn_first=True):
        self.sx = -SF_SPAWN_X if robot == 0 else SF_SPAWN_X
        self.m, self.P = None, None
        self.pending = 1 if spawn_first else 0

    def new_round(self):
        """Called when /game/state announces a new round: try the spawn update for the next ~1 s."""
        self.pending = SF_SPAWN_WAIT

    def step(self, zm, tm, zo, to, ranges, dt, a=1.0):
        """zm, zo: odometry poses (x, y, yaw) of us / the opponent; tm, to: their twists (v, w);
        ranges: our 8 lidar ranges; a: assumed noise level. Returns our pose and the opponent's (x, y, yaw)."""
        p = _sf_scaled(a)
        if self.m is None:
            self.m = np.array([zm[0], zm[1], zm[2], 0.0, 0.0, 0.0, zo[0], zo[1], zo[2], 0.0, 0.0, 0.0])
            self.P = _sf_p0(p)
        else:
            m, P = self.m.copy(), self.P
            F = np.eye(12)
            q = []
            for off, (v, w) in ((0, tm), (6, to)):
                c, s = math.cos(m[off + 2]), math.sin(m[off + 2])
                m[off] += v * c * dt
                m[off + 1] += v * s * dt
                m[off + 2] = _sf_wrap(m[off + 2] + w * dt)
                F[off, off + 2], F[off + 1, off + 2] = -v * s * dt, v * c * dt
                qv = (p["sig_v"] * dt) ** 2 + (p["slip_v"] * dt) ** 2
                q += [qv, qv, (p["sig_w"] * dt) ** 2 + (p["slip_w"] * dt) ** 2,
                      p["drift_p"] ** 2 * dt + 1e-9, p["drift_p"] ** 2 * dt + 1e-9, p["drift_yaw"] ** 2 * dt + 1e-9]
            P = F.dot(P).dot(F.T) + np.diag(q)
            z = np.array([zm[0], zm[1], zm[2], zo[0], zo[1], zo[2]])
            y = z - m[SF_POSE] - m[SF_BIAS]
            y[2], y[5] = _sf_wrap(y[2]), _sf_wrap(y[5])
            if max(math.hypot(y[0], y[1]), math.hypot(y[3], y[4])) > SF_JUMP:
                # teleport (new round): restart both poses from the odometry, keep the drift
                m[SF_POSE] = z - m[SF_BIAS]
                P[SF_POSE, :] = 0.0
                P[:, SF_POSE] = 0.0
                for i, sg in zip(SF_POSE, [p["sig_p"], p["sig_p"], p["sig_yaw"]] * 2):
                    P[i, i] = sg ** 2 + (1e-3 if i in (2, 8) else 1e-4)
                self.pending = SF_SPAWN_WAIT
            else:
                H = np.zeros((6, 12))
                H[np.arange(6), SF_POSE] = 1.0
                H[np.arange(6), SF_BIAS] = 1.0
                R = np.diag([p["sig_p"] ** 2 + 1e-6, p["sig_p"] ** 2 + 1e-6, p["sig_yaw"] ** 2 + 1e-6] * 2)
                pht = P.dot(H.T)
                K = np.linalg.solve(H.dot(pht) + R, pht.T).T
                m = m + K.dot(y)
                P = P - K.dot(pht.T)
            m[2], m[8] = _sf_wrap(m[2]), _sf_wrap(m[8])
            self.m, self.P = m, 0.5 * (P + P.T)
        if self.pending > 0:
            self.pending = 0 if self._spawn() else self.pending - 1
        self._lidar(ranges, p)
        m = self.m
        return (float(m[0]), float(m[1]), float(m[2])), (float(m[6]), float(m[7]), float(m[8]))

    def _spawn(self):
        m, P = self.m, self.P
        y0 = 0.0 if self.sx < 0 else math.pi
        z = np.array([self.sx, 0.0, y0, -self.sx, 0.0, _sf_wrap(y0 + math.pi)])
        y = z - m[SF_POSE]
        y[2], y[5] = _sf_wrap(y[2]), _sf_wrap(y[5])
        if max(math.hypot(y[0], y[1]), math.hypot(y[3], y[4])) > SF_SPAWN_GATE:
            return False
        H = np.zeros((6, 12))
        H[np.arange(6), SF_POSE] = 1.0
        R = np.diag([s ** 2 for s in SF_SPAWN_SIG] * 2)
        pht = P.dot(H.T)
        K = np.linalg.solve(H.dot(pht) + R, pht.T).T
        m = m + K.dot(y)
        m[2], m[8] = _sf_wrap(m[2]), _sf_wrap(m[8])
        P = P - K.dot(pht.T)
        self.m, self.P = m, 0.5 * (P + P.T)
        return True

    def _lidar(self, ranges, p):
        """Each beam that hit something saw the opponent: its centre is ~ p + (r + R_robot) d, precise along
        the beam (range noise; the beam may hit off-centre), +-R_robot/sqrt(3) across it. Our yaw is not
        corrected here (its error only widens the across-beam variance), which keeps the update robust."""
        m, P = self.m, self.P
        ang = m[2] + np.array(SF_ANGLES)
        dx, dy = np.cos(ang), np.sin(ang)
        r = np.asarray(ranges, dtype=np.float64)
        qx, qy = m[6] - m[0], m[7] - m[1]
        y_al = r + SF_ROBOT_R - (dx * qx + dy * qy)
        y_pp = -(-dy * qx + dx * qy)
        ok = (r < SF_RMAX - max(0.005, 3.0 * p["sig_r"])) & (np.hypot(y_al, y_pp) < SF_LIDAR_GATE)
        if not ok.any():
            return
        H = np.zeros((2 * SF_BEAMS, 12))
        H[0::2, 0], H[0::2, 1], H[0::2, 6], H[0::2, 7] = -dx, -dy, dx, dy
        H[1::2, 0], H[1::2, 1], H[1::2, 6], H[1::2, 7] = dy, -dx, -dy, dx
        ok2 = np.repeat(ok, 2)
        H[~ok2] = 0.0
        y = np.where(ok2, np.stack([y_al, y_pp], 1).ravel(), 0.0)
        r_al = p["sig_r"] ** 2 + 0.01 ** 2
        r_pp = SF_ROBOT_R ** 2 / 3.0 + (r + SF_ROBOT_R) ** 2 * P[2, 2]
        Rd = np.where(ok2, np.stack([np.full(SF_BEAMS, r_al), r_pp], 1).ravel(), 1.0)
        pht = P.dot(H.T)
        K = np.linalg.solve(H.dot(pht) + np.diag(Rd), pht.T).T
        m = m + K.dot(y)
        m[2], m[8] = _sf_wrap(m[2]), _sf_wrap(m[8])
        P = P - K.dot(pht.T)
        self.m, self.P = m, 0.5 * (P + P.T)
# <<< numpy-end


def cast_opponent_np(px, py, yaw, ox, oy):
    """Lidar ranges of a robot at (px, py, yaw) against the opponent circle (psr_sim model, no walls)."""
    ang = yaw + np.array(SF_ANGLES)
    dx, dy = np.cos(ang), np.sin(ang)
    fx, fy = px - ox, py - oy
    b = dx * fx + dy * fy
    disc = b * b - (fx * fx + fy * fy - SF_ROBOT_R ** 2)
    t = -b - np.sqrt(np.maximum(disc, 0.0))
    t = np.where((disc >= 0) & (t >= 0), t, np.inf)
    return np.clip(t, SF_RMIN, SF_RMAX)


class SumoSensorNoiseNp:
    """numpy noise for the real psr_sim (eval_sumo.py --noise): corrupts the odometry and the lidar of both
    robots at a fixed level. The drift keeps accumulating over the rounds (the odometry is not reset)."""

    def __init__(self, level, seed=0, drop=0.02):
        self.s, self.drop = float(level), drop
        self.rng = np.random.default_rng(seed)
        self.bias = np.zeros((2, 3))

    def apply(self, robots, dt):
        s, N, g = self.s, SF, self.rng
        out = []
        for r in robots:
            i = r["id"]
            self.bias[i, :2] += N["drift_p"] * s * math.sqrt(dt) * g.standard_normal(2)
            self.bias[i, 2] += N["drift_yaw"] * s * math.sqrt(dt) * g.standard_normal()
            q = dict(r)
            q["x"] = r["x"] + self.bias[i, 0] + N["sig_p"] * s * g.standard_normal()
            q["y"] = r["y"] + self.bias[i, 1] + N["sig_p"] * s * g.standard_normal()
            q["yaw"] = _sf_wrap(r["yaw"] + self.bias[i, 2] + N["sig_yaw"] * s * g.standard_normal())
            q["v"] = r["v"] + N["sig_v"] * s * g.standard_normal()
            q["w"] = r["w"] + N["sig_w"] * s * g.standard_normal()
            r0 = np.asarray(r["ranges"], dtype=float)
            rg = np.where(r0 < SF_RMAX, r0 + N["sig_r"] * s * g.standard_normal(len(r0)), SF_RMAX)
            if s > 0:
                rg = np.where(g.random(len(rg)) < self.drop, SF_RMAX, rg)
            q["ranges"] = np.clip(rg, SF_RMIN, SF_RMAX).tolist()
            out.append(q)
        return out


# ======================================================================================================
# torch: noise model and the batched copy of the filter (training)
# ======================================================================================================
import torch  # noqa: E402

_ANG = None


def _angles():
    global _ANG
    if _ANG is None:
        _ANG = torch.tensor(SF_ANGLES, dtype=torch.float32, device="cuda")
    return _ANG


def wrap_t(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def cast_opponent_t(px, py, yaw, ox, oy):
    """(b,) each -> (b, 8) ranges against the opponent circle."""
    ang = yaw.view(-1, 1) + _angles().view(1, -1)
    dx, dy = torch.cos(ang), torch.sin(ang)
    fx, fy = (px - ox).view(-1, 1), (py - oy).view(-1, 1)
    b = dx * fx + dy * fy
    disc = b * b - (fx * fx + fy * fy - SF_ROBOT_R ** 2)
    t = -b - torch.sqrt(disc.clamp_min(0))
    t = torch.where((disc >= 0) & (t >= 0), t, torch.full_like(t, float("inf")))
    return t.clamp(SF_RMIN, SF_RMAX)


class SumoSensorNoise:
    """Per-episode noise level and the noisy sensors of both robots (training only)."""

    def __init__(self, n, gen, s_max=2.0, p_clean=0.2, drop=0.02, fixed_level=None, t0_max=120.0):
        self.n, self.gen = n, gen
        self.s_max, self.p_clean, self.drop, self.fixed, self.t0_max = s_max, p_clean, drop, fixed_level, t0_max
        self.level = torch.zeros(n, device="cuda")
        self.bias = torch.zeros(2, n, 3, device="cuda")

    def _randn(self, *shape):
        return torch.randn(*shape, device="cuda", generator=self.gen)

    def _rand(self, *shape):
        return torch.rand(*shape, device="cuda", generator=self.gen)

    def reset(self, idx, std=None):
        """std: the episodes that start at the standard spawn. The others (random / clinch starts) stand
        for moments inside a round, when the spawn update corrected the drift at most ~40 s before."""
        k = idx.numel()
        if self.fixed is not None:
            lv = torch.full((k,), float(self.fixed), device="cuda")
        else:
            lv = self.s_max * self._rand(k)
            lv = torch.where(self._rand(k) < self.p_clean, torch.zeros_like(lv), lv)
        self.level[idx] = lv
        # drift accumulated in the earlier rounds of the match
        t0 = self.t0_max if std is None else torch.where(std, torch.full_like(lv, self.t0_max), torch.full_like(lv, 40.0))
        sq = torch.sqrt(t0 * self._rand(k)) * lv
        b = torch.zeros(2, k, 3, device="cuda")
        b[..., :2] = SF["drift_p"] * sq.view(1, -1, 1) * self._randn(2, k, 2)
        b[..., 2] = SF["drift_yaw"] * sq.view(1, -1) * self._randn(2, k)
        self.bias[:, idx] = b

    def measure(self, me, opp, dt, idx):
        """me, opp (b, 5) true (x, y, yaw, v, w) of the envs idx. Returns odometry poses (2, b, 3),
        twists (2, b, 2) and our lidar (b, 8)."""
        b = idx.numel()
        lvl = self.level[idx]
        s = lvl.view(1, -1)
        N = SF
        rob = torch.stack([me, opp])
        bias = self.bias[:, idx]
        bias[..., :2] += N["drift_p"] * s.unsqueeze(2) * math.sqrt(dt) * self._randn(2, b, 2)
        bias[..., 2] += N["drift_yaw"] * s * math.sqrt(dt) * self._randn(2, b)
        self.bias[:, idx] = bias
        pose = rob[..., :3] + bias
        pose[..., :2] += N["sig_p"] * s.unsqueeze(2) * self._randn(2, b, 2)
        pose[..., 2] = wrap_t(pose[..., 2] + N["sig_yaw"] * s * self._randn(2, b))
        tw = rob[..., 3:5].clone()
        tw[..., 0] += N["sig_v"] * s * self._randn(2, b)
        tw[..., 1] += N["sig_w"] * s * self._randn(2, b)
        r = cast_opponent_t(me[:, 0], me[:, 1], me[:, 2], opp[:, 0], opp[:, 1])
        r = torch.where(r < SF_RMAX, r + N["sig_r"] * lvl.view(-1, 1) * self._randn(b, SF_BEAMS), r)
        dropped = self._rand(b, SF_BEAMS) < self.drop * (lvl.view(-1, 1) > 0).float()
        r = torch.where(dropped, torch.full_like(r, SF_RMAX), r).clamp(SF_RMIN, SF_RMAX)
        return pose, tw, r


class SumoLevelTorch:
    def __init__(self, n):
        self.prev = torch.zeros(2, n, 3, device="cuda")
        self.has = torch.zeros(2, n, dtype=torch.bool, device="cuda")
        self.v = torch.zeros(n, device="cuda")
        self.init = torch.zeros(n, dtype=torch.bool, device="cuda")

    def forget(self, idx):
        self.has[:, idx] = False
        self.init[idx] = False

    def update(self, z, w, dt, idx):
        """z (2, b, 3), w (2, b) -> level (b,)."""
        c = SF["drift_yaw"] ** 2 * dt + 2 * SF["sig_yaw"] ** 2 + (SF["sig_w"] * dt) ** 2
        v = torch.where(self.init[idx], self.v[idx], torch.full_like(self.v[idx], c))
        pv, has = self.prev[:, idx], self.has[:, idx]
        ok = has & (torch.hypot(z[..., 0] - pv[..., 0], z[..., 1] - pv[..., 1]) <= SF_JUMP)
        e = wrap_t(z[..., 2] - pv[..., 2] - w * dt)
        e2 = torch.where(ok, (e * e).clamp(max=0.04), torch.zeros_like(e))
        cnt = ok.float().sum(0)
        v = torch.where(cnt > 0, v + SumoLevel.BETA * (e2.sum(0) / cnt.clamp_min(1) - v), v)
        self.v[idx], self.init[idx] = v, True
        self.prev[:, idx], self.has[:, idx] = z, True
        return torch.sqrt(v.clamp_min(0) / c).clamp(SF_LV_MIN, SF_LV_MAX)


def _sc(k, a):
    return SF[k] * a


class SumoFusionTorch:
    """Batched copy of SumoLevel + SumoEKF (robot 0 = the agent, spawn at -0.18)."""

    def __init__(self, n, dt, sx=-SF_SPAWN_X):
        self.n, self.dt, self.sx = n, dt, sx
        self.m = torch.zeros(n, 12, device="cuda")
        self.P = torch.zeros(n, 12, 12, device="cuda")
        self.init = torch.zeros(n, dtype=torch.bool, device="cuda")
        self.pending = torch.zeros(n, dtype=torch.long, device="cuda")
        self.level = SumoLevelTorch(n)
        self.a = torch.ones(n, device="cuda")
        self.ip = torch.tensor(SF_POSE, device="cuda")
        self.ib = torch.tensor(SF_BIAS, device="cuda")
        H = torch.zeros(6, 12, device="cuda")
        H[torch.arange(6), self.ip] = 1.0
        self.Hs = H.clone()  # spawn: pose only
        H[torch.arange(6), self.ib] = 1.0
        self.Ho = H  # odometry: pose + drift
        y0 = 0.0 if sx < 0 else math.pi
        self.zs = torch.tensor([sx, 0.0, y0, -sx, 0.0, (y0 + 2 * math.pi) % (2 * math.pi) - math.pi], device="cuda")
        self.Rs = torch.diag(torch.tensor([s ** 2 for s in SF_SPAWN_SIG] * 2, device="cuda"))

    def forget(self, idx, spawn=None):
        """idx: envs that start a new episode; spawn (bool, same size): they start at the standard spawn."""
        self.init[idx] = False
        self.level.forget(idx)
        self.pending[idx] = 0 if spawn is None else spawn.long()

    def _p0(self, a):
        sp, sy, dp, dy = (_sc(k, a) for k in ("sig_p", "sig_yaw", "drift_p", "drift_yaw"))
        pose = [sp ** 2 + 1e-4, sp ** 2 + 1e-4, sy ** 2 + 1e-3]
        bias = [dp ** 2 * SF_BIAS_T0 + 1e-6, dp ** 2 * SF_BIAS_T0 + 1e-6, dy ** 2 * SF_BIAS_T0 + 1e-6]
        return torch.diag_embed(torch.stack(pose + bias + pose + bias, 1))

    @staticmethod
    def _kalman(m, P, H, y, R):
        """H (b or 1, k, 12), y (b, k), R (b, k, k) -> updated m, P."""
        pht = P @ H.transpose(-1, -2)
        S = H @ pht + R
        K = torch.linalg.solve(S, pht.transpose(1, 2)).transpose(1, 2)
        m = m + (K @ y.unsqueeze(2)).squeeze(2)
        P = P - K @ pht.transpose(1, 2)
        return m, P

    def step(self, pose, tw, ranges, idx):
        """pose (2, b, 3), tw (2, b, 2), ranges (b, 8) of the envs idx -> estimated me, opp (b, 3) each."""
        dt = self.dt
        b = idx.numel()
        a = self.level.update(pose, tw[..., 1], dt, idx)
        self.a[idx] = a
        sp, sy, dp, dy, sv, sw = (_sc(k, a) for k in ("sig_p", "sig_yaw", "drift_p", "drift_yaw", "sig_v", "sig_w"))
        m, P, init, pend = self.m[idx].clone(), self.P[idx].clone(), self.init[idx], self.pending[idx].clone()
        # predict
        F = torch.eye(12, device="cuda").expand(b, 12, 12).clone()
        for j, off in enumerate((0, 6)):
            v, w = tw[j, :, 0], tw[j, :, 1]
            c, s = torch.cos(m[:, off + 2]), torch.sin(m[:, off + 2])
            m[:, off] += v * c * dt
            m[:, off + 1] += v * s * dt
            m[:, off + 2] = wrap_t(m[:, off + 2] + w * dt)
            F[:, off, off + 2], F[:, off + 1, off + 2] = -v * s * dt, v * c * dt
        qv = (sv * dt) ** 2 + SF["slip_v"] ** 2 * dt ** 2
        qw = (sw * dt) ** 2 + SF["slip_w"] ** 2 * dt ** 2
        q1 = [qv, qv, qw, dp ** 2 * dt + 1e-9, dp ** 2 * dt + 1e-9, dy ** 2 * dt + 1e-9]
        P = F @ P @ F.transpose(1, 2) + torch.diag_embed(torch.stack(q1 + q1, 1))
        # odometry update
        z = torch.cat([pose[0], pose[1]], 1)  # (b, 6)
        y = z - m[:, self.ip] - m[:, self.ib]
        y[:, 2], y[:, 5] = wrap_t(y[:, 2]), wrap_t(y[:, 5])
        jump = (torch.maximum(torch.hypot(y[:, 0], y[:, 1]), torch.hypot(y[:, 3], y[:, 4])) > SF_JUMP) & init
        r1 = [sp ** 2 + 1e-6, sp ** 2 + 1e-6, sy ** 2 + 1e-6]
        mu, Pu = self._kalman(m, P, self.Ho.unsqueeze(0), y, torch.diag_embed(torch.stack(r1 + r1, 1)))
        mj, Pj = m.clone(), P.clone()
        mj[:, self.ip] = z - m[:, self.ib]
        Pj[:, self.ip, :] = 0.0
        Pj[:, :, self.ip] = 0.0
        for i, sg, e in zip(SF_POSE, [sp, sp, sy] * 2, [1e-4, 1e-4, 1e-3] * 2):
            Pj[:, i, i] = sg ** 2 + e
        jj = jump.view(-1, 1)
        m = torch.where(jj, mj, mu)
        P = torch.where(jj.unsqueeze(2), Pj, Pu)
        pend = torch.where(jump, torch.full_like(pend, SF_SPAWN_WAIT), pend)
        m[:, 2], m[:, 8] = wrap_t(m[:, 2]), wrap_t(m[:, 8])
        P = 0.5 * (P + P.transpose(1, 2))
        # first step of an episode: initialise from the odometry
        new = ~init
        m0 = torch.cat([pose[0], torch.zeros_like(pose[0]), pose[1], torch.zeros_like(pose[1])], 1)
        m = torch.where(new.view(-1, 1), m0, m)
        P = torch.where(new.view(-1, 1, 1), self._p0(a), P)
        # spawn update
        ys = self.zs.view(1, -1) - m[:, self.ip]
        ys[:, 2], ys[:, 5] = wrap_t(ys[:, 2]), wrap_t(ys[:, 5])
        gate = torch.maximum(torch.hypot(ys[:, 0], ys[:, 1]), torch.hypot(ys[:, 3], ys[:, 4])) <= SF_SPAWN_GATE
        do = (pend > 0) & gate
        ms, Ps = self._kalman(m, P, self.Hs.unsqueeze(0), ys, self.Rs.expand(b, 6, 6))
        ms[:, 2], ms[:, 8] = wrap_t(ms[:, 2]), wrap_t(ms[:, 8])
        m = torch.where(do.view(-1, 1), ms, m)
        P = torch.where(do.view(-1, 1, 1), 0.5 * (Ps + Ps.transpose(1, 2)), P)
        pend = torch.where(pend > 0, torch.where(do, torch.zeros_like(pend), pend - 1), pend)
        # lidar update
        m, P = self._lidar(m, P, ranges, a)
        self.m[idx], self.P[idx], self.init[idx], self.pending[idx] = m, P, True, pend
        return m[:, 0:3], m[:, 6:9]

    def _lidar(self, m, P, r, a):
        b = m.shape[0]
        ang = m[:, 2:3] + _angles().view(1, -1)
        dx, dy = torch.cos(ang), torch.sin(ang)
        qx, qy = (m[:, 6] - m[:, 0]).view(-1, 1), (m[:, 7] - m[:, 1]).view(-1, 1)
        y_al = r + SF_ROBOT_R - (dx * qx + dy * qy)
        y_pp = -(-dy * qx + dx * qy)
        thr = SF_RMAX - (3.0 * SF["sig_r"] * a).clamp_min(0.005).view(-1, 1)
        ok = (r < thr) & (torch.hypot(y_al, y_pp) < SF_LIDAR_GATE)
        H = torch.zeros(b, SF_BEAMS, 2, 12, device="cuda")
        H[:, :, 0, 0], H[:, :, 0, 1], H[:, :, 0, 6], H[:, :, 0, 7] = -dx, -dy, dx, dy
        H[:, :, 1, 0], H[:, :, 1, 1], H[:, :, 1, 6], H[:, :, 1, 7] = dy, -dx, -dy, dx
        H = (H * ok.view(b, SF_BEAMS, 1, 1)).reshape(b, 2 * SF_BEAMS, 12)
        ok2 = ok.repeat_interleave(2, 1)
        y = torch.where(ok2, torch.stack([y_al, y_pp], 2).reshape(b, -1), torch.zeros(b, 2 * SF_BEAMS, device="cuda"))
        r_al = ((SF["sig_r"] * a) ** 2 + 0.01 ** 2).view(-1, 1).expand(b, SF_BEAMS)
        r_pp = SF_ROBOT_R ** 2 / 3.0 + (r + SF_ROBOT_R) ** 2 * P[:, 2, 2].view(-1, 1)
        Rd = torch.where(ok2, torch.stack([r_al, r_pp], 2).reshape(b, -1), torch.ones(b, 2 * SF_BEAMS, device="cuda"))
        mu, Pu = self._kalman(m, P, H, y, torch.diag_embed(Rd))
        mu[:, 2], mu[:, 8] = wrap_t(mu[:, 2]), wrap_t(mu[:, 8])
        any_ok = ok.any(1)
        m = torch.where(any_ok.view(-1, 1), mu, m)
        P = torch.where(any_ok.view(-1, 1, 1), 0.5 * (Pu + Pu.transpose(1, 2)), P)
        return m, P
