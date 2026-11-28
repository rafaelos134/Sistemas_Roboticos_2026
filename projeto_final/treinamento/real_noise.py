"""Sensor noise of the competition kit (psr-hero-kit-v2-ruido, ros/psr_hero/scripts/hero_bridge.py), ported
so the offline evaluators feed the nodes exactly what the noisy bridge publishes: zero-mean Gaussian with a
standard deviation that drifts slowly (mean-reverting walk, tau 5 s) between a min and a max, per robot/channel.
Enabled with the env var REAL_NOISE=1 (eval_sumo / eval_ctf / eval_soccer); PSR_KIT points to the kit."""
import math
import random

TAU = 5.0
NOISE = {
    "xy": (0.01, 0.05, 0.15),
    "yaw": (math.radians(1.0), math.radians(5.0), math.radians(15.0)),
    "v": (0.005, 0.02, 0.06),
    "w": (0.02, 0.10, 0.30),
    "laser": (0.002, 0.008, 0.024),
    "ball": (0.005, 0.02, 0.06),
    "ball_v": (0.005, 0.02, 0.06),
}
LIDAR_MIN, LIDAR_MAX = 0.037, 0.5


class _Varying:
    def __init__(self, rng, lo, hi, clip):
        self.rng, self.lo, self.hi, self.clip = rng, lo, hi, clip
        self.s, self.t = rng.random(), None

    def sample(self, t):
        if self.t is not None:
            dt = min(max(t - self.t, 0.0), 1.0)
            self.s += -(self.s - 0.5) * dt / TAU + 0.5 * math.sqrt(dt / TAU) * self.rng.gauss(0.0, 1.0)
            self.s = min(max(self.s, 0.0), 1.0)
        self.t = t
        sigma = self.lo + (self.hi - self.lo) * self.s
        return min(max(self.rng.gauss(0.0, sigma), -self.clip), self.clip)


class KitNoise:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.g = {}

    def __call__(self, key, kind, t):
        g = self.g.get(key)
        if g is None:
            g = self.g[key] = _Varying(self.rng, *NOISE[kind])
        return g.sample(t)

    def robots(self, robots, t):
        out = []
        for r in robots:
            i = r["id"]
            yaw = r["yaw"] + self(("yaw", i), "yaw", t)
            rr = dict(r, x=r["x"] + self(("x", i), "xy", t), y=r["y"] + self(("y", i), "xy", t),
                      yaw=math.atan2(math.sin(yaw), math.cos(yaw)), v=r["v"] + self(("v", i), "v", t),
                      w=r["w"] + self(("w", i), "w", t))
            if "ranges" in r:
                rr["ranges"] = [d if d >= LIDAR_MAX - 1e-6 else min(max(d + self(("laser", i, k), "laser", t), LIDAR_MIN), LIDAR_MAX)
                                for k, d in enumerate(r["ranges"])]
            out.append(rr)
        return out

    def ball(self, p, t):
        if p is None:
            return p
        return dict(p, x=p["x"] + self("ball_x", "ball", t), y=p["y"] + self("ball_y", "ball", t),
                    vx=p["vx"] + self("ball_vx", "ball_v", t), vy=p["vy"] + self("ball_vy", "ball_v", t))

    def flag(self, team, xy, t):
        return (xy[0] + self("flag_%s_x" % team, "ball", t), xy[1] + self("flag_%s_y" % team, "ball", t))


def enabled():
    import os
    return os.environ.get("REAL_NOISE", "0") == "1"
