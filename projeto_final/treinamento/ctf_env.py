"""Vectorized capture-the-flag environment (2 x 2 HeRo robots) on Genesis, for RL.

Same physics as psr_sim (kit/sim/psr_sim): robot URDF, walls of the scenario and all constants are
imported from there, and the referee below is a copy of CaptureTheFlag.referee (kit/sim/psr_sim/
scenarios/ctf.py) in torch, in the same order: carried flags follow their carrier, holds, pickups
(the kit does not check whether the robot is held), points, 90 s draw. A point or a draw ends the episode.

Robots: 0 blue attacker, 1 blue defender, 2 red attacker, 3 red defender (the launch file of the kit).
Team frame: each team sees the arena as if its base were at (-0.62, 0) (red rotated by 180 deg), so one
network per role plays both sides.

Training episodes: a round of the kit lasts up to 90 s; for more episode ends per update an episode is cut
after `episode_s` (truncation: the value is bootstrapped) and every episode starts at a random round time,
so the round clock in the observation still covers 0..90 s (and the 90 s draw still happens).

One robot learns: mode "attack" -> robot 0, mode "defend" -> robot 1. The other three are, per episode:
    stopped | script (ctf_scripts.py, gains drawn per episode) | a frozen network of that role (pool).
The "main" opponent (red defender when attacking, red attacker when defending) follows `main_mix`;
the teammate and the other opponent follow `other_mix`.
"""

from __future__ import annotations

import math
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path

import torch

SIM_DIR = Path(__file__).resolve().parents[1] / "kit" / "sim"
sys.path.insert(0, str(SIM_DIR))

from psr_sim.scenarios.ctf import (  # noqa: E402
    BASE_RADIUS, BASE_X, HOLD_TIME, IMMUNE_TIME, PICKUP_DISTANCE, ROUND_TIME, SIZE, TAG_DISTANCE, CaptureTheFlag,
)
from psr_sim.world import (  # noqa: E402
    DT, FORCE_MAX, KV_LIN, KV_YAW, ROBOT_FRICTION, TORQUE_MAX, V_MAX, W_MAX, Z0, World, hat_mesh, hero_urdf,
)

from ctf_obs import OBS_DIM, observation_torch, to_team_frame  # noqa: E402,F401
from ctf_scripts import ATT_STYLES, DEF_STYLES, attacker_actions, defender_actions, draw_gains, draw_styles  # noqa: E402

HALF = SIZE / 2
TEAM = (0, 0, 1, 1)
ROLE = (0, 1, 0, 1)


@dataclass
class AttackReward:
    score: float = 10.0  # our team scored
    conceded: float = -5.0
    pickup: float = 3.0  # I picked up the enemy flag
    tagged: float = -1.0  # I was held
    tagged_carrying: float = -2.0  # extra, I lost the flag
    tag: float = 1.0  # I held an intruder (in my half)
    # potential: carrying -> -w_home * d(me, home); else -w_flag * d(me, enemy flag) - w_home * d0
    shape_flag: float = 2.0
    shape_home: float = 2.0
    d0: float = 1.3
    # potential: -avoid * max(0, avoid_r - d(me, closest active opponent)) while I am in their half and can
    # be held (not immune): keeps a distance from the defenders without forbidding the flag
    avoid: float = 2.0
    avoid_r: float = 0.18
    immune_pickup: float = 1.0  # extra for a pickup while I am immune (just released: cannot be held)


@dataclass
class DefendReward:
    conceded: float = -10.0
    score: float = 2.0
    tag: float = 2.0  # I held an intruder
    tag_carrier: float = 3.0  # extra, it carried my flag
    enemy_pickup: float = -3.0  # an opponent picked up my flag
    tagged: float = -2.0  # I was held (I went into their half)
    enemy_half: float = -0.01  # per step in their half
    # potential: -w * d(me, target); target = the carrier of my flag, else the guard point
    # (on the segment home -> their attacker, guard_d from the flag)
    shape_guard: float = 1.0
    guard_d: float = 0.10


class CtfEnv:
    def __init__(self, n_envs=1024, control_dt=0.05, mode="attack", gamma=0.995, att_reward=None, def_reward=None,
                 episode_s=30.0, random_start=0.2, easy_start=0.4, obs_noise=0.002, seed=0, main_mix=None, other_mix=None,
                 att_pool=None, def_pool=None, att_styles=None, def_styles=None, strong=0.0, dr_speed=0.0,
                 dr_delay=0.0, yaw_noise=0.0):
        import genesis as gs

        if not getattr(gs, "_initialized", False):
            gs.init(backend=gs.gpu, logging_level="warning", seed=seed)
        self.gs, self.n, self.gamma, self.mode = gs, n_envs, gamma, mode
        self.agent = 0 if mode == "attack" else 1
        self.main = 3 if mode == "attack" else 2
        self.arew = att_reward or AttackReward()
        self.drew = def_reward or DefendReward()
        self.random_start, self.easy_start, self.obs_noise = random_start, easy_start, obs_noise
        # script styles (ctf_scripts.py; None = only the standard style), fraction of reference gains
        self.att_styles = att_styles or (1.0,) + (0.0,) * (ATT_STYLES - 1)
        self.def_styles = def_styles or (1.0,) + (0.0,) * (DEF_STYLES - 1)
        self.strong = strong
        # domain randomization: command scale per robot and episode in [1 - dr_speed, 1 + dr_speed], one step
        # of command latency for the learning robot in a fraction dr_delay of the episodes, yaw noise (rad)
        self.dr_speed, self.dr_delay, self.yaw_noise = dr_speed, dr_delay, yaw_noise
        self.substeps = max(1, round(control_dt / DT))
        self.dt = self.substeps * DT
        self.max_steps = int(round(ROUND_TIME / self.dt))  # the kit's draw
        self.ep_max = int(round(episode_s / self.dt))
        self.gen = torch.Generator(device="cuda").manual_seed(seed)
        self.pools = (att_pool, def_pool)
        self.set_mix(main_mix or {"script": 1.0}, other_mix or {"script": 1.0})

        self.tmp = tempfile.TemporaryDirectory(prefix="ctf_rl_")
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, 0.0)),
            rigid_options=gs.options.RigidOptions(enable_collision=True, enable_self_collision=False,
                                                  enable_joint_limit=False),
            show_viewer=False,
        )
        scenario = CaptureTheFlag(4, 0)
        fake = types.SimpleNamespace(gs=gs, scene=self.scene, tmp=self.tmp)
        for s in scenario.statics():
            if s.solid:  # walls only (flags have no collision in psr_sim either)
                World._add_static(fake, s)
        self.robots = []
        for i, rgb in enumerate(scenario.colors()):
            hat = hat_mesh(Path(self.tmp.name) / f"hat_{i}.obj", rgb)
            path = Path(self.tmp.name) / f"hero_{i}.urdf"
            path.write_text(hero_urdf(f"hero_{i}", rgb, hat))
            self.robots.append(self.scene.add_entity(gs.morphs.URDF(file=str(path), fixed=True, pos=(0, 0, Z0))))
        self.scene.build(n_envs=n_envs)
        c = lambda x: torch.tensor(x, device="cuda", dtype=torch.float32)  # noqa: E731
        self.dofs = torch.arange(3, device="cuda")
        for r in self.robots:
            r.set_dofs_armature(c([1e-6] * 3), self.dofs)
            r.set_dofs_kp(c([0.0] * 3), self.dofs)
            r.set_dofs_kv(c([KV_LIN, KV_LIN, KV_YAW]), self.dofs)
            r.set_dofs_force_range(c([-FORCE_MAX, -FORCE_MAX, -TORQUE_MAX]), c([FORCE_MAX, FORCE_MAX, TORQUE_MAX]),
                                   self.dofs)
            r.get_link("base_link").set_friction(ROBOT_FRICTION)

        n = n_envs
        dev = "cuda"
        self.home = torch.tensor([[-BASE_X, 0.0], [BASE_X, 0.0]], device=dev)  # (2, 2) world frame
        self.steps = torch.zeros(n, dtype=torch.long, device=dev)  # round clock (steps)
        self.ep_steps = torch.zeros(n, dtype=torch.long, device=dev)  # steps since the episode started
        self.prev = torch.zeros(4, n, 2, device=dev)
        self.hold_until = torch.zeros(4, n, device=dev)
        self.immune_until = torch.zeros(4, n, device=dev)
        self.carrier = torch.full((2, n), -1, dtype=torch.long, device=dev)  # carrier[team]: robot with that flag
        self.flag = self.home.unsqueeze(1).repeat(1, n, 1)  # (2, n, 2)
        self.kind = torch.zeros(4, n, dtype=torch.long, device=dev)  # 0 stopped, 1 script, 2 + k pool slot k
        self.gains = torch.zeros(4, n, 7, device=dev)
        self.style = torch.zeros(4, n, dtype=torch.long, device=dev)
        self.scale = torch.ones(4, n, device=dev)
        self.delay = torch.zeros(n, dtype=torch.bool, device=dev)
        self.pending = torch.zeros(n, 2, device=dev)  # command of the learning robot waiting one step (latency)
        self.state = None
        self.reset_all()

    # --- opponents -----------------------------------------------------------------------------

    def set_mix(self, main_mix: dict, other_mix: dict):
        """{"stopped": w, "script": w, "latest": w, "pool": w}: weights of each kind of controller."""
        def prep(mix):
            names = [k for k, w in mix.items() if w > 0]
            p = torch.tensor([float(mix[k]) for k in names], device="cuda")
            return names, p / p.sum()
        self.main_mix, self.other_mix = prep(main_mix), prep(other_mix)

    def _draw(self, slot, k):
        names, p = self.main_mix if slot == self.main else self.other_mix
        pool = self.pools[ROLE[slot]]
        npool = len(pool) if pool is not None else 0
        pick = torch.multinomial(p, k, replacement=True, generator=self.gen)
        kinds = torch.ones(k, dtype=torch.long, device="cuda")  # script by default (also when the pool is empty)
        for j, name in enumerate(names):
            rows = pick == j
            if not rows.any():
                continue
            if name == "stopped":
                kinds[rows] = 0
            elif name == "latest" and npool:
                kinds[rows] = 2 + npool - 1
            elif name == "pool" and npool:
                kinds[rows] = 2 + torch.randint(0, max(1, npool - 1), (int(rows.sum()),), device="cuda",
                                                generator=self.gen)
        return kinds

    # --- state ---------------------------------------------------------------------------------

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device="cuda")

    def _spawn(self, k):
        """Poses (4, k, 3) and the robot that starts carrying the enemy flag (k,) or -1."""
        u = lambda lo, hi, *s: lo + (hi - lo) * self._rand(*(s or (k,)))  # noqa: E731
        z = torch.zeros(k, device="cuda")
        x0 = BASE_X - 0.15
        poses = torch.stack([  # the kit start (CaptureTheFlag.respawn)
            torch.stack([-x0 + z, 0.28 + u(-.06, .06), z], 1), torch.stack([-x0 + z, -0.28 + u(-.06, .06), z], 1),
            torch.stack([x0 + z, -0.28 + u(-.06, .06), z + math.pi], 1),
            torch.stack([x0 + z, 0.28 + u(-.06, .06), z + math.pi], 1)])
        carry = torch.full((k,), -1, dtype=torch.long, device="cuda")
        r = self._rand(k)
        rnd = r < self.random_start
        easy = (~rnd) & (r < self.random_start + self.easy_start)
        m = int(rnd.sum())
        if m:  # everybody anywhere in its own half
            pr = torch.stack([u(0.05, HALF - 0.08, 4, m), u(-HALF + 0.08, HALF - 0.08, 4, m),
                              u(-math.pi, math.pi, 4, m)], 2)
            pr[:2, :, 0] *= -1
            poses[:, rnd] = pr
        m = int(easy.sum())
        if m:
            # attack: robot 0 near the red flag (half of them already carrying it);
            # defend: the red attacker (2) near the blue flag, half of them carrying it
            who, fx, sgn = (0, BASE_X, 1.0) if self.mode == "attack" else (2, -BASE_X, -1.0)
            ang = u(-math.pi, math.pi, m)
            d = u(0.10, 0.55, m)
            px = (fx - sgn * d * torch.cos(ang).abs()).clamp(-HALF + 0.06, HALF - 0.06)
            py = (d * torch.sin(ang)).clamp(-HALF + 0.06, HALF - 0.06)
            sub = poses[:, easy]
            sub[who] = torch.stack([px, py, u(-math.pi, math.pi, m)], 1)
            if self.mode == "defend":  # my defender somewhere in my half
                sub[1] = torch.stack([-u(0.1, 0.7, m), u(-0.6, 0.6, m), u(-math.pi, math.pi, m)], 1)
            poses[:, easy] = sub
            ce = torch.where(self._rand(m) < 0.5, torch.full((m,), who, device="cuda"), torch.full((m,), -1, device="cuda"))
            carry[easy] = ce
        # overlaps: back to the kit start
        p = poses[:, :, :2].permute(1, 0, 2)
        d = torch.cdist(p, p) + 10 * torch.eye(4, device="cuda")
        bad = d.amin((1, 2)) < 0.09
        if bad.any():
            std = torch.stack([torch.tensor([-x0, 0.28, 0.0]), torch.tensor([-x0, -0.28, 0.0]),
                               torch.tensor([x0, -0.28, math.pi]), torch.tensor([x0, 0.28, math.pi])]).cuda()
            poses[:, bad] = std.unsqueeze(1).expand(-1, int(bad.sum()), -1)
            carry[bad] = -1
        return poses, carry

    def _reset(self, idx):
        if idx.numel() == 0:
            return
        k = idx.numel()
        poses, carry = self._spawn(k)
        for i, r in enumerate(self.robots):
            r.set_dofs_position(poses[i], self.dofs, envs_idx=idx, zero_velocity=True)
        self.steps[idx] = (self._rand(k) * (self.max_steps - self.ep_max // 2)).long()
        self.ep_steps[idx] = 0
        self.prev[:, idx] = 0.0
        self.hold_until[:, idx] = 0.0
        self.immune_until[:, idx] = 0.0
        self.flag[:, idx] = self.home.unsqueeze(1).expand(-1, k, -1)
        self.carrier[:, idx] = -1
        # a robot that starts carrying: it holds the flag of the other team
        has = carry >= 0
        if has.any():
            team_flag = 1 - torch.tensor(TEAM, device="cuda")[carry.clamp(min=0)]
            for t in (0, 1):
                rows = has & (team_flag == t)
                self.carrier[t, idx[rows]] = carry[rows]
        for slot in range(4):
            self.kind[slot, idx] = self._draw(slot, k)
            self.gains[slot, idx] = draw_gains(k, self.gen, strong=self.strong)
            self.style[slot, idx] = draw_styles(k, self.att_styles if ROLE[slot] == 0 else self.def_styles, self.gen)
            self.scale[slot, idx] = 1.0 + self.dr_speed * (2 * self._rand(k) - 1)
        self.delay[idx] = self._rand(k) < self.dr_delay
        self.pending[idx] = 0.0

    def reset_all(self):
        self._reset(torch.arange(self.n, device="cuda"))
        self.ep_steps[:] = (self._rand(self.n) * self.ep_max).long()  # the envs do not all end together
        self.state = self._read()
        self._follow_flags(self.state)
        return self.obs()

    def _read(self):
        rob = []
        for r in self.robots:
            p, v = r.get_dofs_position(), r.get_dofs_velocity()
            yaw = torch.remainder(p[:, 2] + math.pi, 2 * math.pi) - math.pi
            fwd = v[:, 0] * torch.cos(yaw) + v[:, 1] * torch.sin(yaw)
            rob.append(torch.stack([p[:, 0], p[:, 1], yaw, fwd, v[:, 2]], 1))
        return torch.stack(rob)  # (4, n, 5)

    @property
    def t(self):
        return self.steps.float() * self.dt

    def _remaining(self):
        t = self.t
        held = self.hold_until > t
        hold = (self.hold_until - t).clamp(min=0)
        imm = torch.where(held, torch.zeros_like(hold), (self.immune_until - t).clamp(min=0))
        return hold, imm

    def view(self, team, rob):
        """Everything a node of `team` knows, in its team frame."""
        hold, imm = self._remaining()
        if team == 0:
            mine, theirs = rob[0:2], rob[2:4]
            own, enemy = self.flag[0], self.flag[1]
            carry = torch.stack([self.carrier[1] == 0, self.carrier[1] == 1])
            taken = self.carrier[0] >= 0
            order = [0, 1, 2, 3]
        else:
            mine, theirs = to_team_frame(rob[2:4]), to_team_frame(rob[0:2])
            own, enemy = -self.flag[1], -self.flag[0]
            carry = torch.stack([self.carrier[0] == 2, self.carrier[0] == 3])
            taken = self.carrier[1] >= 0
            order = [2, 3, 0, 1]
        return mine, theirs, own, enemy, carry, taken, hold[order], imm[order]

    def _obs_slot(self, slot, rob, rows=None):
        mine, theirs, own, enemy, carry, taken, hold, imm = self.view(TEAM[slot], rob)
        t, prev = self.t, self.prev[slot]
        if rows is not None:
            mine, theirs, carry, hold, imm = (z[:, rows] for z in (mine, theirs, carry, hold, imm))
            own, enemy, taken, t, prev = (z[rows] for z in (own, enemy, taken, t, prev))
        return observation_torch(ROLE[slot], mine, theirs, own, enemy, carry, taken, hold, imm, t, prev)

    def obs(self):
        rob = self.state
        if self.obs_noise > 0:
            rob = rob.clone()
            rob[:, :, :2] += self.obs_noise * torch.randn(4, self.n, 2, device="cuda", generator=self.gen)
        if self.yaw_noise > 0:
            rob = rob.clone()
            rob[:, :, 2] += self.yaw_noise * torch.randn(4, self.n, device="cuda", generator=self.gen)
        return self._obs_slot(self.agent, rob)

    # --- others --------------------------------------------------------------------------------

    @torch.no_grad()
    def _others(self, rob):
        act = torch.zeros(4, self.n, 2, device="cuda")
        for slot in range(4):
            if slot == self.agent:
                continue
            kind = self.kind[slot]
            sc = kind == 1
            if sc.any():
                mine, theirs, own, enemy, carry, taken, hold, imm = self.view(TEAM[slot], rob)
                if ROLE[slot] == 0:
                    a = attacker_actions(mine, theirs, own, enemy, carry, hold, self.gains[slot], self.style[slot])
                else:
                    a = defender_actions(mine, theirs, own, taken, hold, imm, self.gains[slot], self.style[slot],
                                         enemy, carry)
                act[slot] = torch.where(sc.unsqueeze(1), a, act[slot])
            pool = self.pools[ROLE[slot]]
            if pool is None or not len(pool):
                continue
            for k in range(len(pool)):
                rows = (kind == 2 + k).nonzero(as_tuple=False).flatten()
                if rows.numel():
                    act[slot, rows] = pool.act(k, self._obs_slot(slot, rob, rows))
        return act

    # --- referee (copy of CaptureTheFlag.referee) ------------------------------------------------

    def _follow_flags(self, rob):
        for team in (0, 1):
            c = self.carrier[team]
            has = c >= 0
            if has.any():
                r = rob[c.clamp(min=0), torch.arange(self.n, device="cuda")]  # (n, 5)
                pos = torch.stack([r[:, 0] - 0.012 * torch.cos(r[:, 2]), r[:, 1] - 0.012 * torch.sin(r[:, 2])], 1)
                self.flag[team] = torch.where(has.unsqueeze(1), pos, self.flag[team])

    def _referee(self, rob):
        n, t = self.n, self.t
        ev = {k: torch.zeros(4, n, dtype=torch.bool, device="cuda") for k in ("tagged", "tagger", "tag_carrier", "pickup")}
        self._follow_flags(rob)
        x = rob[:, :, 0]
        own_half = lambda team, xx: (xx < 0) if team == 0 else (xx > 0)  # noqa: E731
        for i in range(4):
            for j in range(4):
                ti, tj = TEAM[i], TEAM[j]
                if ti == tj:
                    continue
                d = torch.hypot(rob[i, :, 0] - rob[j, :, 0], rob[i, :, 1] - rob[j, :, 1])
                ok = ((self.hold_until[i] <= t) & (self.hold_until[j] <= t) & (t >= self.immune_until[j])
                      & own_half(ti, x[i]) & own_half(ti, x[j]) & (d < TAG_DISTANCE))
                if not ok.any():
                    continue
                self.hold_until[j] = torch.where(ok, t + HOLD_TIME, self.hold_until[j])
                self.immune_until[j] = torch.where(ok, t + HOLD_TIME + IMMUNE_TIME, self.immune_until[j])
                was = ok & (self.carrier[ti] == j)
                self.carrier[ti] = torch.where(was, torch.full_like(self.carrier[ti], -1), self.carrier[ti])
                self.flag[ti] = torch.where(was.unsqueeze(1), self.home[ti].expand(n, 2), self.flag[ti])
                ev["tagged"][j] |= ok
                ev["tagger"][i] |= ok
                ev["tag_carrier"][i] |= was
        for i in range(4):
            enemy = 1 - TEAM[i]
            d = torch.hypot(rob[i, :, 0] - self.flag[enemy, :, 0], rob[i, :, 1] - self.flag[enemy, :, 1])
            ok = (self.carrier[enemy] < 0) & (d < PICKUP_DISTANCE)
            self.carrier[enemy] = torch.where(ok, torch.full_like(self.carrier[enemy], i), self.carrier[enemy])
            ev["pickup"][i] |= ok
        point = torch.full((n,), -1, dtype=torch.long, device="cuda")
        for i in range(4):
            team = TEAM[i]
            d = torch.hypot(rob[i, :, 0] - self.home[team, 0], rob[i, :, 1] - self.home[team, 1])
            ok = (point < 0) & (self.carrier[1 - team] == i) & (d < BASE_RADIUS)
            point = torch.where(ok, torch.full_like(point, team), point)
        return point, ev

    # --- rewards ---------------------------------------------------------------------------------

    def _phi(self, rob):
        me = rob[self.agent]
        if self.mode == "attack":
            carrying = self.carrier[1] == 0
            d_home = torch.hypot(me[:, 0] - self.home[0, 0], me[:, 1] - self.home[0, 1])
            d_flag = torch.hypot(me[:, 0] - self.flag[1, :, 0], me[:, 1] - self.flag[1, :, 1])
            r = self.arew
            phi = torch.where(carrying, -r.shape_home * d_home, -r.shape_flag * d_flag - r.shape_home * r.d0)
            if r.avoid > 0:
                t = self.t
                d = torch.stack([torch.hypot(me[:, 0] - rob[j, :, 0], me[:, 1] - rob[j, :, 1]) for j in (2, 3)])
                active = torch.stack([self.hold_until[j] <= t for j in (2, 3)])
                d = torch.where(active, d, torch.full_like(d, 9.0)).amin(0)
                exposed = (me[:, 0] > 0) & (self.hold_until[0] <= t) & (t >= self.immune_until[0])
                phi = phi - r.avoid * exposed.float() * (r.avoid_r - d).clamp(min=0)
            return phi
        att = rob[2]
        u = att[:, :2] - self.home[0]
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-3)
        guard = self.home[0] + self.drew.guard_d * u
        taken = self.carrier[0] >= 0
        tgt = torch.where(taken.unsqueeze(1), self.flag[0], guard)
        return -self.drew.shape_guard * torch.hypot(me[:, 0] - tgt[:, 0], me[:, 1] - tgt[:, 1])

    # --- step ------------------------------------------------------------------------------------

    def step(self, action):
        """action: (n, 2) in [-1, 1] for the learning robot."""
        rob = self.state
        acts = self._others(rob)
        action = action.clamp(-1.0, 1.0)
        if self.dr_delay > 0:  # latency: the command applied now is the one of the previous step
            applied = torch.where(self.delay.unsqueeze(1), self.pending, action)
            self.pending = action.clone()
            action = applied
        acts[self.agent] = action
        phi0 = self._phi(rob)
        held = self.hold_until > self.t  # (4, n): commands ignored
        acts = torch.where(held.unsqueeze(2), torch.zeros_like(acts), acts.clamp(-1.0, 1.0))
        v, w = acts[..., 0] * V_MAX * self.scale, acts[..., 1] * W_MAX * self.scale
        for _ in range(self.substeps):
            for i, r in enumerate(self.robots):
                yaw = r.get_dofs_position()[:, 2]
                r.control_dofs_velocity(torch.stack([v[i] * torch.cos(yaw), v[i] * torch.sin(yaw), w[i]], 1), self.dofs)
            self.scene.step()
        self.prev = acts
        self.steps += 1
        self.ep_steps += 1
        self.state = rob = self._read()
        taken_before = self.carrier[0] >= 0
        carrying_before = self.carrier[1] == self.agent
        point, ev = self._referee(rob)
        scored, conceded = point == 0, point == 1
        timeout = (self.steps >= self.max_steps) & (point < 0)  # the kit's 90 s draw: a real end of the round
        truncated = (self.ep_steps >= self.ep_max) & (point < 0) & ~timeout
        phi1 = self._phi(rob)
        a = self.agent
        if self.mode == "attack":
            r = self.arew
            rew = (self.gamma * phi1 - phi0 + r.score * scored.float() + r.conceded * conceded.float()
                   + r.pickup * ev["pickup"][a].float() + r.tagged * ev["tagged"][a].float()
                   + r.tag * ev["tagger"][a].float())
            rew = rew + r.tagged_carrying * (ev["tagged"][a] & carrying_before).float()
            immune_now = (self.hold_until[a] <= self.t) & (self.t < self.immune_until[a])
            rew = rew + r.immune_pickup * (ev["pickup"][a] & immune_now).float()
        else:
            r = self.drew
            enemy_pick = (ev["pickup"][2] | ev["pickup"][3]) & ~taken_before
            rew = (self.gamma * phi1 - phi0 + r.conceded * conceded.float() + r.score * scored.float()
                   + r.tag * ev["tagger"][a].float() + r.tag_carrier * ev["tag_carrier"][a].float()
                   + r.enemy_pickup * enemy_pick.float() + r.tagged * ev["tagged"][a].float()
                   + r.enemy_half * (rob[a, :, 0] > 0).float())
        terminated = scored | conceded | timeout
        done = terminated | truncated
        info = {"scored": scored, "conceded": conceded, "timeout": timeout, "truncated": truncated, "main_kind": self.kind[self.main].clone(),
                "pickup": ev["pickup"][a] if self.mode == "attack" else (ev["pickup"][2] | ev["pickup"][3]),
                "tagged": ev["tagged"][a], "tagger": ev["tagger"][a]}
        info["final_obs"] = self.obs()
        idx = done.nonzero(as_tuple=False).flatten()
        if idx.numel():
            self._reset(idx)
            self.state = self._read()
            self._follow_flags(self.state)
        return self.obs(), rew, terminated, truncated, info
