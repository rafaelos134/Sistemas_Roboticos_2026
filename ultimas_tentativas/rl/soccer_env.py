"""Vectorized soccer environment (3 x 3 HeRo robots + golf ball) on Genesis, for RL.

Same physics as psr_sim (kit/sim/psr_sim): the robot URDF, the ball (planar sphere
with Coulomb friction as a force limit), the walls of the scenario and all constants are
imported from there. A goal ends the episode, like the referee of psr_sim.

Team frame: every team sees the field as if it attacked +x. For the red team the
states are rotated by 180 deg (x, y -> -x, -y; yaw -> yaw + pi). Velocity commands are
in the robot frame, so they need no conversion. One network therefore plays on both sides.

Roles (ids inside a team): 0 = goalkeeper / defender, 1 and 2 = attackers.

Who controls each robot is set per environment:
    blue attackers (1, 2): "learn" (the agent) or a frozen policy
    blue keeper (0):       "stopped" | "script" | "policy" (defender network)
    red team (3, 4, 5):    "stopped" | "script" | a frozen copy of our team (attacker + defender nets)

The agents of an environment are returned stacked: obs (n_agents * n_envs, OBS_DIM),
with agent k of env e at row k * n_envs + e.
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

from psr_sim.scenarios.soccer import FIELD_L, FIELD_W, GOAL_W, Soccer  # noqa: E402
from psr_sim.world import (  # noqa: E402
    DT, FORCE_MAX, KV_LIN, KV_YAW, ROBOT_FRICTION, TORQUE_MAX, V_MAX, W_MAX, Z0, World, hat_mesh, hero_urdf, puck_urdf,
)

from soccer_obs import OBS_DIM, observation_torch, to_team_frame  # noqa: E402
from soccer_scripts import keeper_actions, team_script_actions  # noqa: E402

HALF_L, HALF_W = FIELD_L / 2, FIELD_W / 2


@dataclass
class SoccerReward:
    """Team reward of the attackers (both attackers get the same value)."""

    goal: float = 10.0  # we scored
    conceded: float = -10.0  # they scored
    still: float = -0.02  # per attacker per step with |v| < still_speed
    still_speed: float = 0.02
    # potential-based shaping (Ng et al., 1999): Phi = -w_ball * d(ball, their goal) - w_reach * min_i d(attacker_i, ball)
    shape_ball: float = 3.0
    shape_reach: float = 1.0
    shape_own: float = 0.0  # optional: +w * d(ball, our goal) (defender uses it)
    # potential: -w * how deep the ball is in the band next to the long walls (|y| > HALF_W - wall_band):
    # a ball against a wall can only slide along it, so the attackers should keep it in the middle
    shape_wall: float = 0.0
    wall_band: float = 0.15
    # not potential based, on purpose: a jammed ball (moved < stall_dist in stall_s) costs `stall` per step,
    # because in a real match nothing resets a jam (psr_sim has no time limit or fouls)
    stall: float = -0.05
    stall_s: float = 2.0
    stall_dist: float = 0.03


@dataclass
class DefenderReward:
    """Reward of the defender (blue keeper, id 0). Purely defensive."""

    conceded: float = -10.0
    scored: float = 0.0  # our goals are not its job
    clear: float = 0.5  # per step in contact with the ball while it leaves towards +x faster than clear_speed
    clear_speed: float = 0.15
    leave_zone: float = -0.05  # per step outside the defensive zone (x > zone_x in team frame)
    zone_x: float = -0.30
    # potential: +w_ball * d(ball, our goal) - w_guard * distance to the guard point (between ball and goal)
    shape_ball: float = 2.0
    shape_guard: float = 1.0
    stall_own_half: float = -0.05  # per step while the ball is jammed (see SoccerReward.stall) in our half


class SoccerEnv:
    def __init__(self, n_envs=1024, control_dt=0.05, episode_s=60.0, mode="attack", gamma=0.99,
                 reward: SoccerReward | None = None, def_reward: DefenderReward | None = None,
                 random_start=0.3, easy_start=0.3, obs_noise=0.002, seed=0, red_mix: dict | None = None, keeper="stopped",
                 att_pool=None, def_pool=None, critic="sym"):
        import genesis as gs

        if not getattr(gs, "_initialized", False):
            gs.init(backend=gs.gpu, logging_level="warning", seed=seed)
        self.gs, self.n, self.gamma = gs, n_envs, gamma
        self.mode = mode  # "attack": the agents are blue 1, 2; "defend": the agent is blue 0
        self.rew = reward or SoccerReward()
        self.drew = def_reward or DefenderReward()
        self.random_start, self.easy_start, self.obs_noise = random_start, easy_start, obs_noise
        self.substeps = max(1, round(control_dt / DT))
        self.control_dt = self.substeps * DT
        self.max_steps = int(round(episode_s / self.control_dt))
        self.gen = torch.Generator(device="cuda").manual_seed(seed)
        self.keeper = keeper
        self.att_pool, self.def_pool = att_pool, def_pool
        self.set_red_mix(red_mix or {"stopped": 1.0})

        self.tmp = tempfile.TemporaryDirectory(prefix="soccer_rl_")
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, 0.0)),
            rigid_options=gs.options.RigidOptions(enable_collision=True, enable_self_collision=False,
                                                  enable_joint_limit=False),
            show_viewer=False,
        )
        scenario = Soccer(6, 0)
        fake = types.SimpleNamespace(gs=gs, scene=self.scene, tmp=self.tmp)
        for s in scenario.statics():
            if s.solid:  # walls only: the paint does not collide
                World._add_static(fake, s)
        self.robots = []
        for i, rgb in enumerate(scenario.colors()):
            hat = hat_mesh(Path(self.tmp.name) / f"hat_{i}.obj", rgb)
            path = Path(self.tmp.name) / f"hero_{i}.urdf"
            path.write_text(hero_urdf(f"hero_{i}", rgb, hat))
            self.robots.append(self.scene.add_entity(gs.morphs.URDF(file=str(path), fixed=True, pos=(0, 0, Z0))))
        spec = scenario.puck_spec()
        path = Path(self.tmp.name) / "puck.urdf"
        path.write_text(puck_urdf(spec.radius, spec.mass, spec.color))
        self.puck = self.scene.add_entity(gs.morphs.URDF(file=str(path), fixed=True, pos=(0, 0, Z0)))
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
        fric = spec.friction * spec.mass * 9.81
        self.puck.set_dofs_armature(c([1e-6] * 3), self.dofs)
        self.puck.set_dofs_kp(c([0.0] * 3), self.dofs)
        self.puck.set_dofs_kv(c([1.0, 1.0, 1e-3]), self.dofs)
        self.puck.set_dofs_force_range(c([-fric, -fric, -fric * spec.radius]), c([fric, fric, fric * spec.radius]),
                                       self.dofs)
        self.puck.get_link("puck_link").set_friction(0.1)
        self.ball_r = spec.radius

        n = n_envs
        self.steps = torch.zeros(n, dtype=torch.long, device="cuda")
        self.prev = torch.zeros(6, n, 2, device="cuda")  # previous action of every robot
        self.red_kind = torch.zeros(n, dtype=torch.long, device="cuda")
        self.script_p = torch.zeros(6, n, 4, device="cuda")  # per-robot script gains
        self.zero3 = torch.zeros(n, 3, device="cuda")
        self.ball_ref = torch.zeros(n, 2, device="cuda")  # ball position when it last moved stall_dist
        self.ball_still = torch.zeros(n, dtype=torch.long, device="cuda")  # steps since then
        self.state = None
        # asymmetric critic (soccer_critic.py): "sym" = the critic sees the actor observation only
        self.critic = critic
        self.grid = None
        if critic == "asym_dij":
            if mode != "attack":
                raise ValueError("the Dijkstra features are defined for the attackers only")
            from soccer_critic import GridDijkstra
            self.grid = GridDijkstra(n)
        self.reset_all()

    # --- opponents ------------------------------------------------------------------------------

    def set_red_mix(self, mix: dict):
        """{"stopped": w, "script": w, "pool": w, "latest": w}"""
        self.mix_names = [k for k, w in mix.items() if w > 0]
        self.mix_p = torch.tensor([float(mix[k]) for k in self.mix_names], device="cuda")
        self.mix_p /= self.mix_p.sum()

    def _draw_red(self, k):
        pick = torch.multinomial(self.mix_p, k, replacement=True, generator=self.gen)
        kinds = torch.zeros(k, dtype=torch.long, device="cuda")  # 0 stopped, 1 script, 2+j pool slot j
        npool = len(self.att_pool) if self.att_pool is not None else 0
        for j, name in enumerate(self.mix_names):
            rows = pick == j
            if not rows.any():
                continue
            if name == "script":
                kinds[rows] = 1
            elif name == "latest" and npool:
                kinds[rows] = 2 + npool - 1
            elif name == "pool" and npool:
                old = max(1, npool - 1)
                kinds[rows] = 2 + torch.randint(0, old, (int(rows.sum()),), device="cuda", generator=self.gen)
        return kinds

    # --- state --------------------------------------------------------------------------------------

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device="cuda")

    def _spawn(self, k):
        """Robot poses (6, k, 3) and ball (k, 2): kickoff like psr_sim, or random positions."""
        u = lambda lo, hi: lo + (hi - lo) * self._rand(k)  # noqa: E731
        z = torch.zeros(k, device="cuda")
        blue = [(-0.78 + z, z, z), (-0.35 + u(-.05, .05), 0.25 + u(-.05, .05), z), (-0.35 + u(-.05, .05), -0.25 + u(-.05, .05), z)]
        red = [(0.78 + z, z, z + math.pi), (0.35 + u(-.05, .05), -0.25 + u(-.05, .05), z + math.pi),
               (0.35 + u(-.05, .05), 0.25 + u(-.05, .05), z + math.pi)]
        poses = torch.stack([torch.stack(p, 1) for p in blue + red])  # (6, k, 3)
        ball = torch.stack([z, u(-0.05, 0.05)], 1)
        rnd = self._rand(k) < self.random_start
        m = int(rnd.sum())
        if m:
            # random: robots anywhere (keepers near their goal), ball anywhere, no overlaps (resolved by retries)
            for _ in range(8):
                pr = torch.stack([(2 * self._rand(6, m) - 1) * (HALF_L - 0.08), (2 * self._rand(6, m) - 1) * (HALF_W - 0.08),
                                  (2 * self._rand(6, m) - 1) * math.pi], 2)
                pr[0, :, 0] = -HALF_L + 0.06 + 0.25 * self._rand(m)
                pr[3, :, 0] = HALF_L - 0.06 - 0.25 * self._rand(m)
                br = torch.stack([(2 * self._rand(m) - 1) * (HALF_L - 0.10), (2 * self._rand(m) - 1) * (HALF_W - 0.06)], 1)
                pts = torch.cat([pr[:, :, :2], br.unsqueeze(0)], 0)  # (7, m, 2)
                d = torch.cdist(pts.permute(1, 0, 2), pts.permute(1, 0, 2))  # (m, 7, 7)
                d = d + 10 * torch.eye(7, device="cuda")
                ok = (d[:, :6, :6].amin((1, 2)) > 0.09) & (d[:, 6, :6].amin(1) > 0.07)
                if ok.all():
                    break
                pr[:, ~ok] = poses[:, rnd][:, ~ok]
                br[~ok] = ball[rnd][~ok]
            # half of the random starts: ball against a wall or in a corner (where play gets stuck)
            wall = self._rand(m) < 0.5
            side = (self._rand(m) * 4).long()
            br[wall & (side == 0), 1] = HALF_W - 0.025
            br[wall & (side == 1), 1] = -(HALF_W - 0.025)
            br[wall & (side == 2), 0] = HALF_L - 0.025
            br[wall & (side == 3), 0] = -(HALF_L - 0.025)
            near_goal = (br[:, 0].abs() > HALF_L - 0.03) & (br[:, 1].abs() < GOAL_W / 2 + 0.03)
            br[near_goal, 1] = torch.sign(br[near_goal, 1] + 1e-6) * (GOAL_W / 2 + 0.05)  # not inside the goal mouth
            for i in range(6):  # a robot on top of the moved ball: move it 0.12 m towards the centre
                d = torch.hypot(pr[i, :, 0] - br[:, 0], pr[i, :, 1] - br[:, 1])
                bad = d < 0.08
                pr[i, bad, 0] = br[bad, 0] - 0.12 * torch.sign(br[bad, 0])
                pr[i, bad, 1] = br[bad, 1] - 0.12 * torch.sign(br[bad, 1])
            poses[:, rnd] = pr
            ball[rnd] = br
        # easy starts (curriculum): ball 0.05..1.2 m in front of the goal we attack (team frame +x for the
        # attack mode, our own goal for the defend mode) with one attacker right behind it, facing the goal
        easy = (~rnd) & (self._rand(k) < self.easy_start / max(1e-6, 1.0 - self.random_start))
        m = int(easy.sum())
        if m:
            dist = 0.05 + 1.15 * self._rand(m) ** 1.5
            by = (2 * self._rand(m) - 1) * 0.18
            if self.mode == "attack":
                bx = HALF_L - dist
                ux, uy = HALF_L - bx, -by
                who = 1 + (self._rand(m) < 0.5).long()  # attacker 1 or 2
            else:  # red attacker 4 or 5 behind the ball, shooting at our goal (-x)
                bx = -HALF_L + dist
                ux, uy = -HALF_L - bx, -by
                who = 4 + (self._rand(m) < 0.5).long()
            un = torch.hypot(ux, uy).clamp_min(1e-6)
            back = 0.06 + 0.06 * self._rand(m)
            px, py = bx - back * ux / un, by - back * uy / un
            yaw = torch.atan2(uy, ux) + 0.3 * (2 * self._rand(m) - 1)
            sub = poses[:, easy]
            rows = torch.arange(m, device="cuda")
            sub[who, rows] = torch.stack([px, py, yaw], 1)
            poses[:, easy] = sub
            ball[easy] = torch.stack([bx, by], 1)
            # move away any other robot that overlaps the ball or the shooter
            for i in range(6):
                pi = poses[i, easy]
                close = (torch.hypot(pi[:, 0] - bx, pi[:, 1] - by) < 0.08) & (who != i)
                close |= (torch.hypot(pi[:, 0] - px, pi[:, 1] - py) < 0.09) & (who != i)
                if close.any() and i not in (0, 3):
                    pi[close, 1] = torch.where(by[close] > 0, by[close] - 0.25, by[close] + 0.25)
                    poses[i, easy] = pi
        return poses, ball

    def _reset(self, idx):
        if idx.numel() == 0:
            return
        k = idx.numel()
        poses, ball = self._spawn(k)
        for i, r in enumerate(self.robots):
            r.set_dofs_position(poses[i], self.dofs, envs_idx=idx, zero_velocity=True)
        bp = torch.cat([ball, torch.zeros(k, 1, device="cuda")], 1)
        self.puck.set_dofs_position(bp, self.dofs, envs_idx=idx, zero_velocity=True)
        self.steps[idx] = 0
        self.prev[:, idx] = 0.0
        self.ball_ref[idx] = ball
        self.ball_still[idx] = 0
        self.red_kind[idx] = self._draw_red(k)
        u = lambda lo, hi: lo + (hi - lo) * self._rand(6, k)  # noqa: E731
        self.script_p[:, idx] = torch.stack([u(2.0, 6.0), u(0.3, 0.7), u(0.6, 1.0), u(0.06, 0.10)], 2)

    def reset_all(self):
        self._reset(torch.arange(self.n, device="cuda"))
        self.state = self._read()
        return self.obs()

    def _read(self):
        rob = []
        for r in self.robots:
            p, v = r.get_dofs_position(), r.get_dofs_velocity()
            yaw = torch.remainder(p[:, 2] + math.pi, 2 * math.pi) - math.pi
            fwd = v[:, 0] * torch.cos(yaw) + v[:, 1] * torch.sin(yaw)
            rob.append(torch.stack([p[:, 0], p[:, 1], yaw, fwd, v[:, 2]], 1))
        bp, bv = self.puck.get_dofs_position(), self.puck.get_dofs_velocity()
        ball = torch.stack([bp[:, 0], bp[:, 1], bv[:, 0], bv[:, 1]], 1)
        return torch.stack(rob), ball  # (6, n, 5), (n, 4)

    def _team_views(self, rob, ball):
        """States in the frame of each team: blue as is, red rotated by 180 deg."""
        blue = (rob[:3], rob[3:], ball)
        red = (to_team_frame(rob[3:]), to_team_frame(rob[:3]), to_team_frame(ball, is_ball=True))
        return blue, red

    def _obs_of(self, view, role, prev):
        mine, theirs, ball = view
        return observation_torch(mine, theirs, ball, role, prev)

    def obs(self):
        rob, ball = self.state
        if self.obs_noise > 0:
            rob = rob.clone()
            ball = ball.clone()
            rob[:, :, :2] += self.obs_noise * torch.randn(6, self.n, 2, device="cuda", generator=self.gen)
            ball[:, :2] += self.obs_noise * torch.randn(self.n, 2, device="cuda", generator=self.gen)
        blue, _ = self._team_views(rob, ball)
        if self.mode == "attack":
            return torch.cat([self._obs_of(blue, 1, self.prev[1]), self._obs_of(blue, 2, self.prev[2])], 0)
        return self._obs_of(blue, 0, self.prev[0])

    def critic_obs(self):
        """Privileged critic input (noise free), rows stacked like obs(). None for the symmetric critic."""
        if self.critic == "sym":
            return None
        from soccer_critic import dijkstra_features, field_view, still_features

        rob, ball = self.state
        blue, _ = self._team_views(rob, ball)
        roles = (1, 2) if self.mode == "attack" else (0,)
        common = torch.cat([field_view(*blue), still_features(self.ball_still, self.control_dt, self.rew.stall_s)], 1)
        dij = dijkstra_features(self.grid, *blue, roles) if self.grid is not None else None
        rows = []
        for r in roles:
            parts = [self._obs_of(blue, r, self.prev[r]), common]
            if dij is not None:
                parts.append(dij[r])
            rows.append(torch.cat(parts, 1))
        return torch.cat(rows, 0)

    # --- step ------------------------------------------------------------------------------------------

    @torch.no_grad()
    def _others(self, rob, ball):
        """Actions (6, n, 2) of every robot not controlled by the learning agent."""
        act = torch.zeros(6, self.n, 2, device="cuda")
        blue, red = self._team_views(rob, ball)
        # red team
        is_script = self.red_kind == 1
        if is_script.any():
            a = team_script_actions(red, self.script_p[3:])
            act[3:] = torch.where(is_script.view(1, -1, 1), a, act[3:])
        if self.att_pool is not None and len(self.att_pool):
            for k in range(len(self.att_pool)):
                rows = (self.red_kind == 2 + k).nonzero(as_tuple=False).flatten()
                if rows.numel() == 0:
                    continue
                v = tuple(x[:, rows] if x.dim() == 3 else x[rows] for x in red)
                for role in (1, 2):
                    o = self._obs_of(v, role, self.prev[3 + role][rows])
                    act[3 + role, rows] = self.att_pool.act(k, o)
                if self.def_pool is not None and len(self.def_pool):
                    o = self._obs_of(v, 0, self.prev[3][rows])
                    act[3, rows] = self.def_pool.act(min(k, len(self.def_pool) - 1), o)
                else:
                    act[3, rows] = keeper_actions(tuple(x[:, rows] if x.dim() == 3 else x[rows] for x in red),
                                                  self.script_p[3][rows])
        # blue keeper (attack mode) / blue attackers (defend mode)
        if self.mode == "attack":
            if self.keeper == "script":
                act[0] = keeper_actions(blue, self.script_p[0])
            elif self.keeper == "policy" and self.def_pool is not None and len(self.def_pool):
                act[0] = self.def_pool.act(len(self.def_pool) - 1, self._obs_of(blue, 0, self.prev[0]))
        else:
            if self.att_pool is not None and len(self.att_pool):
                for role in (1, 2):
                    act[role] = self.att_pool.act(len(self.att_pool) - 1, self._obs_of(blue, role, self.prev[role]))
        return act

    def _phi_attack(self, rob, ball):
        # distance to a point inside their goal, plus a penalty for being outside the goal band (|y| > 0.2):
        # without it the corners next to the goal are a local optimum (ball pinned there, nobody scores)
        d_goal = torch.hypot(HALF_L + 0.05 - ball[:, 0], ball[:, 1]) + 0.8 * torch.clamp(ball[:, 1].abs() - GOAL_W / 2, min=0.0)
        reach = torch.minimum(torch.hypot(rob[1, :, 0] - ball[:, 0], rob[1, :, 1] - ball[:, 1]),
                              torch.hypot(rob[2, :, 0] - ball[:, 0], rob[2, :, 1] - ball[:, 1]))
        d_own = torch.hypot(-HALF_L - ball[:, 0], ball[:, 1])
        wall = torch.clamp(ball[:, 1].abs() - (HALF_W - self.rew.wall_band), min=0.0)
        return (-self.rew.shape_ball * d_goal - self.rew.shape_reach * reach + self.rew.shape_own * d_own
                - self.rew.shape_wall * wall / self.rew.wall_band)

    def _phi_defend(self, rob, ball):
        d_own = torch.hypot(-HALF_L - ball[:, 0], ball[:, 1])
        gx, gy = -HALF_L, 0.0  # guard point: on the segment goal -> ball, 0.12 m in front of the goal (or the ball)
        dx, dy = ball[:, 0] - gx, ball[:, 1] - gy
        dist = torch.hypot(dx, dy).clamp_min(1e-6)
        s = torch.clamp(0.12 / dist, max=1.0)
        px, py = gx + s * dx, gy + s * dy
        guard = torch.hypot(rob[0, :, 0] - px, rob[0, :, 1] - py)
        return self.drew.shape_ball * d_own - self.drew.shape_guard * guard

    def step(self, action):
        """action: (2n, 2) in attack mode (rows: attacker 1 then attacker 2), (n, 2) in defend mode."""
        a = action.clamp(-1.0, 1.0)
        rob, ball = self.state
        acts = self._others(rob, ball)
        if self.mode == "attack":
            acts[1], acts[2] = a[: self.n], a[self.n:]
            phi0 = self._phi_attack(rob, ball)
        else:
            acts[0] = a
            phi0 = self._phi_defend(rob, ball)
        acts = acts.clamp(-1.0, 1.0)
        v, w = acts[..., 0] * V_MAX, acts[..., 1] * W_MAX  # (6, n)
        for _ in range(self.substeps):
            for i, r in enumerate(self.robots):
                yaw = r.get_dofs_position()[:, 2]
                r.control_dofs_velocity(torch.stack([v[i] * torch.cos(yaw), v[i] * torch.sin(yaw), w[i]], 1), self.dofs)
            self.puck.control_dofs_velocity(self.zero3, self.dofs)
            self.scene.step()
        self.prev = acts
        self.state = self._read()
        rob, ball = self.state
        self.steps += 1

        moved = torch.hypot(ball[:, 0] - self.ball_ref[:, 0], ball[:, 1] - self.ball_ref[:, 1]) > self.rew.stall_dist
        self.ball_ref[moved] = ball[moved, :2]
        self.ball_still = torch.where(moved, torch.zeros_like(self.ball_still), self.ball_still + 1)
        stalled = self.ball_still > int(self.rew.stall_s / self.control_dt)
        inside = ball[:, 1].abs() < GOAL_W / 2
        scored = (ball[:, 0] > HALF_L) & inside
        conceded = (ball[:, 0] < -HALF_L) & inside
        terminated = scored | conceded
        truncated = (self.steps >= self.max_steps) & ~terminated

        if self.mode == "attack":
            phi1 = torch.where(terminated, torch.zeros_like(phi0), self._phi_attack(rob, ball))
            r = self.gamma * phi1 - phi0 + self.rew.goal * scored.float() + self.rew.conceded * conceded.float()
            r = r + self.rew.stall * stalled.float()
            r1 = r + self.rew.still * (rob[1, :, 3].abs() < self.rew.still_speed).float()
            r2 = r + self.rew.still * (rob[2, :, 3].abs() < self.rew.still_speed).float()
            reward = torch.cat([r1, r2])
        else:
            phi1 = torch.where(terminated, torch.zeros_like(phi0), self._phi_defend(rob, ball))
            # ball leaving our side fast while the defender is touching it (per step of contact)
            cleared = (ball[:, 2] > self.drew.clear_speed) & \
                      (torch.hypot(rob[0, :, 0] - ball[:, 0], rob[0, :, 1] - ball[:, 1]) < 0.08)
            reward = (self.gamma * phi1 - phi0 + self.drew.conceded * conceded.float() + self.drew.scored * scored.float()
                      + self.drew.clear * cleared.float() + self.drew.leave_zone * (rob[0, :, 0] > self.drew.zone_x).float()
                      + self.drew.stall_own_half * (stalled & (ball[:, 0] < 0)).float())

        info = {"scored": scored, "conceded": conceded, "timeout": truncated, "red_kind": self.red_kind.clone(),
                "stalled": stalled}
        info["final_obs"] = self.obs()
        info["final_cobs"] = self.critic_obs()
        done = terminated | truncated
        idx = done.nonzero(as_tuple=False).flatten()
        if idx.numel():
            self._reset(idx)
            self.state = self._read()
        rep = 2 if self.mode == "attack" else 1
        return self.obs(), reward, terminated.repeat(rep), truncated.repeat(rep), info
