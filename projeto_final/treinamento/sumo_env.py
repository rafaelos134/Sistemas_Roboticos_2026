"""Vectorized sumo environment for RL, built on Genesis (many rings in parallel on the GPU).

The physics is the same as psr_sim (kit/sim/psr_sim/world.py): the HeRo URDF,
mass, friction, velocity gains and force limits are imported from there. The only
simplification is that the fall from the raised ring is not simulated: an episode
ends as soon as a robot centre leaves the ring (radius 0.40 m), which is exactly
the referee rule of psr_sim.

    hero_0 = the learning agent
    hero_1 = the opponent: scripted (opponents.py) or a frozen copy of the policy
             (PolicyPool); which one is drawn per episode by `opponent_mix`.

All tensors live on the GPU, shape (n_envs, ...).
"""

from __future__ import annotations

import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

SIM_DIR = Path(__file__).resolve().parents[1] / "kit" / "sim"
sys.path.insert(0, str(SIM_DIR))

from psr_sim.world import (  # noqa: E402  (same physics constants as the simulator)
    DT, FORCE_MAX, KV_LIN, KV_YAW, ROBOT_FRICTION, TORQUE_MAX, V_MAX, W_MAX, Z0, hat_mesh, hero_urdf,
)
from psr_sim.scenarios.sumo import RING_RADIUS, ROUND_TIME  # noqa: E402

from opponents import N_PARAMS, SCRIPTED, PolicyPool, sample_params, scripted_actions  # noqa: E402
from sumo_obs import OBS_DIM, observation_torch  # noqa: E402


@dataclass
class RewardConfig:
    """Reward terms. The first three are the base reward asked for the project;
    the shaping terms are potential based (Ng et al., 1999), so they speed up
    learning without changing which policy is optimal. Set them to 0 to train
    with the base reward only."""

    win: float = 10.0  # opponent centre left the ring
    still: float = -0.05  # per step with |own speed| < still_speed (robot standing still)
    self_out: float = -10.0  # own centre left the ring (loses the round)
    draw: float = 0.0  # round ended in a draw (40 s timeout or both out); phase 2/3: -3
    still_speed: float = 0.02  # m/s
    shape_approach: float = 1.0  # potential: -distance(me, opponent)
    shape_push: float = 2.0  # potential: +distance(opponent, ring centre)
    shape_center: float = 0.0  # potential: -distance(me, ring centre); phase 2/3: 1.0
    time_cost: float = 0.0  # per step while the round goes on (phase 4: -0.01). The draw penalty
    # alone arrives after 800 steps and is worth gamma^800 ~ 3e-4 at the start, so looping is free


class SumoEnv:
    def __init__(
        self,
        n_envs: int = 2048,
        control_dt: float = 0.05,  # 20 Hz, the rate of the ROS node
        reward: RewardConfig | None = None,
        gamma: float = 0.99,
        random_start: float = 0.3,  # fraction of episodes that start anywhere in the ring
        clinch_start: float = 0.0,  # fraction that start in contact, face to face, anywhere (phase 5)
        obs_noise: float = 0.002,  # m, noise added to the positions seen by the policy
        show_viewer: bool = False,
        seed: int = 0,
        opponent_mix: dict | None = None,  # {kind name or "pool"/"latest": weight}; default: stopped
        pool: PolicyPool | None = None,
        sensing: str = "truth",  # truth | raw (noisy odometry, no filter) | fusion (noisy sensors + SumoEKF)
        noise_max: float = 2.0,  # noise level drawn per episode in U(0, noise_max) (20 % clean)
        noise_level: float | None = None,  # fixed noise level (evaluation)
    ):
        import genesis as gs

        if not getattr(gs, "_initialized", False):
            gs.init(backend=gs.gpu, logging_level="warning", seed=seed)
        self.gs = gs
        self.n = n_envs
        self.rew = reward or RewardConfig()
        self.gamma = gamma
        self.random_start = random_start
        self.clinch_start = clinch_start
        self.obs_noise = obs_noise
        self.substeps = max(1, round(control_dt / DT))
        self.control_dt = self.substeps * DT
        self.max_steps = int(round(ROUND_TIME / self.control_dt))
        self.gen = torch.Generator(device="cuda").manual_seed(seed)

        self.tmp = tempfile.TemporaryDirectory(prefix="sumo_rl_")
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, 0.0)),
            rigid_options=gs.options.RigidOptions(
                enable_collision=True, enable_self_collision=False, enable_joint_limit=False,
            ),
            viewer_options=gs.options.ViewerOptions(camera_pos=(0.0, -1.6, 1.4), camera_lookat=(0, 0, 0)),
            show_viewer=show_viewer,
        )
        self.robots = []
        for i, rgb in enumerate([(0.2, 0.55, 0.9), (0.95, 0.6, 0.15)]):
            hat = hat_mesh(Path(self.tmp.name) / f"hat_{i}.obj", rgb)
            path = Path(self.tmp.name) / f"hero_{i}.urdf"
            path.write_text(hero_urdf(f"hero_{i}", rgb, hat, with_z=False))
            self.robots.append(self.scene.add_entity(gs.morphs.URDF(file=str(path), fixed=True, pos=(0, 0, Z0))))
        self.scene.build(n_envs=n_envs)

        self.dofs = torch.arange(3, device="cuda")
        kv = torch.tensor([KV_LIN, KV_LIN, KV_YAW], device="cuda")
        fmax = torch.tensor([FORCE_MAX, FORCE_MAX, TORQUE_MAX], device="cuda")
        for r in self.robots:
            r.set_dofs_armature(torch.full((3,), 1e-6, device="cuda"), self.dofs)
            r.set_dofs_kp(torch.zeros(3, device="cuda"), self.dofs)
            r.set_dofs_kv(kv, self.dofs)
            r.set_dofs_force_range(-fmax, fmax, self.dofs)
            r.get_link("base_link").set_friction(ROBOT_FRICTION)

        self.steps = torch.zeros(n_envs, dtype=torch.long, device="cuda")
        self.prev_action = torch.zeros(n_envs, 2, device="cuda")
        self.pool = pool
        self.set_opponent_mix(opponent_mix or {"stopped": 1.0})
        self.opp_kind = torch.zeros(n_envs, dtype=torch.long, device="cuda")
        self.opp_params = torch.zeros(n_envs, N_PARAMS, device="cuda")
        self.opp_prev = torch.zeros(n_envs, 2, device="cuda")
        self.opp_hold = torch.zeros(n_envs, 2, device="cuda")
        self.sensing = sensing
        self.std_start = torch.zeros(n_envs, dtype=torch.bool, device="cuda")
        if sensing != "truth":
            from sumo_fusion import SumoFusionTorch, SumoSensorNoise

            self.noise = SumoSensorNoise(n_envs, self.gen, s_max=noise_max, fixed_level=noise_level)
            self.fus = SumoFusionTorch(n_envs, self.control_dt)
            self.est = [torch.zeros(n_envs, 5, device="cuda"), torch.zeros(n_envs, 5, device="cuda")]
            self.raw_pose = torch.zeros(2, n_envs, 3, device="cuda")  # last odometry poses (diagnostics)
        self.state = None
        self.reset_all()

    # --- state -----------------------------------------------------------------------

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device="cuda")

    def _spawn(self, k: int):
        """Poses (k, 3) for hero_0 and hero_1. Default: the psr_sim start (-0.18 / +0.18,
        small y offset, facing each other) with some yaw noise; a fraction of the
        episodes start at random places so the policy learns to recover near the edge."""
        u = lambda lo, hi: lo + (hi - lo) * self._rand(k)  # noqa: E731
        p0 = torch.stack([torch.full((k,), -0.18, device="cuda"), u(-0.06, 0.06), u(-0.3, 0.3)], 1)
        p1 = torch.stack([torch.full((k,), 0.18, device="cuda"), u(-0.06, 0.06), math.pi + u(-0.3, 0.3)], 1)
        rnd = self._rand(k) < self.random_start
        if rnd.any():
            m = int(rnd.sum())
            for p in (p0, p1):
                r = 0.30 * torch.sqrt(self._rand(m))
                a = 2 * math.pi * self._rand(m)
                p[rnd, 0], p[rnd, 1] = r * torch.cos(a), r * torch.sin(a)
                p[rnd, 2] = 2 * math.pi * self._rand(m) - math.pi
            # keep the two robots apart (> 2 radii + margin)
            d = torch.hypot(p0[:, 0] - p1[:, 0], p0[:, 1] - p1[:, 1])
            bad = rnd & (d < 0.10)
            p1[bad, 0], p1[bad, 1] = -p0[bad, 0], -p0[bad, 1]
            d = torch.hypot(p0[:, 0] - p1[:, 0], p0[:, 1] - p1[:, 1])
            bad = rnd & (d < 0.10)
            p0[bad, 0], p0[bad, 1], p1[bad, 0], p1[bad, 1] = -0.18, 0.0, 0.18, 0.0
        cl = (self._rand(k) < self.clinch_start) & ~rnd
        if cl.any():
            m = int(cl.sum())
            r = 0.25 * torch.sqrt(self._rand(m))
            a = 2 * math.pi * self._rand(m)
            th = 2 * math.pi * self._rand(m) - math.pi  # direction of the line between the two
            cx, cy = r * torch.cos(a), r * torch.sin(a)
            h = 0.038  # half the centre distance: just touching
            p0[cl, 0], p0[cl, 1] = cx - h * torch.cos(th), cy - h * torch.sin(th)
            p1[cl, 0], p1[cl, 1] = cx + h * torch.cos(th), cy + h * torch.sin(th)
            noise = lambda: 0.4 * (2 * self._rand(m) - 1)  # noqa: E731
            p0[cl, 2] = th + noise()
            p1[cl, 2] = th + math.pi + noise()
        return p0, p1, ~(rnd | cl)

    def _reset(self, idx: torch.Tensor):
        if idx.numel() == 0:
            return
        p0, p1, std = self._spawn(idx.numel())
        self.std_start[idx] = std
        if self.sensing != "truth":
            self.noise.reset(idx, std)
            self.fus.forget(idx, std)
        for r, p in zip(self.robots, (p0, p1)):
            r.set_dofs_position(p, self.dofs, envs_idx=idx, zero_velocity=True)
        self.steps[idx] = 0
        self.prev_action[idx] = 0.0
        self.opp_kind[idx] = self._draw_kinds(idx.numel())
        self.opp_params[idx] = sample_params(self.opp_kind[idx], self.gen)
        self.opp_prev[idx] = 0.0
        self.opp_hold[idx] = 2 * self._rand(idx.numel(), 2) - 1

    # --- opponents ----------------------------------------------------------------------

    def set_opponent_mix(self, mix: dict):
        """mix: weights by name: the scripted kinds (opponents.SCRIPTED), "pool" (uniform
        over the frozen checkpoints except the last slot) and "latest" (the last slot,
        refreshed with the current policy)."""
        names, weights = [], []
        for k, wgt in mix.items():
            if wgt <= 0:
                continue
            if k not in SCRIPTED and k not in ("pool", "latest"):
                raise ValueError("unknown opponent %r" % k)
            names.append(k)
            weights.append(float(wgt))
        self.mix_names = names
        self.mix_p = torch.tensor(weights, device="cuda") / sum(weights)

    def _draw_kinds(self, k: int) -> torch.Tensor:
        pick = torch.multinomial(self.mix_p, k, replacement=True, generator=self.gen)
        kinds = torch.zeros(k, dtype=torch.long, device="cuda")
        base = len(SCRIPTED)
        npool = len(self.pool) if self.pool is not None else 0
        for j, name in enumerate(self.mix_names):
            rows = pick == j
            if not rows.any():
                continue
            if name in SCRIPTED:
                kinds[rows] = SCRIPTED.index(name)
            elif name == "latest" and npool:
                kinds[rows] = base + npool - 1
            elif name == "pool" and npool:
                old = max(1, npool - 1)  # all slots but the "latest" one (if there are 2+)
                kinds[rows] = base + torch.randint(0, old, (int(rows.sum()),), device="cuda", generator=self.gen)
        return kinds

    def _opponent_actions(self, me, opp):
        # RANDOM: draw a new held action with probability 1/20 per step (~1 s)
        redraw = self._rand(self.n) < 0.05
        self.opp_hold[redraw] = 2 * self._rand(int(redraw.sum()), 2) - 1
        act = scripted_actions(self.opp_kind, self.opp_params, opp, me, self.opp_hold)
        if self.pool is not None and len(self.pool):
            pol = self.pool.actions(self.opp_kind, opp, me, self.opp_prev)
            act = torch.where((self.opp_kind >= len(SCRIPTED)).unsqueeze(1), pol, act)
        return act

    def reset_all(self):
        self._reset(torch.arange(self.n, device="cuda"))
        self.state = self._read()
        self._sense(torch.arange(self.n, device="cuda"))
        return self.obs()

    def _sense(self, idx):
        """Noisy sensors -> the state the node would see (raw odometry, or the SumoEKF estimate)."""
        if self.sensing == "truth" or idx.numel() == 0:
            return
        me, opp = self.state
        pose, tw, ranges = self.noise.measure(me[idx], opp[idx], self.control_dt, idx)
        self.raw_pose[:, idx] = pose
        if self.sensing == "raw":
            em, eo = pose[0], pose[1]
        else:
            em, eo = self.fus.step(pose, tw, ranges, idx)
        self.est[0][idx] = torch.cat([em, tw[0]], 1)
        self.est[1][idx] = torch.cat([eo, tw[1]], 1)

    def critic_obs(self):
        """Noise-free observation (asymmetric critic), None when the policy already sees the truth."""
        if self.sensing == "truth":
            return None
        me, opp = self.state
        return observation_torch(me, opp, self.prev_action)

    def _read(self):
        pos = [r.get_dofs_position() for r in self.robots]  # (n, 3): x, y, yaw
        vel = [r.get_dofs_velocity() for r in self.robots]
        out = []
        for p, v in zip(pos, vel):
            yaw = torch.remainder(p[:, 2] + math.pi, 2 * math.pi) - math.pi
            fwd = v[:, 0] * torch.cos(yaw) + v[:, 1] * torch.sin(yaw)  # like odom twist.linear.x
            out.append(torch.stack([p[:, 0], p[:, 1], yaw, fwd, v[:, 2]], 1))
        return out  # [me (n,5), opp (n,5)]

    def obs(self):
        if self.sensing != "truth":
            return observation_torch(self.est[0], self.est[1], self.prev_action)
        me, opp = self.state
        if self.obs_noise > 0:
            me = me.clone()
            opp = opp.clone()
            me[:, :2] += self.obs_noise * torch.randn(self.n, 2, device="cuda", generator=self.gen)
            opp[:, :2] += self.obs_noise * torch.randn(self.n, 2, device="cuda", generator=self.gen)
        return observation_torch(me, opp, self.prev_action)

    # --- step -------------------------------------------------------------------------

    def _potential(self, me, opp):
        d_mo = torch.hypot(me[:, 0] - opp[:, 0], me[:, 1] - opp[:, 1])
        d_oc = torch.hypot(opp[:, 0], opp[:, 1])
        d_mc = torch.hypot(me[:, 0], me[:, 1])
        return -self.rew.shape_approach * d_mo + self.rew.shape_push * d_oc - self.rew.shape_center * d_mc

    def step(self, action: torch.Tensor):
        """action (n, 2) in [-1, 1]: (linear, angular) as a fraction of V_MAX / W_MAX."""
        a = action.clamp(-1.0, 1.0)
        v = a[:, 0] * V_MAX
        w = a[:, 1] * W_MAX
        me, opp = self.state
        phi0 = self._potential(me, opp)
        oa = self._opponent_actions(me, opp).clamp(-1.0, 1.0)
        ov, ow = oa[:, 0] * V_MAX, oa[:, 1] * W_MAX
        for _ in range(self.substeps):
            for robot, lin, ang in ((self.robots[0], v, w), (self.robots[1], ov, ow)):
                yaw = robot.get_dofs_position()[:, 2]
                target = torch.stack([lin * torch.cos(yaw), lin * torch.sin(yaw), ang], 1)
                robot.control_dofs_velocity(target, self.dofs)
            self.scene.step()
        self.opp_prev = oa
        self.state = self._read()
        me, opp = self.state
        self.steps += 1
        self.prev_action = a

        me_out = torch.hypot(me[:, 0], me[:, 1]) > RING_RADIUS
        opp_out = torch.hypot(opp[:, 0], opp[:, 1]) > RING_RADIUS
        win = opp_out & ~me_out
        lose = me_out & ~opp_out
        timeout = (self.steps >= self.max_steps) & ~(me_out | opp_out)
        draw = (me_out & opp_out) | timeout
        # with a draw penalty the 40 s limit is a real end of the round (terminal);
        # without it (phase 1) it is a truncation and the critic bootstraps
        if self.rew.draw != 0.0:
            terminated, truncated = me_out | opp_out | timeout, torch.zeros_like(timeout)
        else:
            terminated, truncated = me_out | opp_out, timeout

        phi1 = torch.where(terminated, torch.zeros_like(phi0), self._potential(me, opp))
        reward = self.gamma * phi1 - phi0  # potential shaping
        reward = reward + self.rew.win * win.float() + self.rew.self_out * lose.float() + self.rew.draw * draw.float()
        reward = reward + self.rew.still * (me[:, 3].abs() < self.rew.still_speed).float()
        reward = reward + self.rew.time_cost

        info = {"win": win, "lose": lose, "draw": draw, "truncated": truncated, "opp_kind": self.opp_kind.clone()}
        # observation of the final state, for bootstrapping truncated episodes
        self._sense(torch.arange(self.n, device="cuda"))
        info["final_obs"] = self.obs()
        info["final_cobs"] = self.critic_obs()
        done = terminated | truncated
        idx = done.nonzero(as_tuple=False).flatten()
        if idx.numel():
            self._reset(idx)
            self.state = self._read()
            self._sense(idx)
        return self.obs(), reward, terminated, truncated, info
