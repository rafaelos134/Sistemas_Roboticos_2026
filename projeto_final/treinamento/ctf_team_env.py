"""Capture the flag, TEAM training: the blue attacker (robot 0) and the blue defender (robot 1) learn together.

Same physics, referee, opponents (stopped / script styles / frozen pools) and domain randomization as
CtfEnv; both red robots are drawn from `main_mix`. Each blue robot keeps its own network (the exported node
has one network per role), but the rewards share the team result, so the defender is no longer told to stay
home: it can learn to go out (draw the enemy defender, a 2-vs-1 at the flag, take the flag itself).

Per robot reward:
  shared:    our point +score, their point +conceded, a pickup of the enemy flag by any of us +team_pickup
  attacker:  its own AttackReward events and potential (flag / home / avoid), as in CtfEnv
  defender:  tag / tag of the carrier / its own pickup / held, enemy pickup; potential: carrying -> home,
             my flag taken -> the flag (the carrier), else (weak) the guard point
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ctf_env import TEAM, V_MAX, W_MAX, CtfEnv
from ctf_scripts import GAINS_REF, defender_actions


@dataclass
class TeamReward:
    score: float = 10.0
    conceded: float = -10.0
    team_pickup: float = 1.5  # any of us picks up the enemy flag (both robots)
    # defender
    d_pickup: float = 3.0  # the defender itself picks up the enemy flag
    d_tag: float = 2.0
    d_tag_carrier: float = 3.0
    d_enemy_pickup: float = -3.0
    d_tagged: float = -1.0
    d_tagged_carrying: float = -2.0
    d_home: float = 2.0  # potential, carrying: -d_home * d(me, home)
    d_chase: float = 1.0  # potential, my flag taken: -d_chase * d(me, my flag)
    d_guard: float = 0.3  # potential, otherwise: -d_guard * d(me, guard point)
    guard_d: float = 0.10
    # "a draw is bad": a cost per second without a point (both robots) and an extra at the 90 s draw
    time_cost: float = 0.0  # per second
    draw: float = 0.0


class CtfTeamEnv(CtfEnv):
    def __init__(self, team_reward=None, dual=False, gate=None, def_script=False, **kw):
        """dual: robot 1 is a second ATTACKER (attacker reward and potential, as robot 0).
        gate: supervisor of robot 1 (as export_ctf.py --def-gate): when my flag is taken or an active opponent
        has x < gate, robot 1 plays the scripted defender (reference gains) unless it carries the enemy flag;
        info["gated"] marks those steps (their actions are not the policy's: masked in the PPO loss)."""
        self.dual, self.gate = dual, gate
        # def_script: robot 1 is ALWAYS the standard scripted defender (reference gains): guard point on the
        # segment own flag -> their attacker, intercept the carrier (as export_ctf.py --def-script)
        self.def_script = def_script
        kw["mode"] = "attack"  # robot 0 = self.agent for the attacker parts of CtfEnv
        self.trew = team_reward or TeamReward()
        self.agents = (0, 1)
        self.pfsp = [None, None]  # per role: draw weights of the older pool versions (PFSP), None = uniform
        super().__init__(**kw)
        self.pending2 = torch.zeros(2, self.n, 2, device="cuda")

    # both red robots follow main_mix
    def _draw(self, slot, k):
        saved = self.main
        if TEAM[slot] == 1:
            self.main = slot
        try:
            kinds = super()._draw(slot, k)
        finally:
            self.main = saved
        w = self.pfsp[slot % 2] if TEAM[slot] == 1 else None
        if w is not None:
            npool = len(self.pools[slot % 2])
            old = (kinds >= 2) & (kinds < 2 + npool - 1)
            m = int(old.sum())
            if m:
                kinds[old] = 2 + torch.multinomial(w, m, replacement=True, generator=self.gen)
        return kinds

    def _reset(self, idx):
        super()._reset(idx)
        if hasattr(self, "pending2") and idx.numel():
            self.pending2[:, idx] = 0.0

    def obs(self):
        rob = self.state
        if self.obs_noise > 0 or self.yaw_noise > 0:
            rob = rob.clone()
            rob[:, :, :2] += self.obs_noise * torch.randn(4, self.n, 2, device="cuda", generator=self.gen)
            rob[:, :, 2] += self.yaw_noise * torch.randn(4, self.n, device="cuda", generator=self.gen)
        return torch.stack([self._obs_slot(0, rob), self._obs_slot(1, rob)])  # (2, n, OBS_DIM)

    def _phi_def(self, rob):
        r = self.trew
        me = rob[1]
        home = self.home[0]
        carrying = self.carrier[1] == 1
        taken = self.carrier[0] >= 0
        att = rob[2]
        u = att[:, :2] - home
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-3)
        guard = home + r.guard_d * u
        d_home = torch.hypot(me[:, 0] - home[0], me[:, 1] - home[1])
        d_flag = torch.hypot(me[:, 0] - self.flag[0, :, 0], me[:, 1] - self.flag[0, :, 1])
        d_guard = torch.hypot(me[:, 0] - guard[:, 0], me[:, 1] - guard[:, 1])
        return torch.where(carrying, -r.d_home * d_home, torch.where(taken, -r.d_chase * d_flag, -r.d_guard * d_guard))

    def _phi_att(self, rob, i):
        """Attacker potential (as CtfEnv._phi in attack mode) for my robot i (0 or 1)."""
        r = self.arew
        me = rob[i]
        carrying = self.carrier[1] == i
        d_home = torch.hypot(me[:, 0] - self.home[0, 0], me[:, 1] - self.home[0, 1])
        d_flag = torch.hypot(me[:, 0] - self.flag[1, :, 0], me[:, 1] - self.flag[1, :, 1])
        phi = torch.where(carrying, -r.shape_home * d_home, -r.shape_flag * d_flag - r.shape_home * r.d0)
        if r.avoid > 0:
            t = self.t
            d = torch.stack([torch.hypot(me[:, 0] - rob[j, :, 0], me[:, 1] - rob[j, :, 1]) for j in (2, 3)])
            active = torch.stack([self.hold_until[j] <= t for j in (2, 3)])
            d = torch.where(active, d, torch.full_like(d, 9.0)).amin(0)
            exposed = (me[:, 0] > 0) & (self.hold_until[i] <= t) & (t >= self.immune_until[i])
            phi = phi - r.avoid * exposed.float() * (r.avoid_r - d).clamp(min=0)
        return phi

    def _phis(self, rob):
        return torch.stack([self._phi_att(rob, 0), self._phi_att(rob, 1) if self.dual else self._phi_def(rob)])

    def step(self, action):
        """action: (2, n, 2) in [-1, 1] for robots 0 (attacker) and 1 (defender)."""
        rob = self.state
        acts = self._others(rob)
        action = action.clamp(-1.0, 1.0)
        if self.dr_delay > 0:
            applied = torch.where(self.delay.view(1, -1, 1), self.pending2, action)
            self.pending2 = action.clone()
            action = applied
        acts[0], acts[1] = action[0], action[1]
        gated = torch.zeros(self.n, dtype=torch.bool, device="cuda")
        if self.def_script:
            gated = self.hold_until[1] <= self.t
            if gated.any():
                mine, theirs, own, enemy, carry, taken, hold, imm = self.view(0, rob)
                g = torch.tensor(GAINS_REF, device="cuda").expand(self.n, -1)
                sd = defender_actions(mine, theirs, own, taken, hold, imm, g)
                acts[1] = torch.where(gated.unsqueeze(1), sd, acts[1])
        elif self.gate is not None:
            t = self.t
            threat = self.carrier[0] >= 0
            for j in (2, 3):
                threat = threat | ((rob[j, :, 0] < self.gate) & (self.hold_until[j] <= t))
            gated = threat & (self.carrier[1] != 1) & (self.hold_until[1] <= t)
            if gated.any():
                mine, theirs, own, enemy, carry, taken, hold, imm = self.view(0, rob)
                g = torch.tensor(GAINS_REF, device="cuda").expand(self.n, -1)
                sd = defender_actions(mine, theirs, own, taken, hold, imm, g)
                acts[1] = torch.where(gated.unsqueeze(1), sd, acts[1])
        phi0 = self._phis(rob)
        held = self.hold_until > self.t
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
        carrying_before = torch.stack([self.carrier[1] == 0, self.carrier[1] == 1])
        point, ev = self._referee(rob)
        scored, conceded = point == 0, point == 1
        timeout = (self.steps >= self.max_steps) & (point < 0)
        truncated = (self.ep_steps >= self.ep_max) & (point < 0) & ~timeout
        phi1 = self._phis(rob)
        ar, tr = self.arew, self.trew
        f = lambda x: x.float()  # noqa: E731
        shared = tr.score * f(scored) + tr.conceded * f(conceded) + tr.team_pickup * f(ev["pickup"][0] | ev["pickup"][1])
        enemy_pick = (ev["pickup"][2] | ev["pickup"][3]) & ~taken_before
        t = self.t
        immune0 = (self.hold_until[0] <= t) & (t < self.immune_until[0])
        r_att = (ar.pickup * f(ev["pickup"][0]) + ar.tagged * f(ev["tagged"][0]) + ar.tag * f(ev["tagger"][0])
                 + ar.tagged_carrying * f(ev["tagged"][0] & carrying_before[0])
                 + ar.immune_pickup * f(ev["pickup"][0] & immune0))
        r_def = (tr.d_pickup * f(ev["pickup"][1]) + tr.d_tag * f(ev["tagger"][1]) + tr.d_tag_carrier * f(ev["tag_carrier"][1])
                 + tr.d_enemy_pickup * f(enemy_pick) + tr.d_tagged * f(ev["tagged"][1])
                 + tr.d_tagged_carrying * f(ev["tagged"][1] & carrying_before[1]))
        if self.dual:  # robot 1: the same attacker events as robot 0
            immune1 = (self.hold_until[1] <= t) & (t < self.immune_until[1])
            r_def = (ar.pickup * f(ev["pickup"][1]) + ar.tagged * f(ev["tagged"][1]) + ar.tag * f(ev["tagger"][1])
                     + ar.tagged_carrying * f(ev["tagged"][1] & carrying_before[1])
                     + ar.immune_pickup * f(ev["pickup"][1] & immune1))
        shared = shared - tr.time_cost * self.dt + tr.draw * f(timeout)
        rew = self.gamma * phi1 - phi0 + shared.unsqueeze(0) + torch.stack([r_att, r_def])
        terminated = scored | conceded | timeout
        done = terminated | truncated
        info = {"scored": scored, "conceded": conceded, "timeout": timeout, "truncated": truncated,
                "main_kind": self.kind[3].clone(),
                "pickup": ev["pickup"][0] | ev["pickup"][1], "pickup_def": ev["pickup"][1],
                "tagged": ev["tagged"][0] | ev["tagged"][1], "tagger": ev["tagger"][0] | ev["tagger"][1],
                "enemy_half_def": rob[1, :, 0] > 0, "gated": gated}
        info["final_obs"] = self.obs()
        idx = done.nonzero(as_tuple=False).flatten()
        if idx.numel():
            self._reset(idx)
            self.state = self._read()
            self._follow_flags(self.state)
        return self.obs(), rew, terminated, truncated, info
