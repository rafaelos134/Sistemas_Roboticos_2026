"""Capture the flag, team training FROM SCRATCH (train_ctf_zero.py): CtfTeamEnv + the teams estimated by
reverse engineering (eng_reversa/estimados/ctf/ajuste.json, state machines of eng_reversa/modelos/ctf_maquina.py)
as opponents.

The red team is ONE controller per episode (both red robots of the same kind):
    stopped | script (ctf_scripts.py styles, gains drawn per episode) | er:<team> (estimated state machine)
drawn from `main_mix` = {"stopped": w, "script": w, "er": w}; with "er" the team is uniform over the teams.
kind codes: 0 stopped, 1 script, 100 + j estimated team j (ER_TEAMS[j]).

The estimated machines are numpy (the same code as the kit nodes in eng_reversa/estimados/ctf/<Team>.py) and
run on the CPU for the rows that use them; they need a little memory per robot (when it was last released
and where it was held), kept here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from ctf_env import TEAM, V_MAX, W_MAX
from ctf_team_env import CtfTeamEnv

ER_DIR = Path(__file__).resolve().parent / "eng_reversa"
ER_KIND = 100


def load_er_teams(names=None):
    """[(name, module, params)]: the estimated state machines of the CTF teams."""
    sys.path.insert(0, str(ER_DIR))
    import ajustar  # noqa: E402

    fits = json.loads((ER_DIR / "estimados" / "ctf" / "ajuste.json").read_text())
    out = []
    for name, f in fits.items():
        if names and name not in names:
            continue
        params = {"att": f["att"]["params"], "def": f["def"]["params"]}
        mod, _ = ajustar.carregar_modelo("ctf", params, name)
        out.append((name, mod, params))
    return out


class CtfZeroEnv(CtfTeamEnv):
    def __init__(self, er_teams=None, easy_mate=0.0, **kw):
        """easy_mate: fraction of episodes in which robot 1 (my defender) starts near the enemy flag (half of
        them carrying it): so it finds out that it can take the flag too (the standard script defender only
        watches the enemy attacker)."""
        self.er = er_teams or []
        self.easy_mate = easy_mate
        super().__init__(**kw)
        n = self.n
        self.er_was_held = torch.zeros(2, n, dtype=torch.bool, device="cuda")  # red att, red def
        self.er_release = torch.full((2, n), -1e9, device="cuda")
        self.er_held_at = torch.zeros(2, n, 2, device="cuda")

    # one kind for the whole red team
    def _draw_red(self, k):
        names, p = self.main_mix
        pick = torch.multinomial(p, k, replacement=True, generator=self.gen)
        kinds = torch.ones(k, dtype=torch.long, device="cuda")
        for j, name in enumerate(names):
            rows = pick == j
            if not rows.any():
                continue
            if name == "stopped":
                kinds[rows] = 0
            elif name == "er" and self.er:
                kinds[rows] = ER_KIND + torch.randint(0, len(self.er), (int(rows.sum()),), device="cuda",
                                                      generator=self.gen)
        return kinds

    def _spawn(self, k):
        poses, carry = super()._spawn(k)
        if self.easy_mate <= 0:
            return poses, carry
        import math

        sel = (self._rand(k) < self.easy_mate) & (carry < 0)
        m = int(sel.sum())
        if m:
            ang = (self._rand(m) * 2 - 1) * math.pi
            d = 0.10 + 0.40 * self._rand(m)
            px = (0.62 - d * torch.cos(ang).abs()).clamp(0.06, 0.74)
            py = (d * torch.sin(ang)).clamp(-0.74, 0.74)
            sub = poses[:, sel]
            sub[1] = torch.stack([px, py, (self._rand(m) * 2 - 1) * math.pi], 1)
            p = sub[:, :, :2].permute(1, 0, 2)
            dd = (torch.cdist(p, p) + 10 * torch.eye(4, device="cuda")).amin((1, 2))
            ok = dd >= 0.09
            poses[:, sel] = torch.where(ok.view(1, -1, 1), sub, poses[:, sel])
            c = torch.where(self._rand(m) < 0.5, torch.ones(m, dtype=torch.long, device="cuda"),
                            torch.full((m,), -1, dtype=torch.long, device="cuda"))
            carry[sel] = torch.where(ok, c, carry[sel])
        return poses, carry

    def _reset(self, idx):
        super()._reset(idx)
        if idx.numel() == 0:
            return
        kinds = self._draw_red(idx.numel())
        self.kind[2, idx] = kinds
        self.kind[3, idx] = kinds
        if hasattr(self, "er_was_held"):
            self.er_was_held[:, idx] = False
            self.er_release[:, idx] = -1e9
            self.er_held_at[:, idx] = 0.0

    @torch.no_grad()
    def _others(self, rob):
        act = super()._others(rob)  # stopped / script (estimated rows stay at zero here)
        if not self.er:
            return act
        mine, theirs, own, enemy, carry, taken, hold, imm = self.view(1, rob)  # red team frame
        t = self.t
        held = hold[0:2] > 0  # red att, red def
        start = held & ~self.er_was_held
        self.er_held_at = torch.where(start.unsqueeze(2), mine[:, :, :2], self.er_held_at)
        rel = ~held & self.er_was_held
        self.er_release = torch.where(rel, t.unsqueeze(0).expand(2, -1), self.er_release)
        self.er_was_held = held
        er_rows = (self.kind[2] >= ER_KIND).nonzero(as_tuple=False).flatten()
        if er_rows.numel() == 0:
            return act
        # opponents of red sorted by x (team frame): o1 = the most advanced (intruder)
        th_h = hold[2:4] > 0
        a_first = theirs[0, :, 0] <= theirs[1, :, 0]
        o1 = torch.where(a_first.unsqueeze(1), theirs[0], theirs[1])
        o2 = torch.where(a_first.unsqueeze(1), theirs[1], theirs[0])
        h1 = torch.where(a_first, th_h[0], th_h[1])
        h2 = torch.where(a_first, th_h[1], th_h[0])
        cols = {"o1x": o1[:, 0], "o1y": o1[:, 1], "o1h": h1, "o2x": o2[:, 0], "o2y": o2[:, 1], "o2h": h2,
                "efx": enemy[:, 0], "efy": enemy[:, 1], "ofx": own[:, 0], "ofy": own[:, 1]}
        per = {}
        for r in (0, 1):
            per[r] = {"x": mine[r, :, 0], "y": mine[r, :, 1], "yaw": mine[r, :, 2], "carry": carry[r],
                      "t_rel": t - self.er_release[r], "hx_prev": self.er_held_at[r, :, 0],
                      "hy_prev": self.er_held_at[r, :, 1]}
        sel = er_rows
        C = {k: v[sel].cpu().numpy() for k, v in cols.items()}
        Pr = {r: {k: v[sel].cpu().numpy() for k, v in per[r].items()} for r in (0, 1)}
        team = (self.kind[2, sel] - ER_KIND).cpu().numpy()
        out = np.zeros((2, len(team), 2), dtype=np.float32)
        for j in np.unique(team):
            m = team == j
            _, mod, params = self.er[j]
            for r, papel in ((0, "att"), (1, "def")):
                F = {k: v[m] for k, v in C.items()}
                F.update({k: v[m] for k, v in Pr[r].items()})
                F["o1h"], F["o2h"], F["carry"] = F["o1h"].astype(bool), F["o2h"].astype(bool), F["carry"].astype(bool)
                v, w = mod.FAMILIAS[papel][0](F, params[papel])
                out[r, m, 0] = np.asarray(v, dtype=np.float32) / V_MAX
                out[r, m, 1] = np.asarray(w, dtype=np.float32) / W_MAX
        o = torch.from_numpy(out).cuda().clamp(-1, 1)
        act[2, sel] = o[0]
        act[3, sel] = o[1]
        return act


assert TEAM == (0, 0, 1, 1)
