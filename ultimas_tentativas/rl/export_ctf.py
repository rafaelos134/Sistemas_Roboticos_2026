"""Export the capture-the-flag networks as one ROS 1 node for the psr-hero-kit (single file, numpy only).

    ~/ros2_ws/.venv/bin/python export_ctf.py --att runs/ctf_att_3/best.pt --def runs/ctf_def_3/best.pt \
        --out exports/ctf/ctf_rl.py

One node per robot, like the kit template: ~role chooses the network (attacker / defender). Same rules as
the other exports: only standard Python + numpy + rospy + messages + psr_hero.utils.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import torch

from export_soccer import net_arrays
from export_sumo import numpy_block, wrap
from export_soccer import b64

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "kit" / "exercicios" / "ctf" / "ctf_script.py"


def gru_arrays(prefix, ckpt):
    """GRU memory of a recurrent checkpoint (ppo_rnn.py), [] for an MLP one."""
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    if "gru.weight_ih" not in sd:
        return []
    return [(prefix + n, sd[k].numpy()) for n, k in (("gru_wih", "gru.weight_ih"), ("gru_whh", "gru.weight_hh"),
            ("gru_bih", "gru.bias_ih"), ("gru_bhh", "gru.bias_hh"), ("head_w", "head.weight"), ("head_b", "head.bias"))]


def dqn_arrays(prefix, ckpt):
    """Action table of a DQN checkpoint (train_ctf_dqn.py: the node takes the argmax of Q), [] otherwise."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    return [(prefix + "actions", ck["actions"].numpy())] if "actions" in ck else []


def script_block() -> str:
    """Gains, drive() and defender() copied from the scripted reference team (kit/exercicios/ctf/ctf_script.py)."""
    text = SCRIPT.read_text()
    tree = ast.parse(text)
    out = []
    for node in tree.body:
        if (isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "KW" or
                any(getattr(e, "id", None) == "KW" for e in getattr(t, "elts", [])) for t in node.targets)) \
                or (isinstance(node, ast.FunctionDef) and node.name in ("drive", "defender")):
            out.append(ast.get_source_segment(text, node))
    return "\n\n\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--att", required=True)
    p.add_argument("--def", dest="defender", default=None, help="defender network (omit with --def-script)")
    p.add_argument("--def-script", action="store_true", help="hybrid: scripted reference defender")
    p.add_argument("--guard-flag", action="store_true", help="with --def-script: the defender stands on the flag")
    p.add_argument("--def-gate", type=float, default=None,
                   help="supervisor for a defender network (e.g. the dual-attack one): it plays the network while the "
                        "base is safe and the scripted defender when an active opponent has x < DEF_GATE (team frame) "
                        "or my flag is taken (unless it carries the enemy flag)")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if not args.def_script and not args.defender:
        p.error("--def or --def-script is required")
    arrays, n_att = net_arrays("att_", args.att)
    arrays += gru_arrays("att_", args.att) + dqn_arrays("att_", args.att)
    n_def = 0
    if not args.def_script:
        a, n_def = net_arrays("def_", args.defender)
        arrays += a + gru_arrays("def_", args.defender) + dqn_arrays("def_", args.defender)
    weights = "\n".join("    %r: (%r, (\n%s\n    ))," % (n, tuple(x.shape), wrap(b64(x))) for n, x in arrays)
    if args.guard_flag and not args.def_script:
        p.error("--guard-flag needs --def-script")
    dname = (("scripted, on the flag" if args.guard_flag else "scripted (ctf_script.py)") if args.def_script
             else "%s/%s" % (Path(args.defender).parent.name, Path(args.defender).name))
    info = "attacker: %s/%s; defender: %s" % (Path(args.att).parent.name, Path(args.att).name, dname)
    src = (TEMPLATE.replace("@@INFO@@", info).replace("@@N_ATT@@", str(n_att)).replace("@@N_DEF@@", str(n_def))
           .replace("@@OBS@@", numpy_block(HERE / "ctf_obs.py")).replace("@@WEIGHTS@@", weights)
           .replace("@@DEF_SCRIPT@@", str(bool(args.def_script)))
           .replace("@@GUARD_FLAG@@", str(bool(args.guard_flag)))
           .replace("@@DEF_GATE@@", repr(args.def_gate))
           .replace("@@SCRIPT@@", (script_block() + (FN_ON_FLAG if args.guard_flag else ""))
                    if (args.def_script or args.def_gate is not None) else "# (defender network: no script)\n"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(src)
    out.chmod(0o777)
    print("wrote %s (%.0f kB): %s" % (out, out.stat().st_size / 1024, info))


FN_ON_FLAG = r'''

def defender_on_flag(me, own_flag, flag_taken):
    """Guard ON the flag: the robot stands over it (flags have no collision). An attacker cannot get its
    center closer than 2 robot radii (0.074 m) to mine, which is more than the pickup radius (0.07 m), and
    it is held (< 0.10 m) on the way. Leaves only to chase the carrier when the flag was taken anyway."""
    if flag_taken and own_flag[0] < 0:
        return drive(me, own_flag[0] + 0.05, own_flag[1])
    return drive(me, -BASE_X, 0.0, stop_r=0.005)
'''

TEMPLATE = r'''#!/usr/bin/env python3
"""Capture the flag with policies trained by reinforcement learning (PPO). One node per robot (~role).

Generated by rl/export_ctf.py (@@INFO@@).
Do not edit by hand: train again and export. Documentation: rl/docs/documentacao.md

  attacker: MLP 38 -> 128 -> 128 -> 2 (tanh), trained against frozen defenders (and scripts)
  defender: MLP 38 -> 128 -> 128 -> 2 (tanh), trained against frozen attackers (and scripts)
The two roles were trained in alternation (rl/ctf_league.sh); with DEF_SCRIPT the defender is scripted. Everything is computed in the team frame
(red sees the arena rotated by 180 deg), so the same file plays blue or red (~team). It reads the
odometry of the four robots, both flags, /hero_N/has_flag and /game/state (held robots, carriers,
round time). Made for 2 x 2; with -n 2 the missing robots are assumed at their bases.
Only standard Python, numpy, rospy, the ROS messages and psr_hero.utils are used.

Run:  ./run.sh run ctf blue:=ctf_rl.py red:=none
"""
import base64
import json
import math

import numpy as np
import rospy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String

from psr_hero.utils import V_MAX, W_MAX, clamp, pose_from_odom, wrap_angle

CONTROL_HZ = 20
HALF = 0.8  # arena 1.6 x 1.6 m
BASE_X = 0.62  # m, blue base at -BASE_X, red at +BASE_X
HOLD_TIME, IMMUNE_TIME, ROUND_TIME = 4.0, 1.5, 90.0  # s (kit rules)
DEF_SCRIPT = @@DEF_SCRIPT@@  # True: hybrid node, the defender is the scripted reference (ctf_script.py)
GUARD_FLAG = @@GUARD_FLAG@@  # True: the scripted defender stands exactly on its flag
DEF_GATE = @@DEF_GATE@@  # float: the defender network plays only while no active opponent has x < DEF_GATE


# --- observation (copied from rl/ctf_obs.py) --------------------------------
@@OBS@@

# --- scripted defender (copied from kit/exercicios/ctf/ctf_script.py) ---------------------
@@SCRIPT@@

# --- networks -------------------------------------------------------------------------------
WEIGHTS = {
@@WEIGHTS@@
}


def _load(name):
    shape, chunks = WEIGHTS[name]
    return np.frombuffer(base64.b64decode("".join(chunks)), dtype="<f4").reshape(shape).astype(np.float64)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Policy:
    """Deterministic actor (mean of the Gaussian policy). With a GRU memory (rl/ppo_rnn.py) the mean is
    actor(x) + head(h), h = GRUCell(x, h); h is cleared at every new round (reset)."""

    def __init__(self, prefix, n_layers):
        self.mean = _load(prefix + "norm_mean")
        self.std = np.sqrt(_load(prefix + "norm_var") + 1e-5)
        self.layers = [(_load(prefix + "W%d" % j), _load(prefix + "b%d" % j)) for j in range(n_layers)]
        self.gru = None
        if prefix + "gru_wih" in WEIGHTS:
            self.gru = [_load(prefix + n) for n in ("gru_wih", "gru_whh", "gru_bih", "gru_bhh", "head_w", "head_b")]
        # DQN (rl/train_ctf_dqn.py): the last layer gives Q of each discrete action, the command is the argmax
        self.actions = _load(prefix + "actions") if prefix + "actions" in WEIGHTS else None
        self.reset()

    def reset(self):
        self.h = np.zeros(self.gru[1].shape[1]) if self.gru is not None else None

    def __call__(self, obs):
        x0 = np.clip((np.asarray(obs, dtype=np.float64) - self.mean) / self.std, -5.0, 5.0)
        x = x0
        for j, (w, b) in enumerate(self.layers):
            x = w.dot(x) + b
            if j < len(self.layers) - 1:
                x = np.tanh(x)
        if self.gru is not None:  # torch.nn.GRUCell, gates in the order r, z, n
            wih, whh, bih, bhh, hw, hb = self.gru
            gi, gh = wih.dot(x0) + bih, whh.dot(self.h) + bhh
            k = len(self.h)
            r = _sigmoid(gi[:k] + gh[:k])
            z = _sigmoid(gi[k:2 * k] + gh[k:2 * k])
            n = np.tanh(gi[2 * k:] + r * gh[2 * k:])
            self.h = (1.0 - z) * n + z * self.h
            x = x + hw.dot(self.h) + hb
        if self.actions is not None:
            return self.actions[int(np.argmax(x))]
        return np.clip(x, -1.0, 1.0)


class CaptureTheFlag:
    def __init__(self):
        rospy.init_node("ctf")
        self.robot = rospy.get_param("~robot", 0)
        self.team = rospy.get_param("~team", "blue" if self.robot == 0 else "red")
        self.role = rospy.get_param("~role", "attacker")
        self.opponents = list(rospy.get_param("~opponents", [1 if self.robot == 0 else 0]))
        self.blue = self.team == "blue"
        self.flip = not self.blue
        self.home = (-BASE_X if self.blue else BASE_X, 0.0)
        if len(self.opponents) >= 2:  # 2 x 2: blue 0 (att), 1 (def); red 2 (att), 3 (def)
            first = 0 if self.blue else 2
            self.mine = [first, first + 1]
            self.theirs = list(self.opponents[:2])
        else:  # 1 x 1: one robot per team; the missing ones are placed at their bases
            self.mine = [self.robot, None] if self.role == "attacker" else [None, self.robot]
            self.theirs = [self.opponents[0], None]
        self.r = 0 if self.role == "attacker" else 1
        if self.r == 0:
            self.net = Policy("att_", @@N_ATT@@)
        else:
            self.net = None if DEF_SCRIPT else Policy("def_", @@N_DEF@@)

        self.me = None  # (x, y, yaw), filled by the callbacks (kit template fields)
        self.opp = {}
        self.enemy_flag = None
        self.own_flag = None
        self.carrying = False
        self.poses = {}  # id -> (x, y, yaw), every robot
        self.vel = {}  # id -> (v, w)
        self.game = {}  # last /game/state
        self.now = 0.0
        self.prev = (0.0, 0.0)
        self.t_on = {}  # id -> time it was last held
        self.was_held = {}
        self.last_round_t = None

        self.pub = rospy.Publisher("/hero_%d/cmd_vel" % self.robot, Twist, queue_size=1)
        for k in [i for i in self.mine + self.theirs if i is not None]:
            rospy.Subscriber("/hero_%d/odom" % k, Odometry, self.on_odom, callback_args=k, queue_size=1)
        mine, theirs = ("blue", "red") if self.blue else ("red", "blue")
        rospy.Subscriber("/flag/" + theirs, PointStamped, self.on_enemy_flag, queue_size=1)
        rospy.Subscriber("/flag/" + mine, PointStamped, self.on_own_flag, queue_size=1)
        rospy.Subscriber("/hero_%d/has_flag" % self.robot, Bool, self.on_has_flag, queue_size=1)
        rospy.Subscriber("/game/state", String, self.on_game, queue_size=1)

    # --- callbacks ---------------------------------------------------------------------------
    def on_odom(self, msg, k):
        p = pose_from_odom(msg)
        self.poses[k] = p
        self.vel[k] = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)
        if k == self.robot:
            self.me = p
        elif k in self.opponents:
            self.opp[k] = p

    def on_enemy_flag(self, msg):
        self.enemy_flag = (msg.point.x, msg.point.y)

    def on_own_flag(self, msg):
        self.own_flag = (msg.point.x, msg.point.y)

    def on_has_flag(self, msg):
        self.carrying = msg.data

    def on_game(self, msg):
        try:
            self.game = json.loads(msg.data)
        except ValueError:
            pass

    # --- state in the team frame -----------------------------------------------------------------
    def robot_state(self, k, base_x):
        if k is None or k not in self.poses:  # missing robot: at its base, still
            x, y, yaw = base_x * (1.0 if self.blue else -1.0), 0.0, 0.0
            return team_frame_np((x, y, yaw, 0.0, 0.0), self.flip)
        x, y, yaw = self.poses[k]
        v, w = self.vel.get(k, (0.0, 0.0))
        return team_frame_np((x, y, yaw, v, w), self.flip)

    def timers(self, k):
        """(remaining hold, remaining immunity) of robot k, in s, from the 'held' list of /game/state."""
        held = self.game.get("held", [])
        h = k is not None and k < len(held) and bool(held[k])
        if k is not None:
            if h and not self.was_held.get(k, False):
                self.t_on[k] = self.now
            self.was_held[k] = h
        t_on = self.t_on.get(k, -1e9)
        if h:
            return max(0.0, t_on + HOLD_TIME - self.now), 0.0
        return 0.0, max(0.0, t_on + HOLD_TIME + IMMUNE_TIME - self.now)

    def compute_command(self, me, opponents, enemy_flag, own_flag, carrying):
        if self.robot not in self.poses:  # fed only through the template arguments
            self.poses[self.robot] = me
            for k, p in zip(self.opponents, opponents):
                self.poses[k] = p
        round_t = float(self.game.get("round_time", 0.0))
        if self.last_round_t is not None and round_t + 0.5 < self.last_round_t:  # new round: reset the memory
            self.prev, self.t_on, self.was_held = (0.0, 0.0), {}, {}
            if self.net is not None:
                self.net.reset()
        self.last_round_t = round_t
        s = -1.0 if self.flip else 1.0
        mine = [self.robot_state(self.mine[0], -BASE_X), self.robot_state(self.mine[1], -BASE_X)]
        theirs = [self.robot_state(self.theirs[0], BASE_X), self.robot_state(self.theirs[1], BASE_X)]
        ef = (s * enemy_flag[0], s * enemy_flag[1])
        of = (s * own_flag[0], s * own_flag[1])
        cl = self.game.get("carrying", [])
        car = lambda k: k is not None and k < len(cl) and bool(cl[k])  # noqa: E731
        carry = [car(self.mine[0]), car(self.mine[1])]
        carry[self.r] = carry[self.r] or bool(carrying)
        taken = car(self.theirs[0]) or car(self.theirs[1]) or math.hypot(of[0] + BASE_X, of[1]) > 0.02
        tm = [self.timers(k) for k in (self.mine[0], self.mine[1], self.theirs[0], self.theirs[1])]
        hold, immune = [t[0] for t in tm], [t[1] for t in tm]
        if self.net is None:  # hybrid: scripted defender (team frame)
            att_free = hold[2] == 0.0 and immune[2] == 0.0
            if GUARD_FLAG:
                return defender_on_flag(mine[1][:3], of, taken)
            return defender(mine[1][:3], theirs[0][:3], of, taken, att_free)
        if self.r == 1 and DEF_GATE is not None and not carry[1]:
            threat = taken or any(theirs[k][0] < DEF_GATE and hold[2 + k] == 0.0 for k in (0, 1))
            if threat:  # back home: scripted defender
                att_free = hold[2] == 0.0 and immune[2] == 0.0
                v, w = defender(mine[1][:3], theirs[0][:3], of, taken, att_free)
                self.prev = (v / V_MAX, w / W_MAX)
                return v, w
        obs = observation_np(self.r, mine, theirs, of, ef, carry, taken, hold, immune, round_t, self.prev)
        a = self.net(obs)
        self.prev = (float(a[0]), float(a[1]))
        return float(a[0]) * V_MAX, float(a[1]) * W_MAX

    def run(self):
        rate = rospy.Rate(CONTROL_HZ)
        while not rospy.is_shutdown():
            if (self.me is None or len(self.opp) < len(self.opponents)
                    or self.enemy_flag is None or self.own_flag is None):
                rospy.logwarn_throttle(2.0, "waiting for odometry and flags")
            else:
                self.now = rospy.get_time()
                v, w = self.compute_command(self.me, [self.opp[k] for k in self.opponents], self.enemy_flag,
                                            self.own_flag, self.carrying)
                cmd = Twist()
                cmd.linear.x, cmd.angular.z = v, w
                self.pub.publish(cmd)
            rate.sleep()


if __name__ == "__main__":
    CaptureTheFlag().run()
'''


if __name__ == "__main__":
    main()
