#!/usr/bin/env python3
"""Capture the flag, scripted reference team (deterministic). One node per robot (~role).

Same logic as rl/ctf_scripts.py (the opponents used while training the RL policies), fixed gains:
  attacker: carrying -> straight back to x = 0 (reverse allowed), then home. Not carrying -> to the
            enemy flag; in the enemy half pushed away from an active defender closer than AVOID; if the
            defender guards the flag (closer than WAIT to it, and closer than me) it waits at the border.
  defender: an opponent carries my flag -> intercept a little ahead of it; an active intruder in my half
            closer than CHASE to my flag -> at it; else guard on the segment flag -> their attacker.

Everything is computed in the team frame (base at -x). Standard Python + rospy + psr_hero.utils only.
Run:  ./run.sh run ctf blue:=ctf_script.py red:=none
"""
import json
import math

import rospy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String

from psr_hero.utils import V_MAX, W_MAX, clamp, pose_from_odom, wrap_angle

BASE_X, HALF = 0.62, 0.8
HOLD_TIME, IMMUNE_TIME = 4.0, 1.5
KW, ALIGN, SPEED, AVOID, WAIT, GUARD, CHASE = 4.0, 0.6, 1.0, 0.18, 0.18, 0.10, 0.45


def drive(me, tx, ty, stop_r=0.02):
    """Heading controller that drives backwards when the target is behind. Returns (v, w) in m/s, rad/s."""
    dx, dy = tx - me[0], ty - me[1]
    if math.hypot(dx, dy) < stop_r:
        return 0.0, 0.0
    e = wrap_angle(math.atan2(dy, dx) - me[2])
    back = abs(e) > math.pi / 2
    if back:
        e = wrap_angle(e + math.pi)
    w = clamp(KW * e / W_MAX, -1.0, 1.0)
    v = SPEED if abs(e) < ALIGN else 0.0
    return (-v if back else v) * V_MAX, w * W_MAX


def attacker(me, dfn, enemy_flag, carrying, def_active):
    if carrying:
        if me[0] > -0.03:
            return drive(me, -0.15, me[1])
        return drive(me, -BASE_X, 0.0)
    d_flag_def = math.hypot(dfn[0] - enemy_flag[0], dfn[1] - enemy_flag[1])
    d_flag_me = math.hypot(me[0] - enemy_flag[0], me[1] - enemy_flag[1])
    if def_active and d_flag_def < WAIT and d_flag_me > d_flag_def + 0.05:
        return drive(me, -0.06, -0.3 if dfn[1] > 0 else 0.3)
    fx, fy = enemy_flag[0] - me[0], enemy_flag[1] - me[1]
    n = max(math.hypot(fx, fy), 1e-3)
    tx, ty = me[0] + 0.2 * fx / n, me[1] + 0.2 * fy / n
    d_def = max(math.hypot(me[0] - dfn[0], me[1] - dfn[1]), 1e-3)
    if def_active and me[0] > -0.05 and d_def < AVOID:
        k = min(AVOID / d_def, 3.0)
        tx += 0.2 * k * (me[0] - dfn[0]) / d_def
        ty += 0.2 * k * (me[1] - dfn[1]) / d_def
    return drive(me, clamp(tx, -HALF + 0.05, HALF - 0.05), clamp(ty, -HALF + 0.05, HALF - 0.05))


def defender(me, att, own_flag, flag_taken, att_free):
    hx, hy = -BASE_X, 0.0
    if flag_taken and own_flag[0] < 0:
        tx, ty = own_flag[0] + 0.05, own_flag[1]
    elif att_free and att[0] < 0 and math.hypot(att[0] - hx, att[1] - hy) < CHASE:
        tx, ty = att[0], att[1]
    else:
        ux, uy = att[0] - hx, att[1] - hy
        n = max(math.hypot(ux, uy), 1e-3)
        tx, ty = min(hx + GUARD * ux / n, -0.04), hy + GUARD * uy / n
    return drive(me, clamp(tx, -HALF + 0.05, -0.02), clamp(ty, -HALF + 0.05, HALF - 0.05), stop_r=0.015)


class CaptureTheFlag:
    def __init__(self):
        rospy.init_node("ctf")
        self.robot = rospy.get_param("~robot", 0)
        self.team = rospy.get_param("~team", "blue" if self.robot == 0 else "red")
        self.role = rospy.get_param("~role", "attacker")
        self.opponents = list(rospy.get_param("~opponents", [1 if self.robot == 0 else 0]))
        self.blue = self.team == "blue"
        self.s = 1.0 if self.blue else -1.0  # world -> team frame (red: rotated by 180 deg)
        self.me = None
        self.opp = {}
        self.enemy_flag = None
        self.own_flag = None
        self.carrying = False
        self.game = {}
        self.now = 0.0
        self.t_on = {}  # robot id -> time it was last held
        self.was_held = {}
        self.pub = rospy.Publisher("/hero_%d/cmd_vel" % self.robot, Twist, queue_size=1)
        rospy.Subscriber("/hero_%d/odom" % self.robot, Odometry, self.on_me, queue_size=1)
        for k in self.opponents:
            rospy.Subscriber("/hero_%d/odom" % k, Odometry, self.on_opp, callback_args=k, queue_size=1)
        mine, theirs = ("blue", "red") if self.blue else ("red", "blue")
        rospy.Subscriber("/flag/" + theirs, PointStamped, self.on_enemy_flag, queue_size=1)
        rospy.Subscriber("/flag/" + mine, PointStamped, self.on_own_flag, queue_size=1)
        rospy.Subscriber("/hero_%d/has_flag" % self.robot, Bool, self.on_has_flag, queue_size=1)
        rospy.Subscriber("/game/state", String, self.on_game, queue_size=1)

    def on_me(self, msg):
        self.me = pose_from_odom(msg)

    def on_opp(self, msg, k):
        self.opp[k] = pose_from_odom(msg)

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

    def free(self, k):
        """Robot k can be held right now (not held, not immune), from /game/state."""
        held = self.game.get("held", [])
        h = bool(held[k]) if k < len(held) else False
        if h and not self.was_held.get(k, False):
            self.t_on[k] = self.now
        self.was_held[k] = h
        return not h and self.now >= self.t_on.get(k, -1e9) + HOLD_TIME + IMMUNE_TIME

    def tf(self, p):
        """World -> team frame (x, y, yaw)."""
        if self.blue:
            return p
        return (-p[0], -p[1], wrap_angle(p[2] + math.pi))

    def compute_command(self, me, opponents, enemy_flag, own_flag, carrying):
        opp = [self.opp.get(k) for k in self.opponents]
        att = opp[0]
        dfn = opp[1] if len(opp) > 1 else (BASE_X * self.s, 0.0, 0.0)
        me, att, dfn = self.tf(me), self.tf(att), self.tf(dfn)
        ef = (self.s * enemy_flag[0], self.s * enemy_flag[1])
        of = (self.s * own_flag[0], self.s * own_flag[1])
        taken = math.hypot(of[0] + BASE_X, of[1]) > 0.02
        if self.role == "attacker":
            return attacker(me, dfn, ef, carrying, len(opp) < 2 or self.free(self.opponents[1]))
        return defender(me, att, of, taken, self.free(self.opponents[0]))

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if (self.me is None or len(self.opp) < len(self.opponents)
                    or self.enemy_flag is None or self.own_flag is None):
                rospy.logwarn_throttle(2.0, "waiting for odometry and flags")
            else:
                self.now = rospy.get_time()
                v, w = self.compute_command(self.me, list(self.opp.values()), self.enemy_flag,
                                            self.own_flag, self.carrying)
                cmd = Twist()
                cmd.linear.x, cmd.angular.z = v, w
                self.pub.publish(cmd)
            rate.sleep()


if __name__ == "__main__":
    CaptureTheFlag().run()
