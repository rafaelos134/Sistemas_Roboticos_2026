#!/usr/bin/env python3
"""Capture the flag. One file drives every robot of the team (parameter ~role).

The arena is 1.6 x 1.6 m, split at x = 0. Blue (base on the left, x = -0.62 m) has
hero_0 and hero_1; red (base on the right, x = +0.62 m) has hero_2 and hero_3.

Rules (see README.md): touching the enemy flag (< 0.07 m) picks it up; carrying it into
the own base circle (0.14 m) scores. In YOUR half, touching an opponent (< 0.10 m) holds
it for 4 s (then it is immune for 1.5 s); if it carried your flag, the flag goes home.

Defender strategy
  GUARD           sit exactly on our flag: an attacker cannot get within 0.07 m of the
                  flag without getting within 0.10 m of us first, and the referee checks
                  tags before pick-ups, so he is held before he can grab it.
  INTERCEPT       an opponent not held (immune or not) comes within TRIGGER of our flag:
                  drive at him and tag him.
  HELD_WITH_FLAG  referee quirk: the pick-up check does not skip held robots, so an
                  opponent tagged within 0.07 m of our flag's home picks the flag up again
                  on the same tick and keeps it while held. We wait behind him (flag side)
                  and, PUSH_LEAD s before his release, start pushing him away from the flag,
                  so we are already moving when he is released, stay glued to him during
                  his 1.5 s of immunity and re-tag him the moment it ends. The flag then
                  returns home with him far enough not to pick it up again.
  HELD_NO_FLAG    a held opponent (without our flag) close to our flag: push him away for
                  NO_FLAG_PUSH_TIME s after the tag, then come back and sit on the flag well
                  before he is released.
  CHASE_CARRIER   an opponent carries our flag in our half: pursue him.
The defender never crosses into the enemy half (it could be tagged there).

Attacker strategy (uses the same pick-up quirk against the enemy defender)
  GO_TO_FLAG      drive straight at the enemy flag; a defender sitting on it tags us ~10 cm
                  away, without the flag.
  PREPARE / GRAB  while immune after that tag, wait STAGE_R from the flag and, GRAB_LEAD s
                  before the immunity ends, drive into it.
  BAIT            holding the flag, still on the spot, a free enemy close: keep pressing on
                  the spot so his tag lands there and we keep the flag while held.
  RETURN          released with the flag (1.5 s immune): reverse straight home without
                  turning; once SAFE_X inside our half, turn around and drive to the base.

Run:  ./run.sh run ctf      (see README.md in this folder)
"""
import json
import math

import rospy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String

from psr_hero.utils import V_MAX, W_MAX, clamp, pose_from_odom, wrap_angle

BASE_X = 0.62  # m, blue at -BASE_X, red at +BASE_X
BASE_RADIUS = 0.14  # m

# referee constants (sim/psr_sim/scenarios/ctf.py)
HOLD_TIME = 4.0  # s an opponent stays held
IMMUNE_TIME = 1.5  # s of immunity after the release

# controller gains
K_V = 2.0  # (m/s)/m, linear speed per metre of distance to the target
K_W = 6.0  # (rad/s)/rad, angular speed per radian of heading error
PARK_TOL = 0.01  # m, "on the flag" tolerance

# defender parameters
TRIGGER = 0.17  # m, an opponent (not held) this close to our flag is attacked
X_MARGIN = 0.04  # m, the defender stays at least this far inside our half
PUSH_LEAD = 1.0  # s before the release of a held carrier when we start pushing him
STAGE_GAP = 0.08  # m, distance behind the opponent where a push starts
PUSH_AIM = 0.30  # m, aim point beyond the opponent along the push direction
PUSH_RANGE = 0.30  # m, held opponents farther than this from our flag are left alone
RETURN_MARGIN = 0.3  # s, extra time to be back on the flag before a release
NO_FLAG_PUSH_TIME = 2.2  # s after the tag during which a held opponent without our flag is pushed

# attacker parameters
GRAB_LEAD = 0.5  # s before our immunity ends when we drive into the enemy flag
STAGE_R = 0.09  # m from the enemy flag home where we wait while immune
BAIT_RANGE = 0.15  # m, wait on the flag spot (to be tagged there) only if a free enemy is this close
BAIT_GRACE = 0.2  # s after our immunity ends that we keep waiting for the tag
SAFE_X = 0.05  # m into our half before turning around to drive forward


class CaptureTheFlag:
    def __init__(self):
        rospy.init_node("ctf")
        self.robot = rospy.get_param("~robot", 0)
        self.team = rospy.get_param("~team", "blue" if self.robot == 0 else "red")
        self.role = rospy.get_param("~role", "attacker")
        self.opponents = list(rospy.get_param("~opponents", [1 if self.robot == 0 else 0]))
        self.blue = self.team == "blue"
        self.side = -1.0 if self.blue else 1.0  # sign of x in our half
        self.home = (-BASE_X if self.blue else BASE_X, 0.0)  # your base and your flag
        self.enemy_home = (-self.home[0], 0.0)  # where the enemy flag sits (and returns to)
        self.me = None  # (x, y, yaw)
        self.opp = {}  # opponent id -> (x, y, yaw)
        self.enemy_flag = None  # (x, y)
        self.own_flag = None  # (x, y): away from self.home means an opponent has it
        self.carrying = False  # True while you hold the enemy flag

        # from /game/state
        self.now = None  # simulation time (s)
        self.held = []  # held[i]: robot i is held
        self.carriers = []  # carriers[i]: robot i carries its enemy's flag
        self.held_since = {}  # robot id -> time it was last held
        self.flag_at_release = False  # we carried the enemy flag when our last hold ended
        self.state = None  # last behaviour, for logging

        self.pub = rospy.Publisher("/hero_%d/cmd_vel" % self.robot, Twist, queue_size=1)

        rospy.Subscriber("/hero_%d/odom" % self.robot, Odometry, self.on_me, queue_size=1)
        for k in self.opponents:
            rospy.Subscriber("/hero_%d/odom" % k, Odometry, self.on_opp, k, queue_size=1)
        enemy_team, own_team = ("red", "blue") if self.blue else ("blue", "red")
        rospy.Subscriber("/flag/" + enemy_team, PointStamped, self.on_enemy_flag, queue_size=1)
        rospy.Subscriber("/flag/" + own_team, PointStamped, self.on_own_flag, queue_size=1)
        rospy.Subscriber("/hero_%d/has_flag" % self.robot, Bool, self.on_has_flag, queue_size=1)
        rospy.Subscriber("/game/state", String, self.on_game, queue_size=1)

    # --- callbacks ------------------------------------------------------------------

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
            game = json.loads(msg.data)
        except ValueError:
            return
        now = game.get("t")
        held = list(game.get("held", []))
        if now is None:
            return
        carriers = list(game.get("carrying", []))
        for i, h in enumerate(held):
            was = i < len(self.held) and self.held[i]
            if h and not was:
                self.held_since[i] = now
            if i == self.robot and was and not h:  # we were just released
                self.flag_at_release = i < len(carriers) and bool(carriers[i])
        self.now = now
        self.held = held
        self.carriers = carriers

    # --- game state helpers -------------------------------------------------------

    def is_held(self, k):
        return k < len(self.held) and bool(self.held[k])

    def has_our_flag(self, k):
        """Opponent k carries its enemy's flag, which is ours."""
        return k < len(self.carriers) and bool(self.carriers[k])

    def is_immune(self, k):
        t0 = self.held_since.get(k)
        return (self.now is not None and t0 is not None and not self.is_held(k)
                and self.now < t0 + HOLD_TIME + IMMUNE_TIME)

    def in_own_half(self, x, margin=0.0):
        return self.side * x > margin

    def clamp_own_half(self, p):
        """Move a target point back inside our half (X_MARGIN from the line)."""
        x = min(p[0], -X_MARGIN) if self.blue else max(p[0], X_MARGIN)
        return (x, p[1])

    # --- low-level control --------------------------------------------------------

    def steer(self, me, target, fast=False):
        """Heading P-controller towards target. fast: full speed (pursuit), else slow down near it."""
        dx, dy = target[0] - me[0], target[1] - me[1]
        dist = math.hypot(dx, dy)
        err = wrap_angle(math.atan2(dy, dx) - me[2])
        w = clamp(K_W * err, -W_MAX, W_MAX)
        speed = V_MAX if fast else min(V_MAX, K_V * dist)
        v = speed * max(0.0, math.cos(err))  # turn in place when the target is behind
        return v, w

    def steer_reverse(self, me, target, fast=True):
        """Like steer, but driving backwards (the robot's back leads)."""
        dx, dy = target[0] - me[0], target[1] - me[1]
        dist = math.hypot(dx, dy)
        err = wrap_angle(math.atan2(dy, dx) - (me[2] + math.pi))
        w = clamp(K_W * err, -W_MAX, W_MAX)
        speed = V_MAX if fast else min(V_MAX, K_V * dist)
        v = -speed * max(0.0, math.cos(err))
        return v, w

    def face(self, me, yaw):
        return 0.0, clamp(K_W * wrap_angle(yaw - me[2]), -W_MAX, W_MAX)

    def keep_in_own_half(self, me, v, w):
        """Defender safety filter: never drive across X_MARGIN towards the middle line."""
        depth_rate = self.side * v * math.cos(me[2])  # d(side * x)/dt
        if depth_rate < 0.0 and self.side * me[0] + 0.1 * depth_rate < X_MARGIN:
            v = 0.0
        return v, w

    def push(self, me, enemy, u):
        """Push enemy along unit vector u: get behind him (opposite side of u), then drive through."""
        behind = (enemy[0] - STAGE_GAP * u[0], enemy[1] - STAGE_GAP * u[1])
        ahead = (enemy[0] - me[0]) * u[0] + (enemy[1] - me[1]) * u[1]  # > 0: we are behind him
        if ahead < 0.03 and math.hypot(behind[0] - me[0], behind[1] - me[1]) > 0.03:
            return self.steer(me, self.clamp_own_half(behind), fast=True)
        aim = (enemy[0] + PUSH_AIM * u[0], enemy[1] + PUSH_AIM * u[1])
        return self.steer(me, aim, fast=True)

    def away_from_flag(self, p):
        """Unit vector from our flag home towards p (towards our back wall if p is on the flag)."""
        dx, dy = p[0] - self.home[0], p[1] - self.home[1]
        d = math.hypot(dx, dy)
        if d < 0.01:
            return (self.side, 0.0)
        return (dx / d, dy / d)

    def set_state(self, name):
        if name != self.state:
            rospy.loginfo("hero_%d (%s): %s", self.robot, self.role, name)
            self.state = name

    # --- behaviours -----------------------------------------------------------------

    def defend(self, me):
        home = self.home
        now = self.now if self.now is not None else 0.0
        opps = [(k, self.opp[k]) for k in self.opponents if k in self.opp]

        def to_flag(p):
            return math.hypot(p[0] - home[0], p[1] - home[1])

        # 1) INTERCEPT: an opponent (not held) close to our flag
        # (immune or not: an immune opponent is pushed off the flag, and tagged once it expires)
        threats = [(to_flag(p), k, p) for k, p in opps
                   if not self.is_held(k) and self.in_own_half(p[0]) and to_flag(p) < TRIGGER]
        if threats:
            _, k, p = min(threats)
            self.set_state("INTERCEPT hero_%d" % k)
            return self.steer(me, self.clamp_own_half(p), fast=True)

        # 2) HELD_WITH_FLAG: held carrier (pick-up quirk), until its immunity ends
        for k, p in opps:
            t0 = self.held_since.get(k)
            if (self.has_our_flag(k) and t0 is not None and now < t0 + HOLD_TIME + IMMUNE_TIME
                    and (self.is_held(k) or self.is_immune(k)) and self.in_own_half(p[0])):
                u = self.away_from_flag(p)
                if now < t0 + HOLD_TIME - PUSH_LEAD:
                    self.set_state("HELD_WITH_FLAG hero_%d: wait" % k)
                    stage = self.clamp_own_half((p[0] - STAGE_GAP * u[0], p[1] - STAGE_GAP * u[1]))
                    if math.hypot(stage[0] - me[0], stage[1] - me[1]) > PARK_TOL:
                        return self.steer(me, stage)
                    return self.face(me, math.atan2(p[1] - me[1], p[0] - me[0]))
                self.set_state("HELD_WITH_FLAG hero_%d: push" % k)
                return self.push(me, p, u)

        # 3) CHASE_CARRIER: an opponent runs with our flag in our half
        for k, p in opps:
            if self.has_our_flag(k) and not self.is_held(k) and self.in_own_half(p[0]):
                self.set_state("CHASE_CARRIER hero_%d" % k)
                return self.steer(me, self.clamp_own_half(p), fast=True)

        # 4) HELD_NO_FLAG: push a held opponent away from our flag, back in time for the release
        for k, p in opps:
            t0 = self.held_since.get(k)
            if (self.is_held(k) and not self.has_our_flag(k) and t0 is not None
                    and self.in_own_half(p[0]) and to_flag(p) < PUSH_RANGE):
                remaining = t0 + HOLD_TIME - now
                if now < t0 + NO_FLAG_PUSH_TIME and remaining > to_flag(me) / V_MAX + RETURN_MARGIN:
                    self.set_state("HELD_NO_FLAG hero_%d: push" % k)
                    return self.push(me, p, self.away_from_flag(p))

        # 5) GUARD: sit on the flag, facing the closest free opponent (the middle line if none),
        #    so we are already pointed at him if he comes within TRIGGER
        self.set_state("GUARD")
        if to_flag(me) > PARK_TOL:
            # flag behind us (e.g. after pushing an opponent away): back up instead of turning
            fwd_err = wrap_angle(math.atan2(home[1] - me[1], home[0] - me[0]) - me[2])
            if math.cos(fwd_err) < 0.0:
                return self.steer_reverse(me, home, fast=False)
            return self.steer(me, home)
        free = [(math.hypot(p[0] - me[0], p[1] - me[1]), p) for k, p in opps if not self.is_held(k)]
        if free:
            _, p = min(free)
            return self.face(me, math.atan2(p[1] - me[1], p[0] - me[0]))
        return self.face(me, 0.0 if self.blue else math.pi)

    def attack(self, me, enemy_flag, carrying):
        """Attacker built on the pick-up quirk (see the module docstring)."""
        r = self.robot
        now = self.now if self.now is not None else 0.0
        t0 = self.held_since.get(r)
        imm_end = t0 + HOLD_TIME + IMMUNE_TIME if t0 is not None else -1.0

        # 1) HELD: our commands are ignored
        if self.is_held(r):
            self.set_state("HELD" + (" with flag" if carrying else ""))
            return 0.0, 0.0

        if carrying:
            # 3) BAIT: we grabbed the flag during our immunity; stay on the flag spot so the
            #    defender's tag lands there and we keep the flag while held
            in_window = t0 is not None and t0 + HOLD_TIME <= now < imm_end + BAIT_GRACE
            enemy_close = any(
                not self.is_held(k) and math.hypot(p[0] - me[0], p[1] - me[1]) < BAIT_RANGE
                for k, p in self.opp.items())
            if in_window and not self.flag_at_release and enemy_close and not self.in_own_half(me[0]):
                self.set_state("BAIT")
                return self.steer(me, self.enemy_home, fast=True)

            # 2) RETURN: reverse home while in the enemy half, turn around once safe
            if self.in_own_half(me[0], SAFE_X):
                self.set_state("RETURN forward")
                return self.steer(me, self.home, fast=True)
            fwd_err = wrap_angle(math.atan2(self.home[1] - me[1], self.home[0] - me[0]) - me[2])
            if math.cos(fwd_err) > 0.5:  # already facing home (e.g. grabbed from behind)
                self.set_state("RETURN forward")
                return self.steer(me, self.home, fast=True)
            self.set_state("RETURN reverse")
            return self.steer_reverse(me, self.home)

        # 4) PREPARE / GRAB: immune without the flag; wait next to it, grab it just before
        #    the immunity ends
        if self.is_immune(r):
            if now >= imm_end - GRAB_LEAD:
                self.set_state("GRAB")
                return self.steer(me, self.enemy_home, fast=True)
            self.set_state("PREPARE")
            ex, ey = self.enemy_home
            dx, dy = me[0] - ex, me[1] - ey
            d = math.hypot(dx, dy)
            if d < 1e-3:
                dx, dy, d = self.side, 0.0, 1.0  # fallback: the side of the flag facing our half
            stage = (ex + STAGE_R * dx / d, ey + STAGE_R * dy / d)
            if math.hypot(stage[0] - me[0], stage[1] - me[1]) > 0.015:
                return self.steer(me, stage)
            return self.face(me, math.atan2(ey - me[1], ex - me[0]))

        # 5) GO_TO_FLAG
        self.set_state("GO_TO_FLAG")
        return self.steer(me, enemy_flag, fast=True)

    def compute_command(self, me, opponents, enemy_flag, own_flag, carrying):
        """Choose (linear_x, angular_z) for this robot's role."""
        if self.role == "defender":
            v, w = self.defend(me)
            return self.keep_in_own_half(me, v, w)
        return self.attack(me, enemy_flag, carrying)

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if (self.me is None or len(self.opp) < len(self.opponents)
                    or self.enemy_flag is None or self.own_flag is None):
                rospy.logwarn_throttle(2.0, "waiting for odometry and flags")
                v, w = 0.0, 0.0
            else:
                try:
                    v, w = self.compute_command(self.me, list(self.opp.values()), self.enemy_flag,
                                                self.own_flag, self.carrying)
                except Exception as e:  # never crash during a match
                    rospy.logerr_throttle(1.0, "compute_command failed: %r" % (e,))
                    v, w = 0.0, 0.0
            cmd = Twist()
            cmd.linear.x, cmd.angular.z = v, w
            self.pub.publish(cmd)
            rate.sleep()


if __name__ == "__main__":
    try:
        CaptureTheFlag().run()
    except rospy.ROSInterruptException:
        pass
