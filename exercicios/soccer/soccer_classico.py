#!/usr/bin/env python3
"""Soccer 3x3, classical strategy (one node drives the three robots of a team).

Run one node per team: `_team:=blue` (robots 0, 1, 2) attacks +x and
`_team:=red` (robots 3, 4, 5) attacks -x. The field is 1.8 x 1.2 m with the
centre at (0, 0) and a 0.40 m wide goal in the middle of each short side.

Strategy
--------
Everything is computed in a *team frame* where we always attack towards +x
(for red the world is rotated by 180 deg). (v, w) commands are frame-invariant,
so the same code plays on both sides.

Roles, re-evaluated every cycle:
  * GOALKEEPER (fixed: first robot of the team) - slides on a line in front of
    the goal, turned sideways so it only needs linear speed. Covers the point
    where the ball trajectory crosses its line; clears the ball out of the goal
    area when it is slow and no opponent is closer.
  * STRIKER (the field robot with the lowest time-to-ball, with hysteresis) -
    state machine APPROACH -> PUSH. APPROACH goes around the ball to a point
    behind it on the ball->aim line; PUSH drives through the ball towards the
    aim point inside the opponent goal.
  * SUPPORT (the other field robot) - in our half it shadows the ball on the
    ball->own-goal line; in the opponent half it waits for the rebound on the
    far side. It never stands in the striker's corridor.

Low level: bidirectional go-to-point (the HeRo is round, so driving backwards
is as good as forwards), repulsion from nearby robots and a stuck detector that
backs off when a robot is commanded to move but does not.

Topics:
  /hero_N/odom   nav_msgs/Odometry   pose of every robot (N = 0..5)
  /ball/odom     nav_msgs/Odometry   ball position and velocity (world frame)
  /hero_N/cmd_vel  geometry_msgs/Twist   our commands, one topic per robot
  /game/state    std_msgs/String       JSON with the score (logged)

Run:  ./run.sh run soccer      (see README.md in this folder)
"""
import json
import math

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from psr_hero.utils import V_MAX, W_MAX, clamp, pose_from_odom, wrap_angle

FIELD_L, FIELD_W = 1.8, 1.2
GOAL_W = 0.40
TEAMS = {"blue": (0, +1), "red": (3, -1)}  # first robot id, attack direction on x

HALF_L, HALF_W = FIELD_L / 2, FIELD_W / 2
R_ROBOT = 0.037  # HeRo body radius
R_BALL = 0.0214
CONTACT = R_ROBOT + R_BALL  # centre distance when touching the ball

# --- controller gains -------------------------------------------------------
K_V = 1.5  # linear: v = min(v_max, K_V * distance)
K_W = 5.0  # angular: w = K_W * heading error
K_V_GK = 4.0  # goalkeeper: stiffer, it only makes small corrections
REVERSE_HYST = math.radians(110)  # switch driving direction only past this error

# --- striker -----------------------------------------------------------------
BEHIND = CONTACT + 0.025  # distance from the ball to the approach point
ALIGN_DIST = 0.035  # how close to the approach point to start pushing
ALIGN_ANGLE = math.radians(30)  # heading tolerance to start pushing
PUSH_LOST_LAT = 0.035  # leave PUSH when the ball is this far off the push line
PUSH_LOST_DIST = CONTACT + 0.07  # ... or this far ahead of the robot
AROUND_R = 0.11  # clearance when going around the ball
AIM_X = HALF_L + 0.05  # aim behind the goal line
AIM_YS = [-0.14, -0.09, -0.045, 0.0, 0.045, 0.09, 0.14]
SWITCH_MARGIN = 0.35  # s, time advantage needed to steal the striker role

# --- goalkeeper ----------------------------------------------------------------
GK_X = -HALF_L + R_ROBOT + 0.004  # keeper line: right on the goal line, so a ball
#                                  sliding along the back wall cannot pass behind it
GK_POST_Y = GOAL_W / 2 + 0.005  # keeper y when closing a post
GK_Y_MAX = GOAL_W / 2 - 0.02
GK_AREA_X = -HALF_L + 0.24  # clear the ball when it is slow inside this x...
GK_AREA_Y = 0.30  # ... and |y| below this

# --- support -------------------------------------------------------------------
SHADOW_DIST = 0.25
CORRIDOR = 0.12  # half-width of the striker's corridor kept free

# --- stuck detector ------------------------------------------------------------
STUCK_TIME = 0.9  # s commanded to move without moving
STUCK_MOVE = 0.015  # m
ESCAPE_TIME = 0.5  # s
BALL_STALL_TIME = 2.5  # s without the ball moving BALL_STALL_MOVE -> ram it
BALL_STALL_MOVE = 0.03  # m
RAM_TIME = 1.5  # s


def norm(x, y):
    return math.hypot(x, y)


def unit(x, y):
    n = math.hypot(x, y)
    return (x / n, y / n) if n > 1e-9 else (1.0, 0.0)


def seg_dist(p, a, b):
    """Distance from point p to segment a-b."""
    ax, ay = b[0] - a[0], b[1] - a[1]
    L2 = ax * ax + ay * ay
    t = 0.0 if L2 < 1e-12 else clamp(((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / L2, 0.0, 1.0)
    return norm(p[0] - a[0] - t * ax, p[1] - a[1] - t * ay)


def inside_field(p, margin):
    return abs(p[0]) <= HALF_L - margin and abs(p[1]) <= HALF_W - margin


def clamp_field(p, margin):
    return (clamp(p[0], -HALF_L + margin, HALF_L - margin), clamp(p[1], -HALF_W + margin, HALF_W - margin))


def predict_ball(b, v, t):
    """Ball position after t seconds: constant velocity, mild decay, bounces on the walls."""
    k = 0.4  # 1/s, rolling decay (the ball has low friction)
    s = t if k * t < 1e-6 else (1.0 - math.exp(-k * t)) / k
    x, y = b[0] + v[0] * s, b[1] + v[1] * s
    lim_y = HALF_W - R_BALL
    for _ in range(3):
        if y > lim_y:
            y = 2 * lim_y - y
        elif y < -lim_y:
            y = -2 * lim_y - y
    lim_x = HALF_L - R_BALL
    if abs(y) > GOAL_W / 2:  # back walls (outside the goal mouth)
        if x > lim_x:
            x = 2 * lim_x - x
        elif x < -lim_x:
            x = -2 * lim_x - x
    return x, y


# --- telemetria para os logs do torneio ---------------------------------------
# A cada TEL_PERIOD s (relógio do ROS) escreve no log uma linha "TEL {json}" com a
# posição de todos os robôs, bola/bandeiras, estado do jogo e o comando enviado.
# Mudanças de placar/estado do jogo saem na hora como "EVT {json}", e os parâmetros
# do nó saem uma vez como "TELINIT {json}". Os logs aparecem em api/log do site
# do torneio. Isso só gera log: o comportamento do robô continua igual.
TIPO = "classico"
TEL_PERIOD = 1.0  # s entre linhas TEL; 0 desliga (site guarda ~400 linhas)
TEL_IGNORAR = ("field", "ring_radius", "base_x", "base_radius")  # constantes: não vão para o log
TEL_RELOGIO = ("t", "time", "round_time", "held", "message")  # não disparam EVT (vêm junto de placar/tags)


def _tel_r(p, n=3):
    """Arredonda (x, y[, yaw, ...]) para caber no log; None continua None."""
    if p is None:
        return None
    return [round(float(x), n) for x in p]


class Telemetria:
    def __init__(self, period=TEL_PERIOD):
        self.period, self.last, self.ultimo_jogo, self.t0 = period, -1e9, None, None
        self.init, self.eventos = None, True  # eventos=False: este nó não escreve EVT

    @staticmethod
    def _log(tag, campos):
        try:
            rospy.loginfo("%s %s", tag, json.dumps(campos, separators=(",", ":")))
        except Exception as e:  # nunca derrubar o nó por causa do log
            rospy.logwarn_throttle(10.0, "telemetria: %s" % e)

    def inicio(self, eventos=True, **campos):
        self.init, self.eventos = campos, eventos

    def __call__(self, now, game=None, **campos):
        if self.period <= 0:
            return
        if self.t0 is None:
            self.t0 = now
            if self.init is not None:
                self._log("TELINIT", self.init)
        t = round(now - self.t0, 2)  # s desde o início do nó
        game = {k: v for k, v in (game or {}).items() if k not in TEL_IGNORAR}
        chave = json.dumps({k: v for k, v in game.items() if k not in TEL_RELOGIO}, sort_keys=True)
        if self.eventos and self.ultimo_jogo is not None and chave != self.ultimo_jogo:
            self._log("EVT", dict(t=t, jogo=game))
        self.ultimo_jogo = chave
        if now - self.last < self.period:
            return
        self.last = now
        self._log("TEL", dict(t=t, jogo=game, **campos))


class Soccer:
    def __init__(self):
        rospy.init_node("soccer")
        team = rospy.get_param("~team", "blue")
        self.team = team
        self.first, self.attack = TEAMS[team]
        self.ids = [self.first + k for k in range(3)]
        self.opp_ids = [i for i in range(6) if i not in self.ids]
        self.poses = {}  # id -> (x, y, yaw), world frame, for all 6 robots
        self.ball = None  # (x, y), world frame
        self.ball_vel = (0.0, 0.0)  # (vx, vy), world frame
        self.score = None
        self.now = 0.0
        self.init_state()
        self.game = {}  # último /game/state (telemetria)
        self.tel = Telemetria()
        self.tel.inicio(jogo="soccer", tipo=TIPO, team=team, ids=self.ids, opp_ids=self.opp_ids)

        self.pubs = {i: rospy.Publisher("/hero_%d/cmd_vel" % i, Twist, queue_size=1) for i in self.ids}
        for i in range(6):
            rospy.Subscriber("/hero_%d/odom" % i, Odometry, self.on_odom, callback_args=i, queue_size=1)
        rospy.Subscriber("/ball/odom", Odometry, self.on_ball, queue_size=1)
        rospy.Subscriber("/game/state", String, self.on_state, queue_size=1)

    def init_state(self):
        """Controller memory (kept separate from ROS so it can be tested offline)."""
        self.gk = self.ids[0]
        self.striker = self.ids[1]
        self.push = {i: False for i in self.ids}  # striker sub-state: PUSH or APPROACH
        self.drive_dir = {i: 1 for i in self.ids}  # +1 forwards, -1 backwards
        self.aim_y = 0.0
        self.gk_clearing = False
        self.hist = {i: [] for i in self.ids}  # [(t, x, y)] for the stuck detector
        self.busy_since = {i: None for i in self.ids}
        self.escape_until = {i: -1.0 for i in self.ids}
        self.escape_cmd = {i: (0.0, 0.0) for i in self.ids}
        self.side_pref = 1.0  # support: which side it waits on
        self.ball_anchor = None  # (x, y, t): where/when the ball last moved
        self.ram_until = -1.0
        self.next_ram = -1.0

    # ------------------------------------------------------------------ ROS
    def on_odom(self, msg, i):
        self.poses[i] = pose_from_odom(msg)

    def on_ball(self, msg):
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        self.ball = (p.x, p.y)
        self.ball_vel = (v.x, v.y)

    def on_state(self, msg):
        try:
            s = json.loads(msg.data)
            self.game = s
            score = (s.get("blue"), s.get("red"))
        except (ValueError, AttributeError):
            return
        if score != self.score:
            self.score = score
            rospy.loginfo("[%s] placar azul %s x %s vermelho (t=%.1f s)", self.team, score[0], score[1],
                          s.get("time", -1.0))

    # ------------------------------------------------------------ frames
    def to_team(self, p):
        a = self.attack
        if len(p) == 3:
            return a * p[0], a * p[1], wrap_angle(p[2] + (0.0 if a > 0 else math.pi))
        return a * p[0], a * p[1]

    # --------------------------------------------------------- low level
    def go_to(self, i, pose, target, v_max=V_MAX, obstacles=(), slow=True, tol=0.01, k_v=K_V):
        """Bidirectional go-to-point with repulsion. Returns (v, w)."""
        x, y, th = pose
        dx, dy = target[0] - x, target[1] - y
        dist = norm(dx, dy)
        if dist < tol:
            return 0.0, 0.0
        gx, gy = dx / dist, dy / dist
        # repulsion from obstacles (x, y, radius)
        for ox, oy, rad in obstacles:
            rx, ry = x - ox, y - oy
            d = norm(rx, ry)
            if 1e-6 < d < rad:
                k = 1.5 * (rad - d) / rad
                gx += k * rx / d
                gy += k * ry / d
        heading = math.atan2(gy, gx)
        return self.drive(i, th, heading, min(v_max, k_v * dist) if slow else v_max)

    def drive(self, i, th, heading, speed):
        """Follow a heading at a given speed, choosing forwards or backwards with hysteresis."""
        d = self.drive_dir[i]
        alpha = wrap_angle(heading - th - (0.0 if d > 0 else math.pi))
        if abs(alpha) > REVERSE_HYST:
            d = -d
            self.drive_dir[i] = d
            alpha = wrap_angle(alpha + math.pi)
        w = clamp(K_W * alpha, -W_MAX, W_MAX)
        c = max(0.0, math.cos(alpha))
        v = d * speed * c * c
        return clamp(v, -V_MAX, V_MAX), w

    def face(self, i, th, heading, sideways_ok=True):
        """Rotate in place towards heading (or its opposite, the robot is round)."""
        alpha = wrap_angle(heading - th)
        if sideways_ok and abs(alpha) > math.pi / 2:
            alpha = wrap_angle(alpha + math.pi)
        return 0.0, clamp(K_W * alpha, -W_MAX, W_MAX)

    # ---------------------------------------------------- shared helpers
    def time_to(self, pose, p):
        """Rough time (s) for a robot to reach point p (bidirectional)."""
        dx, dy = p[0] - pose[0], p[1] - pose[1]
        a = abs(wrap_angle(math.atan2(dy, dx) - pose[2]))
        a = min(a, math.pi - a)
        return norm(dx, dy) / V_MAX + a / 2.5

    def intercept(self, pose, b, bv):
        """First predicted ball position that the robot can reach in time."""
        if norm(*bv) < 0.03:
            return b
        t = 0.0
        while t <= 2.0:
            q = predict_ball(b, bv, t)
            if self.time_to(pose, q) <= t:
                return q
            t += 0.05
        return predict_ball(b, bv, 2.0)

    def push_dir(self, b, aim):
        """Direction to push the ball: towards aim, rotated as little as needed so that
        the approach point stays inside the field and the push never points at our goal."""
        ux, uy = unit(aim[0] - b[0], aim[1] - b[1])
        base = math.atan2(uy, ux)
        for k in range(0, 13):
            for s in ((1, -1) if k else (1,)):
                a = base + s * k * math.radians(12)
                u = (math.cos(a), math.sin(a))
                behind = (b[0] - BEHIND * u[0], b[1] - BEHIND * u[1])
                if not inside_field(behind, R_ROBOT + 0.004):
                    continue
                if u[0] < -0.05:  # would the ball travel into our own goal?
                    y_at = b[1] + u[1] * (-HALF_L - b[0]) / u[0]
                    if abs(y_at) < GOAL_W / 2 + 0.12:
                        continue
                return u
        return ux, uy

    def choose_aim(self, b, opps):
        """Aim point inside the opponent goal with the clearest shooting lane."""
        best, best_score = self.aim_y, -1e9
        for ay in AIM_YS:
            clear = min([seg_dist(o, b, (AIM_X, ay)) for o in opps] + [0.2])
            score = clear - 0.15 * abs(ay) + (0.02 if ay == self.aim_y else 0.0)
            if score > best_score:
                best, best_score = ay, score
        self.aim_y = best
        return AIM_X, best

    # ---------------------------------------------------------- behaviours
    def striker_cmd(self, i, P, b, bv, aim, team_obst, opps):
        pose = P[i]
        x, y, th = pose
        q = self.intercept(pose, b, bv) if not self.push[i] else b
        u = self.push_dir(q, aim)
        n = (-u[1], u[0])
        rel = (x - q[0], y - q[1])
        s = rel[0] * u[0] + rel[1] * u[1]  # < 0: robot behind the ball
        lat = rel[0] * n[0] + rel[1] * n[1]  # signed lateral offset

        if self.push[i]:
            # leave PUSH if the ball escaped the push line or ran away
            bl = (b[0] - x, b[1] - y)
            along = bl[0] * u[0] + bl[1] * u[1]
            off = abs(bl[0] * n[0] + bl[1] * n[1])
            if along < 0.0 or off > PUSH_LOST_LAT or norm(*bl) > PUSH_LOST_DIST:
                self.push[i] = False
            else:
                # aim the robot through the ball and towards the goal
                hx, hy = unit(bl[0] / max(norm(*bl), 1e-6) + u[0], bl[1] / max(norm(*bl), 1e-6) + u[1])
                return self.drive(i, th, math.atan2(hy, hx), V_MAX)

        behind = (q[0] - BEHIND * u[0], q[1] - BEHIND * u[1])
        if s > -0.02 or abs(lat) > 0.10 and s > -0.06:
            # in front of / beside the ball: go around it on our side
            side = 1.0 if lat >= 0 else -1.0
            wp = (q[0] + side * AROUND_R * n[0] - 0.05 * u[0], q[1] + side * AROUND_R * n[1] - 0.05 * u[1])
            wp = clamp_field(wp, R_ROBOT + 0.005)
            obst = team_obst + [(b[0], b[1], AROUND_R - 0.01)]
            return self.go_to(i, pose, wp, obstacles=obst, slow=False)

        # behind the ball: approach point pushed back by the lateral error,
        # so the robot arrives already aligned with the push direction
        back = min(0.15, 1.2 * abs(lat))
        tgt = (behind[0] - back * u[0], behind[1] - back * u[1])
        tgt = clamp_field(tgt, R_ROBOT + 0.004)
        heading_err = abs(wrap_angle(math.atan2(u[1], u[0]) - th))
        heading_err = min(heading_err, math.pi - heading_err)
        aligned = norm(x - behind[0], y - behind[1]) < ALIGN_DIST and heading_err < ALIGN_ANGLE
        # already behind the ball and on the push line: no need to reach the exact point
        on_line = s < -CONTACT + 0.01 and s > -BEHIND - 0.06 and abs(lat) < 0.025 and heading_err < ALIGN_ANGLE
        # contested ball: an opponent is on it, so charge as soon as we are reasonably behind
        contested = min(norm(o[0] - b[0], o[1] - b[1]) for o in opps) < 0.10
        charge = contested and s < -CONTACT + 0.01 and abs(lat) < 0.05 and s > -0.25
        if aligned or on_line or charge:
            bl = (b[0] - x, b[1] - y)
            d_ball = max(norm(*bl), 1e-6)
            self.push[i] = d_ball < PUSH_LOST_DIST
            hx, hy = unit(bl[0] / d_ball + u[0], bl[1] / d_ball + u[1])
            return self.drive(i, th, math.atan2(hy, hx), V_MAX)
        if back < 0.02 and norm(x - behind[0], y - behind[1]) < ALIGN_DIST:
            return self.face(i, th, math.atan2(u[1], u[0]))
        return self.go_to(i, pose, tgt, obstacles=team_obst, slow=False)

    def goalkeeper_cmd(self, i, P, b, bv, opps):
        pose = P[i]
        x, y, th = pose
        speed = norm(*bv)
        in_area = b[0] < GK_AREA_X and abs(b[1]) < GK_AREA_Y
        d_me = norm(b[0] - x, b[1] - y)
        d_opp = min(norm(b[0] - o[0], b[1] - o[1]) for o in opps)
        ball_ahead = b[0] > x + 0.01  # ball on the field side of the keeper
        if self.gk_clearing:
            # abort if the ball is shot at our goal, gets behind the keeper or an opponent arrives
            self.gk_clearing = in_area and ball_ahead and d_opp > 0.05 and bv[0] > -0.10 and speed < 0.30
        elif in_area and speed < 0.15 and d_me + 0.08 < d_opp and ball_ahead:
            self.gk_clearing = True
        if not self.gk_clearing:
            self.push[i] = False
        if self.gk_clearing:
            sy = 1.0 if b[1] >= 0 else -1.0
            aim = (b[0] + 0.5, b[1] + sy * 0.6)  # out to the side and forwards
            return self.striker_cmd(i, P, b, (0.0, 0.0), aim, [], opps)

        # where the ball crosses the keeper line, or a fraction of its y
        y_des = 0.6 * b[1]
        if bv[0] < -0.05 and b[0] > GK_X:
            t = 0.0
            while t < 3.0:
                q = predict_ball(b, bv, t)
                if q[0] <= GK_X:
                    y_des = q[1]
                    break
                t += 0.02
        y_des = clamp(y_des, -GK_Y_MAX, GK_Y_MAX)
        gk_x = GK_X
        if b[0] < GK_X + 0.06:
            if GOAL_W / 2 - 0.04 < abs(b[1]) < GOAL_W / 2 + 0.25:
                # ball running along our goal line from a corner (or already at the
                # post): plug the near post, leaving no gap between body and back wall
                y_des = GK_POST_Y if b[1] >= 0 else -GK_POST_Y
            elif abs(b[1]) <= GOAL_W / 2 - 0.04:
                # ball in front of the mouth, at the keeper's depth: shield it with the body
                y_des = clamp(b[1], -GK_Y_MAX, GK_Y_MAX)
        tgt = (gk_x, y_des)
        if abs(x - gk_x) > 0.006 or abs(y - y_des) > 0.015:
            v_max = V_MAX if abs(x - gk_x) < 0.08 else 0.8 * V_MAX
            v, w = self.go_to(i, pose, tgt, v_max=v_max, tol=0.004, k_v=K_V_GK)
            return v, w
        return self.face(i, th, math.pi / 2)  # stay turned sideways, ready to slide

    def support_cmd(self, i, P, b, aim, striker_pose, team_obst):
        pose = P[i]
        # ball coming at us while we stand between it and our goal: meet it and push it
        # away. Backing off to the shadow point would dribble it into our own goal.
        d_ball = norm(b[0] - pose[0], b[1] - pose[1])
        if b[0] < 0.0 and pose[0] < b[0] - 0.01 and d_ball < 0.15:
            return self.go_to(i, pose, b, slow=False, tol=0.0)
        if b[0] < 0.0:
            # shadow: between the ball and our goal
            gx, gy = unit(-HALF_L - b[0], 0.0 - b[1])
            tgt = (b[0] + SHADOW_DIST * gx, b[1] + SHADOW_DIST * gy)
            tgt = (max(tgt[0], -HALF_L + 0.18), tgt[1])
        else:
            # wait for the rebound, on the side opposite to the ball
            if abs(b[1]) > 0.06:
                self.side_pref = -1.0 if b[1] > 0 else 1.0
            tgt = (clamp(b[0] - 0.30, -0.45, HALF_L - 0.30), self.side_pref * 0.22)
        # keep out of the striker's corridor (ball -> aim)
        d = seg_dist(tgt, b, aim)
        if d < CORRIDOR:
            ux, uy = unit(aim[0] - b[0], aim[1] - b[1])
            n = (-uy, ux)
            sgn = 1.0 if (tgt[0] - b[0]) * n[0] + (tgt[1] - b[1]) * n[1] >= 0 else -1.0
            tgt = (tgt[0] + sgn * (CORRIDOR - d + 0.02) * n[0], tgt[1] + sgn * (CORRIDOR - d + 0.02) * n[1])
        tgt = clamp_field(tgt, R_ROBOT + 0.01)
        obst = team_obst + [(b[0], b[1], 0.10), (striker_pose[0], striker_pose[1], 0.15)]
        v, w = self.go_to(i, pose, tgt, obstacles=obst, tol=0.02)
        if v == 0.0 and w == 0.0:
            return self.face(i, pose[2], math.atan2(b[1] - pose[1], b[0] - pose[0]))
        return v, w

    @staticmethod
    def ram_safe(pose, b):
        """Ramming the ball from pose must not send it towards our own goal."""
        ux, uy = unit(b[0] - pose[0], b[1] - pose[1])
        if b[0] > -0.3 and ux > -0.3:
            return True
        if ux >= 0.0:
            return True
        y_at = b[1] + uy * (-HALF_L - b[0]) / ux
        return abs(y_at) > GOAL_W / 2 + 0.12

    # ------------------------------------------------------- stuck handling
    def unstick(self, i, P, cmd):
        """Back off when a robot is told to move but does not move."""
        x, y, th = P[i]
        t = self.now
        h = self.hist[i]
        h.append((t, x, y))
        while h and t - h[0][0] > STUCK_TIME + 0.3:
            h.pop(0)
        if t < self.escape_until[i]:
            return self.escape_cmd[i]
        if abs(cmd[0]) > 0.08:
            if self.busy_since[i] is None:
                self.busy_since[i] = t
        else:
            self.busy_since[i] = None
        if self.busy_since[i] is not None and t - self.busy_since[i] > STUCK_TIME:
            old = [p for p in h if t - p[0] >= STUCK_TIME]
            if old and norm(x - old[-1][1], y - old[-1][2]) < STUCK_MOVE:
                self.escape_until[i] = t + ESCAPE_TIME
                self.escape_cmd[i] = (-math.copysign(0.15, cmd[0]), 2.5 if (i + int(t)) % 2 else -2.5)
                self.busy_since[i] = None
                self.push[i] = False
                return self.escape_cmd[i]
        return cmd

    # --------------------------------------------------------------- main
    def compute_commands(self, poses, ball):
        """Return a list of three (linear_x, angular_z), one per robot of self.ids."""
        P = {i: self.to_team(p) for i, p in poses.items()}
        b = self.to_team(ball)
        bv = self.to_team(self.ball_vel)
        opps = [P[i][:2] for i in self.opp_ids]
        field = [i for i in self.ids if i != self.gk]

        # --- striker selection with hysteresis
        def cost(i):
            q = self.intercept(P[i], b, bv)
            c = self.time_to(P[i], (q[0] - BEHIND, q[1]))
            if P[i][0] > q[0] + 0.02:  # on the wrong side of the ball
                c += 0.8
            return c

        costs = {i: cost(i) for i in field}
        other = [i for i in field if i != self.striker][0]
        if costs[other] + SWITCH_MARGIN < costs[self.striker]:
            self.push[self.striker] = False
            self.striker = other
        support = [i for i in field if i != self.striker][0]

        aim = self.choose_aim(b, opps)

        # --- deadlock breaker: ball jammed (corner, wall, head-on push) -> ram it
        if self.ball_anchor is None or norm(b[0] - self.ball_anchor[0], b[1] - self.ball_anchor[1]) > BALL_STALL_MOVE:
            self.ball_anchor = (b[0], b[1], self.now)
        stalled_for = self.now - self.ball_anchor[2]
        if stalled_for > BALL_STALL_TIME and self.now > self.next_ram:
            self.ram_until = self.now + RAM_TIME
            self.next_ram = self.now + RAM_TIME + BALL_STALL_TIME
        ram = self.now < self.ram_until and self.ram_safe(P[self.striker], b)

        cmds = {}
        mates = lambda me: [(P[j][0], P[j][1], 0.12) for j in self.ids if j != me]  # noqa: E731
        cmds[self.gk] = self.goalkeeper_cmd(self.gk, P, b, bv, opps)
        if ram:
            self.push[self.striker] = False
            cmds[self.striker] = self.go_to(self.striker, P[self.striker], b, slow=False, tol=0.0)
        else:
            cmds[self.striker] = self.striker_cmd(self.striker, P, b, bv, aim, mates(self.striker), opps)
        cmds[support] = self.support_cmd(support, P, b, aim, P[self.striker], mates(support))
        return [self.unstick(i, P, cmds[i]) for i in self.ids]

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.ball is None or len(self.poses) < 6:
                rospy.logwarn_throttle(2.0, "waiting for odometry and ball")
            else:
                self.now = rospy.get_time()
                try:
                    cmds = self.compute_commands(dict(self.poses), self.ball)
                except Exception as e:  # never let the node die mid-match
                    rospy.logerr_throttle(1.0, "soccer: %s" % e)
                    cmds = [(0.0, 0.0)] * 3
                for i, (v, w) in zip(self.ids, cmds):
                    cmd = Twist()
                    cmd.linear.x, cmd.angular.z = v, w
                    self.pubs[i].publish(cmd)
                self.tel(self.now, self.game, ball=_tel_r(tuple(self.ball) + tuple(self.ball_vel)),
                         robots={i: _tel_r(self.poses[i]) for i in sorted(self.poses)},
                         striker=self.striker,
                         cmd={i: _tel_r(c) for i, c in zip(self.ids, cmds)})
            rate.sleep()


if __name__ == "__main__":
    Soccer().run()
