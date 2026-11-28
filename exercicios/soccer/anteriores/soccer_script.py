#!/usr/bin/env python3
"""Soccer, deterministic baseline team (no learning): keeper + striker + support.

  keeper   (first id):  on its goal line, y = ball y clipped to the goal mouth; faces the ball.
  striker  (the field player closer to the ball): goes behind the ball on the line ball -> goal
           (around the ball if it is on the wrong side) and pushes it to the goal.
  support  (the other one): waits behind the ball at (ball_x - 0.30, ball_y / 2).

Everything is computed in the team frame (the team attacks +x; red sees the field rotated
by 180 deg), so the same file plays blue or red. It is the scripted red team of the RL
training (stage A2), here with fixed gains.

Run:  ./run.sh run soccer blue:=soccer_rl.py red:=soccer_script.py
"""
import math

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from psr_hero.utils import V_MAX, W_MAX, clamp, pose_from_odom, wrap_angle

FIELD_L, FIELD_W = 1.8, 1.2
GOAL_W = 0.40
HALF_L, HALF_W, GOAL_HALF = FIELD_L / 2, FIELD_W / 2, GOAL_W / 2
TEAMS = {"blue": (0, +1), "red": (3, -1)}
KW, ALIGN, BEHIND = 4.0, 0.5, 0.08


def go_to(me, tx, ty, face=None):
    dx, dy = tx - me[0], ty - me[1]
    if face is not None and math.hypot(dx, dy) < 0.03:
        return 0.0, clamp(KW * wrap_angle(face - me[2]), -W_MAX, W_MAX)
    err = wrap_angle(math.atan2(dy, dx) - me[2])
    return (V_MAX if abs(err) < ALIGN else 0.0), clamp(KW * err, -W_MAX, W_MAX)


def keeper(me, ball):
    ty = clamp(ball[1], -GOAL_HALF + 0.03, GOAL_HALF - 0.03)
    return go_to(me, -HALF_L + 0.07, ty, face=math.atan2(ball[1] - me[1], ball[0] - me[0]))


def striker(me, ball):
    ux, uy = HALF_L - ball[0], -ball[1]
    d = max(math.hypot(ux, uy), 1e-6)
    ux, uy = ux / d, uy / d
    bx, by = ball[0] - BEHIND * ux, ball[1] - BEHIND * uy
    rx, ry = ball[0] - me[0], ball[1] - me[1]
    along, cross = rx * ux + ry * uy, -rx * uy + ry * ux
    if along > 0.02 and abs(cross) < 0.035:  # behind the ball and aligned: push
        tx, ty = ball[0] + 0.10 * ux, ball[1] + 0.10 * uy
    elif along < 0:  # wrong side: go around
        side = -1.0 if cross > 0 else 1.0
        tx, ty = bx + side * 0.09 * -uy, by + side * 0.09 * ux
    else:
        tx, ty = bx, by
    return go_to(me, clamp(tx, -HALF_L + 0.04, HALF_L - 0.04), clamp(ty, -HALF_W + 0.04, HALF_W - 0.04))


class Soccer:
    def __init__(self):
        rospy.init_node("soccer")
        team = rospy.get_param("~team", "blue")
        self.first, self.attack = TEAMS[team]
        self.ids = [self.first + k for k in range(3)]
        self.poses = {}
        self.ball = None
        self.pubs = {i: rospy.Publisher("/hero_%d/cmd_vel" % i, Twist, queue_size=1) for i in self.ids}
        for i in range(6):
            rospy.Subscriber("/hero_%d/odom" % i, Odometry, self.on_odom, callback_args=i, queue_size=1)
        rospy.Subscriber("/ball/odom", Odometry, self.on_ball, queue_size=1)

    def on_odom(self, msg, i):
        self.poses[i] = pose_from_odom(msg)

    def on_ball(self, msg):
        self.ball = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def frame(self, p):
        """World -> team frame (rotation by 180 deg for red)."""
        if self.attack > 0:
            return p
        if len(p) == 2:
            return (-p[0], -p[1])
        return (-p[0], -p[1], wrap_angle(p[2] + math.pi))

    def compute_commands(self, poses, ball):
        mine = [self.frame(poses[i]) for i in self.ids]
        b = self.frame(ball)
        out = [keeper(mine[0], b)]
        d1 = math.hypot(mine[1][0] - b[0], mine[1][1] - b[1])
        d2 = math.hypot(mine[2][0] - b[0], mine[2][1] - b[1])
        sx, sy = clamp(b[0] - 0.30, -HALF_L + 0.15, HALF_L - 0.15), b[1] / 2
        if d1 <= d2:
            out += [striker(mine[1], b), go_to(mine[2], sx, sy)]
        else:
            out += [go_to(mine[1], sx, sy), striker(mine[2], b)]
        return out

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.ball is None or len(self.poses) < 6:
                rospy.logwarn_throttle(2.0, "waiting for odometry and ball")
            else:
                for i, (v, w) in zip(self.ids, self.compute_commands(self.poses, self.ball)):
                    cmd = Twist()
                    cmd.linear.x, cmd.angular.z = v, w
                    self.pubs[i].publish(cmd)
            rate.sleep()


if __name__ == "__main__":
    Soccer().run()
