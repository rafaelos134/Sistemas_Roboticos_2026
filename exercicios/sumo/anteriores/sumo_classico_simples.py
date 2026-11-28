#!/usr/bin/env python3
"""Versão anterior do sumô clássico (commit 401b601, Vinicius): perseguição em linha reta.
Mesma lógica do original; só foi movida para compute_command() e a classe renomeada para Sumo,
para poder ser usada por rl/eval_sumo.py."""
import math

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from psr_hero.utils import V_MAX, W_MAX, clamp, pose_from_odom, wrap_angle


class Sumo:
    def __init__(self):
        rospy.init_node("sumo_basic")
        self.robot = rospy.get_param("~robot", 1)
        self.opponent = rospy.get_param("~opponent", 0)
        self.me = None
        self.opp = None
        self.pub = rospy.Publisher("/hero_%d/cmd_vel" % self.robot, Twist, queue_size=1)
        rospy.Subscriber("/hero_%d/odom" % self.robot, Odometry, self.on_me)
        rospy.Subscriber("/hero_%d/odom" % self.opponent, Odometry, self.on_opp)

    def on_me(self, msg):
        self.me = pose_from_odom(msg)

    def on_opp(self, msg):
        self.opp = pose_from_odom(msg)

    def compute_command(self, me, opp, speed, now):
        mx, my, myaw = me
        ox, oy, _ = opp
        dx, dy = ox - mx, oy - my
        distance = math.hypot(dx, dy)
        if distance > 1e-6:
            ux, uy = dx / distance, dy / distance
        else:
            ux, uy = 1.0, 0.0
        target_x, target_y = ox + ux * 0.18, oy + uy * 0.18
        alpha = wrap_angle(math.atan2(target_y - my, target_x - mx) - myaw)
        v = clamp(1.6 * distance, 0.0, V_MAX)
        if abs(alpha) > 0.45:
            v *= 0.25
        w = clamp(5.0 * alpha, -W_MAX, W_MAX)
        return v, w

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.me is not None and self.opp is not None:
                v, w = self.compute_command(self.me, self.opp, 0.0, rospy.get_time())
                cmd = Twist()
                cmd.linear.x, cmd.angular.z = v, w
                self.pub.publish(cmd)
            rate.sleep()


if __name__ == "__main__":
    Sumo().run()
