#!/usr/bin/env python3
"""Basic sumo opponent: follow the other robot in a straight pursuit."""
import math

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from psr_hero.utils import V_MAX, W_MAX, clamp, pose_from_odom, wrap_angle


class SumoBasic:
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

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.me is None or self.opp is None:
                rate.sleep()
                continue

            mx, my, myaw = self.me
            ox, oy, _ = self.opp

            dx = ox - mx
            dy = oy - my
            distance = math.hypot(dx, dy)

            if distance > 1e-6:
                ux = dx / distance
                uy = dy / distance
            else:
                ux = 1.0
                uy = 0.0

            target_x = ox + ux * 0.18
            target_y = oy + uy * 0.18
            alpha = wrap_angle(
                math.atan2(target_y - my, target_x - mx) - myaw
            )

            v = clamp(1.6 * distance, 0.0, V_MAX)
            if abs(alpha) > 0.45:
                v *= 0.25
            w = clamp(5.0 * alpha, -W_MAX, W_MAX)

            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = w
            self.pub.publish(cmd)
            rate.sleep()


if __name__ == "__main__":
    SumoBasic().run()
