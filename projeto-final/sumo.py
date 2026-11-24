#!/usr/bin/env python3
"""Sumo (student template).

Push the other robot out of the circular ring (radius 0.40 m, centre (0, 0)).
A robot loses when its centre leaves the ring; a round ends in a draw after
40 s. There is no wall: use the odometry to know where you are.

Both robots have the same traction, so a head-on push is a stalemate. Think
about angles, timing and not leaving the ring yourself.

Topics:
  /hero_N/odom   nav_msgs/Odometry   pose of every robot
  /hero_N/laser  sensor_msgs/LaserScan   the 8-beam lidar also sees the opponent
  /hero_N/cmd_vel  geometry_msgs/Twist   your command

Run:  ./run.sh run sumo      (see README.md in this folder)
"""
import math  # noqa: F401

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry  # noqa: F401

from psr_hero.utils import V_MAX, W_MAX, clamp, pose_from_odom, wrap_angle  # noqa: F401

RING_RADIUS = 0.40


class Sumo:
    def __init__(self):
        rospy.init_node("sumo")
        self.robot = rospy.get_param("~robot", 0)
        self.opponent = rospy.get_param("~opponent", 1)
        self.me = None  # (x, y, yaw)
        self.speed = 0.0  # own forward speed, m/s
        self.opp = None  # (x, y, yaw)

        # Ready: the publisher for the velocity command.
        self.pub = rospy.Publisher("/hero_%d/cmd_vel" % self.robot, Twist, queue_size=1)

        # TODO 1 (ROS): subscribe to the odometry of your robot (self.robot) -> self.on_me
        #               and of the opponent (self.opponent) -> self.on_opp.
        rospy.Subscriber("/hero_%d/odom" % self.robot, Odometry, self.on_me)
        rospy.Subscriber("/hero_%d/odom" % self.opponent, Odometry, self.on_opp)
        self.stall = 0.0
        self.best_ro = 0.0
        self.wiggle_until = 0.0
        self.wiggle_sign = 1.0

    def on_me(self, msg):
        # TODO 1 (ROS): self.me = pose_from_odom(msg); self.speed = msg.twist.twist.linear.x
        self.me = pose_from_odom(msg)
        self.speed = msg.twist.twist.linear.x

    def on_opp(self, msg):
        # TODO 1 (ROS): self.opp = pose_from_odom(msg)
        self.opp = pose_from_odom(msg)

    def compute_command(self, me, opp, speed, now):
        """TODO 2: choose (linear_x, angular_z).

        me, opp: (x, y, yaw). speed: own forward speed (m/s). now: time (s).
        """
        mx, my, myaw = me
        ox, oy, _ = opp
        dx, dy = ox - mx, oy - my
        d = math.hypot(dx, dy) or 1e-6
        ux, uy = dx / d, dy / d
        r = math.hypot(mx, my) or 1e-6
        ro = math.hypot(ox, oy)
        if ro > 0.06:
            rx, ry = ox / ro, oy / ro
        else:
            rx, ry = -mx / r, -my / r
        c = ux * rx + uy * ry

        if d >= 0.10:
            self.best_ro = ro
            self.stall = now
            return self.go_to(me, ox - 0.12 * rx, oy - 0.12 * ry)

        if c < 0.85:
            err = wrap_angle(math.atan2(-ry, -rx) - math.atan2(-uy, -ux))
            self.stall = now
            return self.orbit(me, ux, uy, d, -1.0 if err > 0.0 else 1.0)

        if now < self.wiggle_until:
            return self.orbit(me, ux, uy, d, self.wiggle_sign)

        if ro > self.best_ro + 0.005:
            self.best_ro = ro
            self.stall = now
        if now - self.stall > 0.6:
            self.wiggle_sign = -self.wiggle_sign
            self.wiggle_until = now + 0.35
            self.stall = now
            self.best_ro = ro
        return self.go_to(me, ox + 0.25 * rx, oy + 0.25 * ry)

    def orbit(self, me, ux, uy, d, s):
        ang = math.atan2(uy, ux) + s * math.radians(60.0)
        ff = -s * V_MAX * math.sin(math.radians(60.0)) / max(d, 0.06)
        return V_MAX, clamp(ff + 8.0 * wrap_angle(ang - me[2]), -W_MAX, W_MAX)

    def go_to(self, me, tx, ty):
        mx, my, myaw = me
        alpha = wrap_angle(math.atan2(ty - my, tx - mx) - myaw)
        direction = 1.0
        if abs(alpha) > math.pi / 2.0:
            direction = -1.0
            alpha = wrap_angle(alpha - math.pi)
        return direction * V_MAX * clamp(math.cos(alpha), 0.0, 1.0), clamp(6.0 * alpha, -W_MAX, W_MAX)

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.me is None or self.opp is None:
                rospy.logwarn_throttle(2.0, "waiting for odometry (TODO 1)")
            else:
                v, w = self.compute_command(self.me, self.opp, self.speed, rospy.get_time())
                cmd = Twist()
                cmd.linear.x, cmd.angular.z = v, w
                self.pub.publish(cmd)
            rate.sleep()


if __name__ == "__main__":
    Sumo().run()