#!/usr/bin/env python3
"""Obstacle avoidance with the 8-beam lidar (student template).

Goal: keep the robot moving in the arena without touching the obstacle in the
middle or the walls.

Lidar: /hero_N/laser (sensor_msgs/LaserScan), 8 beams, range 0.037 .. 0.5 m.
Beam k points at the angle -pi + k * pi/4 from the robot heading
(counter-clockwise positive), so beam 4 looks forward, beams 3 and 5 look 45
degrees to the right and to the left, and beam 0 looks backwards. A beam that
sees nothing reports 0.5. Distances are measured from the robot centre, so a
robot touching a wall reads about 0.037.

Run:  ./run.sh run avoid      (see README.md in this folder)
"""
import csv
import os
import time

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan  # noqa: F401  (you will need it for TODO 1)

from psr_hero.utils import V_MAX, W_MAX, beam_angle, clamp, pose_from_odom  # noqa: F401
from psr_hero.utils import LIDAR_RANGE_MIN

# Beam indices (see header): 2/3 look right-ish, 5/6 look left-ish, 4 is forward.
#K_V = 0.6  # m/s of forward speed per metre of clearance ahead
#K_W = 8.0  # rad/s of turn per metre of left/right imbalance
#D_SLOW = 0.25  # start slowing down / turning harder inside this front distance (m)

K_V = 1.0
K_W = 4.0
D_SLOW = 0.145

class Avoid:
    def __init__(self):
        rospy.init_node("avoid")
        self.robot = rospy.get_param("~robot", 0)
        self.scan = None
        self.pose = None  # (x, y, yaw), logged only, never used in the control law
        self.t0 = rospy.get_time()

        # Ready: the publisher for the velocity command.
        self.pub = rospy.Publisher("/hero_%d/cmd_vel" % self.robot, Twist, queue_size=1)

        # TODO 1 (ROS): subscribe to the lidar topic of this robot
        #   (/hero_<robot>/laser, message type LaserScan) and call self.on_scan.
        self.sub = rospy.Subscriber("/hero_%d/laser" % self.robot, LaserScan, self.on_scan)

        # Pose is only for the trial log (trajectory plots), not for control.
        self.odom_sub = rospy.Subscriber("/hero_%d/odom" % self.robot, Odometry, self.on_odom)

        self._init_log()

    def on_scan(self, scan):
        # TODO 1 (ROS): keep the latest message in self.scan.
        self.scan = scan

    def on_odom(self, msg):
        self.pose = pose_from_odom(msg)

    def _init_log(self):
        log_dir = os.path.dirname(os.path.abspath(__file__))
        default_name = "avoid_log_%s.csv" % time.strftime("%Y%m%d_%H%M%S")
        log_path = rospy.get_param("~log_path", os.path.join(log_dir, default_name))

        self.log_file = open(log_path, "w", newline="")
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow(["t", "x", "y", "yaw", "min_range", "v", "w"])
        rospy.loginfo("avoid: logging trial data to %s", log_path)
        rospy.on_shutdown(self.log_file.close)

    def log(self, ranges, v, w):
        t = rospy.get_time() - self.t0
        x, y, yaw = self.pose if self.pose is not None else (float("nan"),) * 3
        self.log_writer.writerow([t, x, y, yaw, min(ranges), v, w])
        self.log_file.flush()

    def compute_command(self, ranges):
        """TODO 2: turn the 8 lidar ranges into a velocity command.

        ranges: list with 8 floats (m). Returns (linear_x [m/s], angular_z [rad/s]).
        Linear speed is limited to V_MAX = 0.2 m/s and angular speed to W_MAX = 4 rad/s.
        Hint: the robot should keep moving, slow down when something is in front
        and turn away from the side where the obstacle is closer.
        """
        front = min(ranges[3], ranges[4], ranges[5])
        left = ranges[5] + ranges[6]
        right = ranges[2] + ranges[3]

        # Base speed grows with the clearance ahead; turn towards the side
        # that has more free space (positive angular.z turns left in ROS).
        v = clamp(K_V * (front - LIDAR_RANGE_MIN), 0.0, V_MAX)
        w = clamp(K_W * (left - right), -W_MAX, W_MAX)

        if front < D_SLOW:
            # Something close ahead: slow down further and force a stronger
            # turn so the robot does not stall head-on into the obstacle.
            v *= clamp(front / D_SLOW, 0.0, 1.0)
            turn_sign = 1.0 if left >= right else -1.0
            w = clamp(w + turn_sign * 0.4 * W_MAX, -W_MAX, W_MAX)

        return v, w

    def publish(self, v, w):
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.pub.publish(cmd)

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.scan is None:
                rospy.logwarn_throttle(2.0, "no lidar data yet: is the subscriber of TODO 1 in place?")
            else:
                ranges = list(self.scan.ranges)
                v, w = self.compute_command(ranges)
                self.publish(v, w)
                self.log(ranges, v, w)
            rate.sleep()


if __name__ == "__main__":
    Avoid().run()
