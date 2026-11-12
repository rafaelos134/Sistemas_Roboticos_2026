#!/usr/bin/env python3
"""Go-to-goal navigation (student template).

Goal: drive the robot to the goal point, then to the next one the simulator
draws, as many times as possible. A goal counts when the robot centre gets
within 6 cm of it.

Topics:
  /hero_N/odom   nav_msgs/Odometry         pose in the 'world' frame (x, y, yaw)
  /hero_N/goal   geometry_msgs/PointStamped  current goal (world frame)
  /hero_N/cmd_vel  geometry_msgs/Twist      what you publish (linear.x, angular.z)

Run:  ./run.sh run goto      (see README.md in this folder)
"""
import csv
import math  # noqa: F401
import os
import time

import rospy
from geometry_msgs.msg import PointStamped, Twist  # noqa: F401
from nav_msgs.msg import Odometry  # noqa: F401

from psr_hero.utils import V_MAX, W_MAX, clamp, pose_from_odom, wrap_angle  # noqa: F401

K_RHO = 3.0  # m/s of forward speed per metre of distance to the goal
K_ALPHA = 6.0  # rad/s of turn per radian of heading error


class Goto:
    def __init__(self):
        rospy.init_node("goto")
        self.robot = rospy.get_param("~robot", 0)
        self.pose = None  # (x, y, yaw)
        self.goal = None  # (x, y)
        self.t0 = rospy.get_time()

        # Ready: the publisher for the velocity command.
        self.pub = rospy.Publisher("/hero_%d/cmd_vel" % self.robot, Twist, queue_size=1)

        # TODO 1 (ROS): subscribe to /hero_<robot>/odom (Odometry) -> self.on_odom
        #               and to /hero_<robot>/goal (PointStamped) -> self.on_goal.
        self.odom_sub = rospy.Subscriber("/hero_%d/odom" % self.robot, Odometry, self.on_odom)
        self.goal_sub = rospy.Subscriber("/hero_%d/goal" % self.robot, PointStamped, self.on_goal)

        self._init_log()

    def _init_log(self):
        log_dir = os.path.dirname(os.path.abspath(__file__))
        default_name = "goto_log_%s.csv" % time.strftime("%Y%m%d_%H%M%S")
        log_path = rospy.get_param("~log_path", os.path.join(log_dir, default_name))

        self.log_file = open(log_path, "w", newline="")
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow(["t", "x", "y", "yaw", "gx", "gy", "v", "w"])
        rospy.loginfo("goto: logging trial data to %s", log_path)
        rospy.on_shutdown(self.log_file.close)

    def log(self, pose, goal, v, w):
        t = rospy.get_time() - self.t0
        x, y, yaw = pose
        gx, gy = goal
        self.log_writer.writerow([t, x, y, yaw, gx, gy, v, w])
        self.log_file.flush()

    def on_odom(self, msg):
        # TODO 1 (ROS): store the pose. pose_from_odom(msg) returns (x, y, yaw).
        self.pose = pose_from_odom(msg)

    def on_goal(self, msg):
        # TODO 1 (ROS): store the goal point (msg.point.x, msg.point.y).
        self.goal = (msg.point.x, msg.point.y)

    def compute_command(self, pose, goal):
        """TODO 2: a controller that drives the robot to the goal.

        pose: (x, y, yaw), goal: (x, y). Returns (linear_x, angular_z).
        Hint: compute the heading error (use wrap_angle), turn proportionally to
        it and move forward only when the robot is roughly pointing at the goal.
        """
        x, y, yaw = pose
        gx, gy = goal

        rho = math.hypot(gx - x, gy - y)
        alpha = wrap_angle(math.atan2(gy - y, gx - x) - yaw)

        w = clamp(K_ALPHA * alpha, -W_MAX, W_MAX)
        # Full speed when pointing at the goal, tapering to 0 as the heading
        # error approaches +-90 degrees (negative cosine means the goal is
        # behind: turn in place instead of driving backwards).
        v = clamp(K_RHO * rho, 0.0, V_MAX) * clamp(math.cos(alpha), 0.0, 1.0)

        return v, w

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.pose is None or self.goal is None:
                rospy.logwarn_throttle(2.0, "waiting for pose and goal (TODO 1)")
            else:
                v, w = self.compute_command(self.pose, self.goal)
                cmd = Twist()
                cmd.linear.x, cmd.angular.z = v, w
                self.pub.publish(cmd)
                self.log(self.pose, self.goal, v, w)
            rate.sleep()


if __name__ == "__main__":
    Goto().run()
