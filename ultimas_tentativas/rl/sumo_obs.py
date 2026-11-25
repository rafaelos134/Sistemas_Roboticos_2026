"""Observation of the sumo policy: one definition in torch (training) and one in numpy
(the ROS node). Everything is written in the agent's own frame (rotated by -yaw),
so the same policy works as hero_0 or hero_1 and for any rotation of the ring.

Input per robot: (x, y, yaw, v, w), as given by /hero_N/odom
(v = twist.twist.linear.x, the forward speed; w = twist.twist.angular.z).

  0-1  own position relative to the ring centre, in the own frame       / R
  2-3  opponent position relative to the agent, in the own frame         / R
  4-5  opponent position relative to the ring centre, in the own frame   / R
  6-7  cos, sin of (opponent yaw - own yaw)
  8    own forward speed                                                / V_MAX
  9    opponent forward speed                                           / V_MAX
  10-11 previous action (linear, angular) in [-1, 1]
"""

import math

OBS_DIM = 12
RING_R = 0.40
V_MAX = 0.20
W_MAX = 4.0


def observation_torch(me, opp, prev_action):
    import torch

    c, s = torch.cos(me[:, 2]), torch.sin(me[:, 2])

    def rot(dx, dy):
        return c * dx + s * dy, -s * dx + c * dy

    mx, my = rot(me[:, 0], me[:, 1])
    rx, ry = rot(opp[:, 0] - me[:, 0], opp[:, 1] - me[:, 1])
    ox, oy = rot(opp[:, 0], opp[:, 1])
    dyaw = opp[:, 2] - me[:, 2]
    return torch.stack([
        mx / RING_R, my / RING_R, rx / RING_R, ry / RING_R, ox / RING_R, oy / RING_R,
        torch.cos(dyaw), torch.sin(dyaw), me[:, 3] / V_MAX, opp[:, 3] / V_MAX,
        prev_action[:, 0], prev_action[:, 1],
    ], 1)


# >>> numpy-begin (this block is copied verbatim into the ROS node by export_sumo.py)
def observation_np(me, opp, prev_action):
    """me, opp: (x, y, yaw, v, w); prev_action: (lin, ang) in [-1, 1]. Returns a list of 12 floats."""
    c, s = math.cos(me[2]), math.sin(me[2])
    mx, my = c * me[0] + s * me[1], -s * me[0] + c * me[1]
    dx, dy = opp[0] - me[0], opp[1] - me[1]
    rx, ry = c * dx + s * dy, -s * dx + c * dy
    ox, oy = c * opp[0] + s * opp[1], -s * opp[0] + c * opp[1]
    dyaw = opp[2] - me[2]
    return [mx / RING_R, my / RING_R, rx / RING_R, ry / RING_R, ox / RING_R, oy / RING_R,
            math.cos(dyaw), math.sin(dyaw), me[3] / V_MAX, opp[3] / V_MAX,
            float(prev_action[0]), float(prev_action[1])]
# <<< numpy-end
