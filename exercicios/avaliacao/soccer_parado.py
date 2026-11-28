"""Time parado para o rl/eval_soccer.py (equivale a red:=none / blue:=none)."""
TEAMS = {"blue": (0, +1), "red": (3, -1)}
try:
    import rospy
except ImportError:  # pragma: no cover
    rospy = None


class Soccer:
    def __init__(self):
        team = rospy.get_param("~team", "blue") if rospy else "blue"
        first = TEAMS[team][0]
        self.ids = [first, first + 1, first + 2]

    def compute_commands(self, poses, ball):
        return [(0.0, 0.0)] * 3
