"""Play sumo nodes of the kit (exercicios/sumo/*.py) against each other in the real psr_sim
(raised ring, fall, referee), headless and faster than real time, or in the Genesis
window (VNC) in real time.

The node files themselves are imported (rospy and the ROS messages are stubbed) and
their Sumo class is driven exactly as in ROS: same parameters (~robot, ~opponent), same
odometry fields, same compute_command(), at the node's rate. So what is tested is the
code that runs in the kit.

    PY=~/ros2_ws/.venv/bin/python; S=exports/sumo
    $PY eval_sumo.py $S/sumo_rl.py                          # vs a stopped robot (robot1:=none)
    $PY eval_sumo.py $S/sumo_rl.py --opponent $S/sumo_chaser.py --rounds 20
    $PY eval_sumo.py $S/sumo_rl.py --opponent $S/sumo_rl_v1.py --side 1   # policy plays as hero_1
    DISPLAY=:2 $PY eval_sumo.py $S/sumo_rl.py --opponent $S/sumo_flanker.py --viewer   # watch in the VNC
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
import types
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parents[1] / "kit" / "sim"
sys.path.insert(0, str(SIM_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kit" / "ros" / "psr_hero" / "src"))

PARAMS: dict = {}  # values returned by the stub rospy.get_param


def _stub_ros():
    def noop(*a, **k):
        return None

    rospy = types.ModuleType("rospy")
    rospy.init_node = noop
    rospy.get_param = lambda name, default=None: PARAMS.get(name.lstrip("~"), default)
    rospy.Publisher = lambda *a, **k: types.SimpleNamespace(publish=noop)
    rospy.Subscriber = noop
    rospy.logwarn_throttle = rospy.loginfo = rospy.logwarn = noop
    rospy.get_time = lambda: 0.0
    sys.modules["rospy"] = rospy
    for name in ("geometry_msgs", "geometry_msgs.msg", "nav_msgs", "nav_msgs.msg", "std_msgs", "std_msgs.msg"):
        sys.modules.setdefault(name, types.ModuleType(name))
    for mod, cls in (("geometry_msgs.msg", "Twist"), ("nav_msgs.msg", "Odometry"), ("std_msgs.msg", "String"),
                     ("std_msgs.msg", "Bool")):
        setattr(sys.modules[mod], cls, type(cls, (), {}))


def load_node(path: str, alias: str):
    """Import a node file under a unique module name."""
    sys.dont_write_bytecode = True  # no __pycache__ next to the exercise files
    _stub_ros()
    spec = importlib.util.spec_from_file_location(alias, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class NodeDriver:
    """Drives the Sumo class of a node file with the simulator state."""

    def __init__(self, path: str, robot: int, opponent: int, alias: str):
        self.m = load_node(path, alias)
        self.name = Path(path).name
        PARAMS.clear()
        PARAMS.update(robot=robot, opponent=opponent)
        self.node = self.m.Sumo()
        self.robot, self.opponent = robot, opponent
        self.rate = float(getattr(self.m, "CONTROL_HZ", 20))

    def new_round(self):
        if hasattr(self.node, "prev_action"):
            self.node.prev_action = (0.0, 0.0)

    def act(self, robots, now):
        me, opp = robots[self.robot], robots[self.opponent]
        n = self.node
        n.me, n.opp, n.speed = (me["x"], me["y"], me["yaw"]), (opp["x"], opp["y"], opp["yaw"]), me["v"]
        # the extra fields that the RL node reads from the odometry messages
        n.w, n.opp_speed, n.opp_w = me["w"], opp["v"], opp["w"]
        v, w = n.compute_command(n.me, n.opp, n.speed, now)
        return float(v), float(w)


def play(node_a: str, node_b: str | None, side: int, rounds: int, seed: int, backend: str, viewer: bool):
    """node_a drives hero_<side>; node_b (or a stopped robot) the other one. Returns per-round rows."""
    from psr_sim.scenarios import SCENARIOS
    from psr_sim.world import World

    a_i, b_i = side, 1 - side
    drivers = [(a_i, NodeDriver(node_a, a_i, b_i, "node_a"))]
    if node_b:
        drivers.append((b_i, NodeDriver(node_b, b_i, a_i, "node_b")))
    sc = SCENARIOS["sumo"](2, seed)
    w = World(sc, backend=backend, viewer=viewer)
    next_t = {i: 0.0 for i, _ in drivers}
    rows, round_t0, t_start, wall0 = [], w.t, w.t, time.monotonic()
    last = (0, 0, 0)
    state = w.read()
    while len(rows) < rounds:
        if w.pending is None:
            for i, drv in drivers:
                if w.t >= next_t[i]:
                    w.set_cmd(i, *drv.act(state["robots"], w.t))
                    next_t[i] = w.t + 1.0 / drv.rate
        was_pending = w.pending is not None
        state = w.tick()
        if viewer:  # real time, to watch
            delay = (w.t - t_start) - (time.monotonic() - wall0)
            if delay > 0:
                time.sleep(delay)
        if was_pending and w.pending is None:
            for _, drv in drivers:
                drv.new_round()
            round_t0 = w.t
        score = (sc.wins[a_i], sc.wins[b_i], sc.draws)
        if score != last:
            res = ("win", "lose", "draw")[[x != y for x, y in zip(score, last)].index(True)]
            rows.append({"round": len(rows) + 1, "result": res, "time_s": round(w.t - round_t0, 2), "message": w.message})
            print("round %2d: %-4s in %5.2f s  (%s)" % (len(rows), res, w.t - round_t0, w.message), flush=True)
            last = score
    w.close()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("node", help="node file under test")
    p.add_argument("--opponent", default=None, help="node file of the opponent (default: stopped robot)")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--side", type=int, default=0, choices=(0, 1), help="robot driven by the node under test")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--backend", default="gpu")
    p.add_argument("--csv", default=None, help="write one line per round")
    p.add_argument("--viewer", action="store_true", help="open the Genesis window (VNC: DISPLAY=:2) in real time")
    args = p.parse_args()

    rows = play(args.node, args.opponent, args.side, args.rounds, args.seed, args.backend, args.viewer)
    n = len(rows)
    count = {k: sum(r["result"] == k for r in rows) for k in ("win", "lose", "draw")}
    times = [r["time_s"] for r in rows if r["result"] == "win"]
    opp = Path(args.opponent).name if args.opponent else "stopped"
    print("RESULT %s as hero_%d vs %s as hero_%d: win %d, lose %d, draw %d of %d; mean win time %s"
          % (Path(args.node).name, args.side, opp, 1 - args.side, count["win"], count["lose"], count["draw"], n,
             "%.2f s" % (sum(times) / len(times)) if times else "-"))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)


if __name__ == "__main__":
    main()
