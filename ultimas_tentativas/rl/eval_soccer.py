"""Play soccer matches between kit nodes (exercicios/soccer/*.py) in the real psr_sim,
headless and faster than real time (or in the Genesis window with --viewer).

The node files are imported (rospy stubbed) and their Soccer class is driven as in ROS
(same parameters, odometry fields, compute_commands(), 20 Hz).

    PY=~/ros2_ws/.venv/bin/python; S=exports/soccer
    $PY eval_soccer.py --blue $S/soccer_rl.py                              # red stopped (red:=none)
    $PY eval_soccer.py --blue $S/soccer_rl.py --red $S/soccer_script.py --minutes 5 --matches 3
    $PY eval_soccer.py --blue $S/soccer_script.py --red $S/soccer_rl.py   # the RL team as red
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import eval_sumo  # noqa: F401  (sys.path for psr_sim and psr_hero)
from eval_sumo import PARAMS, load_node


class TeamDriver:
    def __init__(self, path, team, alias):
        self.m = load_node(path, alias)
        PARAMS.clear()
        PARAMS.update(team=team)
        self.node = self.m.Soccer()
        self.ids = list(self.node.ids)
        self.name = Path(path).stem

    def new_kickoff(self):
        if hasattr(self.node, "prev"):
            self.node.prev = [(0.0, 0.0)] * 3

    def act(self, robots, puck, now=0.0):
        n = self.node
        n.now = now
        n.poses = {r["id"]: (r["x"], r["y"], r["yaw"]) for r in robots}
        n.vel = {r["id"]: (r["v"], r["w"]) for r in robots}
        n.ball = (puck["x"], puck["y"])
        n.ball_v = (puck["vx"], puck["vy"])
        return list(zip(self.ids, n.compute_commands(n.poses, n.ball)))


def play_match(blue, red, minutes, seed, backend="gpu", viewer=False):
    from psr_sim.scenarios import SCENARIOS
    from psr_sim.world import World

    drivers = [TeamDriver(blue, "blue", "blue_node")]
    if red:
        drivers.append(TeamDriver(red, "red", "red_node"))
    sc = SCENARIOS["soccer"](6, seed)
    w = World(sc, backend=backend, viewer=viewer)
    next_t, t0, wall0 = 0.0, w.t, time.monotonic()
    goals = []
    last = (0, 0)
    state = w.read()
    while w.t - t0 < 60.0 * minutes:
        if w.pending is None and w.t >= next_t:
            for d in drivers:
                for i, (v, om) in d.act(state["robots"], state["puck"], w.t):
                    w.set_cmd(i, v, om)
            next_t = w.t + 0.05
        was_pending = w.pending is not None
        state = w.tick()
        if viewer:
            delay = (w.t - t0) - (time.monotonic() - wall0)
            if delay > 0:
                time.sleep(delay)
        if was_pending and w.pending is None:
            for d in drivers:
                d.new_kickoff()
        s = tuple(sc.score)
        if s != last:
            goals.append({"t_s": round(w.t - t0, 1), "team": "blue" if s[0] > last[0] else "red"})
            last = s
    w.close()
    return sc.score[0], sc.score[1], goals


def _match_subprocess(args, seed):
    import json
    import subprocess
    import sys

    code = ("import json, sys; sys.path.insert(0, %r); import eval_soccer as e; "
            "b, r, g = e.play_match(%r, %r, %r, %d, viewer=%r); print('JSON' + json.dumps([b, r, g]))"
            % (str(Path(__file__).resolve().parent), args.blue, args.red, args.minutes, seed, args.viewer))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    b, r, goals = json.loads([ln for ln in out.splitlines() if ln.startswith("JSON")][-1][4:])
    return b, r, goals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blue", required=True)
    p.add_argument("--red", default=None, help="node of the red team (default: stopped, like red:=none)")
    p.add_argument("--minutes", type=float, default=5.0)
    p.add_argument("--matches", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    rows = []
    for k in range(args.matches):
        if args.matches > 1:  # Genesis can be initialized once per process: one subprocess per match
            b, r, goals = _match_subprocess(args, args.seed + k)
        else:
            b, r, goals = play_match(args.blue, args.red, args.minutes, args.seed + k, viewer=args.viewer)
        rows.append({"match": k + 1, "blue": b, "red": r, "goal_times_s": " ".join(
            "%s@%s" % (g["team"][0], g["t_s"]) for g in goals)})
        print("match %d (%.0f min): %s %d x %d %s" % (k + 1, args.minutes, Path(args.blue).stem, b, r,
                                                      Path(args.red).stem if args.red else "parado"), flush=True)
    tb, tr = sum(x["blue"] for x in rows), sum(x["red"] for x in rows)
    print("RESULT %s (blue) vs %s (red): %d matches of %.0f min, goals %d x %d"
          % (Path(args.blue).stem, Path(args.red).stem if args.red else "parado", len(rows), args.minutes, tb, tr))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)


if __name__ == "__main__":
    main()
