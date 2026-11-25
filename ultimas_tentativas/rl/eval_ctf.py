"""Play capture-the-flag matches (2 x 2) between kit nodes in the real psr_sim, headless and faster than
real time (or in the Genesis window with --viewer). Works with any node in the kit format
(kit/exercicios/ctf/ctf.py): the file is imported with rospy stubbed, one instance per robot with the
launch-file parameters (~robot, ~team, ~role, ~opponents), fed with the fields its callbacks would fill
and driven through compute_command(...) at 20 Hz.

    PY=~/ros2_ws/.venv/bin/python; S=exports/ctf
    $PY eval_ctf.py --blue $S/ctf_rl.py                                    # red stopped (red:=none)
    $PY eval_ctf.py --blue $S/ctf_rl.py --red $S/ctf_script.py --minutes 5 --matches 3
    $PY eval_ctf.py --blue $S/ctf_script.py --red $S/ctf_rl.py --minutes 5 --matches 3 --csv results/x.csv
    $PY eval_ctf.py --blue ../classico/ctf/ctf_classico.py --red $S/ctf_rl.py  # classic x RL
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import sys

import eval_sumo  # noqa: F401  (sys.path for psr_sim and psr_hero)
from eval_sumo import PARAMS, load_node

eval_sumo._stub_ros()  # the sumo stub has no PointStamped (used by the flags)
sys.modules["geometry_msgs.msg"].PointStamped = type("PointStamped", (), {})

TEAMS = {"blue": ((0, "attacker"), (1, "defender")), "red": ((2, "attacker"), (3, "defender"))}
OPP = {"blue": [2, 3], "red": [0, 1]}


class RobotDriver:
    def __init__(self, path, robot, team, role):
        self.m = load_node(path, "%s_%d" % (Path(path).stem, robot))
        PARAMS.clear()
        PARAMS.update(robot=robot, team=team, role=role, opponents=list(OPP[team]))
        self.node = self.m.CaptureTheFlag()
        self.robot, self.team, self.opps = robot, team, OPP[team]

    def act(self, robots, sc, now):
        n = self.node
        pose = {r["id"]: (r["x"], r["y"], r["yaw"]) for r in robots}
        mine, theirs = (0, 1) if self.team == "blue" else (1, 0)
        n.now = now
        n.me = pose[self.robot]
        n.opp = {k: pose[k] for k in self.opps}
        n.enemy_flag = tuple(sc.flag_xy[theirs])
        n.own_flag = tuple(sc.flag_xy[mine])
        n.carrying = sc.carrier[theirs] == self.robot
        # the extra fields of nodes that subscribe to every odometry and to /game/state
        n.poses = pose
        n.vel = {r["id"]: (r["v"], r["w"]) for r in robots}
        n.game = sc.public()
        n.flags = {"blue": tuple(sc.flag_xy[0]), "red": tuple(sc.flag_xy[1])}
        v, w = n.compute_command(n.me, [n.opp[k] for k in self.opps], n.enemy_flag, n.own_flag, n.carrying)
        return float(v), float(w)


def play_match(blue, red, minutes, seed, backend="gpu", viewer=False):
    from psr_sim.scenarios import SCENARIOS
    from psr_sim.world import World

    drivers = []
    for team, path in (("blue", blue), ("red", red)):
        if path:
            drivers += [RobotDriver(path, i, team, role) for i, role in TEAMS[team]]
    sc = SCENARIOS["ctf"](4, seed)
    w = World(sc, backend=backend, viewer=viewer)
    next_t, t0, wall0 = 0.0, w.t, time.monotonic()
    points, pickups = [], [0, 0]
    last_score, last_carrier = (0, 0), [None, None]
    state = w.read()
    while w.t - t0 < 60.0 * minutes:
        if w.pending is None and w.t >= next_t:
            for d in drivers:
                w.set_cmd(d.robot, *d.act(state["robots"], sc, w.t))
            next_t = w.t + 0.05
        state = w.tick()
        if viewer:
            delay = (w.t - t0) - (time.monotonic() - wall0)
            if delay > 0:
                time.sleep(delay)
        for team in (0, 1):  # team's flag picked up by the other team
            c = sc.carrier[team]
            if c is not None and last_carrier[team] is None:
                pickups[1 - team] += 1
            last_carrier[team] = c
        s = tuple(sc.score)
        if s != last_score:
            points.append({"t_s": round(w.t - t0, 1), "team": "blue" if s[0] > last_score[0] else "red"})
            last_score = s
    w.close()
    tags = list(sc.tags)  # times each robot was held
    return {"blue": sc.score[0], "red": sc.score[1], "pickups_blue": pickups[0], "pickups_red": pickups[1],
            "held_blue": tags[0] + tags[1], "held_red": tags[2] + tags[3], "points": points}


def _match_subprocess(args, seed):
    import subprocess
    import sys

    code = ("import json, sys; sys.path.insert(0, %r); import eval_ctf as e; "
            "print('JSON' + json.dumps(e.play_match(%r, %r, %r, %d, viewer=%r)))"
            % (str(Path(__file__).resolve().parent), args.blue, args.red, args.minutes, seed, args.viewer))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    return json.loads([ln for ln in out.splitlines() if ln.startswith("JSON")][-1][4:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blue", default=None, help="node of the blue team (default: stopped)")
    p.add_argument("--red", default=None, help="node of the red team (default: stopped, like red:=none)")
    p.add_argument("--minutes", type=float, default=5.0)
    p.add_argument("--matches", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--csv", default=None)
    args = p.parse_args()
    name = lambda x: Path(x).stem if x else "parado"  # noqa: E731

    rows = []
    for k in range(args.matches):
        r = _match_subprocess(args, args.seed + k) if args.matches > 1 else \
            play_match(args.blue, args.red, args.minutes, args.seed + k, viewer=args.viewer)
        r = {"match": k + 1, **r, "points": " ".join("%s@%s" % (q["team"][0], q["t_s"]) for q in r["points"])}
        rows.append(r)
        print("match %d (%.0f min): %s %d x %d %s | pegadas %d x %d | segurados %d x %d"
              % (k + 1, args.minutes, name(args.blue), r["blue"], r["red"], name(args.red), r["pickups_blue"],
                 r["pickups_red"], r["held_blue"], r["held_red"]), flush=True)
    s = {k: sum(r[k] for r in rows) for k in ("blue", "red", "pickups_blue", "pickups_red", "held_blue", "held_red")}
    print("RESULT %s (blue) vs %s (red): %d matches of %.0f min, points %d x %d, pickups %d x %d, held %d x %d"
          % (name(args.blue), name(args.red), len(rows), args.minutes, s["blue"], s["red"], s["pickups_blue"],
             s["pickups_red"], s["held_blue"], s["held_red"]))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)


if __name__ == "__main__":
    main()
