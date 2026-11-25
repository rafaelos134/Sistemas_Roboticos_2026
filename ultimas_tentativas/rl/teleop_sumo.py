"""Play sumo yourself (keyboard) against a trained policy, in psr_sim with the Genesis window.

Run it in a terminal of the container (the VNC terminal, or `docker exec -it ... bash`):

    cd ~/ros2_ws/rl
    DISPLAY=:2 ~/ros2_ws/.venv/bin/python teleop_sumo.py --fase 3          # you: hero_0 (blue)
    DISPLAY=:2 ~/ros2_ws/.venv/bin/python teleop_sumo.py --fase 1 --side 1 # you: hero_1 (orange)
    DISPLAY=:2 ~/ros2_ws/.venv/bin/python teleop_sumo.py --opponent exports/sumo/sumo_flanker.py

Keys (in this terminal, not in the 3D window):
    w / s  or  up / down      speed +/- 0.05 m/s (max 0.20)
    a / d  or  left / right   turn left / right while held
    space                     stop
    r                         restart the scoreboard
    q                         quit
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

from eval_sumo import NodeDriver

HERE = Path(__file__).resolve().parent
NODES = {1: "sumo_rl_v1.py", 2: "sumo_rl_v2.py", 3: "sumo_rl_v3.py"}
V_STEP, V_MAX, W_TURN, TURN_HOLD = 0.05, 0.20, 3.0, 0.30


class Keyboard:
    """Non-blocking key reader on the terminal (cbreak mode)."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def keys(self):
        out = []
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = os.read(self.fd, 1).decode(errors="ignore")
            if ch == "\x1b":  # arrow keys: ESC [ A/B/C/D
                seq = os.read(self.fd, 2).decode(errors="ignore")
                ch = {"[A": "w", "[B": "s", "[C": "d", "[D": "a"}.get(seq, "")
            out.append(ch.lower())
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fase", type=int, default=3, choices=(1, 2, 3), help="trained policy to play against")
    p.add_argument("--opponent", default=None, help="any sumo node file instead of --fase")
    p.add_argument("--side", type=int, default=0, choices=(0, 1), help="the robot you drive (0 blue, 1 orange)")
    p.add_argument("--backend", default="gpu")
    args = p.parse_args()
    if not sys.stdin.isatty():
        sys.exit("run it in an interactive terminal (VNC terminal or docker exec -it)")
    os.environ.setdefault("DISPLAY", ":2")

    from psr_sim.scenarios import SCENARIOS
    from psr_sim.world import World

    me_i, bot_i = args.side, 1 - args.side
    node = args.opponent or str(HERE / "exports" / "sumo" / NODES[args.fase])
    bot = NodeDriver(node, bot_i, me_i, "bot")
    sc = SCENARIOS["sumo"](2, None)
    print("abrindo o simulador (a janela aparece no VNC)...", flush=True)
    w = World(sc, backend=args.backend, viewer=True)
    print(__doc__.split("Keys")[1].split("\n", 1)[1])
    print("você: hero_%d (%s)   adversário: %s como hero_%d\n" % (me_i, ("azul", "laranja")[me_i], Path(node).name, bot_i))

    v, turn, turn_until = 0.0, 0.0, 0.0
    next_bot, t_start, wall0 = 0.0, w.t, time.monotonic()
    state = w.read()
    last_line = ""
    with Keyboard() as kb:
        while True:
            now = time.monotonic()
            for k in kb.keys():
                if k == "q":
                    print("\nfim. placar: você %d x %d adversário, empates %d" % (sc.wins[me_i], sc.wins[bot_i], sc.draws))
                    w.close()
                    return
                if k == "w":
                    v = min(V_MAX, v + V_STEP)
                elif k == "s":
                    v = max(-V_MAX, v - V_STEP)
                elif k in ("a", "d"):
                    turn, turn_until = (W_TURN if k == "a" else -W_TURN), now + TURN_HOLD
                elif k == " ":
                    v, turn = 0.0, 0.0
                elif k == "r":
                    w.reset()
                    bot.new_round()
            if now > turn_until:
                turn = 0.0
            if w.pending is None:
                w.set_cmd(me_i, v, turn)
                if w.t >= next_bot:
                    w.set_cmd(bot_i, *bot.act(state["robots"], w.t))
                    next_bot = w.t + 1.0 / bot.rate
            else:  # the round is over: stop, so you start the next one standing still
                v, turn = 0.0, 0.0
            was_pending = w.pending is not None
            state = w.tick()
            if was_pending and w.pending is None:
                bot.new_round()
            line = "\rvelocidade %+.2f m/s  giro %+.1f rad/s | você %d x %d adversário, empates %d | %s" % (
                v, turn, sc.wins[me_i], sc.wins[bot_i], sc.draws, (w.message or "").ljust(16))
            if line != last_line:
                sys.stdout.write(line)
                sys.stdout.flush()
                last_line = line
            delay = (w.t - t_start) - (time.monotonic() - wall0)  # real time
            if delay > 0:
                time.sleep(delay)


if __name__ == "__main__":
    main()
