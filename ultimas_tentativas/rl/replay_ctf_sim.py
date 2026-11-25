"""Replays a REAL tournament capture-the-flag match (live recording) in the psr_sim 3D window (VNC).

No physics: at every recorded sample (~9 Hz) the four robots and both flags are put exactly where they
were (World.teleport), and the window caption shows the teams, the score, the clock and the referee.

    # inside the container, in the VNC desktop terminal (DISPLAY=:2):
    cd ~/ros2_ws/rl
    ~/ros2_ws/.venv/bin/python replay_ctf_sim.py            # lists the matches
    ~/ros2_ws/.venv/bin/python replay_ctf_sim.py 672        # plays match 672
    ~/ros2_ws/.venv/bin/python replay_ctf_sim.py 672 --desde 60 --vel 2 --loop

Recordings: data/ctf_ao_vivo/partidas/<id>.jsonl.gz (from torneio_logs/exportar_ctf.py).
Ctrl+C (or closing the window) stops. For pausing / scrubbing use ver_ctf.py (matplotlib).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "kit" / "sim"))
DATA = HERE / "data" / "ctf_ao_vivo"


def ascii_(s):
    """The Genesis window font only has ASCII: drop accents and other symbols."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.split())


def short(name):
    return ascii_(name.split("·")[-1]) if "·" in name else ascii_(name)


def load(path):
    S = []
    with gzip.open(path, "rt") as f:
        for line in f:
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if "t" in o and len(o.get("robots") or []) == 4:
                S.append(o)
    S.sort(key=lambda o: o["t"])
    out = []
    for o in S:  # recordings split by a restart can repeat samples
        if not out or o["t"] > out[-1]["t"] + 1e-6:
            out.append(o)
    return out


def matches():
    with open(DATA / "partidas.csv") as f:
        return {r["partida"]: r for r in csv.DictReader(f)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("partida", nargs="?", help="match id (or a team name to filter the list)")
    p.add_argument("--desde", type=float, default=0.0, help="start at this second of the recording")
    p.add_argument("--ate", type=float, default=None, help="stop at this second")
    p.add_argument("--vel", type=float, default=1.0, help="speed (1 = real time)")
    p.add_argument("--loop", action="store_true", help="start again at the end")
    p.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    a = p.parse_args()

    meta = matches()
    if not a.partida or not a.partida.isdigit():
        for mid, r in meta.items():
            if not a.partida or a.partida.lower() in (r["azul"] + r["vermelho"]).lower():
                print(f"{mid:>5}  {r['data']}  {r['placar_azul']}:{r['placar_vermelho']}  {r['azul']} × {r['vermelho']}")
        print("\nreproduzir: replay_ctf_sim.py <id>")
        return
    path = DATA / "partidas" / f"{a.partida}.jsonl.gz"
    if not path.exists():
        sys.exit(f"não achei {path} (rode torneio_logs/exportar_ctf.py e copie para data/ctf_ao_vivo)")
    S = load(path)
    r = meta.get(a.partida, {})
    blue, red = r.get("azul", "azul"), r.get("vermelho", "vermelho")
    t0 = S[0]["t"]
    S = [o for o in S if o["t"] - t0 >= a.desde and (a.ate is None or o["t"] - t0 <= a.ate)]
    if not S:
        sys.exit("intervalo vazio")
    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":2"  # the container's VNC desktop

    from psr_sim.scenarios.base import Pose
    from psr_sim.scenarios.ctf import CaptureTheFlag
    from psr_sim.world import World

    sc = CaptureTheFlag(n_robots=4)
    world = World(sc, backend=a.backend, viewer=True, fps=30)
    state = {"o": S[0]}

    def scoreboard():
        m = state["o"].get("match") or {}
        s = m.get("score") or [0, 0]
        return ascii_(f"{short(blue)}  {s[0]}  x  {s[1]}  {short(red)}     #{a.partida}  {m.get('clock', 0):.0f}/{m.get('duration', 180):.0f} s")

    sc.scoreboard = scoreboard
    sc.notice = lambda: ""
    print(f"#{a.partida}: {blue} (azul) × {red} (vermelho), {len(S)} amostras, {S[-1]['t'] - S[0]['t']:.0f} s"
          f" a {a.vel:g}x. Ctrl+C para sair.", flush=True)
    try:
        while True:
            start_wall, start_t = time.time(), S[0]["t"]
            for o in S:
                state["o"] = o
                held = {rb["id"]: rb.get("held") for rb in o["robots"]}
                poses = [None] * 4
                for rb in o["robots"]:
                    if 0 <= rb["id"] < 4:
                        poses[rb["id"]] = Pose(rb["x"], rb["y"], rb.get("yaw", 0.0))
                fl = {m["team"]: (m["x"], m["y"]) for m in o.get("markers") or [] if m.get("kind") == "flag"}
                sc.flag_xy = [fl.get("blue", sc.home(0)), fl.get("red", sc.home(1))]
                presos = [f"hero_{i}" for i, h in held.items() if h]
                world.message = ascii_(o.get("message") or "") or (("presos: " + ", ".join(presos)) if presos else "")
                world.teleport(poses)
                world._sync_flags()
                world._sync_captions()
                # real-time pacing (the window refreshes on its own at 30 fps)
                wait = (o["t"] - start_t) / a.vel - (time.time() - start_wall)
                if wait > 0:
                    time.sleep(wait)
            if not a.loop:
                break
        print("fim da gravação", flush=True)
        time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        world.close()


if __name__ == "__main__":
    main()
