"""Our node against a REAL tournament match: one team is our kit node (e.g. the exported RL attacker +
scripted defender), the other team's two robots replay exactly what they did in the tournament ("ghosts",
from the live recording). The psr_sim referee runs as usual (holds, pickups, points, resets).

The ghosts do not react to us (they repeat the recording): it shows how our robots do against the real
moves of that team, not a real match. A ghost that our robot holds stays still for the 4 s and then jumps
back onto the recording; after a point everybody goes to the start and the ghosts continue the recording.

    # inside the container (VNC desktop: DISPLAY=:2 is the default)
    cd ~/ros2_ws/rl; PY=~/ros2_ws/.venv/bin/python
    $PY vs_replay_ctf.py 672 --nosso exports/ctf/ctf_hibrido_real1.py              # window, real time
    $PY vs_replay_ctf.py 672 --nosso exports/ctf/ctf_hibrido_v4.py --sem-janela    # fast, only the result
Side: by default we take R2D2's side (if R2D2 played the match), else --lado azul|vermelho.
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
DATA = HERE / "data" / "ctf_ao_vivo"
R2D2 = "R2D2"


def ascii_(s):
    return " ".join(unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().split())


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
    for o in S:
        if not out or o["t"] > out[-1]["t"] + 1e-6:
            out.append(o)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("partida")
    p.add_argument("--nosso", default="exports/ctf/ctf_hibrido_real1.py", help="our node (kit format)")
    p.add_argument("--lado", choices=("azul", "vermelho"), default=None)
    p.add_argument("--sem-janela", action="store_true", help="no 3D window, as fast as possible")
    p.add_argument("--vel", type=float, default=1.0, help="speed with the window (1 = real time)")
    p.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    a = p.parse_args()

    with open(DATA / "partidas.csv") as f:
        meta = {r["partida"]: r for r in csv.DictReader(f)}
    r = meta.get(a.partida)
    path = DATA / "partidas" / f"{a.partida}.jsonl.gz"
    if r is None or not path.exists():
        sys.exit(f"partida {a.partida} não encontrada em {DATA}")
    if a.lado:
        ours = "blue" if a.lado == "azul" else "red"
    elif R2D2 in r["vermelho"]:
        ours = "red"
    else:
        ours = "blue"
    ghost_team = "red" if ours == "blue" else "blue"
    ghost_name = r["vermelho"] if ghost_team == "red" else r["azul"]
    ghost_ids = (2, 3) if ghost_team == "red" else (0, 1)
    S = [o for o in load(path) if (o.get("match") or {}).get("phase") == "jogo"]
    if not S:
        sys.exit("gravação sem fase de jogo")
    T0 = S[0]["t"]
    dur = S[-1]["t"] - T0
    real_score = (S[-1].get("match") or {}).get("score")

    if not a.sem_janela and not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":2"
    sys.path.insert(0, str(HERE))
    from eval_ctf import TEAMS, RobotDriver  # also sets sys.path for psr_sim and stubs rospy
    from psr_sim.scenarios import SCENARIOS
    import numpy as np
    from psr_sim.world import World

    dofs = np.arange(3)

    drivers = [RobotDriver(a.nosso, i, ours, role) for i, role in TEAMS[ours]]
    sc = SCENARIOS["ctf"](4, 1)
    viewer = not a.sem_janela
    nome_nosso = Path(a.nosso).stem
    left, right = (nome_nosso, ghost_name) if ours == "blue" else (ghost_name, nome_nosso)
    sc.scoreboard = lambda: ascii_(f"{left.split('·')[-1]}  {sc.score[0]}  x  {sc.score[1]}  {right.split('·')[-1]}"
                                   f"     (real: {real_score[0]} x {real_score[1]})" if real_score else "")
    w = World(sc, backend=a.backend, viewer=viewer)
    if viewer:  # the window font only has ASCII
        orig = w._sync_captions

        def captions():
            w.message = ascii_(w.message)
            orig()
        w._sync_captions = captions

    print(f"#{a.partida}: nós = {nome_nosso} ({'azul' if ours == 'blue' else 'vermelho'}) contra a gravação de "
          f"{ghost_name}; {dur:.0f} s de jogo; placar real {real_score}", flush=True)
    k, next_t, t0, wall0 = 0, 0.0, w.t, time.monotonic()
    state = w.read()
    while w.t - t0 < dur:
        # ghosts: recorded pose at this time (frozen while held by our robots or during a reset pause)
        tt = T0 + (w.t - t0)
        while k + 1 < len(S) and S[k + 1]["t"] <= tt:
            k += 1
        if w.pending is None:
            moved = False
            for rb in S[k]["robots"]:
                i = rb["id"]
                if i in ghost_ids and not sc.held(i):
                    # direct pose (World.teleport would add a physics step that also moves our robots)
                    w.robots[i].set_dofs_position(np.array([rb["x"], rb["y"], rb.get("yaw", 0.0)]), dofs,
                                                  zero_velocity=True)
                    moved = True
            if moved:
                state = w.read()
        if w.pending is None and w.t >= next_t:
            for d in drivers:
                w.set_cmd(d.robot, *d.act(state["robots"], sc, w.t))
            next_t = w.t + 0.05
        state = w.tick()
        if viewer:
            delay = (w.t - t0) / a.vel - (time.monotonic() - wall0)
            if delay > 0:
                time.sleep(delay)
    w.close()
    us, them = (0, 1) if ours == "blue" else (1, 0)
    print(f"RESULTADO #{a.partida}: {nome_nosso} {sc.score[us]} x {sc.score[them]} gravação de {ghost_name} "
          f"| segurados: nós {sum(sc.tags[i] for i in range(4) if i not in ghost_ids)}, "
          f"eles {sum(sc.tags[i] for i in ghost_ids)} | placar real da partida: {real_score}", flush=True)


if __name__ == "__main__":
    main()
