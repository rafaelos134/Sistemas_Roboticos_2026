"""Validates exports/ctf/nyx_estimado.py against the real Nyx (live recordings).

1. open loop: every recorded Nyx state is given to the node (in order, so the attacker's wait state
   evolves), and its command is compared with the motion Nyx really made (same MAE as train_ctf_bc.py,
   in units of V_MAX / W_MAX).
2. closed loop: the node plays in psr_sim and the same statistics as in the recordings are measured.

    $PY nyx_validar.py [--node exports/ctf/nyx_estimado.py] [--contra exports/ctf/ctf_hibrido_v4.py] [--partidas 4]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

import eval_ctf  # noqa: F401  (rospy stubs, sys.path)
from eval_ctf import TEAMS, RobotDriver
from eval_sumo import PARAMS, load_node

DATA = Path(__file__).resolve().parent / "data" / "ctf_ao_vivo"
V_MAX, W_MAX = 0.20, 4.0
OPP = {"blue": [2, 3], "red": [0, 1]}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def load(path):
    S = []
    for line in gzip.open(path, "rt"):
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


def roles(S, ids, team):
    """attacker = more time in the enemy half."""
    s = 1 if team == "blue" else -1
    fwd = {i: np.mean([s * o["robots"][i]["x"] > 0 for o in S if (o.get("match") or {}).get("phase") == "jogo"])
           for i in ids}
    a = max(ids, key=lambda i: fwd[i])
    return {a: "attacker", [i for i in ids if i != a][0]: "defender"}


ZONAS = {}
MOV = {}  # só amostras em que o robô real se mexe (parado contra a parede o movimento deduzido é 0)


def open_loop(node_path, matches):
    err = {"attacker": [], "defender": []}
    base = {"attacker": [], "defender": []}
    for mid, team in matches:
        S = load(DATA / "partidas" / f"{mid}.jsonl.gz")
        ids = (0, 1) if team == "blue" else (2, 3)
        rl = roles(S, ids, team)
        nodes = {}
        for i, role in rl.items():
            m = load_node(node_path, "nyx_%d_%d" % (mid, i))
            PARAMS.clear()
            PARAMS.update(robot=i, team=team, role=role, opponents=OPP[team])
            nodes[i] = m.CaptureTheFlag()
        for k in range(len(S) - 1):
            o, o2 = S[k], S[k + 1]
            if (o.get("match") or {}).get("phase") != "jogo":
                continue
            dt = o2["t"] - o["t"]
            pose = {r["id"]: (r["x"], r["y"], r.get("yaw", 0.0)) for r in o["robots"]}
            fl = {m_["team"]: (m_["x"], m_["y"]) for m_ in o.get("markers") or [] if m_.get("kind") == "flag"}
            held = [bool(r.get("held")) for r in sorted(o["robots"], key=lambda r: r["id"])]
            carrying = list(o.get("carrying") or [False] * 4)
            for i, n in nodes.items():
                n.now = o["t"]
                n.me = pose[i]
                n.opp = {j: pose[j] for j in OPP[team]}
                n.enemy_flag = fl["red" if team == "blue" else "blue"]
                n.own_flag = fl[team]
                n.carrying = carrying[i]
                n.game = {"held": held, "carrying": carrying}
                v, w = n.compute_command(n.me, list(n.opp.values()), n.enemy_flag, n.own_flag, n.carrying)
                if held[i] or not (0.04 < dt < 0.3) or (o2["robots"][i].get("held")):
                    continue
                p, q = o["robots"][i], o2["robots"][i]
                dx, dy = q["x"] - p["x"], q["y"] - p["y"]
                if math.hypot(dx, dy) > 0.08:
                    continue
                lin = (dx * math.cos(p["yaw"]) + dy * math.sin(p["yaw"])) / dt
                ang = wrap(q["yaw"] - p["yaw"]) / dt
                real = np.clip([lin / V_MAX, ang / W_MAX], -1, 1)
                pred = np.clip([v / V_MAX, w / W_MAX], -1, 1)
                err[rl[i]].append(np.abs(pred - real).mean())
                base[rl[i]].append(real)
                sx = (1 if team == "blue" else -1) * p["x"]
                zona = ("com bandeira" if carrying[i] else "campo próprio" if sx < 0 else
                        "indo (x<0.45)" if sx < 0.45 else "perto da bandeira")
                if rl[i] == "attacker":
                    ZONAS.setdefault(zona, []).append(np.abs(pred - real))
                if abs(real[0]) > 0.25 or abs(real[1]) > 0.15:  # o robô real estava se mexendo
                    MOV.setdefault(rl[i], []).append(np.abs(pred - real).mean())
    out = {}
    for r in err:
        b = np.array(base[r])
        out[r] = {"n": len(err[r]), "mae": float(np.mean(err[r])),
                  "mae_constante": float(np.abs(b - b.mean(0)).mean()) if len(b) else None}
    return out


def closed_loop(node_path, contra, n_matches, minutes):
    """One match per process (Genesis can only be initialized once per process)."""
    import subprocess
    import sys

    parts = []
    for k in range(n_matches):
        out = subprocess.run([sys.executable, __file__, "--node", node_path, "--contra", contra, "--minutos",
                              str(minutes), "--uma", str(k)], capture_output=True, text=True).stdout
        line = [ln for ln in out.splitlines() if ln.startswith("JSON")]
        if line:
            parts.append(json.loads(line[-1][4:]))
    a = np.array(sum((q["att_dist_def"] for q in parts), [])) if parts else np.array([])
    if not a.size:
        a = np.array([np.nan])
    return {"defensor_na_base": float(np.mean(sum((q["def_na_base"] for q in parts), []))),
            "atacante_dist_defensor_p1_p5_p50": np.nanpercentile(a, [1, 5, 50]).round(3).tolist(),
            "esperas_por_partida": float(np.mean([q["esperas"] for q in parts])),
            "placares_nyx_x_contra": [q["placar"] for q in parts]}


def one_match(node_path, contra, k, minutes):
    from psr_sim.scenarios import SCENARIOS
    from psr_sim.world import World

    stats = {"def_na_base": [], "att_dist_def": [], "esperas": 0, "placar": None}
    if True:
        nyx_team = "blue" if k % 2 == 0 else "red"
        other = "red" if nyx_team == "blue" else "blue"
        drivers = [RobotDriver(node_path, i, nyx_team, role) for i, role in TEAMS[nyx_team]]
        drivers += [RobotDriver(contra, i, other, role) for i, role in TEAMS[other]]
        sc = SCENARIOS["ctf"](4, 10 + k)
        w = World(sc, backend="gpu", viewer=False)
        att, dfn = TEAMS[nyx_team][0][0], TEAMS[nyx_team][1][0]
        odef = TEAMS[other][1][0]
        s = 1 if nyx_team == "blue" else -1
        next_t, t0 = 0.0, w.t
        state = w.read()
        waiting = False
        while w.t - t0 < 60 * minutes:
            if w.pending is None and w.t >= next_t:
                for d in drivers:
                    w.set_cmd(d.robot, *d.act(state["robots"], sc, w.t))
                next_t = w.t + 0.05
            state = w.tick()
            R = {r["id"]: r for r in state["robots"]}
            if w.pending is None:
                stats["def_na_base"].append(math.hypot(s * R[dfn]["x"] + 0.62, R[dfn]["y"]) < 0.05)
                if s * R[att]["x"] > 0.3 and not sc.held(att) and not sc.held(odef) and s * R[odef]["x"] > 0:
                    stats["att_dist_def"].append(math.hypot(R[att]["x"] - R[odef]["x"], R[att]["y"] - R[odef]["y"]))
            nd = drivers[0].node
            if nd.mode == "wait" and not waiting:
                stats["esperas"] += 1
            waiting = nd.mode == "wait"
        me, them = (0, 1) if nyx_team == "blue" else (1, 0)
        stats["placar"] = (sc.score[me], sc.score[them])
        w.close()
    stats["def_na_base"] = [bool(x) for x in stats["def_na_base"]]
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--node", default="exports/ctf/nyx_estimado.py")
    p.add_argument("--contra", default="exports/ctf/ctf_hibrido_v4.py")
    p.add_argument("--partidas", type=int, default=4)
    p.add_argument("--minutos", type=float, default=3.0)
    p.add_argument("--sem-sim", action="store_true")
    p.add_argument("--uma", type=int, default=None, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.uma is not None:
        print("JSON" + json.dumps(one_match(a.node, a.contra, a.uma, a.minutos)))
        return
    with open(DATA / "partidas.csv") as f:
        ms = [(int(r["partida"]), "blue" if "Nyx" in r["azul"] else "red") for r in csv.DictReader(f)
              if "Nyx" in r["azul"] + r["vermelho"]]
    print(f"{len(ms)} partidas do Nyx gravadas")
    ol = open_loop(a.node, ms)
    print("MALHA ABERTA (erro médio do comando, unidades de V_MAX/W_MAX):")
    for r, v in ol.items():
        print(f"  {r:9s} n={v['n']:6d}  MAE {v['mae']:.3f}   (prever a média: {v['mae_constante']:.3f})")
    for r, e in MOV.items():
        print(f"  {r:9s} só quando o real se mexe: n={len(e):6d}  MAE {np.mean(e):.3f}")
    for z, e in ZONAS.items():
        e = np.array(e)
        print(f"     atacante {z:18s} n={len(e):6d}  erro lin {e[:, 0].mean():.3f}  ang {e[:, 1].mean():.3f}")
    if not a.sem_sim:
        cl = closed_loop(a.node, a.contra, a.partidas, a.minutos)
        print("MALHA FECHADA no psr_sim contra", Path(a.contra).stem)
        print("  " + json.dumps(cl))
        print("  real (gravações): defensor_na_base 0.63, atacante_dist_defensor p1/p5/p50 = 0.108/0.126/0.140, "
              "esperas ~1,7 por partida")


if __name__ == "__main__":
    main()
