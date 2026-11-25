"""Dataset of the REAL tournament capture-the-flag matches (live recordings of api/ao-vivo, ~9 Hz).

Each robot of each match becomes a sequence of (observation, action) pairs in the same format the
networks use (ctf_obs.py: team frame, role order, 38 floats), so it can feed behavior cloning
(train_ctf_bc.py) and evaluation. The recordings have no commands: the action is inferred from the
motion between two samples (inverse kinematics of the differential drive):
    lin = (displacement along the heading) / dt / V_MAX,   ang = wrap(dyaw) / dt / W_MAX,   clipped to [-1, 1]

    python3 live_ctf.py --src data/ctf_ao_vivo --out data/live_ctf.npz

--src is the folder made by torneio_logs/exportar_ctf.py (partidas/<id>.jsonl.gz + partidas.csv).
Roles per match and team: the robot that spends more time in the enemy half is the attacker.
Samples dropped: outside the "jogo" phase, held robots (their command is ignored), resets/teleports,
gaps in the recording. Round time = time since the last point / draw / start (the kit's 90 s round).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from collections import defaultdict

import numpy as np

from ctf_obs import BASE_X, HOLD_TIME, IMMUNE_TIME, OBS_DIM, ROUND_TIME, V_MAX, W_MAX, observation_np

TEAM_OF = (0, 0, 1, 1)  # robots 0,1 blue (side a), 2,3 red (side b)
FLAG_TEAM = ("blue", "red")
JUMP = 0.08  # m in one sample: teleport (reset to the start)
DT_MIN, DT_MAX = 0.04, 0.35


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def read(path):
    out = []
    with gzip.open(path, "rt") as f:
        for line in f:
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if "t" in o and len(o.get("robots") or []) == 4:
                out.append(o)
    out.sort(key=lambda o: o["t"])
    dedup = []
    for o in out:  # a recording split by a restart can repeat samples
        if not dedup or o["t"] > dedup[-1]["t"] + 1e-6:
            dedup.append(o)
    return dedup


def flags(o):
    f = {m["team"]: (m["x"], m["y"]) for m in o.get("markers") or [] if m.get("kind") == "flag"}
    return f.get("blue", (-BASE_X, 0.0)), f.get("red", (BASE_X, 0.0))


def match_arrays(S):
    """Per sample: poses (n, 4, 3), held (n, 4), carrying (n, 4), flags (n, 2, 2), t, phase, score, reset."""
    n = len(S)
    pose = np.zeros((n, 4, 3))
    held = np.zeros((n, 4), bool)
    carry = np.zeros((n, 4), bool)
    flag = np.zeros((n, 2, 2))
    t = np.array([o["t"] for o in S])
    for k, o in enumerate(S):
        for r in o["robots"]:
            i = r["id"]
            if 0 <= i < 4:
                pose[k, i] = (r["x"], r["y"], r.get("yaw", 0.0))
                held[k, i] = bool(r.get("held"))
        c = o.get("carrying") or []
        carry[k, :len(c)] = c[:4]
        flag[k] = flags(o)
    phase = [((o.get("match") or {}).get("phase") or "") for o in S]
    score = [tuple((o.get("match") or {}).get("score") or (0, 0)) for o in S]
    msg = [o.get("message") or "" for o in S]
    return pose, held, carry, flag, t, phase, score, msg


def team_frame(p, flip):
    """(x, y, yaw) -> team frame (red rotated by 180 deg)."""
    if not flip:
        return p
    return np.stack([-p[..., 0], -p[..., 1], np.vectorize(wrap)(p[..., 2] + math.pi)], -1)


def build_match(S, mid, names, r2d2):
    pose, held, carry, flag, t, phase, score, msg = match_arrays(S)
    n = len(t)
    # resets: score change, draw, teleport of any robot, start of the "jogo" phase
    reset = np.zeros(n, bool)
    for k in range(1, n):
        jump = np.hypot(*(pose[k, :, :2] - pose[k - 1, :, :2]).T).max()
        reset[k] = (score[k] != score[k - 1] or jump > JUMP or (phase[k] == "jogo") != (phase[k - 1] == "jogo")
                    or ("Empate" in msg[k] and "Empate" not in msg[k - 1]))
    round_start = np.zeros(n)
    rs = t[0]
    for k in range(n):
        if reset[k]:
            rs = t[k]
        round_start[k] = rs
    round_t = np.minimum(t - round_start, ROUND_TIME)
    # remaining hold / immunity per robot (the referee: 4 s held, then 1.5 s immune)
    hold_rem = np.zeros((n, 4))
    imm_rem = np.zeros((n, 4))
    for i in range(4):
        onset, release = None, None
        for k in range(n):
            if held[k, i]:
                if onset is None or (k and not held[k - 1, i]):
                    onset = t[k]
                hold_rem[k, i] = max(0.0, HOLD_TIME - (t[k] - onset))
            else:
                if k and held[k - 1, i]:
                    release = t[k]
                if release is not None:
                    imm_rem[k, i] = max(0.0, IMMUNE_TIME - (t[k] - release))
    # velocities (backward difference, robot frame) for the observation; action labels (forward difference)
    vel = np.zeros((n, 4, 2))
    act = np.full((n, 4, 2), np.nan)
    for k in range(1, n):
        dt = t[k] - t[k - 1]
        if not (DT_MIN <= dt <= DT_MAX) or reset[k]:
            continue
        d = pose[k, :, :2] - pose[k - 1, :, :2]
        yaw = pose[k - 1, :, 2]
        lin = (d[:, 0] * np.cos(yaw) + d[:, 1] * np.sin(yaw)) / dt
        ang = np.array([wrap(a) for a in pose[k, :, 2] - pose[k - 1, :, 2]]) / dt
        vel[k] = np.stack([lin, ang], 1)
        act[k - 1] = np.stack([np.clip(lin / V_MAX, -1, 1), np.clip(ang / W_MAX, -1, 1)], 1)
    jogo = np.array([p == "jogo" for p in phase])

    rows = defaultdict(list)
    for team in (0, 1):
        flip = team == 1
        ids = [i for i in range(4) if TEAM_OF[i] == team]
        opp = [i for i in range(4) if TEAM_OF[i] != team]
        tp = team_frame(pose, flip)  # (n, 4, 3)
        # roles: more time in the enemy half (team frame x > 0) = attacker
        fwd = [(tp[jogo, i, 0] > 0).mean() if jogo.any() else 0 for i in ids]
        mine = ids if fwd[0] >= fwd[1] else ids[::-1]
        fo = [(tp[jogo, i, 0] < 0).mean() if jogo.any() else 0 for i in opp]  # their attacker: more time in MY half
        theirs = opp if fo[0] >= fo[1] else opp[::-1]
        own_f = flag[:, team] * (-1 if flip else 1)
        enemy_f = flag[:, 1 - team] * (-1 if flip else 1)
        team_name = names[team]
        prev = {i: (0.0, 0.0) for i in ids}
        for k in range(n):
            st = {i: (tp[k, i, 0], tp[k, i, 1], tp[k, i, 2], vel[k, i, 0], vel[k, i, 1]) for i in range(4)}
            for role, i in enumerate(mine):
                a = act[k, i]
                ok = jogo[k] and not held[k, i] and not np.isnan(a).any() and (k + 1 < n and not held[k + 1, i])
                if ok:
                    ob = observation_np(
                        role, [st[mine[0]], st[mine[1]]], [st[theirs[0]], st[theirs[1]]],
                        own_f[k], enemy_f[k], [carry[k, mine[0]], carry[k, mine[1]]],
                        bool(carry[k, theirs[0]] or carry[k, theirs[1]]),
                        [hold_rem[k, j] for j in (*mine, *theirs)], [imm_rem[k, j] for j in (*mine, *theirs)],
                        round_t[k], prev[i])
                    rows["obs"].append(ob)
                    rows["act"].append(a.tolist())
                    rows["role"].append(role)
                    rows["team"].append(team_name)
                    rows["match"].append(mid)
                    rows["side"].append(team)
                    rows["r2d2"].append(team_name == r2d2)
                    rows["t"].append(t[k])
                if not np.isnan(a).any():
                    prev[i] = (float(a[0]), float(a[1]))
                elif held[k, i] or reset[k]:
                    prev[i] = (0.0, 0.0)
    final = score[-1] if score else (0, 0)
    return rows, {"match": mid, "blue": names[0], "red": names[1], "score": list(final), "samples": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/ctf_ao_vivo")
    ap.add_argument("--out", default="data/live_ctf.npz")
    ap.add_argument("--r2d2", default="Grupo 4 · R2D2", help="our team (marked in the dataset)")
    a = ap.parse_args()
    with open(os.path.join(a.src, "partidas.csv")) as f:
        meta = {int(r["partida"]): (r["azul"], r["vermelho"]) for r in csv.DictReader(f)}
    allrows, matches = defaultdict(list), []
    for mid in sorted(meta):
        path = os.path.join(a.src, "partidas", f"{mid}.jsonl.gz")
        if not os.path.exists(path):
            continue
        S = read(path)
        if len(S) < 100:
            continue
        rows, info = build_match(S, mid, meta[mid], a.r2d2)
        for k, v in rows.items():
            allrows[k] += v
        matches.append(info)
    obs = np.asarray(allrows["obs"], np.float32)
    assert obs.shape[1] == OBS_DIM
    teams = sorted(set(allrows["team"]))
    np.savez_compressed(
        a.out, obs=obs, act=np.asarray(allrows["act"], np.float32), role=np.asarray(allrows["role"], np.int8),
        team=np.asarray([teams.index(x) for x in allrows["team"]], np.int16), team_names=np.asarray(teams),
        match=np.asarray(allrows["match"], np.int32), side=np.asarray(allrows["side"], np.int8),
        r2d2=np.asarray(allrows["r2d2"], bool), t=np.asarray(allrows["t"], np.float32))
    with open(os.path.splitext(a.out)[0] + "_matches.json", "w") as f:
        json.dump(matches, f, ensure_ascii=False, indent=1)
    print(f"{len(matches)} matches, {len(obs)} samples -> {a.out}")
    tm = np.asarray([teams.index(x) for x in allrows["team"]])
    rl = np.asarray(allrows["role"])
    for k, nm in enumerate(teams):
        print(f"  {nm:32s} att {int(((tm == k) & (rl == 0)).sum()):6d}  def {int(((tm == k) & (rl == 1)).sum()):6d}")


if __name__ == "__main__":
    main()
