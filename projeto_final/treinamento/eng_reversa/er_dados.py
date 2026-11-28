"""Carrega as gravações ao vivo do torneio (data/ao_vivo/<jogo>/<id>.jsonl.gz) em arrays numpy.

Cada partida vira um dict:
  id, jogo, data, azul, vermelho (nomes curtos), lado[k] = time do robô k,
  t (N,), fase (N,) str, P (N, R, 3) poses x y yaw, held (N, R) bool,
  puck (N, 2) (futebol), flags {"blue": (N,2), "red": (N,2)} (ctf), carrying (N, R) (ctf),
  score (N, 2), clock (N,)

Lado azul = lado `a` da partida: sumô robô 0, ctf robôs 0 1, futebol 0 1 2.
Comando deduzido entre amostras consecutivas (k, k+1): v = avanço no sentido do robô / dt, w = dyaw / dt.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
DADOS = AQUI.parent / "data" / "ao_vivo"  # no container (rl/data/ao_vivo)
if not DADOS.exists():  # na cópia de ultimas_tentativas/engenharia_reversa
    DADOS = AQUI.parent / "gravacoes_ao_vivo"
CACHE = AQUI / "cache"
V_MAX, W_MAX = 0.20, 4.0
CURTO = {"Claude (agente da disciplina)": "Claude", "Grupo 3 · Nyx": "Nyx", "Grupo 1 · TuoLing": "TuoLing",
         "Grupo 6 · SoLaDaBot": "SoLaDaBot", "Grupo 5 · Corsa": "Corsa", "Grupo 4 · R2D2": "R2D2",
         "Grupo 2 · RTech Tabajara": "RTech", "Grupo 7 · Tech": "Tech"}
EQUIPES = ["Claude", "TuoLing", "RTech", "Nyx", "R2D2", "Corsa", "SoLaDaBot", "Tech"]
N_ROB = {"sumo": 2, "ctf": 4, "soccer": 6}


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def lado_dos_robos(jogo, n):
    if jogo == "sumo":
        return ["blue", "red"]
    if jogo == "ctf":
        return ["blue", "blue", "red", "red"] if n == 4 else ["blue", "red"]
    return ["blue"] * 3 + ["red"] * 3


def _ler(path, jogo):
    S = []
    for linha in gzip.open(path, "rt"):
        try:
            o = json.loads(linha)
        except ValueError:
            continue
        if "t" in o and o.get("robots"):
            S.append(o)
    if not S:
        return None
    R = max(len(o["robots"]) for o in S)
    S = [o for o in S if len(o["robots"]) == R]
    S.sort(key=lambda o: o["t"])
    out = []
    for o in S:  # descarta repetidas
        if not out or o["t"] > out[-1]["t"] + 1e-6:
            out.append(o)
    S = out
    N = len(S)
    P = np.zeros((N, R, 3))
    held = np.zeros((N, R), bool)
    for k, o in enumerate(S):
        for r in o["robots"]:
            P[k, r["id"]] = (r["x"], r["y"], r.get("yaw", 0.0))
            held[k, r["id"]] = bool(r.get("held"))
    d = {"t": np.array([o["t"] for o in S]), "ts": np.array([o.get("ts", 0.0) for o in S]),
         "fase": np.array([(o.get("match") or {}).get("phase", "") for o in S]),
         "clock": np.array([(o.get("match") or {}).get("clock") or 0.0 for o in S], float),
         "score": np.array([(o.get("match") or {}).get("score") or [0, 0] for o in S], float),
         "msg": [o.get("message") or "" for o in S], "P": P, "held": held, "R": R}
    if jogo == "soccer":
        d["puck"] = np.array([(o["puck"]["x"], o["puck"]["y"]) if o.get("puck") else (np.nan, np.nan) for o in S])
    if jogo == "ctf":
        fl = {"blue": np.full((N, 2), np.nan), "red": np.full((N, 2), np.nan)}
        for k, o in enumerate(S):
            for m in o.get("markers") or []:
                if m.get("kind") == "flag":
                    fl[m["team"]][k] = (m["x"], m["y"])
        d["flags"] = fl
        d["carrying"] = np.array([list(o.get("carrying") or [False] * R) for o in S], bool)
    return d


def carregar(jogo, usar_cache=True):
    """Lista de partidas do jogo (com cache em eng_reversa/cache/<jogo>.pkl)."""
    CACHE.mkdir(exist_ok=True)
    cp = CACHE / f"{jogo}.pkl"
    arquivos = sorted((DADOS / jogo).glob("*.jsonl.gz"), key=lambda p: int(p.name.split(".")[0]))
    if usar_cache and cp.exists() and cp.stat().st_mtime > max(p.stat().st_mtime for p in arquivos):
        return pickle.load(open(cp, "rb"))
    meta = {}
    with open(DADOS / "partidas.csv") as f:
        for r in csv.DictReader(f):
            if r["jogo"] == jogo:
                meta[int(r["partida"])] = r
    ms = []
    for p in arquivos:
        pid = int(p.name.split(".")[0])
        r = meta.get(pid)
        if r is None or r["azul"] == "?" or r["vermelho"] == "?":
            continue
        d = _ler(p, jogo)
        if d is None or len(d["t"]) < 30:
            continue
        d.update(id=pid, jogo=jogo, data=r["data"], azul=CURTO.get(r["azul"], r["azul"]),
                 vermelho=CURTO.get(r["vermelho"], r["vermelho"]))
        d["lado"] = lado_dos_robos(jogo, d["R"])
        d["equipe"] = [d["azul"] if s == "blue" else d["vermelho"] for s in d["lado"]]
        ms.append(d)
    pickle.dump(ms, open(cp, "wb"))
    return ms


def comandos(m, i, dt_min=0.04, dt_max=0.3, salto=0.08):
    """(v, w, ok) deduzidos para o robô i em cada amostra k (usando k e k+1); ok=False onde não dá."""
    P, t = m["P"][:, i], m["t"]
    N = len(t)
    v, w, ok = np.zeros(N), np.zeros(N), np.zeros(N, bool)
    dt = np.diff(t)
    dx, dy = np.diff(P[:, 0]), np.diff(P[:, 1])
    yaw = P[:-1, 2]
    lin = (dx * np.cos(yaw) + dy * np.sin(yaw)) / np.maximum(dt, 1e-6)
    ang = wrap(np.diff(P[:, 2])) / np.maximum(dt, 1e-6)
    good = (dt > dt_min) & (dt < dt_max) & (np.hypot(dx, dy) < salto) & ~m["held"][:-1, i] & ~m["held"][1:, i]
    good &= (m["fase"][:-1] == "jogo") & (m["fase"][1:] == "jogo")
    v[:-1], w[:-1], ok[:-1] = lin, ang, good
    return v, w, ok


def livre(m, i, d_rob=0.095, x_lim=None, y_lim=None):
    """True onde o robô i não encosta em outro robô (< d_rob) nem na parede (|x| > x_lim ou |y| > y_lim):
    em contato o movimento deixa de refletir o comando."""
    P = m["P"]
    ok = np.ones(len(m["t"]), bool)
    for j in range(P.shape[1]):
        if j != i:
            ok &= np.hypot(P[:, i, 0] - P[:, j, 0], P[:, i, 1] - P[:, j, 1]) > d_rob
    if x_lim is not None:
        ok &= np.abs(P[:, i, 0]) < x_lim
    if y_lim is not None:
        ok &= np.abs(P[:, i, 1]) < y_lim
    ok[:-1] &= ok[1:]
    return ok


def frame(lado):
    """Sinal do referencial do time: azul ataca +x (s=+1), vermelho -x (s=-1)."""
    return 1.0 if lado == "blue" else -1.0


def para_time(p, s):
    """(x, y, yaw) do mundo -> referencial do time (gira 180 graus se s=-1)."""
    p = np.asarray(p, float).copy()
    p[..., 0] *= s
    p[..., 1] *= s
    if p.shape[-1] >= 3 and s < 0:
        p[..., 2] = wrap(p[..., 2] + np.pi)
    return p


if __name__ == "__main__":
    for j in ("sumo", "ctf", "soccer"):
        ms = carregar(j, usar_cache=False)
        n = {}
        for m in ms:
            for e in {m["azul"], m["vermelho"]}:
                n[e] = n.get(e, 0) + 1
        print(j, len(ms), "partidas", n)
