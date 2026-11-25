"""Carrega as partidas do Nyx em arrays no referencial do time do Nyx (base em (-0.62, 0), ataca +x)."""
import csv, gzip, json, math, os
import numpy as np
G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gravacoes_ao_vivo", "ctf")

def wrap(a): return (a + np.pi) % (2*np.pi) - np.pi

def partidas(time="Nyx"):
    out = []
    for r in csv.DictReader(open(os.path.join(G, "partidas.csv"))):
        if time in r["azul"]: out.append((int(r["partida"]), 0, r["vermelho"]))
        elif time in r["vermelho"]: out.append((int(r["partida"]), 1, r["azul"]))
    return out

def carregar(mid, lado):
    S = []
    for l in gzip.open(os.path.join(G, f"{mid}.jsonl.gz"), "rt"):
        try: o = json.loads(l)
        except ValueError: continue
        if "t" in o and len(o.get("robots") or []) == 4: S.append(o)
    S.sort(key=lambda o: o["t"]); D = []
    for o in S:
        if not D or o["t"] > D[-1]["t"] + 1e-6: D.append(o)
    n = len(D); P = np.zeros((n, 4, 3)); H = np.zeros((n, 4), bool); C = np.zeros((n, 4), bool); F = np.zeros((n, 2, 2))
    for k, o in enumerate(D):
        for r in o["robots"]: P[k, r["id"]] = (r["x"], r["y"], r["yaw"]); H[k, r["id"]] = r.get("held", False)
        c = o.get("carrying") or [False]*4; C[k, :len(c)] = c[:4]
        fl = {m["team"]: (m["x"], m["y"]) for m in o.get("markers") or [] if m.get("kind") == "flag"}
        F[k, 0] = fl.get("blue", (-0.62, 0)); F[k, 1] = fl.get("red", (0.62, 0))
    t = np.array([o["t"] for o in D]); ph = np.array([(o.get("match") or {}).get("phase") == "jogo" for o in D])
    sc = np.array([(o.get("match") or {}).get("score") or [0, 0] for o in D])
    # referencial do Nyx
    if lado == 1:
        P = P[:, [2, 3, 0, 1]].copy(); P[..., :2] *= -1; P[..., 2] = wrap(P[..., 2] + np.pi)
        H = H[:, [2, 3, 0, 1]]; C = C[:, [2, 3, 0, 1]]; F = -F[:, [1, 0]]; sc = sc[:, ::-1]
    # robôs 0,1 = Nyx; 2,3 = adversário; F[:,0] = bandeira do Nyx, F[:,1] = do adversário
    # papéis: atacante = mais tempo no campo inimigo
    a = 0 if (P[ph, 0, 0] > 0).mean() >= (P[ph, 1, 0] > 0).mean() else 1
    o_att = 2 if (P[ph, 2, 0] < 0).mean() >= (P[ph, 3, 0] < 0).mean() else 3
    idx = [a, 1-a, o_att, 5-o_att]
    return dict(t=t, P=P[:, idx], H=H[:, idx], C=C[:, idx], F=F, jogo=ph, score=sc, mid=mid)
