"""Ajusta a máquina de estados de cada equipe e gera os nós estimados.

    $PY ajustar.py sumo [--equipes R2D2 Nyx ...] [--aleat 400]
    -> estimados/<jogo>/<equipe>.py  (nó no formato do kit, com os parâmetros da equipe)
       estimados/<jogo>/ajuste.json  (parâmetros, erro de treino / validação / prever a média)
"""
from __future__ import annotations

import argparse
import json
import pprint
import sys
import types
from pathlib import Path

import numpy as np

import er_ajuste as A
import er_dados as E

AQUI = Path(__file__).resolve().parent
MODELOS = {"sumo": AQUI / "modelos" / "sumo_maquina.py", "ctf": AQUI / "modelos" / "ctf_maquina.py",
           "soccer": AQUI / "modelos" / "soccer_maquina.py"}
# partidas excluídas: versões antigas do código (o R2D2 trocou o ctf.py às 10:51 de 25/11)
EXCLUIR = {("ctf", "R2D2"): {639, 644, 651, 654, 657}}


def _stubs():
    from eval_sumo import _stub_ros  # rospy e mensagens falsos

    _stub_ros()
    for mod, cls in (("geometry_msgs.msg", "PointStamped"), ("geometry_msgs.msg", "Twist"), ("std_msgs.msg", "Bool"),
                     ("std_msgs.msg", "String"), ("nav_msgs.msg", "Odometry")):
        if not hasattr(sys.modules[mod], cls):
            setattr(sys.modules[mod], cls, type(cls, (), {}))
    m = types.ModuleType("psr_hero.utils")
    m.V_MAX, m.W_MAX = E.V_MAX, E.W_MAX
    m.clamp = lambda x, lo, hi: max(lo, min(hi, x))
    m.wrap_angle = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    m.pose_from_odom = lambda msg: msg
    sys.modules.setdefault("psr_hero", types.ModuleType("psr_hero"))
    sys.modules["psr_hero.utils"] = m


def carregar_modelo(jogo, params=None, equipe="?"):
    _stubs()
    src = MODELOS[jogo].read_text()
    src = src.replace("__PARAMS__", repr(params or {})).replace("__EQUIPE__", equipe)
    mod = types.ModuleType(f"modelo_{jogo}")
    exec(compile(src, str(MODELOS[jogo]), "exec"), mod.__dict__)
    return mod, src


def gerar_no(jogo, equipe, params):
    _, src = carregar_modelo(jogo, params, equipe)
    src = src.replace(f"PARAMS = {params!r}", "PARAMS = " + pprint.pformat(params, width=110, sort_dicts=False))
    out = AQUI / "estimados" / jogo / f"{equipe}.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(src)
    return out


# ---------------------------------------------------------------- features por jogo
def features_sumo(ms):
    import a_sumo

    T = {}
    for m in ms:
        tr = np.full(len(m["t"]), np.nan)
        for k0 in a_sumo.rodadas(m):
            tr[k0:] = m["t"][k0:] - m["t"][k0]
        for i in range(2):
            v, w, ok = E.comandos(m, i)
            ok &= ~np.isnan(tr)
            D = T.setdefault(m["equipe"][i], {c: [] for c in ("x", "y", "yaw", "ox", "oy", "oyaw", "tr", "vr", "wr", "mid")})
            for c, x in (("x", m["P"][:, i, 0]), ("y", m["P"][:, i, 1]), ("yaw", m["P"][:, i, 2]),
                         ("ox", m["P"][:, 1 - i, 0]), ("oy", m["P"][:, 1 - i, 1]), ("oyaw", m["P"][:, 1 - i, 2]), ("tr", tr), ("vr", v), ("wr", w),
                         ("mid", np.full(len(v), m["id"]))):
                D[c].append(x[ok])
    return {e: {c: np.concatenate(x) for c, x in D.items()} for e, D in T.items()}


def features_ctf(ms, equipes):
    """{equipe: [(papel, F), ...]}; papel pelo launch do kit: robôs 0 e 2 atacantes, 1 e 3 defensores."""
    T = {}
    for m in ms:
        if m["R"] != 4:
            continue
        t = m["t"]
        for lado, ids, inim in (("blue", (0, 1), (2, 3)), ("red", (2, 3), (0, 1))):
            s = E.frame(lado)
            e = m["equipe"][ids[0]]
            P = {i: E.para_time(m["P"][:, i], s) for i in range(4)}
            # invasor = adversário com menor x (no referencial do time)
            a_mais = P[inim[0]][:, 0] <= P[inim[1]][:, 0]
            o1 = np.where(a_mais[:, None], P[inim[0]], P[inim[1]])
            o2 = np.where(a_mais[:, None], P[inim[1]], P[inim[0]])
            h1 = np.where(a_mais, m["held"][:, inim[0]], m["held"][:, inim[1]])
            h2 = np.where(a_mais, m["held"][:, inim[1]], m["held"][:, inim[0]])
            of = m["flags"][lado] * s
            ef = m["flags"]["red" if lado == "blue" else "blue"] * s
            for papel, i in (("att", ids[0]), ("def", ids[1])):
                v, w, ok = E.comandos(m, i)
                ok &= ~np.isnan(of[:, 0]) & ~np.isnan(ef[:, 0]) & E.livre(m, i, x_lim=0.76, y_lim=0.76)
                # memória: fim da última vez preso, e onde foi preso
                h = m["held"][:, i]
                t_rel = np.full(len(t), 1e9)
                hx, hy = np.zeros(len(t)), np.zeros(len(t))
                tr, px, py = -1e9, 0.0, 0.0
                for k in range(len(t)):
                    if k and h[k] and not h[k - 1]:
                        px, py = P[i][k, 0], P[i][k, 1]
                    if k and h[k - 1] and not h[k]:
                        tr = t[k]
                    t_rel[k], hx[k], hy[k] = t[k] - tr, px, py
                D = T.setdefault(e, {}).setdefault(papel, {c: [] for c in (
                    "x", "y", "yaw", "o1x", "o1y", "o1h", "o2x", "o2y", "o2h", "efx", "efy", "ofx", "ofy", "carry",
                    "t_rel", "hx_prev", "hy_prev", "vr", "wr", "mid")})
                for c, x in (("x", P[i][:, 0]), ("y", P[i][:, 1]), ("yaw", P[i][:, 2]), ("o1x", o1[:, 0]),
                             ("o1y", o1[:, 1]), ("o1h", h1), ("o2x", o2[:, 0]), ("o2y", o2[:, 1]), ("o2h", h2),
                             ("efx", ef[:, 0]), ("efy", ef[:, 1]), ("ofx", of[:, 0]), ("ofy", of[:, 1]),
                             ("carry", m["carrying"][:, i]), ("t_rel", t_rel), ("hx_prev", hx), ("hy_prev", hy),
                             ("vr", v), ("wr", w), ("mid", np.full(len(v), m["id"]))):
                    D[c].append(x[ok])
    return {e: [(p, {c: np.concatenate(x) for c, x in D.items()}) for p, D in T[e].items()] for e in equipes}


def features_soccer(ms, equipes):
    """{equipe: [("gk", F), ("p1", F), ("p2", F)]}: robô 0/1/2 do time."""
    T = {}
    for m in ms:
        bola_ok = ~np.isnan(m["puck"][:, 0])
        for lado, ids in (("blue", (0, 1, 2)), ("red", (3, 4, 5))):
            s = E.frame(lado)
            e = m["equipe"][ids[0]]
            B = m["puck"] * s
            BV = np.zeros_like(B)  # velocidade da bola (diferença para a amostra anterior)
            BV[1:] = (B[1:] - B[:-1]) / np.maximum(np.diff(m["t"]), 1e-3)[:, None]
            BV = np.nan_to_num(np.clip(BV, -2, 2))
            P = {i: E.para_time(m["P"][:, i], s) for i in ids}
            d = {i: np.hypot(P[i][:, 0] - B[:, 0], P[i][:, 1] - B[:, 1]) for i in ids}
            for k, papel in enumerate(("gk", "p1", "p2")):
                i = ids[k]
                v, w, ok = E.comandos(m, i)
                ok &= bola_ok & E.livre(m, i, x_lim=0.86, y_lim=0.56)
                outro = d[ids[3 - k]] if k else np.full(len(v), 9.0)
                D = T.setdefault(e, {}).setdefault(papel, {c: [] for c in ("x", "y", "yaw", "bx", "by", "bvx", "bvy", "dm", "dt", "vr", "wr", "mid")})
                for c, x in (("x", P[i][:, 0]), ("y", P[i][:, 1]), ("yaw", P[i][:, 2]), ("bx", B[:, 0]), ("by", B[:, 1]),
                             ("bvx", BV[:, 0]), ("bvy", BV[:, 1]), ("dm", d[i]), ("dt", outro), ("vr", v), ("wr", w), ("mid", np.full(len(v), m["id"]))):
                    D[c].append(x[ok])
    return {e: [(p, {c: np.concatenate(x) for c, x in D.items()}) for p, D in T[e].items()] for e in equipes}


FEATURES = {"sumo": features_sumo, "ctf": features_ctf, "soccer": features_soccer}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jogo", choices=["sumo", "ctf", "soccer"])
    p.add_argument("--equipes", nargs="*", default=E.EQUIPES)
    p.add_argument("--aleat", type=int, default=400)
    a = p.parse_args()
    ms = E.carregar(a.jogo)
    fam, _ = carregar_modelo(a.jogo)
    out_json = AQUI / "estimados" / a.jogo / "ajuste.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    todos = json.loads(out_json.read_text()) if out_json.exists() else {}
    if a.jogo == "sumo":
        T = FEATURES["sumo"](ms)
        grupos = {e: [(e, T[e])] for e in a.equipes}
    else:
        grupos = FEATURES[a.jogo](ms, a.equipes)
    for e in a.equipes:
        pares = {}
        for chave, F in grupos[e]:
            ex = EXCLUIR.get((a.jogo, e), set())
            sel = ~np.isin(F["mid"], list(ex))
            F = {c: (x[sel] if isinstance(x, np.ndarray) and len(x) == len(sel) else x) for c, x in F.items()}
            f = fam
            if hasattr(fam, "FAMILIAS"):
                pol, esp, pad = fam.FAMILIAS[chave]
                f = types.SimpleNamespace(politica=pol, ESPACO=esp, PADRAO=pad)
            P, res = A.ajustar(f, F, F["vr"], F["wr"], F["mid"], n_aleat=a.aleat)
            pares[chave] = (P, res)
            print(f"{e} {chave}: erro validação {res['validacao']:.3f} (v {res['validacao_v_w'][0]:.3f} w {res['validacao_v_w'][1]:.3f})"
                  f"  prever a média {res['constante_validacao']:.3f}  treino {res['treino']:.3f}  n={res['n']}", flush=True)
            print("   ", P, flush=True)
        if a.jogo == "sumo":
            P, res = pares[e]
            todos[e] = {"params": P, **res}
            gerar_no("sumo", e, P)
        else:
            todos[e] = {str(k): {"params": P, **res} for k, (P, res) in pares.items()}
            gerar_no(a.jogo, e, {str(k[1]) if isinstance(k, tuple) else str(k): P for k, (P, _) in pares.items()})
        out_json.write_text(json.dumps(todos, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
